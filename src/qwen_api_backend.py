"""Qwen3-ASR cloud backend via any OpenAI-compatible chat/completions endpoint.

Intended for Aliyun Bailian (DashScope) dedicated workspaces:
    ASR_API_BASE="https://ws-xxxx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    ASR_API_KEY="sk-..."
    ASR_MODEL="qwen3-asr-flash"

Audio is sent as base64 WAV inside an input_audio content block; the endpoint
returns the transcript as chat content. Long recordings are split into
_API_CHUNK_SECONDS pieces and merged with cumulative offsets.
"""

from __future__ import annotations

import base64
import os
import subprocess
import time
import wave

from src.transcriber import Segment

# 16 kHz mono 16-bit WAV = 32 KB/s; 60 s ≈ 1.9 MB raw / 2.6 MB base64,
# safely below OpenAI-compatible request body limits.
_API_CHUNK_SECONDS = 60
_API_MAX_ATTEMPTS = 3


def _env(name: str) -> str:
    return os.getenv(name, "").strip().strip('"')


def _wav_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as audio:
            return audio.getnframes() / audio.getframerate()
    except (wave.Error, EOFError):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True,
        )
        return float(result.stdout.strip())


def _split_audio(path: str, chunk_seconds: int = _API_CHUNK_SECONDS) -> list[tuple[str, float]]:
    """Split long audio into (path, duration) pieces; short files pass through."""
    duration = _wav_duration(path)
    if duration <= chunk_seconds:
        return [(path, duration)]

    base, _ = os.path.splitext(path)
    pieces = []
    start = 0.0
    index = 0
    while start < duration:
        piece = f"{base}_apichunk{index:03d}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ss", str(start), "-t", str(chunk_seconds),
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", piece],
            capture_output=True, text=True,
        )
        pieces.append((piece, _wav_duration(piece)))
        start += chunk_seconds
        index += 1
    return pieces


class QwenApiTranscriber:
    """Transcriber that calls an OpenAI-compatible Qwen3-ASR deployment."""

    def __init__(self, model_id: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-ASR API mode requires the openai package. "
                "Run: pip install -r requirements.txt"
            ) from exc

        self.api_base = _env("ASR_API_BASE")
        self.api_key = _env("ASR_API_KEY")
        self.model = model_id or _env("ASR_MODEL") or "qwen3-asr-flash"
        if not self.api_base or not self.api_key:
            raise RuntimeError(
                "Qwen3-ASR API mode needs ASR_API_BASE and ASR_API_KEY set in .env\n"
                "  ASR_API_BASE example: https://ws-xxxx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
            )
        self.client = OpenAI(base_url=self.api_base, api_key=self.api_key, timeout=120)
        print(f"  Using Qwen3-ASR API: {self.model} @ {self.api_base}")

    def _request(self, audio_b64: str, language_name: str | None) -> str:
        data_uri = f"data:audio/wav;base64,{audio_b64}"
        # Verified against Bailian workspaces: the dedicated asr task accepts a
        # single {"type": "audio", "audio": <data-uri>} item and rejects text
        # blocks, so language is auto-detected. Other gateways expect the
        # OpenAI standard input_audio / vLLM audio_url shapes — try in order.
        audio_blocks = [
            {"type": "audio", "audio": data_uri},
            {"type": "input_audio", "input_audio": {"data": data_uri, "format": "wav"}},
            {"type": "audio_url", "audio_url": {"url": data_uri}},
        ]
        last_error: Exception | None = None
        for block in audio_blocks:
            try:
                message = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": [block]}],
                    temperature=0,
                )
                return message.choices[0].message.content or ""
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                message_text = str(exc)
                if status not in (400, 404, 422, 500) and "audio" not in message_text.lower():
                    raise  # network / auth / rate-limit — not a format problem
                last_error = exc
        raise last_error  # type: ignore[misc]

    def _request_with_retry(self, audio_b64: str, language_name: str | None) -> str:
        last_error: Exception | None = None
        for attempt in range(_API_MAX_ATTEMPTS):
            try:
                return self._request(audio_b64, language_name)
            except Exception as exc:  # network / 5xx / timeout — retry with backoff
                last_error = exc
                if attempt + 1 < _API_MAX_ATTEMPTS:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Qwen3-ASR API failed after {_API_MAX_ATTEMPTS} attempts") from last_error

    @staticmethod
    def _parse(raw: str, language_name: str | None) -> str:
        """Strip 'language X<asr_text>...' wrappers; keep plain transcripts as-is."""
        if not raw:
            return ""
        text = raw.strip()
        # Plain transcripts (Bailian verified) skip the qwen_asr parser —
        # importing it pulls in torch, costing ~15s and hundreds of MB RSS.
        if "<asr_text>" in text or text.startswith("language"):
            try:
                from qwen_asr import parse_asr_output

                _, text = parse_asr_output(raw)
            except Exception:
                pass
        return text.strip()

    def transcribe(self, audio_path: str, language: str = "zh") -> list[Segment]:
        from src.qwen_asr_backend import _language_name

        language_name = _language_name(language)
        pieces = _split_audio(audio_path)
        segments: list[Segment] = []
        offset = 0.0
        temp_paths = [p for p, _ in pieces if p != audio_path]
        try:
            total = len(pieces)
            for index, (piece, duration) in enumerate(pieces, start=1):
                suffix = f" ({index}/{total})" if total > 1 else ""
                print(f"  Transcribing via Qwen3-ASR API: {piece}{suffix}")
                with open(piece, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("ascii")
                raw = self._request_with_retry(audio_b64, language_name)
                text = self._parse(raw, language_name)
                if text:
                    segments.append(Segment(offset, offset + duration, text))
                offset += duration
        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)
        print(f"  Transcription complete: {len(segments)} segment(s)")
        return segments

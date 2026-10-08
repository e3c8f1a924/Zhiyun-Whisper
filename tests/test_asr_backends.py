"""ASR routing, Qwen SDK contracts, subtitles, and monitor reuse without weights."""

import io
import tempfile
import unittest
import wave
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from click.testing import CliRunner

from main import cli
from src.crawler import Lesson
from src.live_monitor import monitor_loop
from src.qwen_asr_backend import QwenTranscriber, _aligned_segments, _language_name
from src.qwen_api_backend import QwenApiTranscriber
from src.transcriber import DEFAULT_MODEL, Segment, load_local_model, transcribe_local


def stamp(text, start, end):
    return SimpleNamespace(text=text, start_time=start, end_time=end)


@contextmanager
def fake_qwen(cuda=False, bf16=False):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: cuda, is_bf16_supported=lambda: bf16,
    )
    torch.float32, torch.float16, torch.bfloat16 = "float32", "float16", "bfloat16"
    sdk = ModuleType("qwen_asr")
    sdk.Qwen3ASRModel = Mock()
    with patch.dict("sys.modules", {"torch": torch, "qwen_asr": sdk}):
        yield sdk.Qwen3ASRModel.from_pretrained


class LocalModelTests(unittest.TestCase):
    def test_qwen_default_and_size_aliases(self):
        for alias, size in (
            (DEFAULT_MODEL, "1.7B"), ("1.7b", "1.7B"), ("0.6b", "0.6B"),
            ("qwen3-asr-0.6b", "0.6B"), ("Qwen/Qwen3-ASR-1.7B", "1.7B"),
            ("Qwen/Qwen3-ASR-0.6B", "0.6B"),
        ):
            with self.subTest(alias=alias), patch("src.qwen_asr_backend.QwenTranscriber") as qwen:
                model = load_local_model(alias)
                self.assertIs(model, qwen.return_value)
                qwen.assert_called_once_with(
                    f"Qwen/Qwen3-ASR-{size}", device="auto", batch_size=1, return_timestamps=True,
                )

    def test_whisper_remains_available_without_qwen(self):
        sdk = ModuleType("faster_whisper")
        sdk.WhisperModel = Mock()
        with patch.dict("sys.modules", {"faster_whisper": sdk, "qwen_asr": None}), redirect_stdout(io.StringIO()):
            model = load_local_model("small", device="cpu")
            sdk.WhisperModel.assert_called_once_with("small", device="cpu", compute_type="int8")
            with patch("src.transcriber.transcribe_with_model", return_value=[]) as transcribe:
                self.assertEqual(model.transcribe("chunk.wav"), [])
                transcribe.assert_called_once_with(sdk.WhisperModel.return_value, "chunk.wav", "zh", 16)

    def test_recordings_request_timestamps_and_pass_options(self):
        with patch("src.transcriber.load_local_model") as load:
            transcribe_local("recording.wav", model_size="0.6b", batch_size=2, language="en")
            load.assert_called_once_with("0.6b", "auto", 2, return_timestamps=True)
            load.return_value.transcribe.assert_called_once_with("recording.wav", "en")

    def test_invalid_batch_and_qwen_size_are_rejected(self):
        for options in ({"batch_size": 0}, {"batch_size": -1}, {"model_size": "qwen3-asr-7b"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                load_local_model(**options)


class QwenAdapterTests(unittest.TestCase):
    def test_devices_dtypes_and_recording_aligner(self):
        for cuda, bf16, device, dtype in (
            (False, False, "cpu", "float32"),
            (True, False, "cuda:0", "float16"),
            (True, True, "cuda:0", "bfloat16"),
        ):
            with self.subTest(device=device, dtype=dtype), fake_qwen(cuda, bf16) as load, redirect_stdout(io.StringIO()):
                QwenTranscriber("Qwen/Qwen3-ASR-1.7B", batch_size=2)
                load.assert_called_once_with(
                    "Qwen/Qwen3-ASR-1.7B", device_map=device, dtype=dtype,
                    max_inference_batch_size=2, max_new_tokens=4096,
                    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                    forced_aligner_kwargs={"device_map": device, "dtype": dtype},
                )

    def test_recording_converts_sdk_timestamps(self):
        with fake_qwen() as load, redirect_stdout(io.StringIO()):
            load.return_value.transcribe.return_value = [SimpleNamespace(
                text="你好。", time_stamps=[stamp("你", 2, 2.2), stamp("好", 2.2, 2.5)],
            )]
            model = QwenTranscriber("Qwen/Qwen3-ASR-0.6B")
            self.assertEqual(model.transcribe("recording.wav"), [Segment(2, 2.5, "你好。")])
            load.return_value.transcribe.assert_called_once_with(
                audio="recording.wav", language="Chinese", return_time_stamps=True,
            )

    def test_live_chunks_reuse_model_without_aligner(self):
        with tempfile.TemporaryDirectory() as tmp, fake_qwen() as load, redirect_stdout(io.StringIO()):
            audio_path = str(Path(tmp) / "chunk.wav")
            with wave.open(audio_path, "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\0" * 8000)
            load.return_value.transcribe.return_value = [SimpleNamespace(text="测试。", time_stamps=None)]
            model = QwenTranscriber("Qwen/Qwen3-ASR-1.7B", return_timestamps=False)
            for _ in range(2):
                self.assertEqual(model.transcribe(audio_path), [Segment(0, 0.5, "测试。")])
            load.assert_called_once()
            self.assertNotIn("forced_aligner", load.call_args.kwargs)
            self.assertEqual(load.return_value.transcribe.call_count, 2)
            self.assertFalse(load.return_value.transcribe.call_args.kwargs["return_time_stamps"])

    def test_silence_does_not_require_alignment(self):
        with fake_qwen() as load, redirect_stdout(io.StringIO()):
            load.return_value.transcribe.return_value = [SimpleNamespace(text=" ", time_stamps=None)]
            self.assertEqual(QwenTranscriber("model").transcribe("silence.wav"), [])

    def test_language_codes_and_auto_detection(self):
        for value, expected in (("zh", "Chinese"), ("en", "English"), ("yue", "Cantonese"),
                                ("zh-TW", "Chinese"), ("Japanese", "Japanese"), ("auto", None)):
            with self.subTest(value=value):
                self.assertEqual(_language_name(value), expected)

    def test_missing_sdk_has_install_instructions(self):
        with patch.dict("sys.modules", {"qwen_asr": None}), self.assertRaisesRegex(RuntimeError, "pip install -r requirements.txt"):
            QwenTranscriber("model")


class QwenSubtitlesTests(unittest.TestCase):
    def test_mixed_text_preserves_punctuation_and_long_audio_offsets(self):
        segments = _aligned_segments("你好，World！再见。", [
            stamp("你", 180.1, 180.3), stamp("好", 180.3, 180.6),
            stamp("World", 180.6, 181), stamp("再", 190.1, 190.3), stamp("见", 190.3, 190.5),
        ])
        self.assertEqual(segments, [Segment(180.1, 181, "你好，World！"), Segment(190.1, 190.5, "再见。")])

    def test_aligner_punctuation_removal_does_not_damage_transcript(self):
        text = "It's 3.14, isn't it?"
        segments = _aligned_segments(text, [
            stamp("It's", 0, 0.2), stamp("314", 0.2, 0.4),
            stamp("isn't", 0.4, 0.6), stamp("it", 0.6, 0.8),
        ])
        self.assertEqual(" ".join(s.text for s in segments), text)

    def test_long_unpunctuated_chinese_is_grouped_into_readable_captions(self):
        text = "课" * 100
        segments = _aligned_segments(text, [stamp("课", i * .1, (i + 1) * .1) for i in range(100)])
        self.assertEqual("".join(s.text for s in segments), text)
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(len(s.text) <= 42 for s in segments))

    def test_missing_or_mismatched_alignment_fails_instead_of_losing_text(self):
        for items in (None, [], [stamp("错", 0, 1)], [stamp("你", 0, 1)]):
            with self.subTest(items=items), self.assertRaises(RuntimeError):
                _aligned_segments("你好", items)


class AsrCommandTests(unittest.TestCase):
    def test_recording_default_and_api_routing(self):
        for mode in ("local", "api"):
            with (
                self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp,
                patch("main._get_session"),
                patch("src.crawler.fetch_lessons", return_value=[Lesson("1", "Lecture", "https://example.invalid/video.mp4")]),
                patch("src.crawler.download_audio", return_value="audio.wav"),
                patch("src.transcriber.transcribe_local", return_value=[]) as local,
                patch("src.transcriber.transcribe_api", return_value=[]) as api,
                patch.dict("os.environ", {"OPENAI_API_KEY": "test"}),
            ):
                result = CliRunner().invoke(cli, ["transcribe", "https://example.invalid/?course_id=1", "--mode", mode, "-o", tmp])
                self.assertEqual(result.exit_code, 0, result.output)
                if mode == "local":
                    self.assertEqual(local.call_args.kwargs["model_size"], "qwen3-asr-1.7b")
                    self.assertIsNone(local.call_args.kwargs["batch_size"])
                    api.assert_not_called()
                else:
                    api.assert_called_once_with(audio_path="audio.wav", api_key="test", language="zh")
                    local.assert_not_called()

    def test_monitor_defaults_and_model_override(self):
        auth = ModuleType("src.auth")
        auth.refresh_token = Mock()
        config = {"ZJU_TOKEN": "test", "DINGTALK_WEBHOOK": "test", "DINGTALK_SECRET": "test",
                  "LLM_API_BASE": "test", "LLM_API_KEY": "test",
                  "LLM_FALLBACK_API_BASE": "", "LLM_FALLBACK_API_KEY": "", "LLM_FALLBACK_MODEL": ""}
        for args, model, batch in (([], "qwen3-asr-1.7b", None), (["--model", "0.6b", "--batch-size", "2"], "0.6b", 2),
                                   (["--model", "small"], "small", None)):
            with (
                self.subTest(args=args), patch.dict("sys.modules", {"src.auth": auth}), patch.dict("os.environ", config),
                patch("src.live_monitor.check_llm_apis"),
                patch("main._get_session"), patch("src.live_monitor.monitor_loop") as monitor,
            ):
                result = CliRunner().invoke(cli, ["monitor", "--course-id", "1", *args])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(monitor.call_args.kwargs["model_size"], model)
                self.assertEqual(monitor.call_args.kwargs["batch_size"], batch)

    def test_invalid_batch_is_rejected_before_startup(self):
        for command in (["monitor"], ["transcribe", "https://example.invalid"]):
            result = CliRunner().invoke(cli, [*command, "--batch-size", "0"])
            self.assertEqual(result.exit_code, 2)


class MonitorLlmConfigTests(unittest.TestCase):
    def invoke(self, fallback_env, check_result=None):
        auth = ModuleType("src.auth")
        auth.refresh_token = Mock()
        env = {
            "ZJU_TOKEN": "test", "DINGTALK_WEBHOOK": "test", "DINGTALK_SECRET": "test",
            "LLM_API_BASE": "https://primary.invalid/v1", "LLM_API_KEY": "primary-key", "LLM_MODEL": "primary-model",
            **fallback_env,
        }
        with (
            patch.dict("sys.modules", {"src.auth": auth}), patch.dict("os.environ", env, clear=True),
            patch("src.live_monitor.check_llm_apis", return_value=check_result or {"primary": True}) as check,
            patch("main._get_session") as session, patch("src.live_monitor.monitor_loop") as monitor,
        ):
            def checked_before_startup(**kwargs):
                session.assert_not_called()
                monitor.assert_not_called()
                return check_result or {"primary": True}

            check.side_effect = checked_before_startup
            result = CliRunner().invoke(cli, ["monitor", "--course-id", "1"])
        return result, monitor, session, check

    def test_missing_or_empty_fallback_config_keeps_it_disabled(self):
        for env in ({}, {"LLM_FALLBACK_API_BASE": "", "LLM_FALLBACK_API_KEY": "", "LLM_FALLBACK_MODEL": ""}):
            with self.subTest(env=env):
                result, monitor, _, check = self.invoke(env)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertNotIn("fallback", monitor.call_args.kwargs["llm_config"])
                check.assert_called_once_with(**monitor.call_args.kwargs["llm_config"], debug=False)

    def test_complete_fallback_config_is_forwarded_separately(self):
        result, monitor, _, check = self.invoke({
            "LLM_FALLBACK_API_BASE": ' "https://fallback.invalid/v1" ',
            "LLM_FALLBACK_API_KEY": ' "fallback-key" ', "LLM_FALLBACK_MODEL": ' "fallback-model" ',
        })
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(monitor.call_args.kwargs["llm_config"], {
            "api_base": "https://primary.invalid/v1", "api_key": "primary-key", "model": "primary-model",
            "fallback": {"api_base": "https://fallback.invalid/v1", "api_key": "fallback-key", "model": "fallback-model"},
        })
        self.assertNotIn("fallback-key", result.output)
        check.assert_called_once_with(**monitor.call_args.kwargs["llm_config"], debug=False)

    def test_partial_fallback_config_fails_before_network_or_model_startup(self):
        keys = ("LLM_FALLBACK_API_BASE", "LLM_FALLBACK_API_KEY", "LLM_FALLBACK_MODEL")
        for mask in range(1, 7):
            env = {key: "test" if mask & (1 << i) else "" for i, key in enumerate(keys)}
            with self.subTest(env=env):
                result, monitor, session, check = self.invoke(env)
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("must all be set", result.output)
                monitor.assert_not_called()
                session.assert_not_called()
                check.assert_not_called()

    def test_failed_startup_checks_do_not_disable_providers_or_abort_monitoring(self):
        for primary_ok, fallback_ok in ((False, True), (True, False), (False, False)):
            with self.subTest(primary_ok=primary_ok, fallback_ok=fallback_ok):
                result, monitor, _, check = self.invoke({
                    "LLM_FALLBACK_API_BASE": "https://fallback.invalid/v1",
                    "LLM_FALLBACK_API_KEY": "fallback-key", "LLM_FALLBACK_MODEL": "fallback-model",
                }, check_result={"primary": primary_ok, "fallback": fallback_ok})
                self.assertEqual(result.exit_code, 0, result.output)
                check.assert_called_once()
                monitor.assert_called_once()
                self.assertIn("fallback", monitor.call_args.kwargs["llm_config"])


class MonitorModelReuseTests(unittest.TestCase):
    def test_monitor_loads_once_and_transcribes_multiple_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"chunk_{i}.wav" for i in range(2)]
            for path in paths:
                path.touch()
            live = ("https://example.invalid/live.m3u8", "sub1")
            with (
                patch("src.live_monitor.fetch_live_url", side_effect=[live, live, None]),
                patch("src.live_monitor.stream_audio_chunks", return_value=iter(map(str, paths))),
                patch("src.transcriber.load_local_model") as load,
                patch("src.live_monitor.is_stream_ended", return_value=False),
                patch("src.live_monitor.check_keywords_pinyin", return_value=None),
                patch("time.sleep"), redirect_stdout(io.StringIO()),
            ):
                load.return_value.transcribe.return_value = [Segment(0, 1, "课堂内容")]
                monitor_loop(Mock(), "course1", [], 30, DEFAULT_MODEL, {}, {}, log_dir=tmp)
            load.assert_called_once_with(model_size=DEFAULT_MODEL, batch_size=None, return_timestamps=False)
            self.assertEqual(load.return_value.transcribe.call_count, 2)
            self.assertTrue(all(not path.exists() for path in paths))
            logs = list(Path(tmp).glob("course1_*.txt"))
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0].read_text().count("课堂内容"), 2)


class QwenApiTranscriberTests(unittest.TestCase):
    API_ENV = {"ASR_API_BASE": "https://asr.invalid/v1", "ASR_API_KEY": "sk-test",
               "ASR_MODEL": "qwen3-asr-flash"}

    @staticmethod
    def _fake_parse(raw, user_language=None):
        s = str(raw).strip()
        if "<asr_text>" in s:
            lang, _, rest = s.partition("<asr_text>")
            return lang.replace("language", "").strip(), rest.replace("</asr_text>", "").strip()
        return "", s

    def _make_wav(self, path, seconds):
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * int(16000 * seconds))

    def _completed(self, content):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def test_model_routing_resolves_api_aliases(self):
        for alias, model_id in (("qwen3-asr-flash", "qwen3-asr-flash"), ("asr-flash", None),
                                ("qwen-asr-api", None)):
            with self.subTest(alias=alias), patch("src.qwen_api_backend.QwenApiTranscriber") as api, redirect_stdout(io.StringIO()):
                load_local_model(alias)
                api.assert_called_once_with(model_id=model_id)

    def test_missing_config_has_setup_instructions(self):
        with patch.dict("os.environ", {"ASR_API_BASE": "", "ASR_API_KEY": ""}):
            self.assertRaisesRegex(RuntimeError, "ASR_API_BASE", QwenApiTranscriber)

    def test_transcribe_sends_base64_audio_and_strips_wrapper(self):
        stub = ModuleType("qwen_asr")
        stub.parse_asr_output = self._fake_parse
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", self.API_ENV), \
             patch.dict("sys.modules", {"qwen_asr": stub}), patch("openai.OpenAI") as openai_cls, \
             redirect_stdout(io.StringIO()):
            audio_path = Path(tmp) / "chunk.wav"
            self._make_wav(audio_path, 0.5)
            openai_cls.return_value.chat.completions.create.return_value = \
                self._completed("language Chinese<asr_text>你好世界</asr_text>")
            model = QwenApiTranscriber()
            segments = model.transcribe(str(audio_path), language="zh")

        self.assertEqual([s.text for s in segments], ["你好世界"])
        self.assertEqual((segments[0].start, segments[0].end), (0.0, 0.5))
        create = openai_cls.return_value.chat.completions.create
        create.assert_called_once()
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], "qwen3-asr-flash")
        # Bailian asr task: single audio block, no text instruction allowed
        (audio_block,) = kwargs["messages"][0]["content"]
        self.assertEqual(audio_block["type"], "audio")
        self.assertTrue(audio_block["audio"].startswith("data:audio/wav;base64,"))

    def test_plain_text_response_passes_through(self):
        stub = ModuleType("qwen_asr")
        stub.parse_asr_output = self._fake_parse
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", self.API_ENV), \
             patch.dict("sys.modules", {"qwen_asr": stub}), patch("openai.OpenAI") as openai_cls, \
             redirect_stdout(io.StringIO()):
            audio_path = Path(tmp) / "chunk.wav"
            self._make_wav(audio_path, 0.5)
            openai_cls.return_value.chat.completions.create.return_value = self._completed("直接返回的文本")
            model = QwenApiTranscriber()
            segments = model.transcribe(str(audio_path), language="zh")
        self.assertEqual([s.text for s in segments], ["直接返回的文本"])

    def test_audio_url_fallback_when_first_format_rejected(self):
        stub = ModuleType("qwen_asr")
        stub.parse_asr_output = self._fake_parse

        class Rejected(Exception):
            status_code = 400

        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", self.API_ENV), \
             patch.dict("sys.modules", {"qwen_asr": stub}), patch("openai.OpenAI") as openai_cls, \
             redirect_stdout(io.StringIO()):
            audio_path = Path(tmp) / "chunk.wav"
            self._make_wav(audio_path, 0.5)
            create = openai_cls.return_value.chat.completions.create
            create.side_effect = [Rejected("audio item not supported"), self._completed("回退成功")]
            model = QwenApiTranscriber()
            segments = model.transcribe(str(audio_path), language="zh")

        self.assertEqual([s.text for s in segments], ["回退成功"])
        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args_list[0].kwargs["messages"][0]["content"][0]["type"], "audio")
        self.assertEqual(create.call_args_list[1].kwargs["messages"][0]["content"][0]["type"], "input_audio")

    def test_long_audio_is_split_and_merged_with_offsets(self):
        stub = ModuleType("qwen_asr")
        stub.parse_asr_output = self._fake_parse
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", self.API_ENV), \
             patch.dict("sys.modules", {"qwen_asr": stub}), patch("openai.OpenAI") as openai_cls, \
             redirect_stdout(io.StringIO()):
            audio_path = Path(tmp) / "long.wav"
            self._make_wav(audio_path, 61)
            calls = []

            def fake_create(**kwargs):
                calls.append(kwargs)
                return self._completed(f"第{len(calls)}段")

            openai_cls.return_value.chat.completions.create.side_effect = fake_create
            model = QwenApiTranscriber()
            segments = model.transcribe(str(audio_path), language="zh")

        self.assertEqual(len(calls), 2)
        self.assertEqual([s.start for s in segments], [0.0, 60.0])
        self.assertEqual([s.end for s in segments], [60.0, 61.0])
        self.assertFalse(list(Path(tmp).glob("*_apichunk*")), "临时切片应已清理")


if __name__ == "__main__":
    unittest.main()

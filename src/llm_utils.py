"""
Shared LLM helpers for keyword alert confirmation and course summarization.
"""

from __future__ import annotations


def extract_final_answer(content: str | None) -> str:
    """Extract final text when a provider leaks thinking markers into content.

    Some providers omit the opening <think> tag, e.g. '否</think>否'.
    Only text after the last closing tag is a final answer. A separate
    reasoning_content field must never be used as a fallback answer.
    """
    if not isinstance(content, str):
        raise ValueError("LLM returned no final answer")

    answer = content.rsplit("</think>", 1)[-1].strip()
    if "<think>" in answer:
        raise ValueError("LLM returned an unfinished thinking block")
    if not answer:
        raise ValueError("LLM returned no final answer")
    return answer


def make_client(
    api_base: str,
    api_key: str,
    *,
    max_retries: int = 2,
    timeout: float | None = None,
):
    """Build an OpenAI-compatible client with SDK retries already applied."""
    from openai import OpenAI

    kwargs = {"api_key": api_key, "base_url": api_base, "max_retries": max_retries}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)


def redact_key(text: str, api_key: str | None) -> str:
    """Redact an API key from an error message before logging it."""
    if api_key:
        return text.replace(api_key, "<redacted>")
    return text

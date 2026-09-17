# core/generation/llm_client.py
import re
import time
from dataclasses import dataclass

from google import genai
from google.genai._gaos.lib.compat_errors import APIConnectionError, APITimeoutError, RateLimitError

_RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _ContentBlock:
    text: str


@dataclass
class _Response:
    content: list[_ContentBlock]
    usage: _Usage


class GeminiClient:
    def __init__(self):
        self._client = genai.Client()
        self.messages = self

    def create(self, model: str, max_tokens: int, messages: list[dict], system: str | None = None) -> _Response:
        max_retries = 8
        for attempt in range(max_retries + 1):
            try:
                interaction = self._client.interactions.create(
                    model=model,
                    system_instruction=system,
                    input=messages[0]["content"],
                )
                break
            except _RETRYABLE_ERRORS as exc:
                if attempt == max_retries:
                    raise
                time.sleep(_retry_delay_seconds(exc))
        input_tokens, output_tokens = _extract_usage(interaction)
        return _Response(
            content=[_ContentBlock(text=interaction.output_text)],
            usage=_Usage(input_tokens, output_tokens),
        )


def _retry_delay_seconds(exc: Exception) -> float:
    if not isinstance(exc, RateLimitError):
        return 5.0
    retry_after = getattr(exc, "response", None) and exc.response.headers.get("Retry-After")
    if retry_after:
        return float(retry_after) + 1
    match = re.search(r"retry in ([\d.]+)s", str(exc))
    if match:
        return float(match.group(1)) + 1
    return 30.0


def _extract_usage(interaction) -> tuple[int, int]:
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return 0, 0
    return getattr(usage, "total_input_tokens", 0) or 0, getattr(usage, "total_output_tokens", 0) or 0

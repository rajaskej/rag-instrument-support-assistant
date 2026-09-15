# core/generation/llm_client.py
from dataclasses import dataclass

from google import genai


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
        interaction = self._client.interactions.create(
            model=model,
            system_instruction=system,
            input=messages[0]["content"],
        )
        input_tokens, output_tokens = _extract_usage(interaction)
        return _Response(
            content=[_ContentBlock(text=interaction.output_text)],
            usage=_Usage(input_tokens, output_tokens),
        )


def _extract_usage(interaction) -> tuple[int, int]:
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return 0, 0
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is not None and output_tokens is not None:
        return input_tokens, output_tokens
    return getattr(usage, "prompt_token_count", 0), getattr(usage, "candidates_token_count", 0)

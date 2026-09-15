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
    return getattr(usage, "total_input_tokens", 0) or 0, getattr(usage, "total_output_tokens", 0) or 0

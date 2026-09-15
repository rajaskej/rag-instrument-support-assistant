# tests/test_llm_client.py
from core.generation.llm_client import GeminiClient


class _FakeUsage:
    def __init__(self):
        self.input_tokens = 12
        self.output_tokens = 7


class _FakeInteraction:
    def __init__(self, text):
        self.output_text = text
        self.usage = _FakeUsage()


class _FakeInteractions:
    def __init__(self, text):
        self._text = text
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return _FakeInteraction(self._text)


class _FakeGenaiClient:
    def __init__(self, text):
        self.interactions = _FakeInteractions(text)


def test_gemini_client_create_maps_to_anthropic_style_response(monkeypatch):
    import core.generation.llm_client as llm_client_module

    fake_genai_client = _FakeGenaiClient("Hello world")
    monkeypatch.setattr(llm_client_module.genai, "Client", lambda: fake_genai_client)

    client = GeminiClient()
    response = client.messages.create(
        model="gemini-3.8-flash",
        max_tokens=100,
        system="You are helpful.",
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert response.content[0].text == "Hello world"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 7
    assert fake_genai_client.interactions.last_call["model"] == "gemini-3.8-flash"
    assert fake_genai_client.interactions.last_call["system_instruction"] == "You are helpful."
    assert fake_genai_client.interactions.last_call["input"] == "Hi"

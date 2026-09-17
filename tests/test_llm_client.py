# tests/test_llm_client.py
import httpx
import pytest
from google.genai._gaos.lib.compat_errors import APIConnectionError, APITimeoutError, RateLimitError

from core.generation.llm_client import GeminiClient, _retry_delay_seconds


class _FakeUsage:
    def __init__(self):
        self.total_input_tokens = 12
        self.total_output_tokens = 7


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


def _rate_limit_error(message: str, retry_after_header: str | None = None) -> RateLimitError:
    request = httpx.Request("POST", "https://example.test/interactions")
    headers = {"Retry-After": retry_after_header} if retry_after_header else {}
    response = httpx.Response(429, headers=headers, request=request)
    return RateLimitError(message, response=response, body=None)


def _timeout_error() -> APITimeoutError:
    request = httpx.Request("POST", "https://example.test/interactions")
    return APITimeoutError(request)


class _FlakyThenSucceedsInteractions:
    def __init__(self, text: str, errors: list[Exception]):
        self._text = text
        self._errors = list(errors)
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        if self._errors:
            raise self._errors.pop(0)
        return _FakeInteraction(self._text)


class _AlwaysFailsInteractions:
    def __init__(self, error_factory):
        self._error_factory = error_factory
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        raise self._error_factory()


def test_create_retries_on_rate_limit_error_then_succeeds(monkeypatch):
    import core.generation.llm_client as llm_client_module

    interactions = _FlakyThenSucceedsInteractions("ok", [_rate_limit_error("please retry in 2.5s")])
    fake_genai_client = _FakeGenaiClient("ok")
    fake_genai_client.interactions = interactions
    monkeypatch.setattr(llm_client_module.genai, "Client", lambda: fake_genai_client)
    sleeps = []
    monkeypatch.setattr(llm_client_module.time, "sleep", lambda s: sleeps.append(s))

    client = GeminiClient()
    response = client.messages.create(model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}])

    assert response.content[0].text == "ok"
    assert interactions.call_count == 2
    assert sleeps == [3.5]


def test_create_retries_on_timeout_error_then_succeeds(monkeypatch):
    import core.generation.llm_client as llm_client_module

    interactions = _FlakyThenSucceedsInteractions("ok", [_timeout_error()])
    fake_genai_client = _FakeGenaiClient("ok")
    fake_genai_client.interactions = interactions
    monkeypatch.setattr(llm_client_module.genai, "Client", lambda: fake_genai_client)
    sleeps = []
    monkeypatch.setattr(llm_client_module.time, "sleep", lambda s: sleeps.append(s))

    client = GeminiClient()
    response = client.messages.create(model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}])

    assert response.content[0].text == "ok"
    assert interactions.call_count == 2
    assert sleeps == [5.0]


def test_create_raises_after_max_retries_exhausted(monkeypatch):
    import core.generation.llm_client as llm_client_module

    interactions = _AlwaysFailsInteractions(lambda: _rate_limit_error("retry in 0.1s"))
    fake_genai_client = _FakeGenaiClient("unused")
    fake_genai_client.interactions = interactions
    monkeypatch.setattr(llm_client_module.genai, "Client", lambda: fake_genai_client)
    monkeypatch.setattr(llm_client_module.time, "sleep", lambda s: None)

    client = GeminiClient()
    with pytest.raises(RateLimitError):
        client.messages.create(model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}])

    assert interactions.call_count == 9


def test_create_does_not_retry_on_non_retryable_error(monkeypatch):
    import core.generation.llm_client as llm_client_module

    class _Boom(Exception):
        pass

    class _RaisesBoomInteractions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            self.call_count += 1
            raise _Boom("nope")

    interactions = _RaisesBoomInteractions()
    fake_genai_client = _FakeGenaiClient("unused")
    fake_genai_client.interactions = interactions
    monkeypatch.setattr(llm_client_module.genai, "Client", lambda: fake_genai_client)

    client = GeminiClient()
    with pytest.raises(_Boom):
        client.messages.create(model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}])

    assert interactions.call_count == 1


def test_retry_delay_seconds_prefers_retry_after_header():
    exc = _rate_limit_error("no pattern here", retry_after_header="4")
    assert _retry_delay_seconds(exc) == 5.0


def test_retry_delay_seconds_parses_retry_in_pattern_from_message():
    exc = _rate_limit_error("You exceeded your quota. Please retry in 12.34s.")
    assert _retry_delay_seconds(exc) == pytest.approx(13.34)


def test_retry_delay_seconds_falls_back_to_default_for_unparseable_message():
    exc = _rate_limit_error("rate limited, no timing info")
    assert _retry_delay_seconds(exc) == 30.0


def test_retry_delay_seconds_returns_fixed_delay_for_non_rate_limit_errors():
    assert _retry_delay_seconds(APIConnectionError(request=httpx.Request("POST", "https://example.test"))) == 5.0

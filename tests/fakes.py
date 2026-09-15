# tests/fakes.py
class FakeLLMClient:
    def __init__(self, reply_text: str, input_tokens: int = 10, output_tokens: int = 10):
        self._reply_text = reply_text
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.messages = self

    def create(self, **kwargs):
        return _FakeResponse(self._reply_text, self._input_tokens, self._output_tokens)


class _FakeResponse:
    def __init__(self, text: str, input_tokens: int, output_tokens: int):
        self.content = [_FakeBlock(text)]
        self.usage = _FakeUsage(input_tokens, output_tokens)


class _FakeBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

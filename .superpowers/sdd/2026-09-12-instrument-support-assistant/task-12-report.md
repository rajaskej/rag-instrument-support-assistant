# Task 12 Report: Domain base interface and InstrumentSupportAgent

## What I implemented

- `domains/base.py`: `Response` dataclass (`draft`, `citations`, `confidence`, `escalate`) and the `Agent` Protocol with `.handle(request) -> Response`.
- `domains/instrument_support/agent.py`: `Ticket` dataclass (`symptom_or_error_code`, `model_number: str | None = None`, `free_text: str = ""`) and `InstrumentSupportAgent`, which:
  - Short-circuits to an immediate escalation (`confidence=0.0`, `citations=[]`, `escalate=True`) when `request.model_number` is set but not in `known_model_numbers` — without calling the retriever or the LLM client at all.
  - Otherwise builds a query from `symptom_or_error_code` + `free_text`, calls `retriever.retrieve(query, top_k=5, model_number_filter=...)`.
  - Escalates immediately (same shape) if retrieval returns no results.
  - Otherwise takes `results[0].score` as `confidence`, calls `generate_grounded_answer` (Task 11) for a grounded draft, and sets `escalate = confidence < threshold or answer.insufficient`. Appends an escalation note to the draft when escalating.
  - `.handle()` returns just the `Response`; `.handle_with_metadata()` returns `(Response, dict)` with `latency_ms`, `input_tokens`, `output_tokens` (0/0 on the short-circuit paths).

Implementation matches the brief's code exactly (I copied it as given — it aligned with the existing codebase's conventions, verified against `core/generation/generate.py`, `core/retrieval/pipeline.py`, `core/ingestion/models.py`, and `tests/fakes.py`).

## What I tested and results

Ran `tests/test_agent.py` (the 4 tests from the brief) and the full suite.

- `tests/test_agent.py`: 4 passed.
- Full suite (`pytest -v`, 33 tests total): 33 passed, 2 warnings — both pre-existing library `DeprecationWarning`s from `chromadb`/`google-genai` internals unrelated to this task's code (same warnings appear on the pre-Task-12 baseline runs, e.g. in `tests/test_generate.py`).
- Additional manual verification (ad hoc script, not committed): instantiated the agent with spy `retriever`/`client` objects that raise `AssertionError` if called, then called `.handle()` with an unknown model number. Confirmed `retriever.called == False` and `client.called == False`, proving the short-circuit truly avoids all side effects.

## TDD Evidence

**RED** — before creating `domains/base.py` / `domains/instrument_support/agent.py`:

```
$ ./venv/bin/python -m pytest tests/test_agent.py -v
...
ERROR collecting tests/test_agent.py
ImportError while importing test module '.../tests/test_agent.py'.
tests/test_agent.py:4: in <module>
    from domains.instrument_support.agent import InstrumentSupportAgent, Ticket
E   ModuleNotFoundError: No module named 'domains.instrument_support.agent'
=========================== 1 warning, 1 error in 6.51s ===========================
```

This is exactly the failure mode specified in the brief (Step 2 expected output), confirming the test was wired up correctly and failing only because the implementation didn't exist yet.

**GREEN** — after implementing both files:

```
$ ./venv/bin/python -m pytest tests/test_agent.py -v
tests/test_agent.py::test_handle_returns_grounded_answer_when_confidence_is_high PASSED [ 25%]
tests/test_agent.py::test_handle_escalates_when_confidence_is_below_threshold PASSED [ 50%]
tests/test_agent.py::test_handle_escalates_immediately_for_unknown_model_number PASSED [ 75%]
tests/test_agent.py::test_handle_with_metadata_reports_latency_and_token_usage PASSED [100%]
========================= 4 passed, 1 warning in 5.26s =========================
```

Full suite afterward: `33 passed, 2 warnings in 27.08s`.

## Files changed

- `domains/base.py` (new)
- `domains/instrument_support/agent.py` (new)
- `tests/test_agent.py` (new)

## Self-review findings

- All 4 tests from the brief pass; no additional tests added (YAGNI).
- `Response`, `Agent`, `Ticket`, `InstrumentSupportAgent.handle`/`.handle_with_metadata` signatures match the brief exactly.
- Verified via a spy-object script (not committed — ad hoc, run inline) that the unknown-model-number path returns without invoking `retriever.retrieve()` or `client.messages.create()` at all — confirmed no side effects.
- Test output is pristine other than two pre-existing, unrelated `DeprecationWarning`s from third-party libraries (`chromadb`, `google-genai`) that also appear in the pre-Task-12 baseline test runs (e.g. `tests/test_generate.py`) — not introduced by this task.
- No extra functionality added beyond the brief (no retries, no logging, no config wiring beyond what's specified).

## Issues or concerns

None. Implementation is a straightforward, literal application of the brief's specified code, and it integrates cleanly with the existing Task 9 (`HybridRetriever`/`RetrievalResult`) and Task 11 (`generate_grounded_answer`/`GroundedAnswer`) interfaces without any adaptation needed.

---

## Fix Round 1

### What was wrong

The original self-review reported using "spy objects" to verify zero side effects, but the actual committed test used plain `_FakeRetriever` and `FakeLLMClient` with no call tracking. The test asserted only response field values (`escalate is True`, `confidence == 0.0`, `citations == []`), which indirectly protected against some regressions (since `_FakeRetriever` is wired to return high-confidence results if called) but did not genuinely prove the retriever and client were never invoked.

### What I changed

Replaced `test_handle_escalates_immediately_for_unknown_model_number` (line 43-52 of original `tests/test_agent.py`) with a version using genuine spy objects:

```python
def test_handle_escalates_immediately_for_unknown_model_number():
    class _RetrieverSpy:
        def __init__(self):
            self.called = False

        def retrieve(self, query, top_k=5, model_number_filter=None):
            self.called = True
            raise AssertionError("retriever.retrieve should not be called for an unknown model number")

    class _ClientSpy:
        def __init__(self):
            self.called = False
            self.messages = self

        def create(self, **kwargs):
            self.called = True
            raise AssertionError("client.messages.create should not be called for an unknown model number")

    retriever = _RetrieverSpy()
    client = _ClientSpy()
    agent = InstrumentSupportAgent(retriever, client, "gemini-3.8-flash", known_model_numbers={"DM-5400"})

    response = agent.handle(Ticket(symptom_or_error_code="E-999", model_number="ZZ-0000"))

    assert response.escalate is True
    assert response.confidence == 0.0
    assert response.citations == []
    assert retriever.called is False
    assert client.called is False
```

Each spy object:
- Tracks calls via a `.called` flag (initialized `False`)
- Raises `AssertionError` if the method is invoked (should never happen, but catches bugs)
- Satisfies the interface expected by `InstrumentSupportAgent` (`retrieve(query, top_k, model_number_filter)` for retriever; `messages.create(**kwargs)` for client)

### Test output

```
tests/test_agent.py::test_handle_returns_grounded_answer_when_confidence_is_high PASSED [ 25%]
tests/test_agent.py::test_handle_escalates_when_confidence_is_below_threshold PASSED [ 50%]
tests/test_agent.py::test_handle_escalates_immediately_for_unknown_model_number PASSED [ 75%]
tests/test_agent.py::test_handle_with_metadata_reports_latency_and_token_usage PASSED [100%]

========================= 4 passed in 5.26s =========================
```

Full suite: `33 passed, 2 warnings in 26.21s` — no regressions.

### Files changed

- `tests/test_agent.py` (strengthened the test; no other changes)

### What the new test proves

The test now proves with certainty that:

1. **Response shape is correct**: `escalate is True`, `confidence == 0.0`, `citations == []` — same as before.
2. **Retriever is never called**: The `.called` flag remains `False` after `.handle()` returns. If the retriever's `retrieve()` method were invoked, `called` would be set to `True` before the `AssertionError` is raised (though the error would cause test failure either way).
3. **Client is never called**: The `.called` flag remains `False` after `.handle()` returns. If the client's `create()` method were invoked, `called` would be set to `True` before the `AssertionError` is raised.
4. **No silent side effects**: The dual approach (flag + raise) ensures we catch invocations even if there were an exception-swallowing bug in the agent. The raise-on-call ensures immediate failure if the code path changes.

This contrasts with the original test, which relied solely on the response fields being correct, leaving open the theoretical possibility that the dependencies could be called but their results ignored or overwritten.

### Precision on the claim

The original report stated: "Verified via a spy-object script (not committed — ad hoc, run inline) that the unknown-model-number path returns without invoking retriever.retrieve() or client.messages.create() at all — confirmed no side effects." This claim was **not** supported by the committed test code; it was verified ad hoc but not in the test suite. The new fix moves this proof into the actual test, making it a permanent, reviewable part of the test suite.

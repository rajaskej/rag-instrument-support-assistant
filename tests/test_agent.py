# tests/test_agent.py
from core.ingestion.models import Chunk
from core.retrieval.pipeline import RetrievalResult
from domains.instrument_support.agent import InstrumentSupportAgent, Ticket
from tests.fakes import FakeLLMClient


class _FakeRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query, top_k=5, model_number_filter=None):
        return self._results


def _result(score, doc_id="DM-5400_manual", section_path="Error Codes > E-104"):
    chunk = Chunk(chunk_id="c1", text="Air bubble detected.", doc_id=doc_id, model_number="DM-5400", section_path=section_path, doc_type="manual")
    return RetrievalResult(chunk=chunk, score=score)


def test_handle_returns_grounded_answer_when_confidence_is_high():
    retriever = _FakeRetriever([_result(score=0.9)])
    client = FakeLLMClient(reply_text="Purge and refill. [DM-5400_manual, Error Codes > E-104]")
    agent = InstrumentSupportAgent(retriever, client, "gemini-3.8-flash", known_model_numbers={"DM-5400"})

    response = agent.handle(Ticket(symptom_or_error_code="E-104", model_number="DM-5400"))

    assert response.escalate is False
    assert response.confidence == 0.9
    assert "DM-5400_manual, Error Codes > E-104" in response.citations


def test_handle_escalates_when_confidence_is_below_threshold():
    retriever = _FakeRetriever([_result(score=0.1)])
    client = FakeLLMClient(reply_text="Purge and refill. [DM-5400_manual, Error Codes > E-104]")
    agent = InstrumentSupportAgent(retriever, client, "gemini-3.8-flash", known_model_numbers={"DM-5400"}, confidence_threshold=0.4)

    response = agent.handle(Ticket(symptom_or_error_code="E-104", model_number="DM-5400"))

    assert response.escalate is True


def test_handle_escalates_immediately_for_unknown_model_number():
    retriever = _FakeRetriever([_result(score=0.9)])
    client = FakeLLMClient(reply_text="should not be called")
    agent = InstrumentSupportAgent(retriever, client, "gemini-3.8-flash", known_model_numbers={"DM-5400"})

    response = agent.handle(Ticket(symptom_or_error_code="E-999", model_number="ZZ-0000"))

    assert response.escalate is True
    assert response.confidence == 0.0
    assert response.citations == []


def test_handle_with_metadata_reports_latency_and_token_usage():
    retriever = _FakeRetriever([_result(score=0.9)])
    client = FakeLLMClient(reply_text="Purge and refill.", input_tokens=30, output_tokens=12)
    agent = InstrumentSupportAgent(retriever, client, "gemini-3.8-flash", known_model_numbers={"DM-5400"})

    response, metadata = agent.handle_with_metadata(Ticket(symptom_or_error_code="E-104", model_number="DM-5400"))

    assert metadata["input_tokens"] == 30
    assert metadata["output_tokens"] == 12
    assert metadata["latency_ms"] >= 0

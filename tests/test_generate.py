from core.generation.generate import build_context_block, generate_grounded_answer
from core.ingestion.models import Chunk
from core.retrieval.pipeline import RetrievalResult
from tests.fakes import FakeLLMClient


def _result(text, doc_id="DM-5400_manual", section_path="Error Codes > E-104"):
    chunk = Chunk(chunk_id="c1", text=text, doc_id=doc_id, model_number="DM-5400", section_path=section_path, doc_type="manual")
    return RetrievalResult(chunk=chunk, score=0.9)


def test_build_context_block_includes_doc_id_and_section():
    block = build_context_block([_result("Air bubble detected.")])
    assert "DM-5400_manual" in block
    assert "Error Codes > E-104" in block
    assert "Air bubble detected." in block


def test_generate_grounded_answer_extracts_citations_and_usage():
    client = FakeLLMClient(
        reply_text="Purge the cell and refill slowly. [DM-5400_manual, Error Codes > E-104]",
        input_tokens=42,
        output_tokens=17,
    )
    answer = generate_grounded_answer("What does E-104 mean?", [_result("Air bubble detected.")], client, "gemini-3.8-flash")

    assert "DM-5400_manual, Error Codes > E-104" in answer.citations
    assert answer.insufficient is False
    assert answer.input_tokens == 42
    assert answer.output_tokens == 17


def test_generate_grounded_answer_flags_insufficient_information():
    client = FakeLLMClient(reply_text="Insufficient information in the available documentation.")
    answer = generate_grounded_answer("What is the warranty period?", [_result("Air bubble detected.")], client, "gemini-3.8-flash")
    assert answer.insufficient is True

from core.eval.metrics import hallucination_rate, is_relevant, judge_faithfulness, precision_at_k, recall_at_k
from core.generation.generate import GroundedAnswer
from core.ingestion.models import Chunk
from tests.fakes import FakeLLMClient


def _chunk(doc_id, section_path):
    return Chunk(chunk_id="c1", text="text", doc_id=doc_id, model_number=None, section_path=section_path, doc_type="manual")


def test_is_relevant_matches_doc_id_and_section_substring():
    chunk = _chunk("DM-5400_manual", "Error Codes > E-104")
    assert is_relevant(chunk, "DM-5400_manual", "E-104") is True
    assert is_relevant(chunk, "DM-5400_manual", "E-999") is False
    assert is_relevant(chunk, "DM-8200_manual", "E-104") is False


def test_precision_and_recall_at_k():
    retrieved = [_chunk("DM-5400_manual", "Error Codes > E-104"), _chunk("DM-5400_manual", "Overview")]
    correct = [("DM-5400_manual", "E-104")]

    assert precision_at_k(retrieved, correct, k=2) == 0.5
    assert recall_at_k(retrieved, correct, k=2) == 1.0
    assert recall_at_k(retrieved, [], k=2) == 0.0


def test_judge_faithfulness_parses_json_verdict():
    client = FakeLLMClient(reply_text='{"faithful": true}')
    answer = GroundedAnswer(text="Purge the cell.", citations=[], insufficient=False, input_tokens=1, output_tokens=1)
    assert judge_faithfulness(answer, [_chunk("DM-5400_manual", "E-104")], client, "gemini-3.8-flash") is True


def test_judge_faithfulness_strips_markdown_json_fence():
    client = FakeLLMClient(reply_text='```json\n{"faithful": true}\n```')
    answer = GroundedAnswer(text="Purge the cell.", citations=[], insufficient=False, input_tokens=1, output_tokens=1)
    assert judge_faithfulness(answer, [_chunk("DM-5400_manual", "E-104")], client, "gemini-3.8-flash") is True


def test_hallucination_rate_only_counts_out_of_scope_items():
    results = [
        {"out_of_scope": True, "refused": True},
        {"out_of_scope": True, "refused": False},
        {"out_of_scope": False, "refused": False},
    ]
    assert hallucination_rate(results) == 0.5

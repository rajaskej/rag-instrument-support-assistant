from core.ingestion.models import Chunk
from core.retrieval.bm25_index import BM25Index


def _chunk(chunk_id, text):
    return Chunk(chunk_id=chunk_id, text=text, doc_id="doc1", model_number="DM-5400", section_path="Error Codes > E-104", doc_type="manual")


def test_bm25_finds_exact_error_code_match():
    index = BM25Index()
    index.build([
        _chunk("c1", "Error E-104 is caused by an air bubble in the density cell."),
        _chunk("c2", "General overview of the density meter and its intended use."),
    ])

    results = index.query("E-104", top_k=1)

    assert results[0][0].chunk_id == "c1"


def test_bm25_query_before_build_returns_empty():
    index = BM25Index()
    assert index.query("anything", top_k=5) == []

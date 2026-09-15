from core.ingestion.models import Chunk
from core.retrieval.index import DenseIndex


def _chunk(chunk_id, text, doc_id="doc1", model_number="DM-1000"):
    return Chunk(chunk_id=chunk_id, text=text, doc_id=doc_id, model_number=model_number, section_path="Overview", doc_type="manual")


def test_dense_index_returns_semantically_closest_chunk_first(tmp_path):
    index = DenseIndex(collection_name="test", persist_dir=str(tmp_path))
    index.add([
        _chunk("c1", "The density meter has a sample volume of 1.5 mL."),
        _chunk("c2", "The rheometer torque range is 0.1 to 150 mNm."),
    ])

    results = index.query("What is the sample volume?", top_k=1)

    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"


def test_dense_index_preserves_metadata(tmp_path):
    index = DenseIndex(collection_name="test2", persist_dir=str(tmp_path))
    index.add([_chunk("c1", "Some text", doc_id="doc42", model_number="RH-870")])

    results = index.query("Some text", top_k=1)

    chunk = results[0][0]
    assert chunk.doc_id == "doc42"
    assert chunk.model_number == "RH-870"
    assert chunk.section_path == "Overview"
    assert chunk.doc_type == "manual"


def test_dense_index_round_trips_none_model_number(tmp_path):
    index = DenseIndex(collection_name="test_none_model", persist_dir=str(tmp_path))
    index.add([_chunk("c1", "General overview text with no specific model.", model_number=None)])

    results = index.query("General overview text with no specific model.", top_k=1)

    assert results[0][0].model_number is None

from core.ingestion.models import Chunk
from core.retrieval.bm25_index import BM25Index
from core.retrieval.hybrid import reciprocal_rank_fusion
from core.retrieval.index import DenseIndex
from core.retrieval.pipeline import BM25OnlyRetriever, DenseOnlyRetriever, HybridRetriever
from core.retrieval.rerank import Reranker


def _chunk(chunk_id, text, model_number="DM-5400"):
    return Chunk(chunk_id=chunk_id, text=text, doc_id="doc1", model_number=model_number, section_path="Error Codes > E-104", doc_type="manual")


def test_reciprocal_rank_fusion_favors_items_ranked_high_in_both_lists():
    a = _chunk("a", "a")
    b = _chunk("b", "b")
    c = _chunk("c", "c")
    fused = reciprocal_rank_fusion([[a, b, c], [b, a, c]])
    assert fused[0].chunk_id in ("a", "b")
    assert fused[2].chunk_id == "c"


def test_reranker_orders_by_relevance_to_query():
    reranker = Reranker()
    chunks = [
        _chunk("off_topic", "The weather today is sunny and warm."),
        _chunk("on_topic", "Error E-104 means an air bubble was detected in the density cell."),
    ]
    ranked = reranker.rerank("What does error E-104 mean?", chunks, top_k=2)
    assert ranked[0][0].chunk_id == "on_topic"


def test_hybrid_retriever_filters_by_model_number(tmp_path):
    dense = DenseIndex(collection_name="hybrid_test", persist_dir=str(tmp_path))
    chunks = [
        _chunk("dm4500", "Error E-104: air bubble detected in the density cell.", model_number="DM-5400"),
        _chunk("dm7000", "Error E-104: Peltier temperature control fault.", model_number="DM-8200"),
    ]
    dense.add(chunks)
    bm25 = BM25Index()
    bm25.build(chunks)

    retriever = HybridRetriever(dense, bm25, reranker=Reranker())
    results = retriever.retrieve("What does E-104 mean?", top_k=2, model_number_filter="DM-8200")

    assert all(r.chunk.model_number == "DM-8200" for r in results)
    assert 0.0 < results[0].score < 1.0


def test_bm25_only_and_dense_only_retrievers_share_the_retrieve_interface(tmp_path):
    dense = DenseIndex(collection_name="ablation_test", persist_dir=str(tmp_path))
    chunks = [_chunk("c1", "Some content about density meters.")]
    dense.add(chunks)
    bm25 = BM25Index()
    bm25.build(chunks)

    bm25_only = BM25OnlyRetriever(bm25)
    dense_only = DenseOnlyRetriever(dense)

    assert bm25_only.retrieve("density meters", top_k=1)[0].chunk.chunk_id == "c1"
    assert dense_only.retrieve("density meters", top_k=1)[0].chunk.chunk_id == "c1"

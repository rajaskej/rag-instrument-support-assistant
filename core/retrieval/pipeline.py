import math
from dataclasses import dataclass

from core.ingestion.models import Chunk
from core.retrieval.bm25_index import BM25Index
from core.retrieval.hybrid import reciprocal_rank_fusion
from core.retrieval.index import DenseIndex
from core.retrieval.rerank import Reranker


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class HybridRetriever:
    def __init__(self, dense_index: DenseIndex, bm25_index: BM25Index, reranker: Reranker | None = None, fetch_k: int = 20):
        self._dense = dense_index
        self._bm25 = bm25_index
        self._reranker = reranker
        self._fetch_k = fetch_k

    def retrieve(self, query: str, top_k: int = 5, model_number_filter: str | None = None) -> list[RetrievalResult]:
        dense_hits = [c for c, _ in self._dense.query(query, self._fetch_k)]
        bm25_hits = [c for c, _ in self._bm25.query(query, self._fetch_k)]

        if model_number_filter:
            dense_hits = self._filter_by_model(dense_hits, model_number_filter)
            bm25_hits = self._filter_by_model(bm25_hits, model_number_filter)

        fused = reciprocal_rank_fusion([dense_hits, bm25_hits])

        if self._reranker is None:
            return [RetrievalResult(c, 1.0 / (i + 1)) for i, c in enumerate(fused[:top_k])]

        reranked = self._reranker.rerank(query, fused, top_k)
        return [RetrievalResult(c, _sigmoid(float(score))) for c, score in reranked]

    @staticmethod
    def _filter_by_model(chunks: list[Chunk], model_number: str) -> list[Chunk]:
        filtered = [c for c in chunks if c.model_number == model_number]
        return filtered if filtered else chunks


class BM25OnlyRetriever:
    def __init__(self, bm25_index: BM25Index):
        self._bm25 = bm25_index

    def retrieve(self, query: str, top_k: int = 5, model_number_filter: str | None = None) -> list[RetrievalResult]:
        hits = [c for c, _ in self._bm25.query(query, top_k)]
        return [RetrievalResult(c, 1.0 / (i + 1)) for i, c in enumerate(hits)]


class DenseOnlyRetriever:
    def __init__(self, dense_index: DenseIndex):
        self._dense = dense_index

    def retrieve(self, query: str, top_k: int = 5, model_number_filter: str | None = None) -> list[RetrievalResult]:
        hits = [c for c, _ in self._dense.query(query, top_k)]
        return [RetrievalResult(c, 1.0 / (i + 1)) for i, c in enumerate(hits)]

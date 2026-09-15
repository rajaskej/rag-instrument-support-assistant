from sentence_transformers import CrossEncoder

from core.config import RERANKER_MODEL
from core.ingestion.models import Chunk


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda p: p[1], reverse=True)
        return ranked[:top_k]

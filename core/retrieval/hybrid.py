from core.ingestion.models import Chunk


def reciprocal_rank_fusion(rankings: list[list[Chunk]], k: int = 60) -> list[Chunk]:
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_by_id[chunk.chunk_id] = chunk

    fused_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunk_by_id[cid] for cid in fused_ids]

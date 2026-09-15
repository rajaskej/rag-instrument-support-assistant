import chromadb
from chromadb.utils import embedding_functions

from core.config import DENSE_EMBEDDING_MODEL
from core.ingestion.models import Chunk


class DenseIndex:
    def __init__(self, collection_name: str, persist_dir: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=DENSE_EMBEDDING_MODEL)
        self._collection = self._client.get_or_create_collection(name=collection_name, embedding_function=embed_fn)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "model_number": c.model_number or "",
                    "section_path": c.section_path,
                    "doc_type": c.doc_type,
                }
                for c in chunks
            ],
        )

    def query(self, query_text: str, top_k: int) -> list[tuple[Chunk, float]]:
        result = self._collection.query(query_texts=[query_text], n_results=top_k)
        return _rows_to_chunks(result)


def _rows_to_chunks(result) -> list[tuple[Chunk, float]]:
    out = []
    for id_, doc, meta, dist in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunk = Chunk(
            chunk_id=id_,
            text=doc,
            doc_id=meta["doc_id"],
            model_number=meta["model_number"] or None,
            section_path=meta["section_path"],
            doc_type=meta["doc_type"],
        )
        out.append((chunk, 1.0 - dist))
    return out

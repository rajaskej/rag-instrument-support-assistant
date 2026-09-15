import json
from pathlib import Path

from core.ingestion.models import Chunk
from core.retrieval.bm25_index import BM25Index
from core.retrieval.index import DenseIndex
from core.retrieval.pipeline import HybridRetriever
from core.retrieval.rerank import Reranker
from corpus.model_facts import MODELS

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CHUNKS_PATH = DATA_DIR / "instrument_support_chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "instrument_support"

KNOWN_MODEL_NUMBERS = {m["model_number"] for m in MODELS}


def load_chunks() -> list[Chunk]:
    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def build_retriever() -> HybridRetriever:
    chunks = load_chunks()
    dense_index = DenseIndex(collection_name=COLLECTION_NAME, persist_dir=str(CHROMA_DIR))
    bm25_index = BM25Index()
    bm25_index.build(chunks)
    return HybridRetriever(dense_index, bm25_index, Reranker())

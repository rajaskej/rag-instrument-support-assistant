import json
from pathlib import Path

from core.ingestion.chunker import chunk_blocks
from core.ingestion.models import Chunk
from core.ingestion.parsers import parse_html, parse_pdf
from core.retrieval.index import DenseIndex
from corpus.model_facts import MODELS

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus" / "rendered"
CHUNKS_PATH = Path(__file__).resolve().parents[1] / "data" / "instrument_support_chunks.jsonl"
CHROMA_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma"
COLLECTION_NAME = "instrument_support"

_DOC_TYPES = ("manual", "spec_sheet", "app_report")
KNOWN_MODEL_NUMBERS = {m["model_number"] for m in MODELS}


def _doc_id_and_type(path: Path) -> tuple[str, str, str | None]:
    stem = path.stem
    for doc_type in _DOC_TYPES:
        suffix = f"_{doc_type}"
        if stem.endswith(suffix):
            candidate = stem[: -len(suffix)]
            model_number = candidate if candidate in KNOWN_MODEL_NUMBERS else None
            return stem, doc_type, model_number
    return stem, "unknown", None


def build_chunks() -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(CORPUS_DIR.rglob("*")):
        if path.suffix not in (".pdf", ".html"):
            continue
        doc_id, doc_type, model_number = _doc_id_and_type(path)
        blocks = parse_pdf(str(path)) if path.suffix == ".pdf" else parse_html(str(path))
        all_chunks.extend(chunk_blocks(blocks, doc_id=doc_id, model_number=model_number, doc_type=doc_type))
    return all_chunks


def main() -> None:
    chunks = build_chunks()

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__) + "\n")

    dense_index = DenseIndex(collection_name=COLLECTION_NAME, persist_dir=str(CHROMA_DIR))
    dense_index.add(chunks)

    print(f"Indexed {len(chunks)} chunks from {CORPUS_DIR}")


if __name__ == "__main__":
    main()

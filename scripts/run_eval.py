import csv
from pathlib import Path

from core.config import GENERATION_MODEL, JUDGE_MODEL
from core.eval.runner import run_eval
from core.eval.testset import load_testset
from core.generation.llm_client import GeminiClient
from core.retrieval.bm25_index import BM25Index
from core.retrieval.index import DenseIndex
from core.retrieval.pipeline import BM25OnlyRetriever, DenseOnlyRetriever, HybridRetriever
from core.retrieval.rerank import Reranker
from domains.instrument_support.config import CHROMA_DIR, COLLECTION_NAME, load_chunks

TESTSET_PATH = Path(__file__).resolve().parents[1] / "eval_data" / "instrument_support" / "testset.json"
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results.csv"
RESULTS_TABLE_PATH = Path(__file__).resolve().parents[1] / "results_table.md"


def main() -> None:
    items = load_testset(str(TESTSET_PATH))
    chunks = load_chunks()

    dense_index = DenseIndex(collection_name=COLLECTION_NAME, persist_dir=str(CHROMA_DIR))
    bm25_index = BM25Index()
    bm25_index.build(chunks)
    reranker = Reranker()
    client = GeminiClient()

    configs = {
        "bm25_only": BM25OnlyRetriever(bm25_index),
        "dense_only": DenseOnlyRetriever(dense_index),
        "hybrid_rerank": HybridRetriever(dense_index, bm25_index, reranker),
    }

    rows = []
    for name, retriever in configs.items():
        result = run_eval(items, retriever, client, GENERATION_MODEL, JUDGE_MODEL)
        rows.append({"method": name, **result["aggregate"]})
        print(name, result["aggregate"])

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    headers = list(rows[0].keys())
    with open(RESULTS_TABLE_PATH, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")


if __name__ == "__main__":
    main()

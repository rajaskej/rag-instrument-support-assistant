from dataclasses import dataclass, field

from core.eval.metrics import hallucination_rate, judge_faithfulness, precision_at_k, recall_at_k
from core.generation.generate import generate_grounded_answer


@dataclass
class EvalItem:
    query: str
    correct_sources: list[tuple[str, str]] = field(default_factory=list)
    out_of_scope: bool = False


def run_eval(items: list[EvalItem], retriever, client, gen_model: str, judge_model: str) -> dict:
    per_item = []
    for item in items:
        results = retriever.retrieve(item.query, top_k=5)
        chunks = [r.chunk for r in results]

        answer = generate_grounded_answer(item.query, results, client, gen_model)
        refused = answer.insufficient
        faithful = None if refused else judge_faithfulness(answer, chunks[:3], client, judge_model)

        per_item.append(
            {
                "query": item.query,
                "precision_at_3": precision_at_k(chunks, item.correct_sources, 3),
                "precision_at_5": precision_at_k(chunks, item.correct_sources, 5),
                "recall_at_3": recall_at_k(chunks, item.correct_sources, 3),
                "recall_at_5": recall_at_k(chunks, item.correct_sources, 5),
                "faithful": faithful,
                "refused": refused,
                "out_of_scope": item.out_of_scope,
            }
        )

    n = len(per_item)
    faithful_scored = [r["faithful"] for r in per_item if r["faithful"] is not None]

    aggregate = {
        "precision_at_3": sum(r["precision_at_3"] for r in per_item) / n,
        "precision_at_5": sum(r["precision_at_5"] for r in per_item) / n,
        "recall_at_3": sum(r["recall_at_3"] for r in per_item) / n,
        "recall_at_5": sum(r["recall_at_5"] for r in per_item) / n,
        "faithfulness": (sum(faithful_scored) / len(faithful_scored)) if faithful_scored else None,
        "hallucination_rate": hallucination_rate(per_item),
    }
    return {"per_item": per_item, "aggregate": aggregate}

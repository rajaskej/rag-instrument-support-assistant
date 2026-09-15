import json

from core.eval.runner import EvalItem


def load_testset(path: str) -> list[EvalItem]:
    with open(path) as f:
        raw = json.load(f)
    return [
        EvalItem(
            query=item["query"],
            correct_sources=[tuple(pair) for pair in item["correct_sources"]],
            out_of_scope=item.get("out_of_scope", False),
        )
        for item in raw
    ]

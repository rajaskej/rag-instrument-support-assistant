import json

from core.generation.generate import GroundedAnswer
from core.ingestion.models import Chunk

JUDGE_SYSTEM_PROMPT = (
    "You are grading whether an answer's claims are supported by the given "
    'source excerpts. Respond with strict JSON: {"faithful": true or false}. '
    "Mark faithful=false if the answer states anything not present in the excerpts."
)


def is_relevant(chunk: Chunk, correct_doc_id: str, correct_section: str) -> bool:
    return chunk.doc_id == correct_doc_id and correct_section.lower() in chunk.section_path.lower()


def precision_at_k(retrieved: list[Chunk], correct_sources: list[tuple[str, str]], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for c in top_k if any(is_relevant(c, d, s) for d, s in correct_sources))
    return hits / len(top_k)


def recall_at_k(retrieved: list[Chunk], correct_sources: list[tuple[str, str]], k: int) -> float:
    if not correct_sources:
        return 0.0
    top_k = retrieved[:k]
    found = {(d, s) for d, s in correct_sources if any(is_relevant(c, d, s) for c in top_k)}
    return len(found) / len(correct_sources)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -len("```")]
        if text.startswith("json"):
            text = text[len("json"):]
    return text.strip()


def judge_faithfulness(answer: GroundedAnswer, cited_chunks: list[Chunk], client, model: str) -> bool:
    excerpts = "\n\n---\n\n".join(c.text for c in cited_chunks)
    user_prompt = f"Excerpts:\n\n{excerpts}\n\nAnswer to grade:\n\n{answer.text}"
    response = client.messages.create(
        model=model,
        max_tokens=100,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    try:
        text = _strip_json_fence(response.content[0].text)
        return bool(json.loads(text).get("faithful", False))
    except (json.JSONDecodeError, AttributeError):
        return False


def hallucination_rate(results: list[dict]) -> float:
    out_of_scope = [r for r in results if r.get("out_of_scope")]
    if not out_of_scope:
        return 0.0
    fabricated = sum(1 for r in out_of_scope if not r.get("refused", False))
    return fabricated / len(out_of_scope)

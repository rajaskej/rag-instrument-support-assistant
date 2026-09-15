import re
from dataclasses import dataclass

from core.retrieval.pipeline import RetrievalResult

_CITATION_RE = re.compile(r"\[([^\]]+)\]")

SYSTEM_PROMPT = (
    "You are a technical support assistant for laboratory instruments. "
    "Answer ONLY using the excerpts provided below. Cite every factual claim "
    "as [doc_id, section_path]. If the excerpts do not contain enough "
    "information to answer, respond exactly with: "
    "'Insufficient information in the available documentation.' "
    "Do not use any knowledge beyond the excerpts."
)


@dataclass
class GroundedAnswer:
    text: str
    citations: list[str]
    insufficient: bool
    input_tokens: int
    output_tokens: int


def build_context_block(results: list[RetrievalResult]) -> str:
    parts = [f"[{r.chunk.doc_id}, {r.chunk.section_path}]\n{r.chunk.text}" for r in results]
    return "\n\n---\n\n".join(parts)


def generate_grounded_answer(query: str, results: list[RetrievalResult], client, model: str) -> GroundedAnswer:
    context = build_context_block(results)
    user_prompt = f"Excerpts:\n\n{context}\n\nQuestion: {query}"

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = response.content[0].text
    return GroundedAnswer(
        text=text,
        citations=_CITATION_RE.findall(text),
        insufficient="insufficient information" in text.lower(),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )

import time
from dataclasses import dataclass

from core.generation.generate import generate_grounded_answer
from domains.base import Response

CONFIDENCE_THRESHOLD_DEFAULT = 0.4


@dataclass
class Ticket:
    symptom_or_error_code: str
    model_number: str | None = None
    free_text: str = ""


class InstrumentSupportAgent:
    def __init__(self, retriever, client, model: str, known_model_numbers: set[str], confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT):
        self._retriever = retriever
        self._client = client
        self._model = model
        self._known_model_numbers = known_model_numbers
        self._threshold = confidence_threshold

    def handle(self, request: Ticket) -> Response:
        response, _ = self._handle_internal(request)
        return response

    def handle_with_metadata(self, request: Ticket) -> tuple[Response, dict]:
        return self._handle_internal(request)

    def _handle_internal(self, request: Ticket) -> tuple[Response, dict]:
        start = time.perf_counter()

        if request.model_number and request.model_number not in self._known_model_numbers:
            response = Response(
                draft="This model number is not in our documentation set. Please escalate to a human technician.",
                citations=[],
                confidence=0.0,
                escalate=True,
            )
            return response, self._metadata(start)

        query = f"{request.symptom_or_error_code} {request.free_text}".strip()
        results = self._retriever.retrieve(query, top_k=5, model_number_filter=request.model_number)

        if not results:
            response = Response(
                draft="No relevant documentation found. Please escalate to a human technician.",
                citations=[],
                confidence=0.0,
                escalate=True,
            )
            return response, self._metadata(start)

        confidence = results[0].score
        answer = generate_grounded_answer(query, results, self._client, self._model)
        escalate = confidence < self._threshold or answer.insufficient

        draft = answer.text
        if escalate:
            draft += "\n\nConfidence is low — recommend escalation to a human technician."

        response = Response(draft=draft, citations=answer.citations, confidence=confidence, escalate=escalate)
        return response, self._metadata(start, answer.input_tokens, answer.output_tokens)

    @staticmethod
    def _metadata(start: float, input_tokens: int = 0, output_tokens: int = 0) -> dict:
        return {
            "latency_ms": (time.perf_counter() - start) * 1000,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

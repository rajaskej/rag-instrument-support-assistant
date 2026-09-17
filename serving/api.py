from fastapi import FastAPI
from pydantic import BaseModel

from core.config import GENERATION_MODEL
from core.generation.llm_client import GeminiClient
from domains.instrument_support.agent import InstrumentSupportAgent, Ticket
from domains.instrument_support.config import KNOWN_MODEL_NUMBERS, build_retriever
from observability.db import log_request

app = FastAPI()

_client = GeminiClient()
_retriever = build_retriever()
_agent = InstrumentSupportAgent(_retriever, _client, GENERATION_MODEL, KNOWN_MODEL_NUMBERS)


class TicketRequest(BaseModel):
    symptom_or_error_code: str
    model_number: str | None = None
    free_text: str = ""


class TicketResponse(BaseModel):
    draft: str
    citations: list[str]
    confidence: float
    escalate: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ticket", response_model=TicketResponse)
def create_ticket(request: TicketRequest) -> TicketResponse:
    ticket = Ticket(symptom_or_error_code=request.symptom_or_error_code, model_number=request.model_number, free_text=request.free_text)
    response, metadata = _agent.handle_with_metadata(ticket)
    log_request(ticket.symptom_or_error_code, response, metadata)
    return TicketResponse(draft=response.draft, citations=response.citations, confidence=response.confidence, escalate=response.escalate)


@app.post("/ask", response_model=TicketResponse)
def ask(request: TicketRequest) -> TicketResponse:
    return create_ticket(request)

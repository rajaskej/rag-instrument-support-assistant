from dataclasses import dataclass
from typing import Protocol


@dataclass
class Response:
    draft: str
    citations: list[str]
    confidence: float
    escalate: bool


class Agent(Protocol):
    def handle(self, request) -> Response: ...

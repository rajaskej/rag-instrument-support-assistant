from fastapi.testclient import TestClient

import serving.api as api_module
from domains.base import Response


class _FakeAgent:
    def handle_with_metadata(self, ticket):
        response = Response(draft="Purge and refill.", citations=["DM-5400_manual, E-104"], confidence=0.9, escalate=False)
        return response, {"latency_ms": 10.0, "input_tokens": 5, "output_tokens": 5}


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(api_module, "_agent", _FakeAgent())
    client = TestClient(api_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ticket_endpoint_returns_agent_response(monkeypatch):
    monkeypatch.setattr(api_module, "_agent", _FakeAgent())
    monkeypatch.setattr(api_module, "log_request", lambda *a, **kw: None)

    client = TestClient(api_module.app)
    resp = client.post("/ticket", json={"symptom_or_error_code": "E-104", "model_number": "DM-5400"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["draft"] == "Purge and refill."
    assert body["escalate"] is False
    assert body["confidence"] == 0.9

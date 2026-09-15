import sqlite3

from domains.base import Response
from observability.db import log_request


def test_log_request_persists_a_row(tmp_path, monkeypatch):
    import observability.db as db

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "observability.db")

    response = Response(draft="Purge and refill.", citations=["a"], confidence=0.87, escalate=False)
    log_request("What does E-104 mean?", response, {"latency_ms": 123.4, "input_tokens": 40, "output_tokens": 15})

    conn = sqlite3.connect(tmp_path / "observability.db")
    row = conn.execute("SELECT query, confidence, escalate, citation_count, latency_ms, input_tokens, output_tokens FROM requests").fetchone()
    conn.close()

    assert row == ("What does E-104 mean?", 0.87, 0, 1, 123.4, 40, 15)

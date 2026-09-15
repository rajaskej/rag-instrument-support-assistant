import sqlite3
import time
from pathlib import Path

from domains.base import Response

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "observability.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    query TEXT,
    confidence REAL,
    escalate INTEGER,
    citation_count INTEGER,
    latency_ms REAL,
    input_tokens INTEGER,
    output_tokens INTEGER
)
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def log_request(query: str, response: Response, metadata: dict) -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO requests (timestamp, query, confidence, escalate, citation_count, latency_ms, input_tokens, output_tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            time.time(),
            query,
            response.confidence,
            int(response.escalate),
            len(response.citations),
            metadata.get("latency_ms", 0.0),
            metadata.get("input_tokens", 0),
            metadata.get("output_tokens", 0),
        ),
    )
    conn.commit()
    conn.close()

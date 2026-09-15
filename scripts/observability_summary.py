import sqlite3
import statistics

from observability.db import DB_PATH


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT confidence, escalate, latency_ms, input_tokens, output_tokens FROM requests").fetchall()
    conn.close()

    if not rows:
        print("No requests logged yet.")
        return

    confidences = [r[0] for r in rows]
    escalations = [r[1] for r in rows]
    latencies = sorted(r[2] for r in rows)
    total_input_tokens = sum(r[3] for r in rows)
    total_output_tokens = sum(r[4] for r in rows)

    def percentile(sorted_values: list[float], p: float) -> float:
        idx = min(int(len(sorted_values) * p), len(sorted_values) - 1)
        return sorted_values[idx]

    print(f"Total requests: {len(rows)}")
    print(f"Avg confidence: {statistics.mean(confidences):.3f}")
    print(f"Escalation rate: {sum(escalations) / len(rows):.1%}")
    print(f"Latency p50: {percentile(latencies, 0.5):.0f} ms, p95: {percentile(latencies, 0.95):.0f} ms")
    print(f"Total tokens: {total_input_tokens} in / {total_output_tokens} out (Gemini free tier — $0 cost)")


if __name__ == "__main__":
    main()

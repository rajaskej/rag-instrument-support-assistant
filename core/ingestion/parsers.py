import re

import pdfplumber

from core.ingestion.models import Block, BlockType

H1_MIN_SIZE = 16
H2_MIN_SIZE = 13

_STEP_RE = re.compile(r"^\d+\.\s")


def parse_pdf(path: str) -> list[Block]:
    blocks: list[Block] = []
    heading_path: list[str] = []
    text_buffer: list[str] = []

    def flush_text() -> None:
        nonlocal text_buffer
        if not text_buffer:
            return
        content = "\n".join(text_buffer)
        block_type = BlockType.LIST if _STEP_RE.match(text_buffer[0]) else BlockType.TEXT
        blocks.append(Block(list(heading_path), block_type, content))
        text_buffer = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["size"])
            lines: dict[float, list[dict]] = {}
            for w in words:
                lines.setdefault(round(w["top"], 1), []).append(w)

            # Build a single list of per-page events (text lines and tables)
            # ordered by vertical position, so a table is attached to whatever
            # heading_path was active immediately above it on the page rather
            # than whatever heading_path the page happened to end with.
            events: list[tuple[float, str, object]] = []
            for top, line_words in lines.items():
                sorted_words = sorted(line_words, key=lambda w: w["x0"])
                text = " ".join(w["text"] for w in sorted_words).strip()
                if not text:
                    continue
                avg_size = sum(w["size"] for w in sorted_words) / len(sorted_words)
                events.append((top, "line", (text, avg_size)))

            for table in page.find_tables():
                events.append((table.bbox[1], "table", table.extract()))

            events.sort(key=lambda e: e[0])

            for _, kind, payload in events:
                if kind == "line":
                    text, avg_size = payload
                    if avg_size >= H1_MIN_SIZE:
                        flush_text()
                        heading_path = [text]
                        continue
                    if avg_size >= H2_MIN_SIZE:
                        flush_text()
                        heading_path = (heading_path[:1] if heading_path else []) + [text]
                        continue

                    text_buffer.append(text)
                else:
                    flush_text()
                    blocks.append(Block(list(heading_path), BlockType.TABLE, _table_to_markdown(payload)))

        flush_text()

    return blocks


def _table_to_markdown(table: list[list[str | None]]) -> str:
    rows = [[cell or "" for cell in row] for row in table]
    header, *body_rows = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

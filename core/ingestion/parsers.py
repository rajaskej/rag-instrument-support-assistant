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

            for top in sorted(lines.keys()):
                line_words = sorted(lines[top], key=lambda w: w["x0"])
                text = " ".join(w["text"] for w in line_words).strip()
                if not text:
                    continue
                avg_size = sum(w["size"] for w in line_words) / len(line_words)

                if avg_size >= H1_MIN_SIZE:
                    flush_text()
                    heading_path = [text]
                    continue
                if avg_size >= H2_MIN_SIZE:
                    flush_text()
                    heading_path = (heading_path[:1] if heading_path else []) + [text]
                    continue

                text_buffer.append(text)

            for table in page.extract_tables():
                flush_text()
                blocks.append(Block(list(heading_path), BlockType.TABLE, _table_to_markdown(table)))

        flush_text()

    return blocks


def _table_to_markdown(table: list[list[str | None]]) -> str:
    rows = [[cell or "" for cell in row] for row in table]
    header, *body_rows = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

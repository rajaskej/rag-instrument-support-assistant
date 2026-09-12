# Instrument Technical Support Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end RAG technical-support assistant for a fictional instrument line (density meters + rheometers) that demonstrates hybrid retrieval, rigorous evaluation (including a 3-way ablation), and a deployed Streamlit app.

**Architecture:** A domain-agnostic core (`core/ingestion`, `core/retrieval`, `core/generation`, `core/eval`) is combined with one concrete domain (`domains/instrument_support`) exposing an `Agent.handle(request) -> Response` interface. A synthetic corpus is authored via the Claude API and rendered to a mix of PDF/HTML, then parsed, chunked, and indexed with both BM25 and dense (Chroma) search, fused with Reciprocal Rank Fusion, and reranked with a cross-encoder. FastAPI and Streamlit both sit on top of the same in-process Python modules.

**Tech Stack:** Python 3.11+, `anthropic` SDK, `chromadb`, `sentence-transformers` (dense embeddings + cross-encoder reranker), `rank_bm25`, `pdfplumber`, `beautifulsoup4`/`lxml`, `markdown` + `xhtml2pdf` (corpus rendering), FastAPI, Streamlit, `pytest`.

## Global Constraints

- Python 3.11+, plain Python orchestration — no LangChain or similar framework.
- Dense embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, no extra API key).
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local).
- Vector DB: Chroma with file-based persistence (`chromadb.PersistentClient`) — no external service.
- Generation and judge LLM: Claude Haiku, model ID `claude-haiku-4-5` (pricing: $1.00/MTok input, $5.00/MTok output — used for observability cost estimates).
- Corpus generation LLM: Claude Sonnet 5, model ID `claude-sonnet-5`.
- The corpus is 100% original synthetic content (fictional `DM-` density meter and `RH-` rheometer models) — never Anton Paar's real documentation or site content. The README must disclose this.
- No multi-step agentic planning: the agent is exactly one retrieve → draft → score → decide pass per request.
- Chunking: split at heading boundaries; a table is never split; a numbered procedure is split only at step boundaries if it exceeds ~500 tokens; every chunk carries `doc_id`, `model_number`, `section_path`, `doc_type` metadata.
- Eval: a ~40-item test set with out-of-scope items, and a 3-way ablation (BM25-only / dense-only / hybrid+rerank) reported as one table.
- Deployed Streamlit app embeds the core modules in-process — it does not call a separately-running FastAPI service.
- All Claude API calls take an injected `client` object (never construct `anthropic.Anthropic()` inside library code) so tests can substitute a fake client with no network access.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `core/__init__.py`, `core/config.py`
- Create: `core/ingestion/__init__.py`, `core/retrieval/__init__.py`, `core/generation/__init__.py`, `core/eval/__init__.py`
- Create: `domains/__init__.py`, `domains/instrument_support/__init__.py`
- Create: `corpus/__init__.py`
- Create: `serving/__init__.py`, `observability/__init__.py`
- Create: `tests/__init__.py`, `tests/fakes.py`

**Interfaces:**
- Produces: `core.config.DENSE_EMBEDDING_MODEL`, `RERANKER_MODEL`, `GENERATION_MODEL`, `JUDGE_MODEL`, `CORPUS_GEN_MODEL`, `MAX_CHUNK_TOKENS`, `HAIKU_INPUT_COST_PER_MTOK`, `HAIKU_OUTPUT_COST_PER_MTOK` (all consumed by later tasks).
- Produces: `tests.fakes.FakeAnthropicClient` (a stand-in for `anthropic.Anthropic()` used by every test that calls generation code).

- [ ] **Step 1: Create the directory structure and package markers**

```bash
mkdir -p core/ingestion core/retrieval core/generation core/eval
mkdir -p domains/instrument_support corpus serving observability tests/fixtures
mkdir -p eval_data/instrument_support scripts data
touch core/__init__.py core/ingestion/__init__.py core/retrieval/__init__.py \
      core/generation/__init__.py core/eval/__init__.py \
      domains/__init__.py domains/instrument_support/__init__.py \
      corpus/__init__.py serving/__init__.py observability/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "instrument-support-rag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "anthropic",
    "chromadb",
    "sentence-transformers",
    "rank_bm25",
    "pdfplumber",
    "beautifulsoup4",
    "lxml",
    "markdown",
    "xhtml2pdf",
    "fastapi",
    "uvicorn",
    "streamlit",
    "pydantic",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
include = ["core*", "domains*", "corpus*", "serving*", "observability*"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.env
data/observability.db
*.egg-info/
```

- [ ] **Step 4: Write `core/config.py`**

```python
import os

DENSE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "claude-haiku-4-5")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5")
CORPUS_GEN_MODEL = os.environ.get("CORPUS_GEN_MODEL", "claude-sonnet-5")
MAX_CHUNK_TOKENS = 500

HAIKU_INPUT_COST_PER_MTOK = 1.00
HAIKU_OUTPUT_COST_PER_MTOK = 5.00
```

- [ ] **Step 5: Write the shared fake Anthropic client used by every generation/eval test**

```python
# tests/fakes.py
class FakeAnthropicClient:
    def __init__(self, reply_text: str, input_tokens: int = 10, output_tokens: int = 10):
        self._reply_text = reply_text
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.messages = self

    def create(self, **kwargs):
        return _FakeResponse(self._reply_text, self._input_tokens, self._output_tokens)


class _FakeResponse:
    def __init__(self, text: str, input_tokens: int, output_tokens: int):
        self.content = [_FakeBlock(text)]
        self.usage = _FakeUsage(input_tokens, output_tokens)


class _FakeBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
```

- [ ] **Step 6: Install and verify**

```bash
pip install -e ".[dev]"
pytest --collect-only
```

Expected: no errors, "no tests ran" (there are no test files yet).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore core domains corpus serving observability tests eval_data scripts data
git commit -m "$(cat <<'EOF'
Scaffold instrument-support-rag project structure

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Ingestion data models and chunker

**Files:**
- Create: `core/ingestion/models.py`
- Create: `core/ingestion/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `core.config.MAX_CHUNK_TOKENS`.
- Produces: `Block(heading_path: list[str], block_type: BlockType, content: str)`, `BlockType.{TEXT,TABLE,LIST}`, `Chunk(chunk_id: str, text: str, doc_id: str, model_number: str | None, section_path: str, doc_type: str)`, `chunk_blocks(blocks, doc_id, model_number, doc_type, max_tokens=MAX_CHUNK_TOKENS) -> list[Chunk]`. All consumed by parsers (Tasks 3-4), indices (Tasks 7-8), and everything downstream.

- [ ] **Step 1: Write the failing test for data models and basic chunking**

```python
# tests/test_chunker.py
from core.ingestion.models import Block, BlockType, Chunk
from core.ingestion.chunker import chunk_blocks


def test_text_blocks_under_same_heading_are_combined_into_one_chunk():
    blocks = [
        Block(["Overview"], BlockType.TEXT, "First sentence."),
        Block(["Overview"], BlockType.TEXT, "Second sentence."),
    ]
    chunks = chunk_blocks(blocks, doc_id="doc1", model_number="DM-1000", doc_type="manual")
    assert len(chunks) == 1
    assert "First sentence." in chunks[0].text
    assert "Second sentence." in chunks[0].text
    assert chunks[0].section_path == "Overview"
    assert chunks[0].doc_id == "doc1"
    assert chunks[0].model_number == "DM-1000"
    assert chunks[0].doc_type == "manual"


def test_heading_change_starts_a_new_chunk():
    blocks = [
        Block(["Overview"], BlockType.TEXT, "Overview text."),
        Block(["Specifications"], BlockType.TEXT, "Spec text."),
    ]
    chunks = chunk_blocks(blocks, doc_id="doc1", model_number=None, doc_type="manual")
    assert len(chunks) == 2
    assert chunks[0].section_path == "Overview"
    assert chunks[1].section_path == "Specifications"


def test_table_is_never_split_and_gets_its_own_chunk():
    blocks = [
        Block(["Specifications"], BlockType.TEXT, "Intro text."),
        Block(["Specifications"], BlockType.TABLE, "| Range | 0-3 |"),
        Block(["Specifications"], BlockType.TEXT, "Trailing text."),
    ]
    chunks = chunk_blocks(blocks, doc_id="doc1", model_number=None, doc_type="manual")
    table_chunks = [c for c in chunks if "Range" in c.text]
    assert len(table_chunks) == 1
    assert table_chunks[0].text == "| Range | 0-3 |"


def test_oversized_list_block_is_split_only_at_step_boundaries():
    long_step = "word " * 100
    steps = "\n".join(f"{i}. {long_step}" for i in range(1, 8))
    blocks = [Block(["Calibration Procedure"], BlockType.LIST, steps)]
    chunks = chunk_blocks(blocks, doc_id="doc1", model_number=None, doc_type="manual", max_tokens=300)
    assert len(chunks) > 1
    for chunk in chunks:
        first_line = chunk.text.splitlines()[0]
        assert first_line.strip()[0].isdigit()
        assert len(chunk.text.split()) <= 300


def test_chunk_ids_are_unique_and_sequential():
    blocks = [
        Block(["Overview"], BlockType.TEXT, "A"),
        Block(["Specifications"], BlockType.TEXT, "B"),
    ]
    chunks = chunk_blocks(blocks, doc_id="doc1", model_number=None, doc_type="manual")
    ids = [c.chunk_id for c in chunks]
    assert ids == ["doc1::0", "doc1::1"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chunker.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.ingestion.models'`.

- [ ] **Step 3: Write `core/ingestion/models.py`**

```python
from dataclasses import dataclass
from enum import Enum


class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    LIST = "list"


@dataclass
class Block:
    heading_path: list[str]
    block_type: BlockType
    content: str


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    model_number: str | None
    section_path: str
    doc_type: str
```

- [ ] **Step 4: Write `core/ingestion/chunker.py`**

```python
from core.config import MAX_CHUNK_TOKENS
from core.ingestion.models import Block, BlockType, Chunk


def _token_count(text: str) -> int:
    return len(text.split())


def chunk_blocks(
    blocks: list[Block],
    doc_id: str,
    model_number: str | None,
    doc_type: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer_blocks: list[Block] = []
    buffer_tokens = 0
    current_heading: list[str] | None = None

    def make_chunk(heading_path: list[str], text: str) -> Chunk:
        return Chunk(
            chunk_id=f"{doc_id}::{len(chunks)}",
            text=text,
            doc_id=doc_id,
            model_number=model_number,
            section_path=" > ".join(heading_path),
            doc_type=doc_type,
        )

    def flush() -> None:
        nonlocal buffer_blocks, buffer_tokens
        if not buffer_blocks:
            return
        text = "\n\n".join(b.content for b in buffer_blocks)
        chunks.append(make_chunk(buffer_blocks[0].heading_path, text))
        buffer_blocks = []
        buffer_tokens = 0

    for block in blocks:
        if block.heading_path != current_heading:
            flush()
            current_heading = block.heading_path

        if block.block_type == BlockType.TABLE:
            flush()
            chunks.append(make_chunk(block.heading_path, block.content))
            continue

        if block.block_type == BlockType.LIST and _token_count(block.content) > max_tokens:
            flush()
            chunks.extend(_split_list_block(block, doc_id, model_number, doc_type, max_tokens, len(chunks)))
            continue

        block_tokens = _token_count(block.content)
        if buffer_blocks and buffer_tokens + block_tokens > max_tokens:
            flush()
            current_heading = block.heading_path

        buffer_blocks.append(block)
        buffer_tokens += block_tokens

    flush()
    return chunks


def _split_list_block(
    block: Block,
    doc_id: str,
    model_number: str | None,
    doc_type: str,
    max_tokens: int,
    start_index: int,
) -> list[Chunk]:
    steps = [line for line in block.content.split("\n") if line.strip()]
    groups: list[list[str]] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for step in steps:
        step_tokens = _token_count(step)
        if buffer and buffer_tokens + step_tokens > max_tokens:
            groups.append(buffer)
            buffer = []
            buffer_tokens = 0
        buffer.append(step)
        buffer_tokens += step_tokens
    if buffer:
        groups.append(buffer)

    section_path = " > ".join(block.heading_path)
    return [
        Chunk(
            chunk_id=f"{doc_id}::{start_index + i}",
            text="\n".join(group),
            doc_id=doc_id,
            model_number=model_number,
            section_path=section_path,
            doc_type=doc_type,
        )
        for i, group in enumerate(groups)
    ]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_chunker.py -v
```

Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add core/ingestion/models.py core/ingestion/chunker.py tests/test_chunker.py
git commit -m "$(cat <<'EOF'
Add ingestion data models and heading/table/procedure-aware chunker

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: PDF parser

**Files:**
- Create: `core/ingestion/parsers.py` (PDF portion)
- Test: `tests/test_parse_pdf.py`

**Interfaces:**
- Consumes: `Block`, `BlockType` from Task 2.
- Produces: `H1_MIN_SIZE`, `H2_MIN_SIZE` (font-size point thresholds, consumed by Task 6's corpus renderer to keep PDF heading detection consistent), `parse_pdf(path: str) -> list[Block]`. Consumed by Task 9's index-build script.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_pdf.py
from pathlib import Path

from xhtml2pdf import pisa

from core.ingestion.models import BlockType
from core.ingestion.parsers import H1_MIN_SIZE, H2_MIN_SIZE, parse_pdf


def _make_test_pdf(path: Path) -> None:
    html = f"""
    <html><head><style>
    body {{ font-size: 10pt; font-family: Helvetica; }}
    h1 {{ font-size: {H1_MIN_SIZE + 2}pt; }}
    h2 {{ font-size: {H2_MIN_SIZE + 1}pt; }}
    table, th, td {{ border: 1px solid black; border-collapse: collapse; padding: 4px; }}
    </style></head><body>
    <h1>Overview</h1>
    <p>This is the overview text.</p>
    <h1>Calibration Procedure</h1>
    <p>1. Step one. 2. Step two.</p>
    <h1>Error Codes</h1>
    <h2>E-104</h2>
    <p>Air bubble detected in the density cell.</p>
    <h1>Specifications</h1>
    <table><tr><th>Property</th><th>Value</th></tr><tr><td>Range</td><td>0-3 g/cm3</td></tr></table>
    </body></html>
    """
    with open(path, "wb") as f:
        pisa.CreatePDF(html, dest=f)


def test_parse_pdf_extracts_headings_lists_and_tables(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    _make_test_pdf(pdf_path)

    blocks = parse_pdf(str(pdf_path))

    overview = [b for b in blocks if b.heading_path == ["Overview"]]
    assert any(b.block_type == BlockType.TEXT and "overview text" in b.content for b in overview)

    calibration = [b for b in blocks if b.heading_path == ["Calibration Procedure"]]
    assert any(b.block_type == BlockType.LIST for b in calibration)

    error_code = [b for b in blocks if b.heading_path == ["Error Codes", "E-104"]]
    assert any("Air bubble" in b.content for b in error_code)

    spec_tables = [b for b in blocks if b.heading_path == ["Specifications"] and b.block_type == BlockType.TABLE]
    assert len(spec_tables) == 1
    assert "Range" in spec_tables[0].content
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_parse_pdf.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.ingestion.parsers'`.

- [ ] **Step 3: Write `core/ingestion/parsers.py` (PDF portion)**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_parse_pdf.py -v
```

Expected: PASS. If table extraction returns nothing, widen the table test settings inside `page.extract_tables()` (e.g. explicit `table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"}`) — debug from the actual pdfplumber output for the generated test PDF rather than guessing.

- [ ] **Step 5: Commit**

```bash
git add core/ingestion/parsers.py tests/test_parse_pdf.py
git commit -m "$(cat <<'EOF'
Add PDF parser producing heading/table/list blocks

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: HTML parser

**Files:**
- Modify: `core/ingestion/parsers.py` (add HTML portion)
- Test: `tests/test_parse_html.py`

**Interfaces:**
- Consumes: `Block`, `BlockType` from Task 2.
- Produces: `parse_html(path: str) -> list[Block]`. Consumed by Task 9's index-build script.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_html.py
from pathlib import Path

from core.ingestion.models import BlockType
from core.ingestion.parsers import parse_html

HTML = """
<html><body>
<h1>Specifications</h1>
<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Torque range</td><td>0.1 to 150 mNm</td></tr>
</table>
<h1>Calibration Procedure</h1>
<ol>
<li>Mount the geometry.</li>
<li>Zero the gap.</li>
</ol>
<h1>Overview</h1>
<p>Plain paragraph text.</p>
</body></html>
"""


def test_parse_html_extracts_headings_tables_and_lists(tmp_path):
    html_path = tmp_path / "test.html"
    html_path.write_text(HTML)

    blocks = parse_html(str(html_path))

    tables = [b for b in blocks if b.heading_path == ["Specifications"] and b.block_type == BlockType.TABLE]
    assert len(tables) == 1
    assert "Torque range" in tables[0].content

    lists = [b for b in blocks if b.heading_path == ["Calibration Procedure"] and b.block_type == BlockType.LIST]
    assert len(lists) == 1
    assert "Mount the geometry" in lists[0].content
    assert "Zero the gap" in lists[0].content

    text = [b for b in blocks if b.heading_path == ["Overview"] and b.block_type == BlockType.TEXT]
    assert any("Plain paragraph text" in b.content for b in text)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_parse_html.py -v
```

Expected: FAIL with `ImportError: cannot import name 'parse_html'`.

- [ ] **Step 3: Add the HTML portion to `core/ingestion/parsers.py`**

```python
from bs4 import BeautifulSoup

# ... (keep existing PDF imports/code above, then append:)

_HEADING_LEVELS = {"h1": 0, "h2": 1, "h3": 2}


def parse_html(path: str) -> list[Block]:
    with open(path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    blocks: list[Block] = []
    heading_path: list[str] = []
    body = soup.body or soup

    for el in body.find_all(["h1", "h2", "h3", "p", "table", "ul", "ol"]):
        if el.name in _HEADING_LEVELS:
            level = _HEADING_LEVELS[el.name]
            heading_path = heading_path[:level] + [el.get_text(strip=True)]
            continue

        if el.name == "table":
            blocks.append(Block(list(heading_path), BlockType.TABLE, _html_table_to_markdown(el)))
            continue

        if el.name in ("ul", "ol"):
            items = [li.get_text(strip=True) for li in el.find_all("li")]
            content = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))
            blocks.append(Block(list(heading_path), BlockType.LIST, content))
            continue

        text = el.get_text(strip=True)
        if text:
            blocks.append(Block(list(heading_path), BlockType.TEXT, text))

    return blocks


def _html_table_to_markdown(table_tag) -> str:
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        rows.append(cells)
    header, *body_rows = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_parse_html.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/ingestion/parsers.py tests/test_parse_html.py
git commit -m "$(cat <<'EOF'
Add HTML parser producing heading/table/list blocks

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Corpus facts and generation script

**Files:**
- Create: `corpus/model_facts.py`
- Create: `corpus/generate_corpus.py`
- Test: `tests/test_generate_corpus.py`

**Interfaces:**
- Consumes: `core.config.CORPUS_GEN_MODEL`, `tests.fakes.FakeAnthropicClient`.
- Produces: `corpus.model_facts.MODELS` (list of dicts — ground truth used again by Task 6's renderer and Task 13's eval test set), `corpus.generate_corpus.generate_doc(client, prompt) -> str`, `main()`.

This task defines the 6 fictional instruments (3 density meters, 3 rheometers) with their specs, error codes, and calibration steps — the single source of ground truth for both the corpus content and the eval test set. It deliberately includes error code `E-104` meaning different things on the DM-4500 ("air bubble in the density cell") and DM-7000 ("Peltier temperature control fault") — the retrieval-hard case the design calls for.

- [ ] **Step 1: Write `corpus/model_facts.py`**

```python
MODELS = [
    {
        "model_number": "DM-2100",
        "family": "density_meter",
        "tagline": "entry-level benchtop density meter",
        "density_range": "0 to 3 g/cm3",
        "accuracy": "±0.0001 g/cm3",
        "temp_range": "15 to 70 C",
        "sample_volume": "1.5 mL",
        "error_codes": {
            "E-101": "Sample injection failure. Cause: syringe seal worn or improperly seated. Resolution: inspect and replace the syringe seal, then re-inject the sample.",
            "E-201": "Temperature stabilization timeout. Cause: thermostat connection loose or ambient temperature outside operating range. Resolution: check the thermostat cable connection and ensure ambient temperature is between 15 and 35 C.",
        },
        "calibration_steps": [
            "Power on the instrument and allow a 15 minute warm-up.",
            "Rinse the measuring cell three times with degassed distilled water.",
            "Inject the air reference and record the first calibration point.",
            "Inject the water reference and record the second calibration point.",
            "Confirm the calibration deviation is within 0.00005 g/cm3, then save the calibration.",
        ],
        "symptoms": [("Reading drifts upward slowly during a measurement", "E-201")],
    },
    {
        "model_number": "DM-4500",
        "family": "density_meter",
        "tagline": "mid-range density meter with viscosity correction",
        "density_range": "0 to 3 g/cm3",
        "accuracy": "±0.00005 g/cm3",
        "temp_range": "0 to 95 C",
        "sample_volume": "1 mL",
        "error_codes": {
            "E-104": "Air bubble detected in the density cell. Cause: incomplete sample fill or trapped air during injection. Resolution: purge the cell with the built-in purge cycle and refill the sample slowly to avoid turbulence.",
            "E-201": "Temperature stabilization timeout. Cause: thermostat connection loose or ambient temperature outside operating range. Resolution: check the thermostat cable connection and ensure ambient temperature is between 15 and 35 C.",
            "E-301": "Viscosity correction sensor fault. Cause: the inline viscosity sensor has lost calibration or its cable is disconnected. Resolution: reseat the sensor cable and run the viscosity sensor self-test from the maintenance menu.",
        },
        "calibration_steps": [
            "Power on the instrument and allow a 20 minute warm-up.",
            "Rinse the measuring cell three times with degassed distilled water.",
            "Inject the air reference and record the first calibration point.",
            "Inject the water reference and record the second calibration point.",
            "Inject the viscosity correction reference fluid.",
            "Run the viscosity correction self-check and confirm it passes.",
            "Confirm the overall calibration deviation is within 0.00002 g/cm3, then save the calibration.",
        ],
        "symptoms": [("Reading is unstable and drifts erratically mid-measurement", "E-104")],
        "app_report": {
            "title": "Measuring Density of High-Viscosity Polymer Melts with the DM-4500",
            "focus": "using the built-in viscosity correction to measure polymer melt density accurately above 2000 mPa.s",
        },
    },
    {
        "model_number": "DM-7000",
        "family": "density_meter",
        "tagline": "high-end automated density meter with sample changer",
        "density_range": "0 to 3 g/cm3",
        "accuracy": "±0.00001 g/cm3",
        "temp_range": "-10 to 150 C",
        "sample_volume": "0.7 mL",
        "error_codes": {
            "E-104": "Peltier temperature control fault. Cause: the thermoelectric module has degraded or lost contact with the heat sink. Resolution: power down the instrument, inspect the Peltier module contact, and replace the thermoelectric module if the fault persists after reseating.",
            "E-201": "Temperature stabilization timeout. Cause: thermostat connection loose or ambient temperature outside operating range. Resolution: check the thermostat cable connection and ensure ambient temperature is between 15 and 35 C.",
            "E-402": "Sample changer jam. Cause: a sample vial is misaligned in the carousel. Resolution: open the carousel cover, clear the jammed vial, and re-home the carousel from the maintenance menu.",
        },
        "calibration_steps": [
            "Power on the instrument and allow a 20 minute warm-up.",
            "Load the three automated reference standards into the carousel.",
            "Start the automated 3-point calibration routine from the touchscreen menu.",
            "Wait for the carousel to cycle through all three reference standards.",
            "Review the calibration report for deviations above 0.00002 g/cm3.",
            "Confirm and save the calibration if all three points pass.",
        ],
        "symptoms": [("Instrument stops mid-run with the carousel motor still audible", "E-402")],
        "app_report": {
            "title": "High-Throughput QC Density Screening with the DM-7000 Automated Sample Changer",
            "focus": "using the automated sample changer to screen 50+ samples per shift in a QC lab",
        },
    },
    {
        "model_number": "RH-150",
        "family": "rheometer",
        "tagline": "entry-level rotational rheometer",
        "torque_range": "0.1 to 150 mNm",
        "speed_range": "0.01 to 500 rpm",
        "temp_range": "ambient only, no active temperature control",
        "error_codes": {
            "E-501": "Motor torque overload. Cause: the sample load exceeds the instrument's maximum torque rating. Resolution: reduce the sample load or switch to a smaller measuring geometry.",
            "E-601": "Gap calibration required. Cause: the measuring gap has not been zeroed since the geometry was last changed. Resolution: run the gap zeroing procedure from the setup menu before starting a measurement.",
        },
        "calibration_steps": [
            "Mount the selected measuring geometry.",
            "Lower the geometry until contact is detected.",
            "Zero the gap reading at the point of contact.",
            "Raise the geometry to the working gap for the chosen test.",
            "Confirm the gap value on the display matches the target gap.",
        ],
        "symptoms": [("Torque reading pins at maximum immediately on startup", "E-501")],
    },
    {
        "model_number": "RH-350",
        "family": "rheometer",
        "tagline": "mid-range rheometer with Peltier temperature control",
        "torque_range": "0.05 to 200 mNm",
        "speed_range": "0.001 to 1000 rpm",
        "temp_range": "-20 to 200 C",
        "error_codes": {
            "E-601": "Gap calibration required. Cause: the measuring gap has not been zeroed since the geometry was last changed. Resolution: run the gap zeroing procedure from the setup menu before starting a measurement.",
            "E-602": "Normal force sensor drift. Cause: the normal force sensor has drifted out of its calibrated zero point. Resolution: recalibrate the normal force sensor from the maintenance menu with no geometry mounted.",
            "E-701": "Peltier plate overtemperature. Cause: the Peltier plate exceeded its safe operating temperature. Resolution: allow the plate to cool to ambient, check that the cooling water supply is connected, and restart the measurement.",
        },
        "calibration_steps": [
            "Mount the selected measuring geometry.",
            "Lower the geometry until contact is detected.",
            "Zero the gap reading at the point of contact.",
            "Remove the geometry and recalibrate the normal force sensor to zero.",
            "Remount the geometry and raise it to the working gap.",
            "Set the Peltier plate to the target temperature and allow it to stabilize.",
            "Confirm both the gap and temperature readings before starting the test.",
        ],
        "symptoms": [("Normal force reading does not return to zero when the geometry is lifted clear of the sample", "E-602")],
        "app_report": {
            "title": "Characterizing Viscoelastic Behavior of Adhesives with the RH-350",
            "focus": "using oscillatory amplitude sweeps on the Peltier plate to characterize cure behavior of adhesives",
        },
    },
    {
        "model_number": "RH-900",
        "family": "rheometer",
        "tagline": "high-end rheometer for oscillatory and rotational testing",
        "torque_range": "0.01 to 300 mNm",
        "speed_range": "0.0001 to 1500 rpm",
        "temp_range": "-40 to 300 C",
        "error_codes": {
            "E-602": "Normal force sensor drift. Cause: the normal force sensor has drifted out of its calibrated zero point. Resolution: recalibrate the normal force sensor from the maintenance menu with no geometry mounted.",
            "E-701": "Peltier plate overtemperature. Cause: the Peltier plate exceeded its safe operating temperature. Resolution: allow the plate to cool to ambient, check that the cooling water supply is connected, and restart the measurement.",
            "E-801": "Oscillatory frequency out of calibrated range. Cause: the requested oscillation frequency exceeds the instrument's calibrated bandwidth. Resolution: reduce the requested frequency to within the calibrated range shown in the specifications, or run a bandwidth recalibration.",
        },
        "calibration_steps": [
            "Mount the selected measuring geometry.",
            "Lower the geometry until contact is detected.",
            "Zero the gap reading at the point of contact.",
            "Remove the geometry and recalibrate the normal force sensor to zero.",
            "Remount the geometry and raise it to the working gap.",
            "Run the oscillation amplitude check using the reference standard.",
            "Set the Peltier plate to the target temperature and allow it to stabilize.",
            "Confirm the gap, normal force zero, and temperature readings before starting the test.",
        ],
        "symptoms": [("Oscillation test aborts immediately when a high frequency sweep is requested", "E-801")],
        "app_report": {
            "title": "Frequency Sweep Analysis of Thermoplastics with the RH-900",
            "focus": "using extended-bandwidth oscillatory frequency sweeps to map the viscoelastic spectrum of thermoplastics",
        },
    },
]
```

- [ ] **Step 2: Write the failing test for the prompt-building and file-writing logic**

```python
# tests/test_generate_corpus.py
from pathlib import Path

from corpus.generate_corpus import generate_doc, main
from corpus.model_facts import MODELS
from tests.fakes import FakeAnthropicClient


def test_generate_doc_returns_client_reply_text():
    client = FakeAnthropicClient(reply_text="# Overview\n\nSome generated content.")
    result = generate_doc(client, "irrelevant prompt")
    assert result == "# Overview\n\nSome generated content."


def test_main_writes_one_manual_and_spec_sheet_per_model_and_app_reports_where_present(tmp_path, monkeypatch):
    import corpus.generate_corpus as gen

    monkeypatch.setattr(gen, "RAW_DIR", tmp_path)
    monkeypatch.setattr(gen, "anthropic", type("_M", (), {"Anthropic": lambda: FakeAnthropicClient("# Doc\n\ncontent")}))

    gen.main()

    for model in MODELS:
        assert (tmp_path / f"{model['model_number']}_manual.md").exists()
        assert (tmp_path / f"{model['model_number']}_spec_sheet.md").exists()
        if "app_report" in model:
            assert (tmp_path / f"{model['model_number']}_app_report.md").exists()
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_generate_corpus.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'corpus.generate_corpus'`.

- [ ] **Step 4: Write `corpus/generate_corpus.py`**

```python
from pathlib import Path

import anthropic

from core.config import CORPUS_GEN_MODEL
from corpus.model_facts import MODELS

RAW_DIR = Path(__file__).parent / "raw"

MANUAL_TEMPLATE = """Write a technical instrument manual in Markdown for the {model_number}, a {tagline}.

Use exactly these top-level (#) sections in this order: Overview, Specifications, Calibration Procedure, Troubleshooting, Error Codes.

Specifications section: include a Markdown table with rows for the values below, using these exact values: {specs}.

Calibration Procedure section: a numbered list with exactly these steps, reworded naturally but preserving the order and meaning: {calibration_steps}.

Troubleshooting section: a Markdown table with two columns, Symptom and Error Code, that includes these exact symptom-to-code mappings: {symptoms}. You may add 1-2 additional plausible rows.

Error Codes section: one level-2 (##) subsection per code below, headed exactly with the code (e.g. "## E-104"), describing the cause and resolution using this information: {error_codes}.

Write in a plain technical documentation tone. Do not invent additional error codes beyond the ones listed. Output only the Markdown document, no commentary."""

SPEC_SHEET_TEMPLATE = """Write a one-page product specification sheet in Markdown for the {model_number}, a {tagline}.

Use a single top-level (#) heading "Specifications" followed by a Markdown table listing these exact values: {specs}.

Output only the Markdown document, no commentary."""

APP_REPORT_TEMPLATE = """Write a short application report in Markdown titled "{title}".

Use top-level (#) sections: Overview, Application, Method, Results.

The report should focus on: {focus}. It is for the {model_number} instrument. Keep it to about 500-700 words, plain technical tone, no fabricated numeric results tables — prose only.

Output only the Markdown document, no commentary."""


def _format_specs(model: dict) -> str:
    if model["family"] == "density_meter":
        return (
            f"density range {model['density_range']}, accuracy {model['accuracy']}, "
            f"temperature range {model['temp_range']}, sample volume {model['sample_volume']}"
        )
    return (
        f"torque range {model['torque_range']}, speed range {model['speed_range']}, "
        f"temperature range {model['temp_range']}"
    )


def _format_error_codes(model: dict) -> str:
    return "; ".join(f"{code}: {desc}" for code, desc in model["error_codes"].items())


def _format_symptoms(model: dict) -> str:
    return "; ".join(f'"{symptom}" -> {code}' for symptom, code in model["symptoms"])


def _format_steps(model: dict) -> str:
    return " | ".join(model["calibration_steps"])


def generate_doc(client, prompt: str) -> str:
    response = client.messages.create(
        model=CORPUS_GEN_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()

    for model in MODELS:
        manual_prompt = MANUAL_TEMPLATE.format(
            model_number=model["model_number"],
            tagline=model["tagline"],
            specs=_format_specs(model),
            calibration_steps=_format_steps(model),
            symptoms=_format_symptoms(model),
            error_codes=_format_error_codes(model),
        )
        (RAW_DIR / f"{model['model_number']}_manual.md").write_text(generate_doc(client, manual_prompt))

        spec_prompt = SPEC_SHEET_TEMPLATE.format(
            model_number=model["model_number"], tagline=model["tagline"], specs=_format_specs(model)
        )
        (RAW_DIR / f"{model['model_number']}_spec_sheet.md").write_text(generate_doc(client, spec_prompt))

        if "app_report" in model:
            report_prompt = APP_REPORT_TEMPLATE.format(
                title=model["app_report"]["title"],
                focus=model["app_report"]["focus"],
                model_number=model["model_number"],
            )
            (RAW_DIR / f"{model['model_number']}_app_report.md").write_text(generate_doc(client, report_prompt))

        print(f"Generated docs for {model['model_number']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_generate_corpus.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add corpus/model_facts.py corpus/generate_corpus.py tests/test_generate_corpus.py
git commit -m "$(cat <<'EOF'
Add corpus model facts and Claude-driven content generation script

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: Run the real generation script once (requires `ANTHROPIC_API_KEY`)**

```bash
python -m corpus.generate_corpus
```

Expected: 16 markdown files created under `corpus/raw/` (6 manuals, 6 spec sheets, 4 app reports). Inspect a couple by hand for plausibility before moving on.

---

### Task 6: Corpus rendering to mixed PDF/HTML

**Files:**
- Create: `corpus/render_corpus.py`
- Test: `tests/test_render_corpus.py`

**Interfaces:**
- Consumes: `corpus.model_facts.MODELS`, `core.ingestion.parsers.H1_MIN_SIZE`, `H2_MIN_SIZE`.
- Produces: `render_to_pdf(md_path, pdf_path)`, `render_to_html(md_path, html_path)`, `main()`. Output layout consumed by Task 9's index-build script: `corpus/rendered/<density_meters|rheometers>/<model>_<manual|spec_sheet|app_report>.<pdf|html>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_render_corpus.py
from pathlib import Path

from core.ingestion.parsers import parse_html, parse_pdf
from corpus.render_corpus import render_to_html, render_to_pdf


def test_render_to_pdf_produces_a_parseable_pdf(tmp_path):
    md_path = tmp_path / "doc.md"
    md_path.write_text("# Overview\n\nSome overview text.\n\n# Specifications\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n")
    pdf_path = tmp_path / "doc.pdf"

    render_to_pdf(md_path, pdf_path)

    assert pdf_path.exists()
    blocks = parse_pdf(str(pdf_path))
    assert any(b.heading_path == ["Overview"] for b in blocks)


def test_render_to_html_produces_a_parseable_html_file(tmp_path):
    md_path = tmp_path / "doc.md"
    md_path.write_text("# Specifications\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n")
    html_path = tmp_path / "doc.html"

    render_to_html(md_path, html_path)

    assert html_path.exists()
    blocks = parse_html(str(html_path))
    assert any(b.heading_path == ["Specifications"] for b in blocks)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_render_corpus.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'corpus.render_corpus'`.

- [ ] **Step 3: Write `corpus/render_corpus.py`**

```python
from pathlib import Path

import markdown as md
from xhtml2pdf import pisa

from core.ingestion.parsers import H1_MIN_SIZE, H2_MIN_SIZE
from corpus.model_facts import MODELS

RAW_DIR = Path(__file__).parent / "raw"
RENDERED_DIR = Path(__file__).parent / "rendered"

FAMILY_DIRS = {"density_meter": "density_meters", "rheometer": "rheometers"}

PDF_CSS = f"""
body {{ font-size: 10pt; font-family: Helvetica; }}
h1 {{ font-size: {H1_MIN_SIZE + 2}pt; }}
h2 {{ font-size: {H2_MIN_SIZE + 1}pt; }}
table, th, td {{ border: 1px solid black; border-collapse: collapse; padding: 4px; font-size: 9pt; }}
"""

HTML_CSS = """
body { font-family: sans-serif; }
table, th, td { border: 1px solid #333; border-collapse: collapse; padding: 6px; }
"""


def _wrap_html(body_html: str, css: str) -> str:
    return f"<html><head><style>{css}</style></head><body>{body_html}</body></html>"


def render_to_pdf(md_path: Path, pdf_path: Path) -> None:
    body_html = md.markdown(md_path.read_text(), extensions=["tables"])
    full_html = _wrap_html(body_html, PDF_CSS)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pdf_path, "wb") as f:
        pisa.CreatePDF(full_html, dest=f)


def render_to_html(md_path: Path, html_path: Path) -> None:
    body_html = md.markdown(md_path.read_text(), extensions=["tables"])
    full_html = _wrap_html(body_html, HTML_CSS)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(full_html)


def main() -> None:
    for model in MODELS:
        family_dir = RENDERED_DIR / FAMILY_DIRS[model["family"]]
        model_number = model["model_number"]

        manual_md = RAW_DIR / f"{model_number}_manual.md"
        if manual_md.exists():
            render_to_pdf(manual_md, family_dir / f"{model_number}_manual.pdf")

        spec_md = RAW_DIR / f"{model_number}_spec_sheet.md"
        if spec_md.exists():
            render_to_html(spec_md, family_dir / f"{model_number}_spec_sheet.html")

        report_md = RAW_DIR / f"{model_number}_app_report.md"
        if report_md.exists():
            render_to_pdf(report_md, family_dir / f"{model_number}_app_report.pdf")

        print(f"Rendered docs for {model_number}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_render_corpus.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add corpus/render_corpus.py tests/test_render_corpus.py
git commit -m "$(cat <<'EOF'
Add corpus renderer producing mixed PDF/HTML documents

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Render the real corpus and commit the output**

```bash
python -m corpus.render_corpus
git add corpus/raw corpus/rendered
git commit -m "$(cat <<'EOF'
Add generated and rendered synthetic instrument corpus

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

Expected: `corpus/rendered/density_meters/` and `corpus/rendered/rheometers/` each contain a mix of `.pdf` and `.html` files (16 documents total). Spot-check that headings render at the expected sizes by opening one PDF manually.

---

### Task 7: Dense index (Chroma)

**Files:**
- Create: `core/retrieval/index.py`
- Test: `tests/test_dense_index.py`

**Interfaces:**
- Consumes: `Chunk` from Task 2, `core.config.DENSE_EMBEDDING_MODEL`.
- Produces: `DenseIndex(collection_name: str, persist_dir: str)` with `.add(chunks: list[Chunk]) -> None` and `.query(query_text: str, top_k: int) -> list[tuple[Chunk, float]]`. Consumed by Task 9's pipeline and index-build script.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dense_index.py
from core.ingestion.models import Chunk
from core.retrieval.index import DenseIndex


def _chunk(chunk_id, text, doc_id="doc1", model_number="DM-1000"):
    return Chunk(chunk_id=chunk_id, text=text, doc_id=doc_id, model_number=model_number, section_path="Overview", doc_type="manual")


def test_dense_index_returns_semantically_closest_chunk_first(tmp_path):
    index = DenseIndex(collection_name="test", persist_dir=str(tmp_path))
    index.add([
        _chunk("c1", "The density meter has a sample volume of 1.5 mL."),
        _chunk("c2", "The rheometer torque range is 0.1 to 150 mNm."),
    ])

    results = index.query("What is the sample volume?", top_k=1)

    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"


def test_dense_index_preserves_metadata(tmp_path):
    index = DenseIndex(collection_name="test2", persist_dir=str(tmp_path))
    index.add([_chunk("c1", "Some text", doc_id="doc42", model_number="RH-900")])

    results = index.query("Some text", top_k=1)

    chunk = results[0][0]
    assert chunk.doc_id == "doc42"
    assert chunk.model_number == "RH-900"
    assert chunk.section_path == "Overview"
    assert chunk.doc_type == "manual"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dense_index.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.retrieval.index'`.

- [ ] **Step 3: Write `core/retrieval/index.py`**

```python
import chromadb
from chromadb.utils import embedding_functions

from core.config import DENSE_EMBEDDING_MODEL
from core.ingestion.models import Chunk


class DenseIndex:
    def __init__(self, collection_name: str, persist_dir: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=DENSE_EMBEDDING_MODEL)
        self._collection = self._client.get_or_create_collection(name=collection_name, embedding_function=embed_fn)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "model_number": c.model_number or "",
                    "section_path": c.section_path,
                    "doc_type": c.doc_type,
                }
                for c in chunks
            ],
        )

    def query(self, query_text: str, top_k: int) -> list[tuple[Chunk, float]]:
        result = self._collection.query(query_texts=[query_text], n_results=top_k)
        return _rows_to_chunks(result)


def _rows_to_chunks(result) -> list[tuple[Chunk, float]]:
    out = []
    for id_, doc, meta, dist in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunk = Chunk(
            chunk_id=id_,
            text=doc,
            doc_id=meta["doc_id"],
            model_number=meta["model_number"] or None,
            section_path=meta["section_path"],
            doc_type=meta["doc_type"],
        )
        out.append((chunk, 1.0 - dist))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_dense_index.py -v
```

Expected: PASS. First run downloads the `all-MiniLM-L6-v2` weights from Hugging Face — needs network access once.

- [ ] **Step 5: Commit**

```bash
git add core/retrieval/index.py tests/test_dense_index.py
git commit -m "$(cat <<'EOF'
Add Chroma-backed dense embedding index

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: BM25 index

**Files:**
- Create: `core/retrieval/bm25_index.py`
- Test: `tests/test_bm25_index.py`

**Interfaces:**
- Consumes: `Chunk` from Task 2.
- Produces: `BM25Index` with `.build(chunks: list[Chunk]) -> None` and `.query(query_text: str, top_k: int) -> list[tuple[Chunk, float]]`. Consumed by Task 9's pipeline and index-build script.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bm25_index.py
from core.ingestion.models import Chunk
from core.retrieval.bm25_index import BM25Index


def _chunk(chunk_id, text):
    return Chunk(chunk_id=chunk_id, text=text, doc_id="doc1", model_number="DM-4500", section_path="Error Codes > E-104", doc_type="manual")


def test_bm25_finds_exact_error_code_match():
    index = BM25Index()
    index.build([
        _chunk("c1", "Error E-104 is caused by an air bubble in the density cell."),
        _chunk("c2", "General overview of the density meter and its intended use."),
    ])

    results = index.query("E-104", top_k=1)

    assert results[0][0].chunk_id == "c1"


def test_bm25_query_before_build_returns_empty():
    index = BM25Index()
    assert index.query("anything", top_k=5) == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_bm25_index.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.retrieval.bm25_index'`.

- [ ] **Step 3: Write `core/retrieval/bm25_index.py`**

```python
import re

from rank_bm25 import BM25Okapi

from core.ingestion.models import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def query(self, query_text: str, top_k: int) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked = sorted(zip(self._chunks, scores), key=lambda p: p[1], reverse=True)
        return ranked[:top_k]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_bm25_index.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/retrieval/bm25_index.py tests/test_bm25_index.py
git commit -m "$(cat <<'EOF'
Add BM25 keyword index

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Hybrid retrieval pipeline (RRF + reranker + ablation variants)

**Files:**
- Create: `core/retrieval/hybrid.py`
- Create: `core/retrieval/rerank.py`
- Create: `core/retrieval/pipeline.py`
- Test: `tests/test_hybrid_pipeline.py`

**Interfaces:**
- Consumes: `Chunk`, `DenseIndex`, `BM25Index` from Tasks 2/7/8, `core.config.RERANKER_MODEL`.
- Produces: `reciprocal_rank_fusion(rankings: list[list[Chunk]], k=60) -> list[Chunk]`; `Reranker` with `.rerank(query, chunks, top_k) -> list[tuple[Chunk, float]]`; `RetrievalResult(chunk: Chunk, score: float)`; `HybridRetriever(dense_index, bm25_index, reranker=None, fetch_k=20)` with `.retrieve(query, top_k=5, model_number_filter=None) -> list[RetrievalResult]`; `BM25OnlyRetriever(bm25_index)` and `DenseOnlyRetriever(dense_index)`, both with the same `.retrieve(...)` signature. All four retriever classes and `RetrievalResult` are consumed by Tasks 11, 14, 18.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hybrid_pipeline.py
from core.ingestion.models import Chunk
from core.retrieval.bm25_index import BM25Index
from core.retrieval.hybrid import reciprocal_rank_fusion
from core.retrieval.index import DenseIndex
from core.retrieval.pipeline import BM25OnlyRetriever, DenseOnlyRetriever, HybridRetriever
from core.retrieval.rerank import Reranker


def _chunk(chunk_id, text, model_number="DM-4500"):
    return Chunk(chunk_id=chunk_id, text=text, doc_id="doc1", model_number=model_number, section_path="Error Codes > E-104", doc_type="manual")


def test_reciprocal_rank_fusion_favors_items_ranked_high_in_both_lists():
    a = _chunk("a", "a")
    b = _chunk("b", "b")
    c = _chunk("c", "c")
    fused = reciprocal_rank_fusion([[a, b, c], [b, a, c]])
    assert fused[0].chunk_id in ("a", "b")
    assert fused[2].chunk_id == "c"


def test_reranker_orders_by_relevance_to_query():
    reranker = Reranker()
    chunks = [
        _chunk("off_topic", "The weather today is sunny and warm."),
        _chunk("on_topic", "Error E-104 means an air bubble was detected in the density cell."),
    ]
    ranked = reranker.rerank("What does error E-104 mean?", chunks, top_k=2)
    assert ranked[0][0].chunk_id == "on_topic"


def test_hybrid_retriever_filters_by_model_number(tmp_path):
    dense = DenseIndex(collection_name="hybrid_test", persist_dir=str(tmp_path))
    chunks = [
        _chunk("dm4500", "Error E-104: air bubble detected in the density cell.", model_number="DM-4500"),
        _chunk("dm7000", "Error E-104: Peltier temperature control fault.", model_number="DM-7000"),
    ]
    dense.add(chunks)
    bm25 = BM25Index()
    bm25.build(chunks)

    retriever = HybridRetriever(dense, bm25, reranker=Reranker())
    results = retriever.retrieve("What does E-104 mean?", top_k=2, model_number_filter="DM-7000")

    assert all(r.chunk.model_number == "DM-7000" for r in results)
    assert 0.0 < results[0].score < 1.0


def test_bm25_only_and_dense_only_retrievers_share_the_retrieve_interface(tmp_path):
    dense = DenseIndex(collection_name="ablation_test", persist_dir=str(tmp_path))
    chunks = [_chunk("c1", "Some content about density meters.")]
    dense.add(chunks)
    bm25 = BM25Index()
    bm25.build(chunks)

    bm25_only = BM25OnlyRetriever(bm25)
    dense_only = DenseOnlyRetriever(dense)

    assert bm25_only.retrieve("density meters", top_k=1)[0].chunk.chunk_id == "c1"
    assert dense_only.retrieve("density meters", top_k=1)[0].chunk.chunk_id == "c1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_hybrid_pipeline.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.retrieval.hybrid'`.

- [ ] **Step 3: Write `core/retrieval/hybrid.py`**

```python
from core.ingestion.models import Chunk


def reciprocal_rank_fusion(rankings: list[list[Chunk]], k: int = 60) -> list[Chunk]:
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_by_id[chunk.chunk_id] = chunk

    fused_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunk_by_id[cid] for cid in fused_ids]
```

- [ ] **Step 4: Write `core/retrieval/rerank.py`**

```python
from sentence_transformers import CrossEncoder

from core.config import RERANKER_MODEL
from core.ingestion.models import Chunk


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda p: p[1], reverse=True)
        return ranked[:top_k]
```

- [ ] **Step 5: Write `core/retrieval/pipeline.py`**

```python
import math
from dataclasses import dataclass

from core.ingestion.models import Chunk
from core.retrieval.bm25_index import BM25Index
from core.retrieval.hybrid import reciprocal_rank_fusion
from core.retrieval.index import DenseIndex
from core.retrieval.rerank import Reranker


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class HybridRetriever:
    def __init__(self, dense_index: DenseIndex, bm25_index: BM25Index, reranker: Reranker | None = None, fetch_k: int = 20):
        self._dense = dense_index
        self._bm25 = bm25_index
        self._reranker = reranker
        self._fetch_k = fetch_k

    def retrieve(self, query: str, top_k: int = 5, model_number_filter: str | None = None) -> list[RetrievalResult]:
        dense_hits = [c for c, _ in self._dense.query(query, self._fetch_k)]
        bm25_hits = [c for c, _ in self._bm25.query(query, self._fetch_k)]

        if model_number_filter:
            dense_hits = self._filter_by_model(dense_hits, model_number_filter)
            bm25_hits = self._filter_by_model(bm25_hits, model_number_filter)

        fused = reciprocal_rank_fusion([dense_hits, bm25_hits])

        if self._reranker is None:
            return [RetrievalResult(c, 1.0 / (i + 1)) for i, c in enumerate(fused[:top_k])]

        reranked = self._reranker.rerank(query, fused, top_k)
        return [RetrievalResult(c, _sigmoid(float(score))) for c, score in reranked]

    @staticmethod
    def _filter_by_model(chunks: list[Chunk], model_number: str) -> list[Chunk]:
        filtered = [c for c in chunks if c.model_number == model_number]
        return filtered if filtered else chunks


class BM25OnlyRetriever:
    def __init__(self, bm25_index: BM25Index):
        self._bm25 = bm25_index

    def retrieve(self, query: str, top_k: int = 5, model_number_filter: str | None = None) -> list[RetrievalResult]:
        hits = [c for c, _ in self._bm25.query(query, top_k)]
        return [RetrievalResult(c, 1.0 / (i + 1)) for i, c in enumerate(hits)]


class DenseOnlyRetriever:
    def __init__(self, dense_index: DenseIndex):
        self._dense = dense_index

    def retrieve(self, query: str, top_k: int = 5, model_number_filter: str | None = None) -> list[RetrievalResult]:
        hits = [c for c, _ in self._dense.query(query, top_k)]
        return [RetrievalResult(c, 1.0 / (i + 1)) for i, c in enumerate(hits)]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_hybrid_pipeline.py -v
```

Expected: PASS. First run downloads the `ms-marco-MiniLM-L-6-v2` cross-encoder weights — needs network access once.

- [ ] **Step 7: Commit**

```bash
git add core/retrieval/hybrid.py core/retrieval/rerank.py core/retrieval/pipeline.py tests/test_hybrid_pipeline.py
git commit -m "$(cat <<'EOF'
Add RRF fusion, cross-encoder reranker, and hybrid retrieval pipeline

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Build-index script for the instrument_support domain

**Files:**
- Create: `scripts/build_index.py`
- Create: `domains/instrument_support/config.py`
- Test: `tests/test_build_index.py`

**Interfaces:**
- Consumes: `parse_pdf`, `parse_html` (Tasks 3-4), `chunk_blocks` (Task 2), `DenseIndex` (Task 7), `corpus.model_facts.MODELS` (Task 5).
- Produces: `scripts.build_index.build_chunks() -> list[Chunk]`, `main()`; `domains.instrument_support.config.load_chunks() -> list[Chunk]`, `build_retriever() -> HybridRetriever`, `KNOWN_MODEL_NUMBERS: set[str]`. `build_retriever` and `KNOWN_MODEL_NUMBERS` are consumed by Task 11 (agent), Task 15 (FastAPI), Task 16 (Streamlit).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_index.py
from pathlib import Path

from core.ingestion.models import BlockType
from corpus.render_corpus import render_to_html, render_to_pdf
from scripts.build_index import _doc_id_and_type, build_chunks


def test_doc_id_and_type_parses_filename_convention():
    doc_id, doc_type, model_number = _doc_id_and_type(Path("DM-4500_manual.pdf"))
    assert doc_id == "DM-4500_manual"
    assert doc_type == "manual"
    assert model_number == "DM-4500"


def test_doc_id_and_type_handles_unknown_model_number():
    doc_id, doc_type, model_number = _doc_id_and_type(Path("XX-0000_manual.pdf"))
    assert model_number is None


def test_build_chunks_walks_a_directory_of_mixed_pdf_and_html(tmp_path, monkeypatch):
    import scripts.build_index as bi

    corpus_dir = tmp_path / "rendered" / "density_meters"
    corpus_dir.mkdir(parents=True)

    manual_md = tmp_path / "manual.md"
    manual_md.write_text("# Overview\n\nSome overview text.\n")
    render_to_pdf(manual_md, corpus_dir / "DM-2100_manual.pdf")

    spec_md = tmp_path / "spec.md"
    spec_md.write_text("# Specifications\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n")
    render_to_html(spec_md, corpus_dir / "DM-2100_spec_sheet.html")

    monkeypatch.setattr(bi, "CORPUS_DIR", tmp_path / "rendered")

    chunks = build_chunks()

    assert any(c.doc_id == "DM-2100_manual" and c.model_number == "DM-2100" for c in chunks)
    assert any(c.doc_id == "DM-2100_spec_sheet" and c.doc_type == "spec_sheet" for c in chunks)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_build_index.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_index'`.

- [ ] **Step 3: Write `scripts/build_index.py`**

```python
import json
from pathlib import Path

from core.ingestion.chunker import chunk_blocks
from core.ingestion.models import Chunk
from core.ingestion.parsers import parse_html, parse_pdf
from core.retrieval.index import DenseIndex
from corpus.model_facts import MODELS

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus" / "rendered"
CHUNKS_PATH = Path(__file__).resolve().parents[1] / "data" / "instrument_support_chunks.jsonl"
CHROMA_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma"
COLLECTION_NAME = "instrument_support"

_DOC_TYPES = ("manual", "spec_sheet", "app_report")
KNOWN_MODEL_NUMBERS = {m["model_number"] for m in MODELS}


def _doc_id_and_type(path: Path) -> tuple[str, str, str | None]:
    stem = path.stem
    for doc_type in _DOC_TYPES:
        suffix = f"_{doc_type}"
        if stem.endswith(suffix):
            candidate = stem[: -len(suffix)]
            model_number = candidate if candidate in KNOWN_MODEL_NUMBERS else None
            return stem, doc_type, model_number
    return stem, "unknown", None


def build_chunks() -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(CORPUS_DIR.rglob("*")):
        if path.suffix not in (".pdf", ".html"):
            continue
        doc_id, doc_type, model_number = _doc_id_and_type(path)
        blocks = parse_pdf(str(path)) if path.suffix == ".pdf" else parse_html(str(path))
        all_chunks.extend(chunk_blocks(blocks, doc_id=doc_id, model_number=model_number, doc_type=doc_type))
    return all_chunks


def main() -> None:
    chunks = build_chunks()

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__) + "\n")

    dense_index = DenseIndex(collection_name=COLLECTION_NAME, persist_dir=str(CHROMA_DIR))
    dense_index.add(chunks)

    print(f"Indexed {len(chunks)} chunks from {CORPUS_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_build_index.py -v
```

Expected: PASS.

- [ ] **Step 5: Write `domains/instrument_support/config.py`**

```python
import json
from pathlib import Path

from core.ingestion.models import Chunk
from core.retrieval.bm25_index import BM25Index
from core.retrieval.index import DenseIndex
from core.retrieval.pipeline import HybridRetriever
from core.retrieval.rerank import Reranker
from corpus.model_facts import MODELS

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CHUNKS_PATH = DATA_DIR / "instrument_support_chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "instrument_support"

KNOWN_MODEL_NUMBERS = {m["model_number"] for m in MODELS}


def load_chunks() -> list[Chunk]:
    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def build_retriever() -> HybridRetriever:
    chunks = load_chunks()
    dense_index = DenseIndex(collection_name=COLLECTION_NAME, persist_dir=str(CHROMA_DIR))
    bm25_index = BM25Index()
    bm25_index.build(chunks)
    return HybridRetriever(dense_index, bm25_index, Reranker())
```

- [ ] **Step 6: Commit**

```bash
git add scripts/build_index.py domains/instrument_support/config.py tests/test_build_index.py
git commit -m "$(cat <<'EOF'
Add index-build script and instrument_support domain wiring

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: Build the real index from the rendered corpus**

```bash
python -m scripts.build_index
git add data/instrument_support_chunks.jsonl data/chroma
git commit -m "$(cat <<'EOF'
Build and commit the persisted chunk store and Chroma index

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

Expected: prints a chunk count (roughly 100-200 depending on how sections split); `data/instrument_support_chunks.jsonl` and `data/chroma/` now exist.

---

### Task 11: Grounded generation

**Files:**
- Create: `core/generation/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `RetrievalResult` from Task 9, `tests.fakes.FakeAnthropicClient`.
- Produces: `GroundedAnswer(text: str, citations: list[str], insufficient: bool, input_tokens: int, output_tokens: int)`, `build_context_block(results) -> str`, `generate_grounded_answer(query, results, client, model) -> GroundedAnswer`. Consumed by Task 12 (agent) and Task 13 (eval runner).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate.py
from core.generation.generate import build_context_block, generate_grounded_answer
from core.ingestion.models import Chunk
from core.retrieval.pipeline import RetrievalResult
from tests.fakes import FakeAnthropicClient


def _result(text, doc_id="DM-4500_manual", section_path="Error Codes > E-104"):
    chunk = Chunk(chunk_id="c1", text=text, doc_id=doc_id, model_number="DM-4500", section_path=section_path, doc_type="manual")
    return RetrievalResult(chunk=chunk, score=0.9)


def test_build_context_block_includes_doc_id_and_section():
    block = build_context_block([_result("Air bubble detected.")])
    assert "DM-4500_manual" in block
    assert "Error Codes > E-104" in block
    assert "Air bubble detected." in block


def test_generate_grounded_answer_extracts_citations_and_usage():
    client = FakeAnthropicClient(
        reply_text="Purge the cell and refill slowly. [DM-4500_manual, Error Codes > E-104]",
        input_tokens=42,
        output_tokens=17,
    )
    answer = generate_grounded_answer("What does E-104 mean?", [_result("Air bubble detected.")], client, "claude-haiku-4-5")

    assert "DM-4500_manual, Error Codes > E-104" in answer.citations
    assert answer.insufficient is False
    assert answer.input_tokens == 42
    assert answer.output_tokens == 17


def test_generate_grounded_answer_flags_insufficient_information():
    client = FakeAnthropicClient(reply_text="Insufficient information in the available documentation.")
    answer = generate_grounded_answer("What is the warranty period?", [_result("Air bubble detected.")], client, "claude-haiku-4-5")
    assert answer.insufficient is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_generate.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.generation.generate'`.

- [ ] **Step 3: Write `core/generation/generate.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_generate.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/generation/generate.py tests/test_generate.py
git commit -m "$(cat <<'EOF'
Add grounded generation with citation extraction and usage tracking

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Domain base interface and InstrumentSupportAgent

**Files:**
- Create: `domains/base.py`
- Create: `domains/instrument_support/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `HybridRetriever`, `RetrievalResult` (Task 9), `generate_grounded_answer`, `GroundedAnswer` (Task 11).
- Produces: `Response(draft: str, citations: list[str], confidence: float, escalate: bool)`, `Agent` protocol with `.handle(request) -> Response`; `Ticket(symptom_or_error_code: str, model_number: str | None = None, free_text: str = "")`, `InstrumentSupportAgent(retriever, client, model, known_model_numbers, confidence_threshold=0.4)` with `.handle(request: Ticket) -> Response` and `.handle_with_metadata(request: Ticket) -> tuple[Response, dict]` (metadata has `latency_ms`, `input_tokens`, `output_tokens`). Consumed by Task 15 (FastAPI) and Task 16 (Streamlit).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent.py
from core.ingestion.models import Chunk
from core.retrieval.pipeline import RetrievalResult
from domains.instrument_support.agent import InstrumentSupportAgent, Ticket
from tests.fakes import FakeAnthropicClient


class _FakeRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query, top_k=5, model_number_filter=None):
        return self._results


def _result(score, doc_id="DM-4500_manual", section_path="Error Codes > E-104"):
    chunk = Chunk(chunk_id="c1", text="Air bubble detected.", doc_id=doc_id, model_number="DM-4500", section_path=section_path, doc_type="manual")
    return RetrievalResult(chunk=chunk, score=score)


def test_handle_returns_grounded_answer_when_confidence_is_high():
    retriever = _FakeRetriever([_result(score=0.9)])
    client = FakeAnthropicClient(reply_text="Purge and refill. [DM-4500_manual, Error Codes > E-104]")
    agent = InstrumentSupportAgent(retriever, client, "claude-haiku-4-5", known_model_numbers={"DM-4500"})

    response = agent.handle(Ticket(symptom_or_error_code="E-104", model_number="DM-4500"))

    assert response.escalate is False
    assert response.confidence == 0.9
    assert "DM-4500_manual, Error Codes > E-104" in response.citations


def test_handle_escalates_when_confidence_is_below_threshold():
    retriever = _FakeRetriever([_result(score=0.1)])
    client = FakeAnthropicClient(reply_text="Purge and refill. [DM-4500_manual, Error Codes > E-104]")
    agent = InstrumentSupportAgent(retriever, client, "claude-haiku-4-5", known_model_numbers={"DM-4500"}, confidence_threshold=0.4)

    response = agent.handle(Ticket(symptom_or_error_code="E-104", model_number="DM-4500"))

    assert response.escalate is True


def test_handle_escalates_immediately_for_unknown_model_number():
    retriever = _FakeRetriever([_result(score=0.9)])
    client = FakeAnthropicClient(reply_text="should not be called")
    agent = InstrumentSupportAgent(retriever, client, "claude-haiku-4-5", known_model_numbers={"DM-4500"})

    response = agent.handle(Ticket(symptom_or_error_code="E-999", model_number="ZZ-0000"))

    assert response.escalate is True
    assert response.confidence == 0.0
    assert response.citations == []


def test_handle_with_metadata_reports_latency_and_token_usage():
    retriever = _FakeRetriever([_result(score=0.9)])
    client = FakeAnthropicClient(reply_text="Purge and refill.", input_tokens=30, output_tokens=12)
    agent = InstrumentSupportAgent(retriever, client, "claude-haiku-4-5", known_model_numbers={"DM-4500"})

    response, metadata = agent.handle_with_metadata(Ticket(symptom_or_error_code="E-104", model_number="DM-4500"))

    assert metadata["input_tokens"] == 30
    assert metadata["output_tokens"] == 12
    assert metadata["latency_ms"] >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'domains.instrument_support.agent'`.

- [ ] **Step 3: Write `domains/base.py`**

```python
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
```

- [ ] **Step 4: Write `domains/instrument_support/agent.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_agent.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add domains/base.py domains/instrument_support/agent.py tests/test_agent.py
git commit -m "$(cat <<'EOF'
Add domain Agent interface and InstrumentSupportAgent ticket triage

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Eval metrics

**Files:**
- Create: `core/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: `Chunk` (Task 2), `GroundedAnswer` (Task 11).
- Produces: `is_relevant(chunk, correct_doc_id, correct_section) -> bool`, `precision_at_k(retrieved: list[Chunk], correct_sources: list[tuple[str,str]], k) -> float`, `recall_at_k(...) -> float`, `judge_faithfulness(answer, cited_chunks, client, model) -> bool`, `hallucination_rate(results: list[dict]) -> float`. Consumed by Task 14 (eval runner).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_metrics.py
from core.eval.metrics import hallucination_rate, is_relevant, judge_faithfulness, precision_at_k, recall_at_k
from core.generation.generate import GroundedAnswer
from core.ingestion.models import Chunk
from tests.fakes import FakeAnthropicClient


def _chunk(doc_id, section_path):
    return Chunk(chunk_id="c1", text="text", doc_id=doc_id, model_number=None, section_path=section_path, doc_type="manual")


def test_is_relevant_matches_doc_id_and_section_substring():
    chunk = _chunk("DM-4500_manual", "Error Codes > E-104")
    assert is_relevant(chunk, "DM-4500_manual", "E-104") is True
    assert is_relevant(chunk, "DM-4500_manual", "E-999") is False
    assert is_relevant(chunk, "DM-7000_manual", "E-104") is False


def test_precision_and_recall_at_k():
    retrieved = [_chunk("DM-4500_manual", "Error Codes > E-104"), _chunk("DM-4500_manual", "Overview")]
    correct = [("DM-4500_manual", "E-104")]

    assert precision_at_k(retrieved, correct, k=2) == 0.5
    assert recall_at_k(retrieved, correct, k=2) == 1.0
    assert recall_at_k(retrieved, [], k=2) == 0.0


def test_judge_faithfulness_parses_json_verdict():
    client = FakeAnthropicClient(reply_text='{"faithful": true}')
    answer = GroundedAnswer(text="Purge the cell.", citations=[], insufficient=False, input_tokens=1, output_tokens=1)
    assert judge_faithfulness(answer, [_chunk("DM-4500_manual", "E-104")], client, "claude-haiku-4-5") is True


def test_hallucination_rate_only_counts_out_of_scope_items():
    results = [
        {"out_of_scope": True, "refused": True},
        {"out_of_scope": True, "refused": False},
        {"out_of_scope": False, "refused": False},
    ]
    assert hallucination_rate(results) == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_eval_metrics.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.eval.metrics'`.

- [ ] **Step 3: Write `core/eval/metrics.py`**

```python
import json

from core.generation.generate import GroundedAnswer
from core.ingestion.models import Chunk

JUDGE_SYSTEM_PROMPT = (
    "You are grading whether an answer's claims are supported by the given "
    'source excerpts. Respond with strict JSON: {"faithful": true or false}. '
    "Mark faithful=false if the answer states anything not present in the excerpts."
)


def is_relevant(chunk: Chunk, correct_doc_id: str, correct_section: str) -> bool:
    return chunk.doc_id == correct_doc_id and correct_section.lower() in chunk.section_path.lower()


def precision_at_k(retrieved: list[Chunk], correct_sources: list[tuple[str, str]], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for c in top_k if any(is_relevant(c, d, s) for d, s in correct_sources))
    return hits / len(top_k)


def recall_at_k(retrieved: list[Chunk], correct_sources: list[tuple[str, str]], k: int) -> float:
    if not correct_sources:
        return 0.0
    top_k = retrieved[:k]
    found = {(d, s) for d, s in correct_sources if any(is_relevant(c, d, s) for c in top_k)}
    return len(found) / len(correct_sources)


def judge_faithfulness(answer: GroundedAnswer, cited_chunks: list[Chunk], client, model: str) -> bool:
    excerpts = "\n\n---\n\n".join(c.text for c in cited_chunks)
    user_prompt = f"Excerpts:\n\n{excerpts}\n\nAnswer to grade:\n\n{answer.text}"
    response = client.messages.create(
        model=model,
        max_tokens=100,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    try:
        return bool(json.loads(response.content[0].text).get("faithful", False))
    except (json.JSONDecodeError, AttributeError):
        return False


def hallucination_rate(results: list[dict]) -> float:
    out_of_scope = [r for r in results if r.get("out_of_scope")]
    if not out_of_scope:
        return 0.0
    fabricated = sum(1 for r in out_of_scope if not r.get("refused", False))
    return fabricated / len(out_of_scope)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_eval_metrics.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/eval/metrics.py tests/test_eval_metrics.py
git commit -m "$(cat <<'EOF'
Add retrieval precision/recall, LLM-judge faithfulness, and hallucination rate metrics

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Eval test set and runner

**Files:**
- Create: `eval_data/instrument_support/testset.json`
- Create: `core/eval/testset.py`
- Create: `core/eval/runner.py`
- Test: `tests/test_eval_runner.py`

**Interfaces:**
- Consumes: `precision_at_k`, `recall_at_k`, `judge_faithfulness`, `hallucination_rate` (Task 13); `generate_grounded_answer` (Task 11); `HybridRetriever`-compatible `.retrieve(...)` (Task 9).
- Produces: `EvalItem(query, correct_sources: list[tuple[str,str]], out_of_scope: bool = False)`, `run_eval(items, retriever, client, gen_model, judge_model) -> dict` (`{"per_item": [...], "aggregate": {...}}`); `load_testset(path) -> list[EvalItem]`. Consumed by Task 17 (ablation script).

- [ ] **Step 1: Write the 40-item eval test set**

```json
[
  {"query": "What is the density measurement range of the DM-2100?", "correct_sources": [["DM-2100_spec_sheet", "Specifications"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What does error code E-101 mean on the DM-2100?", "correct_sources": [["DM-2100_manual", "Error Codes > E-101"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "How many calibration reference points does the DM-2100 use during calibration?", "correct_sources": [["DM-2100_manual", "Calibration Procedure"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What should I do if the DM-2100 shows a temperature stabilization timeout?", "correct_sources": [["DM-2100_manual", "Error Codes > E-201"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What is the accuracy of the DM-4500?", "correct_sources": [["DM-4500_spec_sheet", "Specifications"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What does error E-104 mean on the DM-4500?", "correct_sources": [["DM-4500_manual", "Error Codes > E-104"]], "out_of_scope": false, "tag": "collision"},
  {"query": "What is the sample volume required by the DM-4500?", "correct_sources": [["DM-4500_spec_sheet", "Specifications"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What should I do if the DM-4500 reports a viscosity correction sensor fault?", "correct_sources": [["DM-4500_manual", "Error Codes > E-301"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What temperature range does the DM-7000 support?", "correct_sources": [["DM-7000_spec_sheet", "Specifications"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What does error E-104 mean on the DM-7000?", "correct_sources": [["DM-7000_manual", "Error Codes > E-104"]], "out_of_scope": false, "tag": "collision"},
  {"query": "How do I resolve a sample changer jam, error E-402, on the DM-7000?", "correct_sources": [["DM-7000_manual", "Error Codes > E-402"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "How many reference standards does the DM-7000's automated calibration routine use?", "correct_sources": [["DM-7000_manual", "Calibration Procedure"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What is the torque range of the RH-150?", "correct_sources": [["RH-150_spec_sheet", "Specifications"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What does error E-501 mean on the RH-150?", "correct_sources": [["RH-150_manual", "Error Codes > E-501"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What should I do if the RH-150 needs a gap calibration, error E-601?", "correct_sources": [["RH-150_manual", "Error Codes > E-601"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "Does the RH-150 have active temperature control?", "correct_sources": [["RH-150_spec_sheet", "Specifications"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What temperature range can the RH-350's Peltier plate reach?", "correct_sources": [["RH-350_spec_sheet", "Specifications"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What does error E-602 mean on the RH-350?", "correct_sources": [["RH-350_manual", "Error Codes > E-602"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "How do I resolve a Peltier plate overtemperature, error E-701, on the RH-350?", "correct_sources": [["RH-350_manual", "Error Codes > E-701"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What is the torque range of the RH-350?", "correct_sources": [["RH-350_spec_sheet", "Specifications"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What does error E-801 mean on the RH-900?", "correct_sources": [["RH-900_manual", "Error Codes > E-801"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What is the maximum rotational speed of the RH-900?", "correct_sources": [["RH-900_spec_sheet", "Specifications"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "How do I resolve a normal force sensor drift, error E-602, on the RH-900?", "correct_sources": [["RH-900_manual", "Error Codes > E-602"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "What is the minimum operating temperature of the RH-900?", "correct_sources": [["RH-900_spec_sheet", "Specifications"]], "out_of_scope": false, "tag": "straightforward"},
  {"query": "The DM-2100 reading drifts upward slowly during a measurement, what's wrong?", "correct_sources": [["DM-2100_manual", "Troubleshooting"], ["DM-2100_manual", "Error Codes > E-201"]], "out_of_scope": false, "tag": "symptom_two_hop"},
  {"query": "On the DM-4500, the reading is unstable and drifts erratically mid-measurement. What should I check?", "correct_sources": [["DM-4500_manual", "Troubleshooting"], ["DM-4500_manual", "Error Codes > E-104"]], "out_of_scope": false, "tag": "symptom_two_hop"},
  {"query": "The DM-7000 stops mid-run with the carousel motor still audible. What's happening?", "correct_sources": [["DM-7000_manual", "Troubleshooting"], ["DM-7000_manual", "Error Codes > E-402"]], "out_of_scope": false, "tag": "symptom_two_hop"},
  {"query": "On the RH-150, the torque reading pins at maximum immediately on startup. What does that indicate?", "correct_sources": [["RH-150_manual", "Troubleshooting"], ["RH-150_manual", "Error Codes > E-501"]], "out_of_scope": false, "tag": "symptom_two_hop"},
  {"query": "The RH-350's normal force reading does not return to zero when the geometry is lifted clear of the sample. What should I do?", "correct_sources": [["RH-350_manual", "Troubleshooting"], ["RH-350_manual", "Error Codes > E-602"]], "out_of_scope": false, "tag": "symptom_two_hop"},
  {"query": "The RH-900 aborts an oscillation test immediately when a high frequency sweep is requested. Why?", "correct_sources": [["RH-900_manual", "Troubleshooting"], ["RH-900_manual", "Error Codes > E-801"]], "out_of_scope": false, "tag": "symptom_two_hop"},
  {"query": "What is the density range of the DM-9999?", "correct_sources": [], "out_of_scope": true, "tag": "out_of_scope"},
  {"query": "Does this instrument line make a viscosity meter called the VM-100?", "correct_sources": [], "out_of_scope": true, "tag": "out_of_scope"},
  {"query": "What does error code E-999 mean?", "correct_sources": [], "out_of_scope": true, "tag": "out_of_scope"},
  {"query": "Can the RH-150 measure density directly?", "correct_sources": [], "out_of_scope": true, "tag": "out_of_scope"},
  {"query": "What is the warranty period for the DM-4500?", "correct_sources": [], "out_of_scope": true, "tag": "out_of_scope"},
  {"query": "How do I connect the DM-7000 to a LIMS system via its REST API?", "correct_sources": [], "out_of_scope": true, "tag": "out_of_scope"},
  {"query": "What is the calibration procedure for the RH-2000?", "correct_sources": [], "out_of_scope": true, "tag": "out_of_scope"},
  {"query": "Does the DM-2100 support Bluetooth connectivity?", "correct_sources": [], "out_of_scope": true, "tag": "out_of_scope"},
  {"query": "What is the list price of the RH-900?", "correct_sources": [], "out_of_scope": true, "tag": "out_of_scope"},
  {"query": "What does error code E-150 mean on the RH-350?", "correct_sources": [], "out_of_scope": true, "tag": "out_of_scope"}
]
```

- [ ] **Step 2: Write the failing test for the test-set loader and runner**

```python
# tests/test_eval_runner.py
import json

from core.eval.runner import EvalItem, run_eval
from core.eval.testset import load_testset
from core.ingestion.models import Chunk
from core.retrieval.pipeline import RetrievalResult
from tests.fakes import FakeAnthropicClient


def test_load_testset_returns_40_items_with_tuples(tmp_path):
    data = [{"query": "q", "correct_sources": [["doc1", "sec1"]], "out_of_scope": False}]
    path = tmp_path / "testset.json"
    path.write_text(json.dumps(data))

    items = load_testset(str(path))

    assert len(items) == 1
    assert items[0].correct_sources == [("doc1", "sec1")]


def test_real_testset_has_40_items_and_expected_tags():
    items = json.loads(open("eval_data/instrument_support/testset.json").read())
    assert len(items) == 40
    assert sum(1 for i in items if i.get("out_of_scope")) >= 8
    assert sum(1 for i in items if i.get("tag") == "collision") == 2


class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, top_k=5, model_number_filter=None):
        return [RetrievalResult(c, 1.0) for c in self._chunks[:top_k]]


def test_run_eval_computes_aggregate_metrics():
    chunk = Chunk(chunk_id="c1", text="Air bubble detected.", doc_id="DM-4500_manual", model_number="DM-4500", section_path="Error Codes > E-104", doc_type="manual")
    retriever = _FakeRetriever([chunk])
    client = FakeAnthropicClient(reply_text='{"faithful": true}')

    items = [EvalItem(query="What does E-104 mean?", correct_sources=[("DM-4500_manual", "E-104")])]
    result = run_eval(items, retriever, client, "claude-haiku-4-5", "claude-haiku-4-5")

    assert result["aggregate"]["precision_at_5"] == 1.0
    assert result["aggregate"]["recall_at_5"] == 1.0
    assert result["aggregate"]["hallucination_rate"] == 0.0
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_eval_runner.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.eval.runner'`.

- [ ] **Step 4: Write `core/eval/runner.py`**

```python
from dataclasses import dataclass, field

from core.eval.metrics import hallucination_rate, judge_faithfulness, precision_at_k, recall_at_k
from core.generation.generate import generate_grounded_answer


@dataclass
class EvalItem:
    query: str
    correct_sources: list[tuple[str, str]] = field(default_factory=list)
    out_of_scope: bool = False


def run_eval(items: list[EvalItem], retriever, client, gen_model: str, judge_model: str) -> dict:
    per_item = []
    for item in items:
        results = retriever.retrieve(item.query, top_k=5)
        chunks = [r.chunk for r in results]

        answer = generate_grounded_answer(item.query, results, client, gen_model)
        refused = answer.insufficient
        faithful = None if refused else judge_faithfulness(answer, chunks[:3], client, judge_model)

        per_item.append(
            {
                "query": item.query,
                "precision_at_3": precision_at_k(chunks, item.correct_sources, 3),
                "precision_at_5": precision_at_k(chunks, item.correct_sources, 5),
                "recall_at_3": recall_at_k(chunks, item.correct_sources, 3),
                "recall_at_5": recall_at_k(chunks, item.correct_sources, 5),
                "faithful": faithful,
                "refused": refused,
                "out_of_scope": item.out_of_scope,
            }
        )

    n = len(per_item)
    faithful_scored = [r["faithful"] for r in per_item if r["faithful"] is not None]

    aggregate = {
        "precision_at_3": sum(r["precision_at_3"] for r in per_item) / n,
        "precision_at_5": sum(r["precision_at_5"] for r in per_item) / n,
        "recall_at_3": sum(r["recall_at_3"] for r in per_item) / n,
        "recall_at_5": sum(r["recall_at_5"] for r in per_item) / n,
        "faithfulness": (sum(faithful_scored) / len(faithful_scored)) if faithful_scored else None,
        "hallucination_rate": hallucination_rate(per_item),
    }
    return {"per_item": per_item, "aggregate": aggregate}
```

- [ ] **Step 5: Write `core/eval/testset.py`**

```python
import json

from core.eval.runner import EvalItem


def load_testset(path: str) -> list[EvalItem]:
    with open(path) as f:
        raw = json.load(f)
    return [
        EvalItem(
            query=item["query"],
            correct_sources=[tuple(pair) for pair in item["correct_sources"]],
            out_of_scope=item.get("out_of_scope", False),
        )
        for item in raw
    ]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_eval_runner.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add eval_data/instrument_support/testset.json core/eval/testset.py core/eval/runner.py tests/test_eval_runner.py
git commit -m "$(cat <<'EOF'
Add 40-item eval test set and eval runner producing per-item and aggregate metrics

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: 3-way ablation script

**Files:**
- Create: `scripts/run_eval.py`

**Interfaces:**
- Consumes: `load_testset` (Task 14), `load_chunks`, `CHROMA_DIR`, `COLLECTION_NAME` (Task 10), `BM25OnlyRetriever`, `DenseOnlyRetriever`, `HybridRetriever` (Task 9), `run_eval` (Task 14).
- Produces: `results.csv`, `results_table.md` at the repo root.

This script calls the real Anthropic API and is not covered by an automated test — Tasks 11-14 already unit-test every function it composes with a fake client.

- [ ] **Step 1: Write `scripts/run_eval.py`**

```python
import csv
from pathlib import Path

import anthropic

from core.config import GENERATION_MODEL, JUDGE_MODEL
from core.eval.runner import run_eval
from core.eval.testset import load_testset
from core.retrieval.bm25_index import BM25Index
from core.retrieval.index import DenseIndex
from core.retrieval.pipeline import BM25OnlyRetriever, DenseOnlyRetriever, HybridRetriever
from core.retrieval.rerank import Reranker
from domains.instrument_support.config import CHROMA_DIR, COLLECTION_NAME, load_chunks

TESTSET_PATH = Path(__file__).resolve().parents[1] / "eval_data" / "instrument_support" / "testset.json"
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results.csv"
RESULTS_TABLE_PATH = Path(__file__).resolve().parents[1] / "results_table.md"


def main() -> None:
    items = load_testset(str(TESTSET_PATH))
    chunks = load_chunks()

    dense_index = DenseIndex(collection_name=COLLECTION_NAME, persist_dir=str(CHROMA_DIR))
    bm25_index = BM25Index()
    bm25_index.build(chunks)
    reranker = Reranker()
    client = anthropic.Anthropic()

    configs = {
        "bm25_only": BM25OnlyRetriever(bm25_index),
        "dense_only": DenseOnlyRetriever(dense_index),
        "hybrid_rerank": HybridRetriever(dense_index, bm25_index, reranker),
    }

    rows = []
    for name, retriever in configs.items():
        result = run_eval(items, retriever, client, GENERATION_MODEL, JUDGE_MODEL)
        rows.append({"method": name, **result["aggregate"]})
        print(name, result["aggregate"])

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    headers = list(rows[0].keys())
    with open(RESULTS_TABLE_PATH, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_eval.py
git commit -m "$(cat <<'EOF'
Add 3-way retrieval ablation script (BM25-only / dense-only / hybrid+rerank)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Run the ablation against the real corpus and API (requires `ANTHROPIC_API_KEY`)**

```bash
python -m scripts.run_eval
```

Expected: prints three dicts of aggregate metrics; `results.csv` and `results_table.md` are created at the repo root. Read the printed numbers — if `hybrid_rerank` does not clearly beat `bm25_only` and `dense_only` on `precision_at_5`/`recall_at_5`, use `superpowers:systematic-debugging` to investigate before moving on (e.g. check whether the reranker's `fetch_k` is too small, or whether chunk `section_path` metadata doesn't match the eval set's expected strings).

- [ ] **Step 4: Commit the results**

```bash
git add results.csv results_table.md
git commit -m "$(cat <<'EOF'
Add 3-way ablation results

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Observability logging

**Files:**
- Create: `observability/db.py`
- Create: `scripts/observability_summary.py`
- Test: `tests/test_observability.py`

**Interfaces:**
- Consumes: `Response` (Task 12).
- Produces: `DB_PATH`, `log_request(query: str, response: Response, metadata: dict) -> None`. Consumed by Task 17 (FastAPI).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_observability.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_observability.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'observability.db'`.

- [ ] **Step 3: Write `observability/db.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_observability.py -v
```

Expected: PASS.

- [ ] **Step 5: Write `scripts/observability_summary.py`**

```python
import sqlite3
import statistics

from core.config import HAIKU_INPUT_COST_PER_MTOK, HAIKU_OUTPUT_COST_PER_MTOK
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
    total_cost = sum(
        (r[3] / 1_000_000) * HAIKU_INPUT_COST_PER_MTOK + (r[4] / 1_000_000) * HAIKU_OUTPUT_COST_PER_MTOK
        for r in rows
    )

    def percentile(sorted_values: list[float], p: float) -> float:
        idx = min(int(len(sorted_values) * p), len(sorted_values) - 1)
        return sorted_values[idx]

    print(f"Total requests: {len(rows)}")
    print(f"Avg confidence: {statistics.mean(confidences):.3f}")
    print(f"Escalation rate: {sum(escalations) / len(rows):.1%}")
    print(f"Latency p50: {percentile(latencies, 0.5):.0f} ms, p95: {percentile(latencies, 0.95):.0f} ms")
    print(f"Total estimated cost: ${total_cost:.4f} (avg ${total_cost / len(rows):.5f} per query)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add observability/db.py scripts/observability_summary.py tests/test_observability.py
git commit -m "$(cat <<'EOF'
Add SQLite request logging and observability summary script

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 17: FastAPI serving

**Files:**
- Create: `serving/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `InstrumentSupportAgent`, `Ticket` (Task 12), `build_retriever`, `KNOWN_MODEL_NUMBERS` (Task 10), `log_request` (Task 16), `GENERATION_MODEL` (Task 1).
- Produces: FastAPI `app` with `GET /health`, `POST /ticket`, `POST /ask`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
from fastapi.testclient import TestClient

import serving.api as api_module
from domains.base import Response


class _FakeAgent:
    def handle_with_metadata(self, ticket):
        response = Response(draft="Purge and refill.", citations=["DM-4500_manual, E-104"], confidence=0.9, escalate=False)
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
    resp = client.post("/ticket", json={"symptom_or_error_code": "E-104", "model_number": "DM-4500"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["draft"] == "Purge and refill."
    assert body["escalate"] is False
    assert body["confidence"] == 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'serving.api'`.

- [ ] **Step 3: Write `serving/api.py`**

```python
import anthropic
from fastapi import FastAPI
from pydantic import BaseModel

from core.config import GENERATION_MODEL
from domains.instrument_support.agent import InstrumentSupportAgent, Ticket
from domains.instrument_support.config import KNOWN_MODEL_NUMBERS, build_retriever
from observability.db import log_request

app = FastAPI()

_client = anthropic.Anthropic()
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_api.py -v
```

Expected: PASS. This test builds the real retriever and Anthropic client at import time (module-level `_retriever = build_retriever()`), so it requires the committed `data/instrument_support_chunks.jsonl` and `data/chroma` from Task 10 and a network connection for the embedding model — the `_agent`/`log_request` are what's monkeypatched, not the retriever construction.

- [ ] **Step 5: Commit**

```bash
git add serving/api.py tests/test_api.py
git commit -m "$(cat <<'EOF'
Add FastAPI serving layer with /health, /ticket, /ask endpoints

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Manually smoke-test the running server**

```bash
uvicorn serving.api:app --reload
```

In another terminal:

```bash
curl -X POST http://localhost:8000/ticket -H "Content-Type: application/json" -d '{"symptom_or_error_code": "E-104", "model_number": "DM-4500"}'
```

Expected: JSON response with a `draft` mentioning purging an air bubble, `escalate: false`, and a citation referencing `DM-4500_manual`. Stop the server (Ctrl-C) when done.

---

### Task 18: Streamlit app

**Files:**
- Create: `serving/streamlit_app.py`

**Interfaces:**
- Consumes: `InstrumentSupportAgent`, `Ticket` (Task 12), `build_retriever`, `KNOWN_MODEL_NUMBERS` (Task 10), `log_request` (Task 16), `GENERATION_MODEL` (Task 1), `corpus.model_facts.MODELS` (Task 5, for the model-number dropdown).

This is the deployed surface (Streamlit Community Cloud) and embeds the core modules directly — no HTTP call to `serving/api.py`. It is verified manually in the browser, not by an automated test, since it's a thin UI wrapper over already-tested logic.

- [ ] **Step 1: Write `serving/streamlit_app.py`**

```python
import anthropic
import streamlit as st

from core.config import GENERATION_MODEL
from corpus.model_facts import MODELS
from domains.instrument_support.agent import InstrumentSupportAgent, Ticket
from domains.instrument_support.config import KNOWN_MODEL_NUMBERS, build_retriever
from observability.db import log_request


@st.cache_resource
def get_agent() -> InstrumentSupportAgent:
    client = anthropic.Anthropic()
    retriever = build_retriever()
    return InstrumentSupportAgent(retriever, client, GENERATION_MODEL, KNOWN_MODEL_NUMBERS)


st.title("Instrument Technical Support Assistant")
st.caption(
    "Demo assistant over a synthetic, LLM-generated instrument documentation corpus "
    "(no real Anton Paar content). See the README for details."
)

model_numbers = ["(unknown)"] + sorted(m["model_number"] for m in MODELS)
model_number = st.selectbox("Instrument model", model_numbers)
symptom = st.text_input("Symptom or error code", placeholder="e.g. E-104")
free_text = st.text_area("Additional details (optional)")

if st.button("Get resolution") and symptom:
    agent = get_agent()
    ticket = Ticket(
        symptom_or_error_code=symptom,
        model_number=None if model_number == "(unknown)" else model_number,
        free_text=free_text,
    )
    response, metadata = agent.handle_with_metadata(ticket)
    log_request(symptom, response, metadata)

    if response.escalate:
        st.warning("Low confidence — this ticket should be escalated to a human technician.")

    st.write(response.draft)
    st.metric("Confidence", f"{response.confidence:.2f}")

    if response.citations:
        with st.expander("Cited sources"):
            for citation in response.citations:
                st.write(f"- {citation}")
```

- [ ] **Step 2: Run it locally and verify manually in the browser**

```bash
streamlit run serving/streamlit_app.py
```

In the browser: select model `DM-4500`, enter symptom `E-104`, click "Get resolution". Expected: a grounded answer describing purging an air bubble, confidence displayed, an expandable citations list showing `DM-4500_manual`. Then try model `(unknown)` with symptom `E-999` — expected: an escalation warning since the code doesn't exist in the corpus. Stop the server (Ctrl-C) when done.

- [ ] **Step 3: Commit**

```bash
git add serving/streamlit_app.py
git commit -m "$(cat <<'EOF'
Add Streamlit UI embedding the core RAG pipeline in-process

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 19: README and Streamlit Cloud deployment

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: `results_table.md` content (Task 15).

- [ ] **Step 1: Write `README.md`**

Include, at minimum:
- One-paragraph problem framing (technical support assistant for a fictional instrument line).
- Explicit disclosure: the corpus is 100% synthetic, LLM-generated content in the same domain as Anton Paar's real product lines (density meters, rheometers) — not scraped or copied from any real vendor's documentation.
- Architecture summary (domain-agnostic core + `instrument_support` domain, matching `docs/superpowers/specs/2026-09-12-instrument-support-assistant-design.md`).
- The 3-way ablation results table, copied from `results_table.md`.
- A short "what fails and why" section written from the actual `results.csv` and per-item eval output (e.g. which retrieval method under-performs on the `collision` and `symptom_two_hop` tagged items, and the measured hallucination rate on `out_of_scope` items).
- Setup instructions: `pip install -e ".[dev]"`, `python -m corpus.generate_corpus`, `python -m corpus.render_corpus`, `python -m scripts.build_index`, `python -m scripts.run_eval`, `streamlit run serving/streamlit_app.py`.
- Link to the deployed Streamlit Community Cloud app (added after Step 2 below).

- [ ] **Step 2: Deploy to Streamlit Community Cloud**

1. Push the repository to GitHub (confirm with the user before pushing/creating a remote if one doesn't already exist).
2. In Streamlit Community Cloud, create a new app pointing at this repo, branch `main`, main file `serving/streamlit_app.py`.
3. In the app's Secrets, add `ANTHROPIC_API_KEY = "..."` (never commit this value).
4. Deploy and verify the live URL loads and answers a test ticket (e.g. `DM-4500` / `E-104`) the same way the local run did in Task 18.
5. Add the live URL to `README.md`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
Add README with architecture, ablation results, and deployment instructions

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** ingestion/chunking (Tasks 2-4), synthetic corpus + disclosure (Tasks 5-6, 19), hybrid retrieval + reranking (Tasks 7-9), domain-agnostic core + `instrument_support` domain (Tasks 10-12), eval set + metrics + 3-way ablation (Tasks 13-15), observability (Task 16), FastAPI + Streamlit serving with in-process embedding (Tasks 17-18), README + deployment (Task 19) — every section of the design spec has a corresponding task.
- **Model ID / pricing correction:** the original design conversation referenced `claude-haiku-4-5-20251001`; the current canonical ID (verified via the `claude-api` skill) is `claude-haiku-4-5`, and Sonnet 5 is `claude-sonnet-5` — both are used consistently in `core/config.py` and every task that references a model string.
- **Type consistency check:** `Chunk`, `Block`/`BlockType`, `RetrievalResult`, `GroundedAnswer`, `Response`, `Ticket`, and `EvalItem` are defined once (Tasks 2, 9, 11, 12, 14) and referenced with identical field names in every later task that consumes them.

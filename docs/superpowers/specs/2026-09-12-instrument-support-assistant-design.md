# Design: Instrument Technical Support Assistant (RAG + Agent)

Date: 2026-09-12
Status: Approved for planning

## Purpose

A resume-grade, end-to-end RAG system demonstrating retrieval quality,
evaluation rigor, and production concerns for an AI Engineer role focused on
ML applications and GenAI use cases in a lab-instrumentation context. The
system acts as a technical support assistant for a fictional line of
laboratory instruments (density meters and rheometers): given a support
ticket ("Model X, error code Y"), it retrieves the relevant manual section,
drafts a grounded resolution with citations, and flags escalation when
retrieval support is weak rather than hallucinating a fix.

Framing: depth on retrieval + evaluation rigor over breadth of features or
agentic orchestration. The "agent" is a single retrieve → draft → self-score
→ decide loop, not a multi-step planner.

## Non-goals

- Not using Anton Paar's own copyrighted documentation or scraping
  anton-paar.com. The corpus is original synthetic content in the same
  domain (density meters, rheometers), disclosed as synthetic in the README.
- Not building the second ("resume/cover-letter curation") use case in this
  project — only building the core so that instance can be added later
  without rewriting it.
- Not building multi-step agentic planning, tool use, or long-horizon
  orchestration. One retrieve/draft/score/decide pass per ticket.
- Not deploying a separate FastAPI network service in production — the
  deployed Streamlit app embeds the core modules in-process (see Serving).

## Architecture: domain-agnostic core + pluggable domain layer

The system is split into a **generic core** (reusable for any future corpus
+ use case) and a **domain layer** (one concrete implementation now,
structurally ready for a second one later).

### Generic core

- `ingestion/` — PDF/HTML parsing and section/procedure-aware chunking.
  Works on any structured document set, not instrument-specific.
- `retrieval/` — BM25 + dense embeddings + cross-encoder reranker over a
  named Chroma collection. Fully domain-agnostic; operates on chunks +
  metadata, not on instrument concepts.
- `generation/` — grounded-answer utility: given retrieved chunks and a
  prompt template, calls the LLM, extracts citations, refuses when support
  is weak. The prompt template is supplied by the domain layer.
- `eval/` — retrieval precision/recall@k, faithfulness (LLM-as-judge), and
  hallucination-rate metrics, computed against any
  `(query, correct_answer, correct_source)` test set.

### Domain layer

- `DomainConfig` — corpus path, vector-collection name, prompt template, and
  an `Agent` implementation.
- `Agent` interface: `handle(request) -> Response(draft, citations,
  confidence, escalate: bool)`.
- **Concrete instance built now:** `instrument_support` domain, backed by
  `InstrumentSupportAgent`.
- *(Not built now — structurally possible later without touching the core:
  a `resume_curation` domain, where the corpus would be past resumes/cover
  letters, the request would be a job posting, and the agent would draft a
  tailored resume/cover-letter section with citations back to prior
  material.)*

The serving layer selects a domain via a `--domain` flag/config value,
**defaulting to `instrument_support`** so the app runs today with no flag
needed.

## Corpus

Two fictional instrument families in Anton Paar's real product space,
entirely original content:

- **Density meters (DM-series):** ~5 fictional models
- **Rheometers (RH-series):** ~5 fictional models

Each model has a manual and a spec sheet; a couple of models also get an
application report. Roughly 15-20 source documents total.

**Generation process:** content is authored with Gemini, then rendered out
as actual PDF and HTML files (mixed across the corpus, not left as clean
markdown) so the ingestion pipeline performs genuine PDF/HTML parsing and
table extraction rather than reading pre-structured text.

**Deliberately engineered retrieval-hard cases:**
- Tables for calibration specs and unit conversions
- Multi-step calibration/troubleshooting procedures
- Overlapping error codes that mean *different* things on different models
  (e.g., `E-104` is a different failure mode on the DM-5400 vs. the DM-8200)
  — this is the case that motivates hybrid search: dense embeddings blur
  the exact code, BM25 alone can't disambiguate which model's manual
  applies without also matching on model number metadata.

The README discloses the corpus is synthetic (LLM-generated, then rendered
to PDF/HTML) and explains why (avoids copyright/ToS concerns around a
potential employer's real documentation while preserving domain fidelity
and giving exact control over eval ground truth).

## Ingestion & chunking

**Parsing:** `pdfplumber` for PDFs (text + table extraction), `BeautifulSoup`
for HTML. Both normalize into a common intermediate representation: a list
of `(heading_path, block_type, content)` tuples, where `block_type` is one
of `text`, `table`, `list`.

**Chunking rules:**
1. Split at heading boundaries first (H1/H2/H3 → section tree) — a chunk
   never straddles two unrelated sections.
2. A `table` block is never split, regardless of token count. Its preceding
   heading/caption is prepended so it retrieves correctly on its own.
3. A numbered procedure is kept as one chunk when it fits under a ~500 token
   ceiling; if it must split, only at step boundaries, never mid-step.
4. Every chunk carries metadata: `doc_id`, `model_number` (parsed from
   filename/header), `section_path`, `doc_type` (manual / spec sheet / app
   report).

**Justification (goes in README):** technical docs pack meaning into
structural units — a table row means nothing without its table, a
calibration step means nothing without its step number and predecessor.
Fixed-token windows would frequently cut a table in half or separate an
error code from its resolution steps, which is exactly the failure mode
this project exists to fix and measure.

## Retrieval & reranking

**Indexing (per domain collection):**
- Dense: `sentence-transformers/all-MiniLM-L6-v2` (local, no extra API key),
  stored in Chroma (file-based persistence, no external service).
- Keyword: `rank_bm25` (BM25Okapi) over the same chunks — catches exact
  model numbers and error codes that dense embeddings tend to blur.

**Hybrid combination:** top-k=20 from each method, combined via Reciprocal
Rank Fusion (no score-scale calibration needed), then the fused top-~15
passed through a cross-encoder reranker
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, local) to produce the final top-4/5
sent to generation.

**Rationale:** BM25 and dense embeddings fail in different, complementary
ways (BM25 misses paraphrases, dense misses exact codes); RRF recovers both
at the rank level, and the cross-encoder then re-scores the fused
candidates jointly with the query for precision, since RRF's fusion signal
is coarse on its own.

## Generation & agent

**Grounded generation:** retrieved chunks (with `doc_id`/`section_path`/
`model_number` metadata) are assembled into a context block sent to Gemini
3.8 Flash (free tier) with a strict system prompt: answer only from the provided excerpts,
cite each claim as `[doc_id, section]`, explicitly say "insufficient
information" rather than filling gaps from general knowledge.

**`InstrumentSupportAgent` flow:**
1. **Input:** ticket `{model_number, symptom_or_error_code, free_text}`
2. **Retrieve:** hybrid+rerank search, filtered/boosted by `model_number`
   metadata when present
3. **Draft:** grounded generation produces a resolution with citations
4. **Confidence:** derived from the reranker's top-chunk score, with an
   explicit hard-0 case when the ticket's `model_number` doesn't match any
   corpus metadata at all (out-of-scope model)
5. **Escalate:** `true` when confidence falls below a threshold tuned
   against the eval set (not hand-picked); response then states
   insufficient documentation and recommends escalation to a human
   technician instead of guessing

This is the entire "agent" logic in the system — no multi-step planning or
tool loops.

## Serving, observability, deployment

**API (FastAPI, standalone/local use):**
- `POST /ticket` — ticket in, `{draft, citations, confidence, escalate}` out
- `POST /ask` — general grounded QA, no ticket framing
- `GET /health`

**UI (Streamlit):** single page — ticket submission form (model dropdown +
symptom/error code + free text), displays the drafted resolution, expandable
cited source chunks, confidence score, and an escalation banner when
flagged.

**Observability:** every request logged to a local SQLite file: latency
breakdown (retrieval / rerank / generation), token counts, retrieved
chunks + scores, final confidence/escalate decision. A
`scripts/observability_summary.py` prints aggregate stats (p50/p95 latency,
token usage, escalation rate). Cost is $0 — the project runs entirely on
the Gemini API's free tier.

**Deployment:** Streamlit Community Cloud. The deployed app embeds the core
retrieval/generation modules directly in-process (Streamlit calling Python
functions, not making a network call to a separate FastAPI service), since
Streamlit Cloud runs a single process. The standalone FastAPI app remains
documented and runnable for local use. Chroma's persisted index is committed
to the repo (small, synthetic corpus) so cold starts don't require
re-embedding. The Gemini API key is stored in Streamlit's secrets
manager, never committed.

## Evaluation

**Test set (~40 items)**, authored alongside the corpus so ground truth is
exact:
- Straightforward single-chunk lookups
- Retrieval-hard cases: same error code across different models (tests
  model-number disambiguation)
- Cases needing 2 chunks (symptom description mapping to an error code
  defined in a different section)
- ~8-10 out-of-scope questions (asking about a model/feature not in the
  corpus at all) — correct behavior is refusal, not invention

Each item: `{query, correct_answer_summary, correct_source_doc,
correct_section}`.

**Metrics:**
- Retrieval precision/recall@k (k=3, 5)
- Faithfulness: LLM-as-judge (separate Gemini call) checks whether the
  generated answer's claims are supported by the chunks it cited,
  independent of whether the answer is "correct"
- Hallucination rate on out-of-scope questions: % where the system
  fabricates an answer instead of refusing/escalating

**Ablation:** the same eval set run three ways — keyword-only (BM25),
dense-only, hybrid+rerank — producing one results table (precision@5,
recall@5, faithfulness, hallucination rate per method) that is the
headline evidence in the README.

**Harness:** `pytest`, with metric computation as plain, importable
functions; results export to `results.csv` and a markdown table generator
for the README.

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | |
| Orchestration | Plain Python, no LangChain | Full control over the hybrid retrieval/rerank chain; easier to explain internals; avoids framework overhead at this scale |
| Corpus generation | Gemini API, free tier (content) → rendered to PDF/HTML | Keeps ingestion demonstrating real parsing work despite synthetic content, at zero cost |
| PDF/HTML parsing | `pdfplumber`, `BeautifulSoup` | |
| Dense embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) | Free, no extra API key, reproducible |
| Keyword search | `rank_bm25` | |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) | Free, standard baseline |
| Vector DB | Chroma | File-based persistence, no external service, fits Streamlit Cloud |
| Generation LLM | Gemini 3.8 Flash (Gemini API, free tier) | Zero cost, strong instruction-following for strict grounding |
| Serving | FastAPI (standalone) + Streamlit (deployed, embeds core directly) | |
| Eval | `pytest` + plain metric functions | |
| Observability | SQLite request log + summary script | |
| Deployment | Streamlit Community Cloud | Free, purpose-built for this |

## Build order

1. Generate + render synthetic corpus; build ingestion/chunking
2. Dense-only baseline RAG working end-to-end on the happy path
3. Add BM25 + RRF + reranker; build the eval set; run the 3-way ablation
4. Add `InstrumentSupportAgent` (ticket triage, confidence/escalation) on
   top of the generic core
5. FastAPI + Streamlit UI, observability logging, deploy to Streamlit
   Cloud, write README with results table front and center

## Deliverables

- Public GitHub repo with clean README: problem framing, architecture
  diagram, eval results table, explicit "what fails and why" section,
  synthetic-corpus disclosure
- 2-3 resume bullets quantifying eval results (e.g. retrieval precision
  improvement from hybrid search, hallucination rate on out-of-scope
  questions)
- Screen-recorded demo walkthrough (optional, alongside the live Streamlit
  Cloud link)

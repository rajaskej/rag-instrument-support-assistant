# tests/test_eval_runner.py
import json

from core.eval.runner import EvalItem, run_eval
from core.eval.testset import load_testset
from core.ingestion.models import Chunk
from core.retrieval.pipeline import RetrievalResult
from tests.fakes import FakeLLMClient


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
    chunk = Chunk(chunk_id="c1", text="Air bubble detected.", doc_id="DM-5400_manual", model_number="DM-5400", section_path="Error Codes > E-104", doc_type="manual")
    retriever = _FakeRetriever([chunk])
    client = FakeLLMClient(reply_text='{"faithful": true}')

    items = [EvalItem(query="What does E-104 mean?", correct_sources=[("DM-5400_manual", "E-104")])]
    result = run_eval(items, retriever, client, "gemini-3.8-flash", "gemini-3.8-flash")

    assert result["aggregate"]["precision_at_5"] == 1.0
    assert result["aggregate"]["recall_at_5"] == 1.0
    assert result["aggregate"]["hallucination_rate"] == 0.0

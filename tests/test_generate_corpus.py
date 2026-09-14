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

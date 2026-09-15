from pathlib import Path

from core.ingestion.models import BlockType
from corpus.render_corpus import render_to_html, render_to_pdf
from scripts.build_index import _doc_id_and_type, build_chunks


def test_doc_id_and_type_parses_filename_convention():
    doc_id, doc_type, model_number = _doc_id_and_type(Path("DM-5400_manual.pdf"))
    assert doc_id == "DM-5400_manual"
    assert doc_type == "manual"
    assert model_number == "DM-5400"


def test_doc_id_and_type_handles_unknown_model_number():
    doc_id, doc_type, model_number = _doc_id_and_type(Path("XX-0000_manual.pdf"))
    assert model_number is None


def test_build_chunks_walks_a_directory_of_mixed_pdf_and_html(tmp_path, monkeypatch):
    import scripts.build_index as bi

    corpus_dir = tmp_path / "rendered" / "density_meters"
    corpus_dir.mkdir(parents=True)

    manual_md = tmp_path / "manual.md"
    manual_md.write_text("# Overview\n\nSome overview text.\n")
    render_to_pdf(manual_md, corpus_dir / "DM-2600_manual.pdf")

    spec_md = tmp_path / "spec.md"
    spec_md.write_text("# Specifications\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n")
    render_to_html(spec_md, corpus_dir / "DM-2600_spec_sheet.html")

    monkeypatch.setattr(bi, "CORPUS_DIR", tmp_path / "rendered")

    chunks = build_chunks()

    assert any(c.doc_id == "DM-2600_manual" and c.model_number == "DM-2600" for c in chunks)
    assert any(c.doc_id == "DM-2600_spec_sheet" and c.doc_type == "spec_sheet" for c in chunks)

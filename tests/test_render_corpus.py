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

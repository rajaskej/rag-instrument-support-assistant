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


def _make_two_table_test_pdf(path: Path) -> None:
    html = f"""
    <html><head><style>
    body {{ font-size: 10pt; font-family: Helvetica; }}
    h1 {{ font-size: {H1_MIN_SIZE + 2}pt; }}
    table, th, td {{ border: 1px solid black; border-collapse: collapse; padding: 4px; }}
    </style></head><body>
    <h1>Section A</h1>
    <table><tr><th>Property</th><th>Value</th></tr><tr><td>RangeA</td><td>0-1 g/cm3</td></tr></table>
    <h1>Section B</h1>
    <table><tr><th>Property</th><th>Value</th></tr><tr><td>RangeB</td><td>0-2 g/cm3</td></tr></table>
    </body></html>
    """
    with open(path, "wb") as f:
        pisa.CreatePDF(html, dest=f)


def test_parse_pdf_assigns_each_table_to_its_preceding_heading_not_the_pages_last(tmp_path):
    """Regression test: a page with H1 A -> table1 -> H1 B -> table2 must tag
    table1 with heading_path ["Section A"] and table2 with ["Section B"], not
    both with the page's final heading."""
    pdf_path = tmp_path / "two_tables.pdf"
    _make_two_table_test_pdf(pdf_path)

    blocks = parse_pdf(str(pdf_path))

    tables = [b for b in blocks if b.block_type == BlockType.TABLE]
    assert len(tables) == 2

    table_a = next(b for b in tables if "RangeA" in b.content)
    table_b = next(b for b in tables if "RangeB" in b.content)

    assert table_a.heading_path == ["Section A"]
    assert table_b.heading_path == ["Section B"]

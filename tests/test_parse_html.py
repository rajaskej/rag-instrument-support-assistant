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

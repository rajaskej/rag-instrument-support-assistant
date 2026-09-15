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

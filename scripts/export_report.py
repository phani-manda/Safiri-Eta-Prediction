"""Convert report/technical_report.md to report/technical_report.pdf.

Uses the `markdown` package (tables + fenced code extensions) to render the
report to HTML, then `xhtml2pdf` to produce the PDF. Both are dev-only
dependencies pinned in requirements-dev.txt; the runtime requirements are
untouched.

    python scripts/export_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "report" / "technical_report.md"
DST = REPO_ROOT / "report" / "technical_report.pdf"

CSS = """
@page {
    size: a4 portrait;
    margin: 1.2cm;
    @frame footer_frame {
        -pdf-frame-content: footer;
        bottom: 0.4cm; left: 1.2cm; width: 18.6cm; height: 0.6cm;
    }
}
body { font-family: helvetica; font-size: 7.5pt; line-height: 1.2; }
p { margin: 2pt 0; }
ul, ol { margin: 2pt 0 2pt 12pt; }
li { margin: 0.5pt 0; }
h1 { font-size: 12.5pt; color: #1a3c5e; margin: 0 0 3pt 0; }
h2 { font-size: 9.5pt; color: #1a3c5e; margin: 6pt 0 2pt 0;
     border-bottom: 1px solid #cccccc; }
table { border-collapse: collapse; font-size: 6.5pt; margin: 2.5pt 0; }
th { background-color: #eef2f6; padding: 1.2pt 2.5pt; border: 0.5pt solid #999999; }
td { padding: 1.2pt 2.5pt; border: 0.5pt solid #bbbbbb; }
pre { font-family: courier; font-size: 6.2pt; background-color: #f5f5f5;
      padding: 3pt; border: 0.5pt solid #dddddd; margin: 2.5pt 0; }
code { font-family: courier; font-size: 7pt; }
"""


def export() -> int:
    md_text = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    html = (
        "<html><head><meta charset='utf-8'><style>" + CSS + "</style></head>"
        "<body>" + body + "<div id='footer'>Safiri ETA Prediction — Technical Report</div></body></html>"
    )
    with DST.open("wb") as fh:
        status = pisa.CreatePDF(html, dest=fh)
    if status.err:
        print("PDF generation failed", file=sys.stderr)
        return 1
    try:
        from pypdf import PdfReader

        pages = len(PdfReader(str(DST)).pages)
        print(f"written {DST.relative_to(REPO_ROOT).as_posix()}  ({pages} pages)")
    except Exception:
        print(f"written {DST.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(export())
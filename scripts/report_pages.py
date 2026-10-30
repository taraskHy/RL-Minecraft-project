"""Estimate the report's page count without pandoc or LibreOffice (headless Chrome, calibrated).

    python scripts/report_pages.py [REV_OR_FILE ...] [--line 1.80] [--para 6]

Renders the body of docs/report_draft.md (everything before "## References"; the working file by
default, or a git revision, or a path) to HTML in the submission format (Letter, one-inch margins,
Arial 12, 1.5 line spacing, single-spaced tables, figures at their natural size capped to the text
width), prints it to PDF with headless Chrome or Edge and reports the page on which the body ends.

It is an estimate and can be a page off: Chrome lays out tables and lists differently from
LibreOffice. Two real exports (scripts/report_docx.py --pdf) calibrate it: 12c736a's body ends
about half a page into page 21 and 25eced6's two lines into page 21. The defaults read 22 and 21
for them, so they err high. Use scripts/report_docx.py --pdf for the real count (the tools are
installed on the training PC; see its docstring).
"""
import argparse
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "chromium-browser"]


def browser() -> str:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    sys.exit("no Chrome, Chromium or Edge found")


def source(spec: str | None) -> str:
    if spec is None:
        return (ROOT / "docs" / "report_draft.md").read_text(encoding="utf-8")
    p = Path(spec)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{spec}:docs/report_draft.md"], capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout


def png_inches(path: Path) -> float | None:
    """Width in inches from the PNG's pixel size and pHYs density, as pandoc sizes images."""
    b = path.read_bytes()
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w = struct.unpack(">I", b[16:20])[0]
    i, dpi = 8, 96.0
    while i < len(b):
        n = struct.unpack(">I", b[i:i + 4])[0]
        kind = b[i + 4:i + 8]
        if kind == b"pHYs":
            ppu, _, unit = struct.unpack(">IIB", b[i + 8:i + 17])
            if unit == 1:
                dpi = ppu * 0.0254
        if kind == b"IDAT":
            break
        i += 12 + n
    return w / dpi


def html(md: str, line: float, para: float, table_line: float) -> str:
    def img(m):
        f = (ROOT / "docs" / m.group(2)).resolve()
        inch = png_inches(f) if f.exists() else None
        width = f"{min(inch, 6.5):.3f}in" if inch else "6.5in"
        return f'<img alt="{m.group(1)}" src="{f.as_uri()}" style="width:{width}">'
    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)(\{[^}]*\})?", img, md)
    body = markdown.markdown(md, extensions=["tables", "sane_lists"])
    css = f"""
    @page {{ size: 8.5in 11in; margin: 1in; }}
    body {{ font-family: Arial, sans-serif; font-size: 12pt; line-height: {line}; margin: 0; }}
    p, li {{ margin: 0 0 {para}pt 0; }}
    ul, ol {{ margin: 0 0 {para}pt 0; padding-left: 0.5in; }}
    h1 {{ font-size: 16pt; margin: 24pt 0 6pt; line-height: 1.15; }}
    h2 {{ font-size: 14pt; margin: 18pt 0 6pt; line-height: 1.15; }}
    h3 {{ font-size: 12pt; margin: 12pt 0 4pt; line-height: 1.15; }}
    table {{ border-collapse: collapse; font-size: 12pt; line-height: {table_line}; margin: 0 0 {para}pt 0; }}
    td, th {{ padding: 1pt 4pt; vertical-align: top; }}
    th {{ text-align: left; }}
    img {{ display: block; }}
    code {{ font-family: Consolas, monospace; font-size: 11pt; }}
    """
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"


def pages(md: str, line: float, para: float, table_line: float) -> int:
    with tempfile.TemporaryDirectory() as d:
        h, pdf = Path(d) / "report.html", Path(d) / "report.pdf"
        h.write_text(html(md, line, para, table_line), encoding="utf-8")
        subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--allow-file-access-from-files", f"--print-to-pdf={pdf}", h.as_uri()],
                       capture_output=True, timeout=180)
        return len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="*", help="git revisions or files (default: the working docs/report_draft.md)")
    ap.add_argument("--line", type=float, default=1.80, help="CSS line height of the text (calibrated, errs high)")
    ap.add_argument("--para", type=float, default=6.0, help="space after a paragraph, points (calibrated)")
    ap.add_argument("--table-line", type=float, default=1.15)
    a = ap.parse_args(argv)
    for spec in a.specs or [None]:
        md = source(spec)
        body_md = md.split("\n## References")[0]
        words = sum(len(l.split()) for l in body_md.splitlines()
                    if l.strip() and not l.lstrip().startswith("|") and not l.startswith("!["))
        tables = sum(1 for l in body_md.splitlines() if l.lstrip().startswith("|"))
        n = pages(body_md, a.line, a.para, a.table_line)
        print(f"{spec or 'docs/report_draft.md'}: the body ends on about page {n} "
              f"({words} words of text, {tables} table lines)")


if __name__ == "__main__":
    sys.exit(main())

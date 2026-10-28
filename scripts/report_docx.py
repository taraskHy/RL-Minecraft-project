"""The report as Word and PDF in the submission format, with its page count.

    python scripts/report_docx.py [--pdf]

Converts docs/report_draft.md with pandoc into results/report_draft.docx:
Arial 12, 1.5 line spacing for the text (table cells single-spaced),
one-inch margins (the course's format), figures from results/figures/. With --pdf, LibreOffice also writes
results/report_draft.pdf and the script prints the page count with and
without the references (the limit is 8 to 20 pages without them).
Needs pandoc and python-docx (and LibreOffice for --pdf); none of them is
part of setup.cmd.
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "report_draft.md"
OUT = ROOT / "results" / "report_draft.docx"


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def arial(rpr) -> None:
    """Arial in every font slot of a run-properties element, and no theme font overriding it."""
    from docx.oxml import OxmlElement
    fonts = rpr.find(W + "rFonts")
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for k in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        fonts.attrib.pop(W + k, None)
    for k in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(W + k, "Arial")


def reference_doc(path: Path) -> None:
    """pandoc's default styles, set to Arial 12, 1.5 spacing and one-inch margins."""
    from docx import Document
    from docx.shared import Inches, Pt

    subprocess.run(["pandoc", "-o", str(path), "--print-default-data-file", "reference.docx"], check=True)
    doc = Document(str(path))
    for style in doc.styles:
        if not hasattr(style, "font") or style.font is None:
            continue
        style.font.name = "Arial"
        arial(style.element.get_or_add_rPr())
        if style.name in ("Normal", "Body Text", "First Paragraph", "Compact", "Block Text"):
            style.font.size = Pt(12)
        if hasattr(style, "paragraph_format") and not style.name.startswith(("Heading", "Title")):
            style.paragraph_format.line_spacing = 1.5
    defaults = doc.styles.element.find(W + "docDefaults")
    if defaults is not None and defaults.find(W + "rPrDefault") is not None:
        rpr = defaults.find(W + "rPrDefault").find(W + "rPr")
        if rpr is not None:
            arial(rpr)
    for section in doc.sections:
        section.top_margin = section.bottom_margin = Inches(1)
        section.left_margin = section.right_margin = Inches(1)
    doc.save(str(path))


def single_space_tables(path: Path) -> None:
    """Table cells single-spaced (the 1.5 spacing is for the running text)."""
    from docx import Document
    doc = Document(str(path))
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for par in cell.paragraphs:
                    par.paragraph_format.line_spacing = 1.0
    doc.save(str(path))


def page_counts(pdf: Path) -> tuple[int, int | None]:
    """Total pages, and the page on which the References heading starts."""
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    ref = next((i + 1 for i, p in enumerate(pages) if re.search(r"^\s*References\s*$", p, re.M)), None)
    return len(pages), ref


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", action="store_true", help="also write the PDF (LibreOffice) and count pages")
    a = ap.parse_args(argv)
    if not shutil.which("pandoc"):
        sys.exit("pandoc is not installed")
    with tempfile.TemporaryDirectory() as d:
        ref = Path(d) / "reference.docx"
        reference_doc(ref)
        # No implicit figures: the alt text stays alt text; the report writes its own captions.
        subprocess.run(["pandoc", str(SOURCE), "-f", "markdown-implicit_figures", "-o", str(OUT),
                        "--reference-doc", str(ref), "--resource-path", str(SOURCE.parent)], check=True)
    single_space_tables(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")
    if a.pdf:
        office = shutil.which("soffice") or shutil.which("libreoffice")
        if not office:
            sys.exit("LibreOffice is not installed")
        subprocess.run([office, "--headless", "--convert-to", "pdf", "--outdir", str(OUT.parent), str(OUT)],
                       check=True, capture_output=True)
        pdf = OUT.with_suffix(".pdf")
        total, ref_page = page_counts(pdf)
        body = (ref_page - 1) if ref_page else total
        print(f"wrote {pdf.relative_to(ROOT)}: {total} pages; {body} before the references "
              f"(the limit is 8 to 20)")


if __name__ == "__main__":
    main()

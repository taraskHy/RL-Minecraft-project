"""Turn a page written for an artifact's main slot into a supporting page of the same artifact.

    python scripts/artifact_subpage.py IN.html OUT.html [--back-label TEXT]

An artifact's main page is published without <!doctype>, <html>, <head> or <body>: the host wraps it in a
skeleton (charset, viewport with viewport-fit=cover, a small reset). A supporting HTML file of the same artifact
is served as it is, so it needs its own skeleton. This script adds that skeleton with the same reset and moves the
page's own <title>, <meta>, <link> and first <style> into <head>. With --back-label it also puts a link to the
artifact's main page ("./") above the content; leave it out when the main page embeds this one in an <iframe>.
Standard library only.
"""
import argparse
import html
import re
from pathlib import Path

SKELETON_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
/* The artifact host's reset for main pages, repeated for this supporting page. */
:root { color-scheme: light; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
body { margin: 0; font: 14px system-ui, -apple-system, "Segoe UI", sans-serif; }
img { max-width: 100%; }
[hidden] { display: none !important; }
.subpage-back { display: block; padding-block: 10px 0; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); font-size: 15px; }
.subpage-back a { color: inherit; }
</style>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--back-label", default="", help="a link back to the main page, with this text")
    a = ap.parse_args()
    page = a.inp.read_text(encoding="utf-8")
    if re.match(r"\s*<!doctype", page, re.I):
        raise SystemExit(f"{a.inp} already has a doctype: it is a whole document")
    m = re.search(r"<style[\s>].*?</style>", page, re.S | re.I)
    if not m:
        raise SystemExit(f"{a.inp} has no <style>")
    head = re.sub(r'<meta\s+charset=["\']?[\w-]+["\']?\s*/?>\s*', "", page[:m.end()], flags=re.I)
    body = page[m.end():]
    back = (f'<nav class="subpage-back"><a href="./">&larr; {html.escape(a.back_label)}</a></nav>\n'
            if a.back_label else "")
    a.out.write_text(SKELETON_HEAD + head.strip() + "\n</head>\n<body>\n" + back + body.strip() + "\n</body>\n</html>\n",
                     encoding="utf-8")
    print(f"wrote {a.out} ({a.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

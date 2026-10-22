"""Plain CSV/JSON and a dependency-free SVG learning curve."""

import csv
import hashlib
import html
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone


def prepare_output(path: str | Path) -> Path:
    out = Path(path)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output folder is not empty: {out}. Choose a new --out folder.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[dict]):
    if not records:
        raise ValueError("Cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def provenance() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    root = Path(__file__).resolve().parent.parent
    sources = sorted((root / "minecraft_rl").glob("*.py"))
    sources += sorted((root / "configs").glob("*.json"))
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--short"),
            "source_sha256": hashes,
            "environment": "crafting_arena_v1", "evidence": "SIMULATOR ONLY"}


def learning_curve(path: Path, records: list[dict], title: str):
    # Non-overlapping bins show training behavior, including exploratory actions.
    window = max(1, len(records) // 100)
    points = []
    for i in range(0, len(records), window):
        batch = records[i:i+window]
        x = 70 + 690 * batch[-1]["episode"] / len(records)
        y = 330 - 260 * sum(r["success"] for r in batch) / len(batch)
        points.append(f"{x:.1f},{y:.1f}")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="820" height="420" viewBox="0 0 820 420">
<rect width="820" height="420" fill="white"/>
<g font-family="Arial,sans-serif" fill="#172538">
<text x="70" y="30" font-size="20">{html.escape(title)}</text>
<text x="70" y="52" font-size="12">SIMULATOR ONLY | exploratory training success | bin size {window}</text>
<path d="M70 70 V330 H760" stroke="#667085" fill="none"/>
<path d="M70 200 H760 M70 70 H760" stroke="#e4e7ec"/>
<text x="25" y="335">0%</text><text x="20" y="205">50%</text><text x="12" y="75">100%</text>
<text x="65" y="355">0</text><text x="710" y="355">{len(records)}</text>
<text x="365" y="385">Training episode</text>
<polyline points="{' '.join(points)}" stroke="#087f8c" stroke-width="2" fill="none"/>
<text x="70" y="410" font-size="12">Frozen-policy test success is reported separately in summary.json.</text>
</g></svg>'''
    path.write_text(svg, encoding="utf-8")

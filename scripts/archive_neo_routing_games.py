"""Copy this branch's bot runs into results/neo_routing/games/NAME, in parkour.report's format (archive()).

    python scripts/archive_neo_routing_games.py runs/NAME [runs/NAME ...]

courses.json, summary.json, summary.md as they are; episodes.jsonl and server.log gzipped with mtime 0 (the same
records give the same bytes). results/games/ is parkour.report's own output, so this branch keeps its runs here.
"""
import gzip
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "neo_routing" / "games"
GAME_FILES = ("courses.json", "summary.json", "summary.md")  # parkour.report.GAME_FILES


def main(argv):
    for arg in argv:
        run = Path(arg).resolve()
        if not (run / "episodes.jsonl").exists() or not (run / "summary.json").exists():
            raise SystemExit(f"{run}: not a finished bot run (no episodes.jsonl or summary.json)")
        dst = OUT / run.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in GAME_FILES:
            if (run / f).exists():
                shutil.copy2(run / f, dst / f)
        for f in ("episodes.jsonl", "server.log"):
            if (run / f).exists():
                with open(run / f, "rb") as src, gzip.GzipFile(dst / (f + ".gz"), "wb", compresslevel=9, mtime=0) as gz:
                    shutil.copyfileobj(src, gz)
        print(dst)


if __name__ == "__main__":
    main(sys.argv[1:])

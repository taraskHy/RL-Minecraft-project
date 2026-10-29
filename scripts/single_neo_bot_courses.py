"""The first N courses of a single-neo set as a course file for the bots (parkour.live --courses-file).

    python scripts/single_neo_bot_courses.py OUT.json [--family 41] [--n 64]

HANDOFF.md section 3 (rules of 29 Oct, 17:49): after the simulator's test on family 41, the offline bot and, only if
it matches the simulator on every course, real Minecraft play the first 64 test courses (each of the 32 single-neo
types twice), with the policy alone and with the planner. Course i of the file is course i of
parkour.neo_assist.single_neo_courses(family), so the planner's per-request seeds (by course index) are the
simulator run's.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour.neo_assist import single_neo_courses  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--family", type=int, default=41)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; the course file is written once")
    cs = [c.to_json() for c in single_neo_courses(a.family, a.n)]
    a.out.write_text(json.dumps({"courses": cs}), encoding="utf-8")
    print(json.dumps({"out": str(a.out), "family": a.family, "courses": len(cs),
                      "sha256": hashlib.sha256(a.out.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()

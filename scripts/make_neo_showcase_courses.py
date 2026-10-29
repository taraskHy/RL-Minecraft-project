"""Showcase courses with exactly one neo, for the replay page.

    python scripts/make_neo_showcase_courses.py OUT.json

The user saw on 29 October 2026 that the page's "neo jumps" replays show no neo done: those nine courses come from
the advanced levels, where neos are only some of the jumps (three of the nine have one), and the one network fell at
every one of the three. These courses each hold exactly one neo, from parkour.neo_assist.single_neo_courses (the
generator of the single-neo sets of families 40 and 41) with its own seed family 45, never used before: the first 32
courses hold each of the 32 single-neo types once, and the set keeps the even-numbered ones (types 0, 2, ..., 30;
levels 6 and 7), fixed before any outcome was seen. Standard library and the repository's code only.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour.neo_assist import single_neo_courses, single_neo_types  # noqa: E402
from parkour.neo_dagger import neo_takeoffs  # noqa: E402

FAMILY = 45
USED = set(range(0, 6)) | {7, 99} | set(range(11, 45))  # 18-39 the routing work's, 40-41 single neo, 42-44 replays


def courses() -> list[dict]:
    assert FAMILY not in USED
    types = single_neo_types()
    out = []
    for i, c in enumerate(single_neo_courses(FAMILY, 32)):
        if i % 2:
            continue
        ks = neo_takeoffs(c)
        assert len(ks) == 1, (i, ks)  # exactly one neo, as in the single-neo sets
        d = c.to_json()
        d.update(kind="neo-single", family=FAMILY, id=f"neo-single-{i}", neo_type=i % len(types), neo_jump=ks[0],
                 neo_spec=types[i % len(types)][1].key())
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; the course file is written once")
    cs = courses()
    a.out.write_text(json.dumps({"courses": cs}), encoding="utf-8")
    print(json.dumps([{k: d[k] for k in ("id", "level", "neo_type", "neo_jump", "neo_spec")} for d in cs]))


if __name__ == "__main__":
    main()

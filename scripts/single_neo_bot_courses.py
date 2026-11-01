"""The first N courses of a single-neo set as a course file for the bots (parkour.live --courses-file).

    python scripts/single_neo_bot_courses.py OUT.json [--family 41] [--n 64]
    python scripts/single_neo_bot_courses.py OUT.json --set two --family 53   (a two-neo set, 1 Nov)
    python scripts/single_neo_bot_courses.py OUT.json --set water --family 101   (a water set, 1 Nov)

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

from parkour.neo_assist import single_neo_courses, two_neo_courses  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--family", type=int, default=0, help="default: the set's test family (single 41, two 53, "
                                                          "water 101)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--set", choices=("single", "two", "water"), default="single")
    a = ap.parse_args()
    from parkour import neo_assist as NA  # noqa: E402
    families = {"single": (NA.SINGLE_TEST_FAMILY, NA.SINGLE_VALIDATION_FAMILY, NA.SINGLE_FRESH_TEST_FAMILY,
                           NA.SINGLE_FRESH_TEST_FAMILY_2, NA.SINGLE_FRESH_TEST_FAMILY_3, NA.SINGLE_FRESH_TEST_FAMILY_4),
                "two": (NA.TWO_TEST_FAMILY, NA.TWO_VALIDATION_FAMILY),
                "water": (NA.WATER_TEST_FAMILY, 100)}[a.set]
    a.family = a.family or families[0]
    if a.family not in families:
        raise SystemExit(f"--set {a.set} has families {families}")
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; the course file is written once")
    if a.set == "water":
        from parkour.water import water_set  # noqa: E402
        cs = [c.to_json() for c in water_set(a.family, a.n)]
    else:
        cs = [c.to_json() for c in (two_neo_courses if a.set == "two" else single_neo_courses)(a.family, a.n)]
    a.out.write_text(json.dumps({"courses": cs}), encoding="utf-8")
    print(json.dumps({"out": str(a.out), "set": a.set, "family": a.family, "courses": len(cs),
                      "sha256": hashlib.sha256(a.out.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()

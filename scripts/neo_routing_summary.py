"""Per-run breakdown of the neo-routing runs (results/neo_routing/details/*.json.gz): a measurement, no choice.

    python scripts/neo_routing_summary.py NAME [NAME ...]          print one line per run
    python scripts/neo_routing_summary.py --out results/neo_routing/breakdown.json NAME ...

For the single-neo sets, per course: whether the bot reached the neo's takeoff, what the router chose there
first (the neo's landing, a platform further on or back, off the course, nothing), whether that choice was
searched and found, and how the course ended (completed with the neo, completed past it, failed at the neo,
failed before the neo's takeoff, after passing the neo, or past the takeoff without doing the neo; and how many
failed on the jump onto the takeoff). Wrong platforms and failures are the planner's ``failure_modes`` on the
router's decisions.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour import courses as C  # noqa: E402
from parkour.env import CoursePool  # noqa: E402
from parkour.neo_dagger import neo_takeoffs  # noqa: E402
from parkour.planner import failure_modes  # noqa: E402
from parkour.route_neo import course_set  # noqa: E402

DETAILS = ROOT / "results" / "neo_routing" / "details"


def load(name: str) -> dict:
    with gzip.open(DETAILS / f"{name}.json.gz", "rt", encoding="utf-8") as fh:
        return json.load(fh)


def pick_at_takeoff(plats: list, t: int, decisions: list) -> str:
    """The router's first choice the first time it decided on the neo's takeoff t (the 30 Oct check's rule)."""
    idx = [plats.index(tuple(h)) if tuple(h) in plats else -1 for _, h, _ in decisions]
    first = next((i for i, k in enumerate(idx) if k == t), None)
    if first is None:
        return "not_reached"
    chain = decisions[first][2]
    if not chain:
        return "nothing"
    j = plats.index(tuple(chain[0])) if tuple(chain[0]) in plats else -1
    return "neo_target" if j == t + 1 else ("off_course" if j < 0 else ("ahead" if j > t + 1 else "back"))


def breakdown(name: str) -> dict:
    d = load(name)
    row = d["row"]
    courses = course_set(row["set"], row["family"], row.get("per_level", 0))
    if row.get("first_courses_only"):
        courses = courses[:row["first_courses_only"]]
    pool = CoursePool(courses, "cpu", C.GRID)
    pc, ev, dec = d["per_course"], d["events"], d["decisions"]
    success = [c["success"] for c in pc]
    out = {"name": name, "set": row["set"], "family": row["family"], "courses": len(pc), "completed": row["completed"],
           "completed_neo_done": row.get("completed_neo_done"), "config": row["config"], "tag": row.get("tag", ""),
           "failure_modes": failure_modes(pool, [SimpleNamespace(events=e) for e in dec], success)}
    out["searches"] = row.get("searches", {})
    out["plans_played"], out["plans_landed_as_planned"] = row.get("plans_played", 0), row.get("plans_landed_as_planned", 0)
    if row["set"] == "single":
        picks, ends, onto = {}, {}, 0
        for i, c in enumerate(courses):
            plats = [tuple(int(v) for v in r) for r in c.platforms]
            t = neo_takeoffs(c)[0]
            p = pick_at_takeoff(plats, t, dec[i])
            picks[p] = picks.get(p, 0) + 1
            mine = [e for e in ev if e["course"] == i and e["takeoff"] == t and e["target"] == t + 1]
            if pc[i]["success"]:
                end = "completed, neo done" if pc[i]["neo_done"] else "completed past the neo"
            elif pc[i]["stopped_at_neo"]:
                end = ("failed at the neo: no landing found" if mine and not any(e["found"] for e in mine) else
                       "failed at the neo: not searched" if not mine else "failed at the neo after a landing")
            elif pc[i]["neos_passed"]:
                end = "failed after passing the neo"
            elif pc[i]["stopped_at"] < t:
                end = "failed before the neo's takeoff"
                onto += pc[i]["stopped_at"] == t - 1  # on the jump onto the takeoff
            else:  # it went past the takeoff without doing the neo (the router took another way)
                end = "failed past the neo's takeoff without doing the neo"
            ends[end] = ends.get(end, 0) + 1
        out["pick_at_takeoff"] = dict(sorted(picks.items()))
        out["ends"] = dict(sorted(ends.items()))
        out["failed_on_the_jump_onto_the_takeoff"] = onto
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    rows = [breakdown(n) for n in a.names]
    for r in rows:
        print(json.dumps(r))
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    main()

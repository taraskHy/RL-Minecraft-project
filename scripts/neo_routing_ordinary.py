"""The router's ordinary validation sets with the neo routing mode, course by course against the shipped router.

    python scripts/neo_routing_ordinary.py STANDARD_RUN OBSTACLES_RUN [--keep FILE] [--out FILE]

A measurement, no choice. For each set: the shipped router's outcome on every course
(results/routing/router_validation_route-full-f_it29_*.json, fall rule local), the run's (its details file in
results/neo_routing/details/), the courses where they differ, and whether the run searched a blocked-path choice
on each of them. With --keep, the 30 Oct "filter off, head-hitter flags kept" what-if (no planner;
neo-route-check-rulecost.json, variant "keephead") is a third column, so a difference can be told apart as the
filter's (the router's choices change when blocked platforms are selectable) or the planner's.
"""
import argparse
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = {"standard": ROOT / "results/routing/router_validation_route-full-f_it29_standard.json",
           "obstacles": ROOT / "results/routing/router_validation_route-full-f_it29_obstacles.json"}


def details(name):
    with gzip.open(ROOT / "results/neo_routing/details" / f"{name}.json.gz", "rt", encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("standard")
    ap.add_argument("obstacles")
    ap.add_argument("--keep", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    keep = {}
    if a.keep:
        for r in json.loads(Path(a.keep).read_text())["runs"]:
            if r["variant"] == "keephead" and r["jumper"] == "results/final_policy.pt" and r["set"] in SHIPPED:
                keep[r["set"]] = [bool(x) for x in r["per_course"]]
    out = {}
    for name, run in (("standard", a.standard), ("obstacles", a.obstacles)):
        ship = [bool(x) for x in json.loads(SHIPPED[name].read_text())["local"]["router_route"]["per_course"]["success"]]
        d = details(run)
        mine = [c["success"] for c in d["per_course"]]
        searched = {e["course"] for e in d["events"]}
        assert len(ship) == len(mine), (len(ship), len(mine))
        diff = [i for i in range(len(mine)) if mine[i] != ship[i]]
        rows = [{"course": i, "shipped": ship[i], "run": mine[i], **({"keep": keep[name][i]} if name in keep else {}),
                 "searched": i in searched} for i in diff]
        out[name] = {"run": run, "family": d["row"]["family"], "courses": len(mine), "shipped": sum(ship),
                     "run_completed": sum(mine), **({"keep": sum(keep[name])} if name in keep else {}),
                     "courses_searched": len(searched), "searches": len(d["events"]),
                     "searched_and_completed": sum(mine[i] for i in searched),
                     "searched_and_shipped_completed": sum(ship[i] for i in searched),
                     "differ": len(diff), "only_run": sum(r["run"] for r in rows),
                     "only_shipped": sum(r["shipped"] for r in rows), "rows": rows}
        print(json.dumps({k: v for k, v in out[name].items() if k != "rows"}))
        for r in rows:
            print("  ", json.dumps(r))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    main()

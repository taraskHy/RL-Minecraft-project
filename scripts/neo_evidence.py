"""Write the data files behind the neo numbers in the docs (results/neo/).

    python scripts/neo_evidence.py

- by_neo_count.json: the chosen neo policy on the 210 held-out level 6-8 test courses with a neo, split by
  the number of neos in the course (simulator).
- detector_check.json: on the courses of the second DAgger round (seed 2), how many takeoffs the first neo
  detector counted and how many are real neo takeoffs (wall blocks only).
- wide_takeoff_search.json: the discrete demonstration search from 1x1, 2x2 and 3x3 takeoffs, with the wall's
  open end next to the landing or two lanes away.
- demonstrations.json: how many verified demonstrations were collected and used (from runs/neo_demos*.json).
"""
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour import courses as C  # noqa: E402
from parkour.env import CoursePool  # noqa: E402
from parkour.live import load_model  # noqa: E402
from parkour.neo_dagger import neo_takeoffs  # noqa: E402
from parkour.neo_demos import neo_course, search, verify  # noqa: E402
from parkour.train import course_hash, evaluate, held_out_courses  # noqa: E402

OUT = ROOT / "results" / "neo"
NEO_POLICY = "runs/neo-dagger3/checkpoints/best.pt"


def by_neo_count(dev):
    levels = C.load_levels(advanced=True)
    tests = [c for c in held_out_courses(levels, 200, 10) if c.level >= 6 and neo_takeoffs(c)]
    m = load_model(ROOT / NEO_POLICY, dev)
    r = evaluate(m, CoursePool(tests, dev, C.GRID), dev, 200, per_course=True)["per_course"]
    tot, ok = Counter(), Counter()
    for c, s in zip(tests, r["success"]):
        k = min(len(neo_takeoffs(c)), 4)
        tot[k] += 1
        ok[k] += bool(s)
    rows = {("4+" if k == 4 else str(k)): {"courses": tot[k], "completed": ok[k]} for k in sorted(tot)}
    return {"policy": NEO_POLICY, "courses": len(tests), "completed": sum(ok.values()), "by_neos_in_course": rows,
            "method": "held_out_courses(advanced levels, 200 per level, 10 jumps, test family), levels 6-8 with a neo "
                      "(neo_dagger.neo_takeoffs), deterministic policy, simulator"}


def detector_check():
    def old_takeoffs(course):  # the first detector: any block on the line, platforms included
        solid = {tuple(int(v) for v in b) for b in course.blocks()}
        out = []
        p = course.platforms
        for k in range(len(p) - 1):
            a = ((p[k][0] + p[k][1] + 1) / 2, (p[k][2] + p[k][3] + 1) / 2)
            b = ((p[k + 1][0] + p[k + 1][1] + 1) / 2, (p[k + 1][2] + p[k + 1][3] + 1) / 2)
            y0 = int(p[k][4]) + 1
            for t in np.linspace(0.0, 1.0, 41)[1:-1]:
                x, z = math.floor(a[0] + t * (b[0] - a[0])), math.floor(a[1] + t * (b[1] - a[1]))
                if any((x, y, z) in solid for y in (y0, y0 + 1)):
                    out.append(k)
                    break
        return out
    levels = C.load_levels(advanced=True)
    held = held_out_courses(levels, 200, 10) + held_out_courses(levels, 100, 10, family=2)
    exclude = {course_hash(c) for c in held}
    rng = np.random.default_rng([2026, 7, 2])
    cs = []
    while len(cs) < 1024:
        c = C.generate(rng, levels, 6 + len(cs) % 2, int(rng.integers(5, 13)), wide_neo=True)
        if c.n >= 3 and course_hash(c) not in exclude and old_takeoffs(c):
            cs.append(c)
    old = sum(len(old_takeoffs(c)) for c in cs)
    real = sum(len(set(old_takeoffs(c)) & set(neo_takeoffs(c))) for c in cs)
    return {"courses": len(cs), "courses_with_a_real_neo": sum(bool(neo_takeoffs(c)) for c in cs),
            "takeoffs_detected": old, "real_neo_takeoffs": real, "share_real": round(real / old, 4),
            "method": "the courses of DAgger round 2 (neo_dagger seed 2) rebuilt; first detector (any block on the "
                      "centre line at body height) against the fixed one (wall blocks only)"}


def wide_takeoff_search(dev):
    rows = []
    for takeoff in ((1, 1), (2, 2), (3, 3)):
        for gap, dy, w in ((1, 0, 1), (2, 0, 1), (2, -1, 2)):
            for side in (1, -1):
                spec = C.JumpSpec(gap, dy, 0, (1, 1), 1, side, 0, w)
                c = neo_course(spec, 0, takeoff)
                t0 = time.time()
                found = verify(c, search(c, dev, seed=0, yaw_bias=0.0, iterations=40, keep=200), dev)
                land = c.platforms[1]
                wall_z = sorted({int(e[2]) for e in c.extra})
                open_end = wall_z[-1] + 1 if side > 0 else wall_z[0] - 1  # heading +x: right of +x is +z
                rows.append({"takeoff": f"{takeoff[0]}x{takeoff[1]}", "gap": gap, "dy": dy, "wall_at": w, "side": side,
                             "landing_lanes_from_open_end": int(abs(open_end - int(land[2]))),
                             "landings_found": len(found), "of_wanted": 200, "seconds": round(time.time() - t0, 1)})
                print(rows[-1], flush=True)
    return {"method": "parkour.neo_demos.search (discrete CEM, 16,384 samples, 40 iterations, seed 0) then verify, "
                      "one neo after a takeoff of the given size; the landing is 1x1",
            "runs": rows}


def demonstrations():
    full = json.loads((ROOT / "runs" / "neo_demos.json").read_text())
    used = json.loads((ROOT / "runs" / "neo_demos_train.json").read_text())
    per = Counter(d["spec"] for d in full["demos"])
    return {"collected": len(full["demos"]), "layout_sides": len(per), "per_layout_side_max": max(per.values()),
            "per_layout_side_min": min(per.values()), "used_for_training": len(used["demos"]),
            "seconds": full.get("seconds"), "method": full.get("method", "")}


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("demonstrations.json", demonstrations), ("detector_check.json", detector_check),
                     ("by_neo_count.json", lambda: by_neo_count(dev)), ("wide_takeoff_search.json", lambda: wide_takeoff_search(dev))):
        data = fn()
        (OUT / name).write_text(json.dumps(data, indent=1))
        print("wrote", OUT / name, flush=True)


if __name__ == "__main__":
    main()

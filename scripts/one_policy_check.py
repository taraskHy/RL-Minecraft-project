"""One policy for everything? The final policies on levels 6-8, the neo policies on levels 0-5.

    python scripts/one_policy_check.py

The advanced test set (levels 0-8) has exactly the standard test courses at
levels 0 to 5 (checked here by course fingerprint), so the neo policies'
level 0-5 scores in their evaluation logs compare directly with the final
policies'. This script (1) checks that, (2) reads each neo run's held-out
scores at its validation-chosen checkpoint from results/runs/NAME/, and (3)
plays the two final policies (in results/) on the levels 6-8 test courses
and on the 210 of them that contain a neo, in the simulator. Writes
results/neo/one_policy_comparison.json.
"""
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour import courses as C  # noqa: E402
from parkour.env import CoursePool  # noqa: E402
from parkour.live import load_model  # noqa: E402
from parkour.neo_dagger import neo_takeoffs  # noqa: E402
from parkour.train import course_hash, evaluate, held_out_courses, obstacle_courses  # noqa: E402

NEO_RUNS = ("advanced-ft3", "neo-stall", "neo-bc", "neo-dagger", "neo-dagger2", "neo-dagger3")


def row_at(path: Path, iteration: int) -> dict:
    with path.open() as f:
        return next(r for r in csv.DictReader(f) if int(r["iteration"]) == iteration)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    basic, advanced = C.load_levels(), C.load_levels(advanced=True)
    same = all(course_hash(a) == course_hash(b) for a, b in
               zip(held_out_courses(basic, 200, 10), held_out_courses(advanced, 200, 10)))
    same_obst = all(course_hash(a) == course_hash(b) for a, b in
                    zip(obstacle_courses(basic, 200, 10), obstacle_courses(advanced, 200, 10)))
    out = {"method": "simulator, held-out test families 1 (standard) and 4 (obstacles), 200 courses per level, "
                     "deterministic policy; neo runs at their validation-chosen checkpoint (summary.json best)",
           "levels_0_5_courses_identical": {"standard": same, "obstacles": same_obst}, "neo_policies": {}}
    for run in NEO_RUNS:
        d = ROOT / "results" / "runs" / run
        it = json.loads((d / "summary.json").read_text())["best"]["iteration"]
        t, o = row_at(d / "eval_log.csv", it), row_at(d / "eval_obstacles_log.csv", it)
        out["neo_policies"][run] = {
            "iteration": it,
            "standard_levels_0_5": round(100 * sum(float(t[f"success_L{L}"]) for L in range(6)) / 6, 1),
            "obstacles_levels_0_5": round(100 * sum(float(o[f"success_L{L}"]) for L in range(6)) / 6, 1),
            "per_level": {L: round(100 * float(t[f"success_L{L}"]), 1) for L in range(9)}}
    hard = held_out_courses(advanced, 200, 10)[6 * 200:]
    neo = [c for c in hard if neo_takeoffs(c)]
    sets = {"standard_levels_0_5": held_out_courses(basic, 200, 10), "obstacles_levels_0_5": obstacle_courses(basic, 200, 10)}
    out["final_policies"] = {}
    for name in ("final_policy", "final_policy_v1"):
        model = load_model(ROOT / "results" / f"{name}.pt", dev)
        r = evaluate(model, CoursePool(hard, dev, C.GRID), dev, 200, deterministic=True)
        rn = evaluate(model, CoursePool(neo, dev, C.GRID), dev, 200, deterministic=True)
        out["final_policies"][name] = {
            **{k: round(100 * evaluate(model, CoursePool(cs, dev, C.GRID), dev, 200, deterministic=True)["success"], 1)
               for k, cs in sets.items()},
            "levels_6_8": {L: round(100 * v["success"], 1) for L, v in r["per_level"].items()},
            "neo_courses_completed": round(rn["success"] * len(neo)), "neo_courses": len(neo)}
    path = ROOT / "results" / "neo" / "one_policy_comparison.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

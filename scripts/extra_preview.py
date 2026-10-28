"""Head hitters and ice run-ups on whole courses: the final policy zero-shot, alone and with the planner.

    python scripts/extra_preview.py [--per-level 200] [--assist-per-level 50] [--family 1]

Courses: the basic levels 0-5 followed by the extra levels of
configs/jump_levels_extra.json (here levels 6 head hitter, 7 hard head
hitter, 8 ice run-up; half of each course's jumps come from its level, half
from levels 0-5), 10 jumps, held-out seed family 1 (test) or 2 (validation).
The policy was never trained on these kinds. With the planner
(parkour.neo_assist, jumps "advanced": every run-up jump is planned from the
bot's exact state) on the first --assist-per-level courses of each level,
with the setting given by --assist (chosen on the neo validation courses).
Writes results/extra/course_preview_family{F}.json. Simulator only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour import courses as C  # noqa: E402
from parkour.env import CoursePool  # noqa: E402
from parkour.live import load_model  # noqa: E402
from parkour.neo_assist import AssistConfig, evaluate_assisted, summarize  # noqa: E402
from parkour.train import evaluate, held_out_courses  # noqa: E402

NAMES = {6: "head hitter", 7: "hard head hitter", 8: "ice run-up"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "results" / "final_policy.pt"))
    ap.add_argument("--per-level", type=int, default=200)
    ap.add_argument("--assist-per-level", type=int, default=50)
    ap.add_argument("--family", type=int, default=1)
    ap.add_argument("--assist", default='{"jumps": "advanced"}', help="AssistConfig overrides (JSON)")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    table = C.load_levels() + C.load_levels(extra=True)[6:]
    courses = held_out_courses(table, a.per_level, 10, family=a.family)[6 * a.per_level:]
    model = load_model(Path(a.checkpoint), dev)
    t0 = time.time()
    with torch.no_grad():
        ev = evaluate(model, CoursePool(courses, dev, C.GRID), dev, 200, deterministic=True)
    out = {"measured": "simulator (float64 port of prismarine-physics), deterministic policy",
           "checkpoint": Path(a.checkpoint).as_posix().split("RL-Minecraft-project/")[-1], "family": a.family,
           "levels": {str(L): NAMES[L] for L in NAMES}, "courses_per_level": a.per_level,
           "policy_alone": {NAMES[L]: round(100 * v["success"], 1) for L, v in ev["per_level"].items()},
           "policy_alone_fell": {NAMES[L]: round(100 * v["fell"], 1) for L, v in ev["per_level"].items()}}
    cfg = AssistConfig(**json.loads(a.assist))
    out["assist_config"] = cfg.__dict__
    out["with_planner"] = {}
    for L in NAMES:
        sub = [c for c in courses if c.level == L][:a.assist_per_level]
        res = evaluate_assisted(model, CoursePool(sub, dev, C.GRID), dev, cfg)
        s = summarize(res)
        alone = evaluate(model, CoursePool(sub, dev, C.GRID), dev, 200, deterministic=True)["success"]
        out["with_planner"][NAMES[L]] = {"courses": len(sub), "completed": s["completed"],
                                         "completed_policy_alone": round(alone * len(sub)), "searches": s.get("searches"),
                                         "plans_played": s.get("plans_played"),
                                         "plans_landed_as_planned": s.get("plans_landed_as_planned"),
                                         "seconds_per_search": s.get("seconds_per_search")}
        print(json.dumps({NAMES[L]: out["with_planner"][NAMES[L]]}), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    path = ROOT / "results" / "extra" / f"course_preview_family{a.family}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

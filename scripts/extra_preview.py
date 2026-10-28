"""The extra jump kinds on whole courses: the final policy zero-shot, alone and with the planner.

    python scripts/extra_preview.py [--per-level 200] [--assist-per-level 40] [--family 1]

Courses, one set per level of configs/jump_levels_extra.json (head hitter,
hard head hitter, ice run-up, shaped blocks, slime bounce, ladder climb):
the basic levels 0-5 followed by that one level, so half of each course's
jumps are of that kind and half from levels 0-5 (courses.generate's mix),
10 jumps, held-out seed family 1 (test) or 2 (validation), the first
--per-level courses.
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
from parkour.paths import CONFIGS  # noqa: E402
from parkour.train import evaluate, held_out_courses  # noqa: E402

EXTRA = json.loads((CONFIGS / "jump_levels_extra.json").read_text())["levels"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "results" / "final_policy.pt"))
    ap.add_argument("--per-level", type=int, default=200)
    ap.add_argument("--assist-per-level", type=int, default=40)
    ap.add_argument("--family", type=int, default=1)
    ap.add_argument("--assist", default='{"jumps": "advanced"}', help="AssistConfig overrides (JSON)")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    basic = C.load_levels()
    model = load_model(Path(a.checkpoint), dev)
    t0 = time.time()
    cfg = AssistConfig(**json.loads(a.assist))
    out = {"measured": f"simulator (float64 port of prismarine-physics), deterministic policy, {dev.upper()}",
           "checkpoint": Path(a.checkpoint).name, "family": a.family, "arguments": vars(a),
           "courses_per_level": a.per_level, "policy_alone": {}, "policy_alone_fell": {},
           "assist_config": cfg.__dict__, "with_planner": {}}
    for L in EXTRA:
        name = L["name"]
        table = basic + [[C._spec_from(d) for d in L["specs"]]]
        courses = held_out_courses(table, a.per_level, 10, family=a.family)[6 * a.per_level:]
        with torch.no_grad():
            ev = evaluate(model, CoursePool(courses, dev, C.GRID), dev, 200, deterministic=True)
        out["policy_alone"][name] = round(100 * ev["success"], 1)
        out["policy_alone_fell"][name] = round(100 * ev["per_level"][6]["fell"], 1)
        sub = courses[:a.assist_per_level]
        if not sub:
            continue
        res = evaluate_assisted(model, CoursePool(sub, dev, C.GRID), dev, cfg)
        s = summarize(res)
        alone = evaluate(model, CoursePool(sub, dev, C.GRID), dev, 200, deterministic=True)["success"]
        out["with_planner"][name] = {"courses": len(sub), "completed": s["completed"],
                                     "completed_policy_alone": round(alone * len(sub)), "searches": s.get("searches"),
                                     "plans_played": s.get("plans_played"),
                                     "plans_landed_as_planned": s.get("plans_landed_as_planned"),
                                     "seconds_per_search": s.get("seconds_per_search"), "failed_at": s.get("failed_at")}
        print(json.dumps({name: out["with_planner"][name], "alone": out["policy_alone"][name]}), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    path = ROOT / "results" / "extra" / f"course_preview_family{a.family}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

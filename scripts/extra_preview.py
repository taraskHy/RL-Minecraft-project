"""The extra jump kinds on whole courses: the final policy zero-shot, alone and with the planner.

    python scripts/extra_preview.py [--per-level 200] [--assist-per-level 40] [--family 2|1]

Courses, one set per level of configs/jump_levels_extra.json (head hitter,
hard head hitter, ice run-up, shaped blocks, slime bounce, ladder climb):
the basic levels 0-5 followed by that one level, so half of each course's
jumps are of that kind and half from levels 0-5 (courses.generate's mix),
10 jumps, held-out seed family 1 (test) or 2 (validation), the first
--per-level courses.
The policy was never trained on these kinds. With the planner
(parkour.neo_assist) on the first --assist-per-level courses of each level,
once per --assist setting (default: jumps "advanced", every run-up jump
planned from the bot's exact state; and jumps "all", every jump planned).
Settings are compared on family 2 and reported on family 1.
Writes results/extra/course_preview_family{F}.json. Simulator only.

Each kind is saved when it is done (runs/extra_preview/family{F}/), so a
stopped run continues where it was; ``--level NAME`` (repeatable) computes
only those kinds, e.g. one process per kind, and the run that finds every
kind done writes the result file.
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
    ap.add_argument("--assist", action="append", help="AssistConfig overrides (JSON), one run per flag "
                                                     "(default: jumps advanced, then jumps all)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--level", action="append", help="only this extra level (repeatable); the others are left as they are")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    basic = C.load_levels()
    model = load_model(Path(a.checkpoint), dev)
    t0 = time.time()
    settings = a.assist or ['{"jumps": "advanced"}', '{"jumps": "all"}']
    cfgs = {s: AssistConfig(**json.loads(s)) for s in settings}
    same = {"checkpoint": Path(a.checkpoint).name, "per_level": a.per_level, "assist_per_level": a.assist_per_level,
            "family": a.family, "settings": settings}  # what a saved kind must have been computed with
    cache = ROOT / "runs" / "extra_preview" / f"family{a.family}"
    cache.mkdir(parents=True, exist_ok=True)
    names = [L["name"] for L in EXTRA]
    if a.level and set(a.level) - set(names):
        raise SystemExit(f"unknown levels {set(a.level) - set(names)}; known: {names}")
    for L in EXTRA:
        name = L["name"]
        f = cache / (name.replace(" ", "_") + ".json")
        if f.exists():
            if json.loads(f.read_text())["same"] != same:
                raise SystemExit(f"{f} was computed with other arguments: move it away or use them")
            continue
        if a.level and name not in a.level:
            continue
        kind = {"same": same, "with_planner": {}}
        table = basic + [[C._spec_from(d) for d in L["specs"]]]
        courses = held_out_courses(table, a.per_level, 10, family=a.family)[6 * a.per_level:]
        t1 = time.time()
        with torch.no_grad():
            ev = evaluate(model, CoursePool(courses, dev, C.GRID), dev, 200, deterministic=True)
        kind["policy_alone"] = round(100 * ev["success"], 1)
        kind["policy_alone_fell"] = round(100 * ev["per_level"][6]["fell"], 1)
        sub = courses[:a.assist_per_level]
        with torch.no_grad():
            alone = evaluate(model, CoursePool(sub, dev, C.GRID), dev, 200, deterministic=True)["success"] if sub else 0.0
        for setting, cfg in cfgs.items():
            if not sub:
                break
            s = summarize(evaluate_assisted(model, CoursePool(sub, dev, C.GRID), dev, cfg))
            kind["with_planner"][setting] = {
                "courses": len(sub), "completed": s["completed"], "completed_policy_alone": round(alone * len(sub)),
                "searches": s.get("searches"), "plans_played": s.get("plans_played"),
                "plans_landed_as_planned": s.get("plans_landed_as_planned"), "brakes": s.get("brakes"),
                "seconds_per_search": s.get("seconds_per_search"), "failed_at": s.get("failed_at")}
            print(json.dumps({name: kind["with_planner"][setting], "setting": setting,
                              "alone": kind["policy_alone"]}), flush=True)
        kind["seconds"] = round(time.time() - t1, 1)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(kind, indent=1))
        tmp.replace(f)  # a kind is saved whole or not at all
    kinds = {L["name"]: cache / (L["name"].replace(" ", "_") + ".json") for L in EXTRA}
    if not all(f.exists() for f in kinds.values()):
        print(f"done: {[n for n, f in kinds.items() if f.exists()]}; still to do: "
              f"{[n for n, f in kinds.items() if not f.exists()]}")
        return
    out = {"measured": f"simulator (float64 port of prismarine-physics), deterministic policy, {dev.upper()}",
           "checkpoint": Path(a.checkpoint).name, "family": a.family,
           "arguments": {k: v for k, v in vars(a).items() if k != "level"},
           "courses_per_level": a.per_level, "policy_alone": {}, "policy_alone_fell": {},
           "assist_configs": {s: c.__dict__ for s, c in cfgs.items()}, "with_planner": {s: {} for s in settings},
           "seconds": 0.0}
    for name, f in kinds.items():
        kind = json.loads(f.read_text())
        out["policy_alone"][name] = kind["policy_alone"]
        out["policy_alone_fell"][name] = kind["policy_alone_fell"]
        for setting in settings:
            if setting in kind["with_planner"]:
                out["with_planner"][setting][name] = kind["with_planner"][setting]
        out["seconds"] = round(out["seconds"] + kind["seconds"], 1)
    path = ROOT / "results" / "extra" / f"course_preview_family{a.family}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

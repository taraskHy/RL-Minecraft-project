"""Success of a trained policy on every single jump type.

    python -m parkour.jumpeval --checkpoint runs/NAME/checkpoints/best.pt --out results/NAME/jump_eval.json
    python -m parkour.jumpeval --advanced ...   (neo and run-up jumps, levels 6-8)
    python -m parkour.jumpeval --extra ...      (the extra kinds of configs/jump_feasibility_extra.json)

Each test course is a 1x1 takeoff block, the jump under test, and one easy
3x3 platform after it (so the target is not the goal, as in a real course).
The player starts standing on the takeoff block facing the target, with a yaw
error from -60 to +60 degrees in 16 steps, and the deterministic policy plays.
A jump counts as made when the player lands on its target. The table puts the
learned success next to the scripted strategies' forgiveness from
parkour.feasibility (configs/jump_feasibility.json); both start from rest on
the same 1x1 block. Composite kinds keep their second part, as in courses: a
shaped target is followed by its exit jump (``exit_success``: the exit made
too), and a slime pad by its exit, which is the jump's target.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import courses as C
from .env import CoursePool
from .live import load_model
from .paths import CONFIGS
from .train import evaluate

YAW_ERRORS = np.linspace(-math.radians(60), math.radians(60), 16)
GRID = (40, 16, 40)


def jump_course(spec: C.JumpSpec, yaw_error: float) -> C.Course:
    base = C.single_jump(spec, takeoff=(1, 1), grid=GRID)
    plats = list(base.platforms)
    second = [C.exit_spec(spec.block)] if spec.block else [C.bounce_exit(spec)] if spec.bounce else []
    for sp in second + [C.JumpSpec(1, 0, 0, (3, 3))]:
        plats.append(C._platform_after(plats[-1], 0, sp))
    return C.Course(np.stack(plats), base.spawn.copy(), base.spawn_yaw + yaw_error, specs=[spec] + second,
                    extra=base.extra, ice=base.ice, shapes=base.shapes, slime=base.slime, ladders=base.ladders)


def target_index(spec: C.JumpSpec) -> int:
    """The platform whose landing makes the jump: its target, or past a slime pad the exit."""
    return 2 if spec.bounce else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--advanced", action="store_true", help="neo and run-up jump types (levels 6-8)")
    ap.add_argument("--extra", action="store_true", help="the extra jump kinds (head hitters, ice, shapes, slime, ladders)")
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(a.checkpoint), dev)
    name = ("jump_feasibility_extra.json" if a.extra else
            "jump_feasibility_advanced.json" if a.advanced else "jump_feasibility.json")
    feas = json.loads((CONFIGS / name).read_text())["specs"]
    if a.extra:
        from .feasibility import EXTRA_LEVELS
        for r in feas:  # level names instead of numbers: these levels follow whatever table they are added to
            r["level"] = next((lv for lv, rule in EXTRA_LEVELS if rule(r)), None)
    specs = [C._spec_from(r) for r in feas]
    courses = [jump_course(s, e) for s in specs for e in YAW_ERRORS]
    pool = CoursePool(courses, dev, GRID)
    ev = evaluate(model, pool, dev, stall_ticks=200, deterministic=True, per_course=True)
    shape = (len(specs), len(YAW_ERRORS))
    last = np.array([c.n - 1 for c in courses])
    reached = np.rint(np.array(ev["per_course"]["progress"]) * last).astype(int).reshape(shape)
    made = reached >= np.array([target_index(s) for s in specs])[:, None]
    fell = np.array(ev["per_course"]["fell"]).reshape(shape)
    rows = []
    for r, sp, m, f, k in zip(feas, specs, made, fell, reached):
        # A missed jump either fell or stalled (never jumped within the stall limit).
        rows.append({**r, "policy_success": round(float(m.mean()), 4),
                     "policy_fell": round(float((f & ~m).mean()), 4), "policy_stalled": round(float((~f & ~m).mean()), 4)})
        if sp.block:
            rows[-1]["exit_success"] = round(float((k >= 2).mean()), 4)
    by_level = {}
    for L in sorted({r["level"] for r in rows}, key=lambda v: (v is None, v)):
        rl = [r for r in rows if r["level"] == L]
        by_level["impossible" if L is None else str(L)] = {
            "specs": len(rl), "mean_policy_success": round(float(np.mean([r["policy_success"] for r in rl])), 4),
            "specs_policy_never": sum(r["policy_success"] == 0 for r in rl)}
    out = {"checkpoint": str(a.checkpoint), "yaw_errors_deg": [round(math.degrees(e), 2) for e in YAW_ERRORS],
           "by_level": by_level, "specs": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in by_level.items():
        print(f"level {k}: {v}")
    beat = [r for r in rows if r["level"] is None and r["policy_success"] > 0]
    print(f"jump types no scripted strategy made but the policy did: {len(beat)}")
    for r in beat:
        print("  ", {k: r[k] for k in ("gap", "dy", "lateral", "size", "policy_success")})
    return out


if __name__ == "__main__":
    main()

"""Success of a trained policy on every single jump type.

    python -m parkour.jumpeval --checkpoint runs/NAME/checkpoints/best.pt --out results/NAME/jump_eval.json

Each test course is a 1x1 takeoff block, the jump under test, and one easy
3x3 platform after it (so the target is not the goal, as in a real course).
The player starts standing on the takeoff block facing the target, with a yaw
error from -60 to +60 degrees in 16 steps, and the deterministic policy plays.
A jump counts as made when the player lands on its target. The table puts the
learned success next to the scripted strategies' forgiveness from
parkour.feasibility (configs/jump_feasibility.json); both start from rest on
the same 1x1 block.
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
    after = C._platform_after(base.platforms[1], 0, C.JumpSpec(1, 0, 0, (3, 3)))
    plats = np.concatenate([base.platforms, after[None, :]])
    c = C.Course(plats, base.spawn.copy(), base.spawn_yaw + yaw_error, specs=[spec])
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(a.checkpoint), dev)
    feas = json.loads((CONFIGS / "jump_feasibility.json").read_text())["specs"]
    specs = [C.JumpSpec(r["gap"], r["dy"], r["lateral"], tuple(r["size"])) for r in feas]
    courses = [jump_course(s, e) for s in specs for e in YAW_ERRORS]
    pool = CoursePool(courses, dev, GRID)
    ev = evaluate(model, pool, dev, stall_ticks=200, deterministic=True, per_course=True)
    made = (np.array(ev["per_course"]["progress"]) >= 0.5 - 1e-9).reshape(len(specs), len(YAW_ERRORS))
    rows = []
    for r, m in zip(feas, made):
        rows.append({**r, "policy_success": round(float(m.mean()), 4)})
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

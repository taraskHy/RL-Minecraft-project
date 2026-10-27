"""How far does a policy generalise beyond the training distribution?

    python -m parkour.stress --checkpoint runs/NAME/checkpoints/best.pt --out results/stress.json

Each condition generates 100 new courses per difficulty level (seed family 3,
disjoint from training and from the main test set) and plays them with the
deterministic policy in the simulator:

  standard        10 jumps, turns 25%, start facing within 60 degrees (the test distribution)
  long            15 jumps (training used 5 to 12)
  zigzag          a turn at every jump (training: 25%)
  straight        no turns at all
  any_start_yaw   start facing any direction (training: within 60 degrees)
  single_blocks   every platform a single 1x1 block, as in classic parkour maps
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
from .train import evaluate

CONDITIONS = {
    "standard": dict(n_jumps=10),
    "long": dict(n_jumps=15),
    "zigzag": dict(n_jumps=10, turn_prob=1.0),
    "straight": dict(n_jumps=10, turn_prob=0.0),
    "any_start_yaw": dict(n_jumps=10, yaw_noise=math.pi),
    "single_blocks": dict(n_jumps=10, start_size=1, single=True),
}


def make(cond: str, levels, per_level: int, index: int) -> list[C.Course]:
    kw = dict(CONDITIONS[cond])
    lv = levels
    if kw.pop("single", False):
        lv = [[s for s in specs if s.size == (1, 1)] for specs in levels]
    n = kw.pop("n_jumps")
    out = []
    for L in range(len(lv)):
        rng = np.random.default_rng([2026, 3, index, L])
        while sum(c.level == L for c in out) < per_level:
            c = C.generate(rng, lv, L, n, **kw)
            if c.n == n + 1:  # keep only courses with every jump placed
                out.append(c)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-level", type=int, default=100)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(a.checkpoint), dev)
    levels = C.load_levels()
    res = {"checkpoint": str(a.checkpoint), "per_level": a.per_level, "conditions": {}}
    for i, cond in enumerate(CONDITIONS):
        courses = make(cond, levels, a.per_level, i)
        ev = evaluate(model, CoursePool(courses, dev, C.GRID), dev, stall_ticks=200)
        res["conditions"][cond] = {"success": ev["success"],
                                   "per_level": {L: v["success"] for L, v in ev["per_level"].items()}}
        print(cond, round(ev["success"], 4), {L: round(v["success"], 3) for L, v in ev["per_level"].items()}, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()

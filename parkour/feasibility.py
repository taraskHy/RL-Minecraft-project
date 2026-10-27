"""Which jumps are possible, and how forgiving each one is.

For every candidate jump spec we build a two-platform course with a 1x1
takeoff block and run a family of simple scripted strategies in the batched
simulator: start at the back edge or the centre of the block, face an aim
point on the target, hold forward and sprint, and tap jump at tick j. The
strategies also vary sprinting and when forward is released after the jump.
The fraction of strategies that land on the target ("forgiveness") orders specs by
difficulty; a spec no strategy lands is treated as infeasible from a standing
start and is never generated. Chained jumps can carry momentum, so this is a
conservative test: a spec that passes is possible without any run-up.

Run: python -m parkour.feasibility  (writes configs/jump_feasibility.json and
configs/jump_levels.json)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import numpy as np
import torch

from . import physics
from .courses import JumpSpec, single_jump
from .env import CoursePool, standing_on
from .paths import CONFIGS

GAPS = range(0, 6)
DYS = (-3, -2, -1, 0, 1)
LATERALS = (0, 1, 2)
SIZES = ((1, 1), (2, 2), (3, 3))
START_BACK = (0.29, 0.0)  # centre offset behind the block centre: back edge (BB overlaps by 0.01) or centre
AIM_LATERAL = tuple(np.linspace(-0.45, 0.45, 7))
AIM_ALONG = (-0.3, 0.0, 0.3)
JUMP_TICKS = range(0, 16)
SPRINT = (True, False)
RELEASE_AFTER = (None, 2, 5)  # ticks after the jump at which forward (and sprint) are released
CHUNK = 24  # specs simulated per batch
HORIZON = 45
LEVEL_EDGES = (0.30, 0.15, 0.08, 0.04, 0.015, 0.0)  # forgiveness lower bounds for levels 0..5


def candidate_specs():
    for gap, dy, lat, size in itertools.product(GAPS, DYS, LATERALS, SIZES):
        if gap == 0 and dy == 0:
            continue  # adjacent platforms at one height merge into one
        if gap == 0 and lat > 0 and dy <= 0:
            continue
        yield JumpSpec(gap, dy, lat, size)


def _evaluate_chunk(specs, device):
    courses = [single_jump(s) for s in specs]
    pool = CoursePool(courses, device, grid=(24, 16, 24))
    strategies = list(itertools.product(START_BACK, AIM_LATERAL, AIM_ALONG, JUMP_TICKS, SPRINT, RELEASE_AFTER))
    ns, nk = len(specs), len(strategies)
    n = ns * nk

    def col(i, dtype):
        return torch.tensor([s[i] if s[i] is not None else 10_000 for s in strategies], dtype=dtype,
                            device=device).repeat(ns)

    course = torch.arange(ns, device=device).repeat_interleave(nk)
    back, aim_lat, aim_along = (col(i, torch.float64) for i in range(3))
    jtick = col(3, torch.int64)
    sprint = col(4, torch.bool)
    release = jtick + col(5, torch.int64)

    # Every single_jump course heads +X; the aim point is the target centre shifted.
    target = pool.centre[course, 1]
    start = pool.centre[course, 0].clone()
    start[:, 0] -= back
    aim = target.clone()
    aim[:, 0] += aim_along
    aim[:, 2] += aim_lat
    d = aim - start
    yaw = torch.atan2(-d[:, 0], -d[:, 2])
    s = physics.PlayerState.at(start, yaw)
    landed = torch.zeros(n, dtype=torch.bool, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    low = torch.minimum(pool.top[course, 0], pool.top[course, 1])
    for t in range(HORIZON):
        c = torch.zeros(n, 6, dtype=torch.bool, device=device)
        held = t < release
        c[:, physics.FORWARD] = held
        c[:, physics.SPRINT] = held & sprint
        c[:, physics.JUMP] = jtick == t
        s.yaw = yaw
        s = physics.simulate(pool.world, course, s, c)
        landed |= standing_on(pool, s, course)[:, 1] & alive
        # A strategy stops counting once it has fallen below the lower platform.
        alive &= s.pos[:, 1] > low - 1.0
    return landed.view(ns, nk).float().mean(dim=1).cpu().numpy()


def evaluate(specs, device="cuda"):
    out = []
    for i in range(0, len(specs), CHUNK):
        out.append(_evaluate_chunk(specs[i:i + CHUNK], device))
    return np.concatenate(out)


def assign_level(forgiveness: float) -> int | None:
    if forgiveness <= 0:
        return None
    for level, edge in enumerate(LEVEL_EDGES):
        if forgiveness >= edge:
            return level
    return len(LEVEL_EDGES) - 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    specs = list(candidate_specs())
    t0 = time.time()
    frac = evaluate(specs, args.device)
    rows = []
    for s, f in zip(specs, frac):
        rows.append({"gap": s.gap, "dy": s.dy, "lateral": s.lateral, "size": list(s.size),
                     "forgiveness": round(float(f), 5), "level": assign_level(float(f))})
    out = {"method": __doc__.strip().splitlines()[0], "takeoff": [1, 1], "horizon_ticks": HORIZON,
           "strategies_per_spec": (len(START_BACK) * len(AIM_LATERAL) * len(AIM_ALONG) * len(JUMP_TICKS)
                                   * len(SPRINT) * len(RELEASE_AFTER)),
           "level_edges": LEVEL_EDGES, "specs": rows, "seconds": round(time.time() - t0, 1)}
    (CONFIGS / "jump_feasibility.json").write_text(json.dumps(out, indent=1))
    levels = []
    for L in range(len(LEVEL_EDGES)):
        members = [r for r in rows if r["level"] == L]
        # Mirror lateral offsets so courses jump both left and right.
        specs_l = []
        for r in members:
            specs_l.append({k: r[k] for k in ("gap", "dy", "lateral", "size")})
            if r["lateral"]:
                specs_l.append({**{k: r[k] for k in ("gap", "dy", "size")}, "lateral": -r["lateral"]})
        levels.append({"level": L, "min_forgiveness": LEVEL_EDGES[L], "specs": specs_l})
    (CONFIGS / "jump_levels.json").write_text(json.dumps({"source": "parkour.feasibility", "levels": levels}, indent=1))
    feas = sum(1 for r in rows if r["level"] is not None)
    print(f"{len(rows)} specs, {feas} feasible from a standing 1x1 start, {out['seconds']} s")
    for L in levels:
        print(f"level {L['level']}: {len(L['specs'])} specs")
    return out


if __name__ == "__main__":
    main()

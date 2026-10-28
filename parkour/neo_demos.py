"""Neo demonstrations in the policy's own action space, for learning neos by imitation.

    python -m parkour.neo_demos [--out runs/neo_demos.json] [--per-run 400] [--seeds 3]

The policy almost never discovers a neo by itself: only a tiny fraction of
action sequences land one (parkour.cem). Here the cross-entropy method
searches directly in the policy's discrete action space (move, strafe, jump,
sprint and one of the 7 yaw steps per tick, applied exactly as
ParkourEnv.step does) and keeps every distinct sequence that lands, from a
standing start anywhere on the 1x1 takeoff block.

A sequence counts only if its start does not overlap a block (physics.inside_solid),
the player never stands on top of the wall (env.on_obstacle) and never drops
more than a block below the lower of the two platforms (the fall rule). Every
kept sequence is replayed in ParkourEnv on the same jump followed by an easy
platform (as in parkour.jumpeval), and only those the environment scores as
reaching the target are written out.

Layouts: the 16 single-neo layouts the continuous search landed
(configs/jump_cem.json), on both sides of the wall, each with several seeds
and start angles, so the demonstrations cover many start states. (The
observation is relative to the player, so one heading is enough.)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import courses as C
from . import physics
from .env import YAW_STEPS_DEG, CoursePool, EnvConfig, ParkourEnv, on_obstacle, standing_on
from .paths import CONFIGS, RUNS

HORIZON = 32
SAMPLES = 16384
ELITE = 0.05
GRID = (40, 16, 40)


def neo_course(spec: C.JumpSpec, heading: int) -> C.Course:
    """Takeoff (1x1), the neo, then an easy 3x3 platform, so the target is not the goal."""
    base = C.single_jump(spec, takeoff=(1, 1), grid=GRID, heading=heading)
    after = C._platform_after(base.platforms[1], heading, C.JumpSpec(1, 0, 0, (3, 3)))
    plats = np.concatenate([base.platforms, after[None, :]])
    return C.Course(plats, base.spawn.copy(), base.spawn_yaw, specs=[spec], extra=base.extra)


def search(course: C.Course, device, seed: int, yaw_bias: float, iterations: int, keep: int) -> list:
    """Distinct landing sequences: [{'start': [x, y, z], 'yaw': y0, 'actions': [[m, s, j, sp, yaw], ...]}]."""
    gen = torch.Generator(device=device).manual_seed(seed)
    pool = CoursePool([course], device, GRID)
    n, T = SAMPLES, HORIZON
    cidx = torch.zeros(n, dtype=torch.int64, device=device)
    target = pool.centre[0, 1]
    low = min(float(pool.top[0, 0]), float(pool.top[0, 1]))
    steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64, device=device) * (math.pi / 180)
    p = {"move": torch.tensor([0.05, 0.15, 0.8], device=device).repeat(T, 1),
         "strafe": torch.full((T, 3), 1 / 3, device=device),
         "jump": torch.tensor([0.7, 0.3], device=device).repeat(T, 1),
         "sprint": torch.tensor([0.2, 0.8], device=device).repeat(T, 1),
         "yaw": torch.tensor([0.05, 0.1, 0.15, 0.4, 0.15, 0.1, 0.05], device=device).repeat(T, 1)}
    start_mu = torch.tensor([0.0, 0.0, yaw_bias], device=device, dtype=torch.float64)
    start_sd = torch.tensor([0.2, 0.2, math.radians(40)], device=device, dtype=torch.float64)
    found, seen = [], set()
    for it in range(iterations):
        acts = {k: torch.multinomial(v, n, replacement=True, generator=gen).T for k, v in p.items()}  # (n, T) each
        st = start_mu + start_sd * torch.randn(n, 3, device=device, dtype=torch.float64, generator=gen)
        st[:, :2] = st[:, :2].clamp(-0.29, 0.29)  # feet over the 1x1 takeoff block
        pos = pool.spawn[cidx].clone()
        pos[:, 0] += st[:, 0]
        pos[:, 2] += st[:, 1]
        yaw = course.spawn_yaw + st[:, 2]
        s = physics.PlayerState.at(pos, yaw.clone())
        alive = ~physics.inside_solid(pool.world, cidx, pos)
        landed = torch.zeros(n, dtype=torch.bool, device=device)
        land_tick = torch.full((n,), T, dtype=torch.int64, device=device)
        closest = torch.full((n,), 1e9, device=device, dtype=torch.float64)
        for t in range(T):
            # Exactly ParkourEnv.step: turn first, then move.
            y = s.yaw + steps[acts["yaw"][:, t]]
            s.yaw = torch.remainder(y + math.pi, 2 * math.pi) - math.pi
            m = acts["move"][:, t]
            ctrl = torch.stack((m == 2, m == 0, acts["strafe"][:, t] == 0, acts["strafe"][:, t] == 2,
                                acts["jump"][:, t] == 1, acts["sprint"][:, t] == 1), -1)
            s = physics.simulate(pool.world, cidx, s, ctrl)
            alive &= ~on_obstacle(pool, s, cidx) & (s.pos[:, 1] > low - 1.0)
            now = standing_on(pool, s, cidx)[:, 1] & alive & ~landed
            land_tick = torch.where(now, torch.full_like(land_tick, t + 1), land_tick)
            landed |= now
            closest = torch.where(alive & ~landed, torch.minimum(closest, torch.linalg.norm(s.pos - target, dim=-1)), closest)
        idx = torch.nonzero(landed).flatten().tolist()
        if idx:
            a = torch.stack([acts[k] for k in ("move", "strafe", "jump", "sprint", "yaw")], -1).cpu().numpy()
            P, Y, L = pos.cpu().numpy(), yaw.cpu().numpy(), land_tick.cpu().numpy()
            for i in idx:
                seq = a[i, :L[i]]
                key = (round(float(P[i, 0]), 3), round(float(P[i, 2]), 3), round(float(Y[i]), 3), seq.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                found.append({"start": P[i].tolist(), "yaw": float(Y[i]), "actions": seq.tolist()})
        if len(found) >= keep:
            break
        score = landed.double() * 100.0 - torch.where(landed, torch.zeros_like(closest), closest)
        top = torch.topk(score, int(n * ELITE)).indices
        a = 0.7  # smoothing, as in parkour.cem
        for k, v in p.items():
            freq = torch.nn.functional.one_hot(acts[k][top], v.shape[1]).float().mean(0)
            p[k] = (a * freq + (1 - a) * v).clamp(min=0.01)
            p[k] = p[k] / p[k].sum(-1, keepdim=True)
        start_mu = a * st[top].mean(0) + (1 - a) * start_mu
        start_sd = (a * st[top].std(0) + (1 - a) * start_sd).clamp(min=0.02)
    return found[:keep]


def verify(course: C.Course, demos: list, device) -> list:
    """Keep the demonstrations ParkourEnv scores as reaching the target platform, with their landing tick."""
    if not demos:
        return []
    cs = []
    for d in demos:
        c = C.Course(course.platforms, np.array(d["start"]), d["yaw"], specs=course.specs, extra=course.extra)
        cs.append(c)
    pool = CoursePool(cs, device, GRID)
    n = len(cs)
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=10 ** 6, max_ticks=10 ** 6), sampler=None)
    env.reset_all(torch.arange(n, device=device))
    T = max(len(d["actions"]) for d in demos)
    acts = torch.full((n, T, 5), 0, dtype=torch.int64, device=device)
    acts[..., 0], acts[..., 1], acts[..., 4] = 1, 1, 3  # no input after the sequence ends
    for i, d in enumerate(demos):
        acts[i, :len(d["actions"])] = torch.as_tensor(d["actions"], device=device)
    reached = torch.zeros(n, dtype=torch.bool, device=device)
    bad = torch.zeros(n, dtype=torch.bool, device=device)
    for t in range(T):
        _, _, _, _, _, info = env.step(acts[:, t])
        done = info["done"] & ~reached & ~bad
        bad |= done & (info["fell"] | info["truncated"])
        reached |= ~bad & (env.cp >= 1) & ~info["done"]
        reached |= ~bad & info["success"]
    return [d for d, ok in zip(demos, reached.tolist()) if ok]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RUNS / "neo_demos.json"))
    ap.add_argument("--per-run", type=int, default=400, help="distinct sequences kept per search")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    layouts = [r for r in json.loads((CONFIGS / "jump_cem.json").read_text())["specs"] if r["neo"] == 1 and r["landed"]]
    t0 = time.time()
    out, summary = [], []
    for r in layouts:
        for side in (1, -1):
            spec = C.JumpSpec(r["gap"], r["dy"], r["lateral"], tuple(r["size"]), 1, side, 0, r["wall_at"])
            got = 0
            course = neo_course(spec, 0)
            key = f"g{spec.gap}_dy{spec.dy}_w{spec.wall_at}_side{side}"
            for seed in range(a.seeds):
                bias = math.radians((-25.0, 0.0, 25.0, 0.0)[seed % 4]) * side
                demos = verify(course, search(course, a.device, seed=seed, yaw_bias=bias,
                                              iterations=a.iterations, keep=a.per_run), a.device)
                got += len(demos)
                out += [{"spec": key, "course": course.to_json(), **d} for d in demos]
            summary.append({"gap": spec.gap, "dy": spec.dy, "wall_at": spec.wall_at, "side": side, "demos": got})
            print(json.dumps(summary[-1]), f"{time.time() - t0:.0f} s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"method": __doc__.split("\n\n")[1], "seconds": round(time.time() - t0, 1),
                                       "summary": summary, "demos": out}))
    print(f"{len(out)} demonstrations from {sum(s['demos'] > 0 for s in summary)} of {len(summary)} layouts, "
          f"{time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()

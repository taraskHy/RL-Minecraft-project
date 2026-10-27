"""Is a jump possible at all? Search tick-by-tick action sequences directly.

    python -m parkour.cem [--kinds neo] [--iterations 40]

The scripted strategy families in parkour.feasibility can only show that a
jump is possible. For jump types none of them lands (for example double
neos), this module runs the cross-entropy method over open-loop action
sequences: every tick has its own yaw change (Gaussian) and its own
probabilities of forward, strafe, jump and sprint. Each round simulates
16,384 sequences in the batched simulator, keeps the best 5% (by whether
they land and how close they get) and refits the distribution. Results go to
configs/jump_cem.json and are merged into the advanced levels.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from . import physics
from .courses import JumpSpec, single_jump
from .env import CoursePool, standing_on
from .paths import CONFIGS

HORIZON = 32
SAMPLES = 16384
ELITE = 0.05


def search(spec: JumpSpec, device="cuda", iterations: int = 40, seed: int = 0) -> dict:
    gen = torch.Generator(device=device).manual_seed(seed)
    c = single_jump(spec)
    pool = CoursePool([c], device, grid=(24, 16, 24))
    n, T = SAMPLES, HORIZON
    course = torch.zeros(n, dtype=torch.int64, device=device)
    target = pool.centre[0, 1]
    low = min(float(pool.top[0, 0]), float(pool.top[0, 1]))
    base_yaw = c.spawn_yaw
    yaw_mu = torch.zeros(T, device=device, dtype=torch.float64)
    yaw_sd = torch.full((T,), math.radians(25), device=device, dtype=torch.float64)
    p_fwd = torch.full((T,), 0.8, device=device)
    p_jump = torch.full((T,), 0.3, device=device)
    p_sprint = torch.full((T,), 0.8, device=device)
    p_strafe = torch.full((T, 3), 1 / 3, device=device)  # left, none, right
    start_mu = torch.zeros(3, device=device, dtype=torch.float64)  # start offset x, z and initial yaw
    start_sd = torch.tensor([0.2, 0.2, math.radians(40)], device=device, dtype=torch.float64)
    best = {"landed": False, "score": -1e9}
    for it in range(iterations):
        dyaw = yaw_mu + yaw_sd * torch.randn(n, T, device=device, dtype=torch.float64, generator=gen)
        fwd = torch.rand(n, T, device=device, generator=gen) < p_fwd
        jump = torch.rand(n, T, device=device, generator=gen) < p_jump
        sprint = torch.rand(n, T, device=device, generator=gen) < p_sprint
        strafe = torch.multinomial(p_strafe, n, replacement=True, generator=gen).T  # (n, T)
        st = start_mu + start_sd * torch.randn(n, 3, device=device, dtype=torch.float64, generator=gen)
        st[:, :2] = st[:, :2].clamp(-0.29, 0.29)
        pos = pool.spawn[course].clone()
        pos[:, 0] += st[:, 0]
        pos[:, 2] += st[:, 1]
        yaw = base_yaw + st[:, 2]
        s = physics.PlayerState.at(pos, yaw)
        landed = torch.zeros(n, dtype=torch.bool, device=device)
        # A start overlapping a block (the wall) could walk through it: such samples never count.
        alive = ~physics.inside_solid(pool.world, course, pos)
        closest = torch.full((n,), 1e9, device=device, dtype=torch.float64)
        for t in range(T):
            yaw = yaw + dyaw[:, t]
            ctrl = torch.stack((fwd[:, t], torch.zeros_like(fwd[:, t]), strafe[:, t] == 0, strafe[:, t] == 2,
                                jump[:, t], sprint[:, t]), -1)
            s.yaw = yaw
            s = physics.simulate(pool.world, course, s, ctrl)
            landed |= standing_on(pool, s, course)[:, 1] & alive
            alive &= s.pos[:, 1] > low - 1.0
            dist = torch.linalg.norm(s.pos - target, dim=-1)
            closest = torch.where(alive & ~landed, torch.minimum(closest, dist), closest)
        score = landed.double() * 100.0 - torch.where(landed, torch.zeros_like(closest), closest)
        k = int(n * ELITE)
        top = torch.topk(score, k).indices
        if float(score[top[0]]) > best["score"]:
            best = {"landed": bool(landed[top[0]]), "score": float(score[top[0]]), "iteration": it,
                    "landed_fraction": float(landed.float().mean())}
        # Refit on the elite.
        a = 0.7  # smoothing
        yaw_mu = a * dyaw[top].mean(0) + (1 - a) * yaw_mu
        yaw_sd = (a * dyaw[top].std(0) + (1 - a) * yaw_sd).clamp(min=math.radians(0.5))
        p_fwd = a * fwd[top].float().mean(0) + (1 - a) * p_fwd
        p_jump = a * jump[top].float().mean(0) + (1 - a) * p_jump
        p_sprint = a * sprint[top].float().mean(0) + (1 - a) * p_sprint
        onehot = torch.nn.functional.one_hot(strafe[top], 3).float().mean(0)
        p_strafe = (a * onehot + (1 - a) * p_strafe).clamp(min=0.02)
        p_strafe = p_strafe / p_strafe.sum(-1, keepdim=True)
        start_mu = a * st[top].mean(0) + (1 - a) * start_mu
        start_sd = (a * st[top].std(0) + (1 - a) * start_sd).clamp(min=0.01)
        if best["landed"] and it >= 5 and float(landed.float().mean()) > 0.2:
            break
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    feas = json.loads((CONFIGS / "jump_feasibility_advanced.json").read_text())["specs"]
    todo = [r for r in feas if r["kind"] == "neo"]
    out = []
    t0 = time.time()
    for r in todo:
        spec = JumpSpec(r["gap"], r["dy"], r["lateral"], tuple(r["size"]), neo=r["neo"], side=1,
                        wall_at=r.get("wall_at", 1))
        res = search(spec, a.device, a.iterations)
        out.append({**{k: r[k] for k in ("gap", "dy", "lateral", "size", "neo", "wall_at")},
                    "scripted_forgiveness": r["forgiveness"], **res})
        print(f"neo{r['neo']} gap{r['gap']} wall{r.get('wall_at', 1)} dy{r['dy']:+d}: "
              f"{'POSSIBLE' if res['landed'] else 'not found'} (scripted {r['forgiveness']})", flush=True)
    (CONFIGS / "jump_cem.json").write_text(json.dumps({"method": __doc__.strip().splitlines()[0], "horizon": HORIZON,
                                                        "samples": SAMPLES, "seconds": round(time.time() - t0, 1),
                                                        "specs": out}, indent=1))


if __name__ == "__main__":
    main()


def merge_into_levels() -> None:
    """Add neo specs the search proved possible (none of the scripted strategies did) to the advanced levels.

    A double neo goes to level 7 and a triple neo to level 8, like the scripted ones.
    """
    found = json.loads((CONFIGS / "jump_cem.json").read_text())["specs"]
    path = CONFIGS / "jump_levels_advanced.json"
    data = json.loads(path.read_text())
    by_level = {lv["level"]: lv for lv in data["levels"]}
    have = {json.dumps({k: s[k] for k in sorted(s) if k != "side"}) for lv in data["levels"] for s in lv["specs"]}
    added = 0
    for r in found:
        if not r["landed"] or r["scripted_forgiveness"] > 0:
            continue
        base = {"gap": r["gap"], "dy": r["dy"], "lateral": r["lateral"], "size": r["size"], "neo": r["neo"],
                "runway": 0, "wall_at": r["wall_at"]}
        if json.dumps({k: base[k] for k in sorted(base)}) in have:
            continue
        level = min(8, 5 + r["neo"])
        lv = by_level.setdefault(level, {"level": level, "specs": []})
        lv["specs"] += [{**base, "side": 1}, {**base, "side": -1}]
        added += 1
    data["levels"] = [by_level[k] for k in sorted(by_level)]
    data["cem"] = "neo specs proven possible by parkour.cem were added"
    path.write_text(json.dumps(data, indent=1))
    print(f"added {added} neo specs proven by the search")


def demo(spec: JumpSpec, device="cuda", iterations: int = 40, seed: int = 0) -> dict | None:
    """Search like ``search`` and return the first landing sequence as a replayable trajectory."""
    gen = torch.Generator(device=device).manual_seed(seed)
    c = single_jump(spec)
    pool = CoursePool([c], device, grid=(24, 16, 24))
    n, T = SAMPLES, HORIZON
    course = torch.zeros(n, dtype=torch.int64, device=device)
    target = pool.centre[0, 1]
    low = min(float(pool.top[0, 0]), float(pool.top[0, 1]))
    yaw_mu = torch.zeros(T, device=device, dtype=torch.float64)
    yaw_sd = torch.full((T,), math.radians(25), device=device, dtype=torch.float64)
    probs = {k: torch.full((T,), v, device=device) for k, v in (("fwd", 0.8), ("jump", 0.3), ("sprint", 0.8))}
    p_strafe = torch.full((T, 3), 1 / 3, device=device)
    start_mu = torch.zeros(3, device=device, dtype=torch.float64)
    start_sd = torch.tensor([0.2, 0.2, math.radians(40)], device=device, dtype=torch.float64)
    for it in range(iterations):
        dyaw = yaw_mu + yaw_sd * torch.randn(n, T, device=device, dtype=torch.float64, generator=gen)
        fwd = torch.rand(n, T, device=device, generator=gen) < probs["fwd"]
        jump = torch.rand(n, T, device=device, generator=gen) < probs["jump"]
        sprint = torch.rand(n, T, device=device, generator=gen) < probs["sprint"]
        strafe = torch.multinomial(p_strafe, n, replacement=True, generator=gen).T
        st = start_mu + start_sd * torch.randn(n, 3, device=device, dtype=torch.float64, generator=gen)
        st[:, :2] = st[:, :2].clamp(-0.29, 0.29)
        pos = pool.spawn[course].clone()
        pos[:, 0] += st[:, 0]
        pos[:, 2] += st[:, 1]
        yaw = c.spawn_yaw + st[:, 2]
        s = physics.PlayerState.at(pos, yaw)
        traj_pos, traj_yaw = [s.pos.clone()], [yaw.clone()]
        landed = torch.zeros(n, dtype=torch.bool, device=device)
        land_tick = torch.full((n,), T, dtype=torch.int64, device=device)
        alive = ~physics.inside_solid(pool.world, course, pos)
        closest = torch.full((n,), 1e9, device=device, dtype=torch.float64)
        for t in range(T):
            yaw = yaw + dyaw[:, t]
            ctrl = torch.stack((fwd[:, t], torch.zeros_like(fwd[:, t]), strafe[:, t] == 0, strafe[:, t] == 2,
                                jump[:, t], sprint[:, t]), -1)
            s.yaw = yaw
            s = physics.simulate(pool.world, course, s, ctrl)
            traj_pos.append(s.pos.clone())
            traj_yaw.append(yaw.clone())
            now = standing_on(pool, s, course)[:, 1] & alive & ~landed
            land_tick = torch.where(now, torch.full_like(land_tick, t + 1), land_tick)
            landed |= now
            alive &= s.pos[:, 1] > low - 1.0
            closest = torch.where(alive & ~landed, torch.minimum(closest, torch.linalg.norm(s.pos - target, dim=-1)),
                                  closest)
        if bool(landed.any()):
            i = int(torch.nonzero(landed)[0])
            k = int(land_tick[i])
            P = torch.stack(traj_pos)[: k + 1, i].cpu().tolist()
            Y = torch.stack(traj_yaw)[: k + 1, i].cpu().tolist()
            return {"course": c, "pos": P, "yaw": Y, "iteration": it}
        score = -closest
        top = torch.topk(score, int(n * ELITE)).indices
        a = 0.7
        yaw_mu = a * dyaw[top].mean(0) + (1 - a) * yaw_mu
        yaw_sd = (a * dyaw[top].std(0) + (1 - a) * yaw_sd).clamp(min=math.radians(0.5))
        for key, arr in (("fwd", fwd), ("jump", jump), ("sprint", sprint)):
            probs[key] = a * arr[top].float().mean(0) + (1 - a) * probs[key]
        onehot = torch.nn.functional.one_hot(strafe[top], 3).float().mean(0)
        p_strafe = (a * onehot + (1 - a) * p_strafe).clamp(min=0.02)
        p_strafe = p_strafe / p_strafe.sum(-1, keepdim=True)
        start_mu = a * st[top].mean(0) + (1 - a) * start_mu
        start_sd = (a * st[top].std(0) + (1 - a) * start_sd).clamp(min=0.01)
    return None

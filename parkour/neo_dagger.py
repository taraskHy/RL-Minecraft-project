"""Neo demonstrations from the states the bot really arrives in (DAgger with a search expert).

    python -m parkour.neo_dagger --checkpoint runs/neo-bc/checkpoints/latest.pt --out runs/neo_dagger_r1.pt

Demonstrations from a standing start (parkour.neo_demos) teach single neos,
but inside a course the bot reaches a neo's takeoff block moving, off centre
and turned, right after the previous jump. Here the policy plays training
courses with neos (levels 6 and 7, never the held-out ones); every time it
first stands on a neo's takeoff block, its exact state is recorded. From each
recorded state the cross-entropy method searches the policy's discrete
actions (applied as ParkourEnv.step does) for sequences that land on the
neo's target, never stand on the wall and never fall; up to ``--keep``
distinct ones per state are kept. The observations along them are built with
the environment's own observer, so the output is (observation, action) pairs
for the behaviour-cloning loss (train.py --bc-pairs).

With ``--prior`` (default 0.7) the search starts from the policy's own action
probabilities along its own play from each state, so the sequences it finds
differ from what the bot does only where that makes the neo land. Without it
the search's random choices in ticks that do not matter (sprinting in the air,
strafing) become noisy labels, and imitating them on ordinary course states
made a first round collapse.

Then every found sequence is simplified (``--simplify``): tick by tick and
action part by action part (move, strafe, jump, sprint, turn), the bot's own
choice at that state is put back whenever the neo still lands with it. What
is left are only the corrections that matter: clean labels for imitation.

A takeoff is a neo's when the straight line between its centre and the next
platform's centre passes through a wall block (a block that is not part of a
platform) at body height.
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
from .env import (PLATFORM_SLOTS, YAW_STEPS_DEG, CoursePool, EnvConfig, Observer, ParkourEnv, on_obstacle,
                  standing_on)
from .live import load_model
from .train import course_hash, held_out_courses, obstacle_courses

HORIZON = 32
ELITE = 0.05


def neo_takeoffs(course: C.Course) -> list[int]:
    """Platform indices whose jump to the next platform goes around a wall."""
    # Only non-platform blocks count: the line to a higher platform ends inside that platform.
    solid = {tuple(int(v) for v in b) for b in np.asarray(course.extra).reshape(-1, 3)}
    out = []
    p = course.platforms
    for k in range(len(p) - 1):
        a = ((p[k][0] + p[k][1] + 1) / 2, (p[k][2] + p[k][3] + 1) / 2)
        b = ((p[k + 1][0] + p[k + 1][1] + 1) / 2, (p[k + 1][2] + p[k + 1][3] + 1) / 2)
        y0 = int(p[k][4]) + 1
        blocked = False
        for t in np.linspace(0.0, 1.0, 41)[1:-1]:
            x, z = math.floor(a[0] + t * (b[0] - a[0])), math.floor(a[1] + t * (b[1] - a[1]))
            if any((x, y, z) in solid for y in (y0, y0 + 1)):
                blocked = True
                break
        if blocked:
            out.append(k)
    return out


def training_neo_courses(n: int, seed: int) -> list[C.Course]:
    """Level-6/7 courses with at least one neo (also from wide takeoffs), none equal to a held-out course."""
    levels = C.load_levels(advanced=True)
    held = held_out_courses(levels, 200, 10) + held_out_courses(levels, 100, 10, family=2)
    exclude = {course_hash(c) for c in held}
    rng = np.random.default_rng([2026, 7, seed])
    out = []
    while len(out) < n:
        c = C.generate(rng, levels, 6 + len(out) % 2, int(rng.integers(5, 13)), wide_neo=True)
        if c.n >= 3 and course_hash(c) not in exclude and neo_takeoffs(c):
            out.append(c)
    return out


def collect_arrivals(model, courses: list, device, max_ticks: int = 1500) -> list[dict]:
    """The policy's state when it first stands on each neo takeoff block (deterministic play)."""
    pool = CoursePool(courses, device, C.GRID)
    n = len(courses)
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=200, max_ticks=max_ticks), sampler=None)
    obs = env.reset_all(torch.arange(n, device=device))
    neo = [set(neo_takeoffs(c)) for c in courses]
    seen = [set() for _ in range(n)]
    out = []
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    with torch.no_grad():
        for t in range(max_ticks):
            a, _, _ = model.act(obs, deterministic=True)
            obs, _, _, _, _, info = env.step(a)
            finished |= info["done"]
            cp = env.cp.cpu().numpy()
            ground = env.state.on_ground.cpu().numpy()
            fin = finished.cpu().numpy()
            for i in np.flatnonzero(ground & ~fin):
                k = int(cp[i])
                if k in neo[i] and k not in seen[i]:
                    seen[i].add(k)
                    s = env.state
                    out.append({"course": i, "cp": k, "pos": s.pos[i].tolist(), "vel": s.vel[i].tolist(),
                                "yaw": float(s.yaw[i]), "on_ground": bool(s.on_ground[i]),
                                "jump_ticks": int(s.jump_ticks[i]), "collided_h": bool(s.collided_h[i]),
                                "collided_v": bool(s.collided_v[i]), "last_action": env.last_action[i].tolist()})
            if bool(finished.all()):
                break
    return out


def state_of(arr: list[dict], device) -> physics.PlayerState:
    t = lambda key, dt: torch.tensor([a[key] for a in arr], dtype=dt, device=device)  # noqa: E731
    return physics.PlayerState(t("pos", torch.float64), t("vel", torch.float64), t("yaw", torch.float64),
                               t("on_ground", torch.bool), t("jump_ticks", torch.int64), t("collided_h", torch.bool),
                               t("collided_v", torch.bool))


HEADS = ("move", "strafe", "jump", "sprint", "yaw")


def policy_prior(model, pool: CoursePool, arrivals: list[dict], device) -> dict:
    """The policy's action probabilities along its own deterministic play from each arrival, (A, T, K) per head."""
    obs_fn = Observer(device)
    s = state_of(arrivals, device)
    cidx = torch.tensor([a["course"] for a in arrivals], device=device)
    cp = torch.tensor([a["cp"] for a in arrivals], device=device)
    last = torch.tensor([a["last_action"] for a in arrivals], device=device)
    steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64, device=device) * (math.pi / 180)
    probs = {k: [] for k in HEADS}
    with torch.no_grad():
        for t in range(HORIZON):
            o = obs_fn(pool, cidx, s, cp, last)
            logits = model.logits(model.norm(o))
            acts = []
            for k, lg in zip(HEADS, logits):
                probs[k].append(torch.softmax(lg.float(), -1))
                acts.append(lg.argmax(-1))
            a = torch.stack(acts, -1)
            y = s.yaw + steps[a[:, 4]]
            s.yaw = torch.remainder(y + math.pi, 2 * math.pi) - math.pi
            s = physics.simulate(pool.world, cidx, s, torch.stack((a[:, 0] == 2, a[:, 0] == 0, a[:, 1] == 0, a[:, 1] == 2,
                                                                    a[:, 2] == 1, a[:, 3] == 1), -1))
            last = a
    return {k: torch.stack(v, 1) for k, v in probs.items()}


def search(pool: CoursePool, arrivals: list[dict], device, samples: int, iterations: int, keep: int, seed: int,
           prior: dict | None = None, prior_weight: float = 0.0):
    """Batched CEM: one action distribution per arrival, ``samples`` sequences each. Returns lists of actions."""
    gen = torch.Generator(device=device).manual_seed(seed)
    A, S, T = len(arrivals), samples, HORIZON
    n = A * S
    cidx = torch.tensor([a["course"] for a in arrivals], device=device).repeat_interleave(S)
    target = torch.tensor([a["cp"] + 1 for a in arrivals], device=device).repeat_interleave(S)
    tops = pool.top[cidx, target - 1].minimum(pool.top[cidx, target])
    goal = pool.centre[cidx, target]
    steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64, device=device) * (math.pi / 180)
    init = state_of(arrivals, device)
    rep = lambda x: x.repeat_interleave(S, dim=0)  # noqa: E731
    p = {"move": torch.tensor([0.05, 0.15, 0.8], device=device).repeat(A, T, 1),
         "strafe": torch.full((A, T, 3), 1 / 3, device=device),
         "jump": torch.tensor([0.7, 0.3], device=device).repeat(A, T, 1),
         "sprint": torch.tensor([0.2, 0.8], device=device).repeat(A, T, 1),
         "yaw": torch.tensor([0.05, 0.1, 0.15, 0.4, 0.15, 0.1, 0.05], device=device).repeat(A, T, 1)}
    if prior is not None and prior_weight > 0:
        for key in p:
            p[key] = prior_weight * prior[key] + (1 - prior_weight) * p[key]
            p[key] = p[key].clamp(min=0.01)
            p[key] = p[key] / p[key].sum(-1, keepdim=True)
    found = [[] for _ in range(A)]
    seen = [set() for _ in range(A)]
    k_elite = max(1, int(S * ELITE))
    for it in range(iterations):
        acts = {}
        for key, v in p.items():  # (A, T, K) -> samples (A*S, T)
            flat = v.reshape(A * T, -1)
            draw = torch.multinomial(flat, S, replacement=True, generator=gen)  # (A*T, S)
            acts[key] = draw.reshape(A, T, S).permute(0, 2, 1).reshape(n, T)
        s = physics.PlayerState(rep(init.pos), rep(init.vel), rep(init.yaw), rep(init.on_ground), rep(init.jump_ticks),
                                rep(init.collided_h), rep(init.collided_v))
        alive = torch.ones(n, dtype=torch.bool, device=device)
        landed = torch.zeros(n, dtype=torch.bool, device=device)
        land_tick = torch.full((n,), T, dtype=torch.int64, device=device)
        closest = torch.full((n,), 1e9, dtype=torch.float64, device=device)
        for t in range(T):
            y = s.yaw + steps[acts["yaw"][:, t]]
            s.yaw = torch.remainder(y + math.pi, 2 * math.pi) - math.pi
            m = acts["move"][:, t]
            ctrl = torch.stack((m == 2, m == 0, acts["strafe"][:, t] == 0, acts["strafe"][:, t] == 2,
                                acts["jump"][:, t] == 1, acts["sprint"][:, t] == 1), -1)
            s = physics.simulate(pool.world, cidx, s, ctrl)
            alive &= ~on_obstacle(pool, s, cidx) & (s.pos[:, 1] > tops - 1.0)
            on = standing_on(pool, s, cidx)
            now = on.gather(1, target[:, None]).squeeze(1) & alive & ~landed
            land_tick = torch.where(now, torch.full_like(land_tick, t + 1), land_tick)
            landed |= now
            closest = torch.where(alive & ~landed, torch.minimum(closest, torch.linalg.norm(s.pos - goal, dim=-1)), closest)
        lt = land_tick.cpu().numpy()
        L = landed.cpu().numpy()
        if L.any():
            stack = torch.stack([acts[k] for k in ("move", "strafe", "jump", "sprint", "yaw")], -1).cpu().numpy()
            for j in np.flatnonzero(L):
                a = j // S
                if len(found[a]) >= keep:
                    continue
                seq = stack[j, :lt[j]]
                b = seq.tobytes()
                if b not in seen[a]:
                    seen[a].add(b)
                    found[a].append(seq)
        if all(len(f) >= keep for f in found):
            break
        score = (landed.double() * 100.0 - torch.where(landed, torch.zeros_like(closest), closest)).reshape(A, S)
        top = torch.topk(score, k_elite, dim=1).indices  # (A, E)
        a_ = 0.7
        for key, v in p.items():
            sel = acts[key].reshape(A, S, T).gather(1, top[:, :, None].expand(-1, -1, T))  # (A, E, T)
            freq = torch.nn.functional.one_hot(sel, v.shape[-1]).float().mean(1)  # (A, T, K)
            p[key] = (a_ * freq + (1 - a_) * v).clamp(min=0.01)
            p[key] = p[key] / p[key].sum(-1, keepdim=True)
    return found


def _rollout(model, pool, arrivals, idx, given, fixed, device):
    """Play from each arrival: ``given`` actions where ``fixed`` is true, the policy's own choice elsewhere.

    Returns (landed, land_tick, actions actually played)."""
    obs_fn = Observer(device)
    arr = [arrivals[i] for i in idx]
    s = state_of(arr, device)
    cidx = torch.tensor([a["course"] for a in arr], device=device)
    cp = torch.tensor([a["cp"] for a in arr], device=device)
    last = torch.tensor([a["last_action"] for a in arr], device=device)
    target = cp + 1
    tops = pool.top[cidx, cp].minimum(pool.top[cidx, target])
    steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64, device=device) * (math.pi / 180)
    n, T = given.shape[0], given.shape[1]
    played = torch.zeros_like(given)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    landed = torch.zeros(n, dtype=torch.bool, device=device)
    land_tick = torch.full((n,), T, dtype=torch.int64, device=device)
    with torch.no_grad():
        for t in range(T):
            o = obs_fn(pool, cidx, s, cp, last)
            own = torch.stack([lg.argmax(-1) for lg in model.logits(model.norm(o))], -1)
            a = torch.where(fixed[:, t], given[:, t], own)
            a = torch.where(landed[:, None], torch.zeros_like(a), a)
            played[:, t] = a
            y = s.yaw + steps[a[:, 4]]
            s.yaw = torch.remainder(y + math.pi, 2 * math.pi) - math.pi
            s = physics.simulate(pool.world, cidx, s, torch.stack((a[:, 0] == 2, a[:, 0] == 0, a[:, 1] == 0, a[:, 1] == 2,
                                                                    a[:, 2] == 1, a[:, 3] == 1), -1))
            alive &= ~on_obstacle(pool, s, cidx) & (s.pos[:, 1] > tops - 1.0)
            now = standing_on(pool, s, cidx).gather(1, target[:, None]).squeeze(1) & alive & ~landed
            land_tick = torch.where(now, torch.full_like(land_tick, t + 1), land_tick)
            landed |= now
            last = a
    return landed, land_tick, played


def simplify(model, pool: CoursePool, arrivals: list[dict], found: list, device, chunk: int = 4096) -> list:
    """Put the policy's own choice back wherever the sequence still lands (greedy, tick by tick, part by part)."""
    seqs = [(i, f) for i, fs in enumerate(found) for f in fs]
    out = [[] for _ in found]
    for c0 in range(0, len(seqs), chunk):
        part = seqs[c0:c0 + chunk]
        idx = [i for i, _ in part]
        n, T = len(part), HORIZON
        given = torch.zeros(n, T, 5, dtype=torch.int64, device=device)
        fixed = torch.zeros(n, T, 5, dtype=torch.bool, device=device)
        for j, (_, f) in enumerate(part):
            given[j, :len(f)] = torch.as_tensor(f, device=device)
            fixed[j, :len(f)] = True
        lens = fixed[:, :, 0].sum(1)
        for t in range(int(lens.max())):
            for h in range(5):
                trial = fixed.clone()
                trial[:, t, h] = False
                ok, _, _ = _rollout(model, pool, arrivals, idx, given, trial, device)
                keep = ok & (t < lens)
                fixed = torch.where(keep[:, None, None], trial, fixed)
        ok, tick, played = _rollout(model, pool, arrivals, idx, given, fixed, device)
        for j, (i, _) in enumerate(part):
            if bool(ok[j]):
                out[i].append(played[j, :int(tick[j])].cpu().numpy())
    return out


def pairs(pool: CoursePool, arrivals: list[dict], found: list, device):
    """Observations along each found sequence from its arrival state, with the actions."""
    obs_fn = Observer(device)
    seqs = [(i, f) for i, fs in enumerate(found) for f in fs]
    if not seqs:
        return torch.zeros(0, 255), torch.zeros(0, 5, dtype=torch.int64)
    idx = [i for i, _ in seqs]
    arr = [arrivals[i] for i in idx]
    n = len(seqs)
    s = state_of(arr, device)
    cidx = torch.tensor([a["course"] for a in arr], device=device)
    cp = torch.tensor([a["cp"] for a in arr], device=device)
    last = torch.tensor([a["last_action"] for a in arr], device=device)
    T = max(len(f) for _, f in seqs)
    acts = torch.zeros(n, T, 5, dtype=torch.int64, device=device)
    lens = torch.tensor([len(f) for _, f in seqs], device=device)
    for j, (_, f) in enumerate(seqs):
        acts[j, :len(f)] = torch.as_tensor(f, device=device)
    steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64, device=device) * (math.pi / 180)
    obs_all, act_all = [], []
    for t in range(T):
        m = lens > t
        o = obs_fn(pool, cidx, s, cp, last)
        obs_all.append(o[m].cpu())
        act_all.append(acts[m, t].cpu())
        a = acts[:, t]
        y = s.yaw + steps[a[:, 4]]
        s.yaw = torch.remainder(y + math.pi, 2 * math.pi) - math.pi
        s = physics.simulate(pool.world, cidx, s, torch.stack((a[:, 0] == 2, a[:, 0] == 0, a[:, 1] == 0, a[:, 1] == 2,
                                                                a[:, 2] == 1, a[:, 3] == 1), -1))
        last = a
    return torch.cat(obs_all), torch.cat(act_all)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--courses", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=32, help="arrival states searched at once")
    ap.add_argument("--samples", type=int, default=1024, help="action sequences per arrival state and round")
    ap.add_argument("--iterations", type=int, default=25)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior", type=float, default=0.95, help="weight of the policy's own action probabilities in the search's start")
    ap.add_argument("--simplify", type=int, default=1, help="1: keep only the corrections that matter (see above)")
    ap.add_argument("--expert", choices=("cem", "assist"), default="cem",
                    help="assist: parkour.neo_assist's search (same method; finished states drop out, so faster)")
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = load_model(Path(a.checkpoint), dev)
    courses = training_neo_courses(a.courses, a.seed)
    arrivals = collect_arrivals(model, courses, dev)
    print(json.dumps({"courses": len(courses), "arrivals": len(arrivals), "seconds": round(time.time() - t0)}), flush=True)
    pool = CoursePool(courses, dev, C.GRID)
    found = []
    if a.expert == "assist":
        from . import neo_assist as NA
        rules = NA.Rules(pool)
        cfg = NA.AssistConfig(samples=a.samples, iterations=a.iterations, keep=a.keep, prior=a.prior)
    for b in range(0, len(arrivals), a.batch):
        part = arrivals[b:b + a.batch]
        if a.expert == "assist":
            prior = NA.policy_prior(model, pool, part, dev, HORIZON) if a.prior > 0 else None
            found += NA.search(pool, rules, part, dev, cfg, a.seed + b, prior, HORIZON)[0]
            continue
        prior = policy_prior(model, pool, part, dev) if a.prior > 0 else None
        found += search(pool, part, dev, a.samples, a.iterations, a.keep, a.seed + b, prior, a.prior)
        if (b // a.batch) % 10 == 0:
            ok = sum(bool(f) for f in found)
            print(json.dumps({"searched": len(found), "with_landing": ok, "seconds": round(time.time() - t0)}), flush=True)
    if a.simplify:
        found = simplify(model, pool, arrivals, found, dev)
        print(json.dumps({"simplified_sequences": sum(len(f) for f in found), "seconds": round(time.time() - t0)}), flush=True)
    obs, act = pairs(pool, arrivals, found, dev)
    torch.save({"obs": obs, "act": act, "arrivals": len(arrivals), "with_landing": sum(bool(f) for f in found),
                "sequences": sum(len(f) for f in found), "checkpoint": a.checkpoint}, a.out)
    print(json.dumps({"arrivals": len(arrivals), "with_landing": sum(bool(f) for f in found),
                      "sequences": sum(len(f) for f in found), "pairs": int(obs.shape[0]),
                      "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()

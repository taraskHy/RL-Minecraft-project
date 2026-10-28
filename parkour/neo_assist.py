"""Neo assist: an exact-model search at every neo takeoff (simulator only).

    python -m parkour.neo_assist --checkpoint results/final_policy.pt --set official --family 2
    python -m parkour.neo_assist --checkpoint results/final_policy.pt --set random --family 13 --prior 0.5
    python -m parkour.neo_assist --checkpoint results/final_policy.pt --set official --family 2 --off

The policy plays every course as in ``train.evaluate`` (deterministic, the
first episode per course). The first time it stands on a neo's takeoff block
(``neo_dagger.neo_takeoffs``: the line to the next platform passes through a
wall block), a cross-entropy search from its exact state looks for a
sequence of the policy's own discrete actions that lands on the neo's target
without standing on an obstacle or falling: the search of
``parkour.neo_dagger``, with the simulator as the model. A found sequence
replaces the policy's actions until it has landed; then the policy plays on.
With no landing found the policy keeps control, and with ``--retries`` the
search runs again every ``--retry-every`` ticks while the bot still stands
on the takeoff (a bot that stopped there can often be planned from the
standstill; ``--retry-horizon`` gives those plans more ticks, to walk back
before the jump). With ``--brake T`` the bot first stops on the takeoff
(``brake_plan``: "back" for up to 6 ticks, then no keys, T ticks in all,
checked with the physics) and the retry runs as soon as it stands still.
With ``--approach H`` the search also runs when the bot first stands on the
platform before a neo's takeoff, for a plan of up to H ticks over both
jumps, so the bot reaches the takeoff in a state the neo can be made from
(without a landing the takeoff search runs as usual). With
``--jumps advanced`` the search also runs at wide (3 sideways) and run-up
jumps (momentum, head hitters, ice), with ``--jumps all`` at every jump.

This is planning with an exact model of the game's physics. It measures
what the bot does when it plans its neos, not what the network learned. The
bots have it too (``parkour.live --assist``: the planner service ``--serve``,
bridge/assist.js); the offline bot reproduces this module's plans when both
search on the same device (CUDA's random draws differ from the CPU's).

Course sets (``--set``):
  official  the held-out level 6-8 courses that contain a neo (family 1:
            the 210 test courses, family 2: the 218 validation courses).
  random    courses of ordinary jumps (levels 0-5) in which half of the
            jumps are neos (the 32 single-neo jump types of levels 6 and 7),
            10 jumps each, from seed families 12 (test) and 13 (validation),
            which nothing else uses.
  mixed     the same with every special kind instead of only neos (single
            neos, wide and momentum jumps, head hitters, ice run-ups, shaped
            blocks, slime bounces, ladder climbs; ``mixed_level``), families
            14 (test) and 15 (validation).

Settings are chosen on the validation families; the test families are run
once with the chosen setting. Each run appends one JSON line to ``--out``
(default ``results/neo/assist_validation.jsonl`` or ``assist_test.jsonl``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from . import courses as C
from . import physics
from .env import (OBS_DIM_EXT, STAND_EPS, YAW_STEPS_DEG, CoursePool, EnvConfig, Observer, ParkourEnv, action_to_controls,
                  standing_on)
from .live import load_model
from .neo_dagger import ELITE, HORIZON, neo_takeoffs
from .paths import ROOT
from .train import TEST_SEED_FAMILY, VALIDATION_FAMILY, held_out_courses

RANDOM_TEST_FAMILY = 12
RANDOM_VALIDATION_FAMILY = 13
MIXED_TEST_FAMILY = 14
MIXED_VALIDATION_FAMILY = 15
TEST_FAMILIES = (TEST_SEED_FAMILY, RANDOM_TEST_FAMILY, MIXED_TEST_FAMILY)
HEADS = ("move", "strafe", "jump", "sprint", "yaw")


@dataclass
class AssistConfig:
    enabled: bool = True
    samples: int = 1024  # action sequences per arrival state and search round
    iterations: int = 25  # search rounds at most
    keep: int = 1  # stop an arrival's search after this many distinct landings (the first is played)
    horizon: int = HORIZON  # ticks a plan from the takeoff may take
    retry_horizon: int = 0  # ticks for retry plans (0: horizon)
    approach: int = 0  # >0: also plan from the platform before a planned takeoff, over both jumps, in so many ticks
    jumps: str = "neo"  # which jumps are planned: "neo", "advanced" (also wide and momentum jumps, by spec) or "all"
    prior: float = 0.0  # weight of the policy's own action probabilities in the search's start
    retries: int = 0  # extra searches while the bot still stands on the takeoff without a plan
    retry_every: int = 10  # ticks between them
    brake: int = 0  # >0: after a search without a landing, first stop on the takeoff (brake_plan, this many ticks), then retry
    batch: int = 1  # arrival states searched together; 1: every search has its own seed (the bots reproduce it)
    seed: int = 0


KIND_CODE = {"takeoff": 0, "retry": 1, "approach": 2}


def request_seed(seed: int, course: int, cp: int, kind: str, attempt: int) -> int:
    """The search seed of one request: the same in the simulator and in the bots' planner service."""
    return seed * 1_000_003 + course * 4099 + cp * 67 + KIND_CODE[kind] * 7 + attempt


def plan_once(model, pool: CoursePool, rules: "Rules", arrival: dict, kind: str, attempt: int, cfg: "AssistConfig",
              horizon: int, device):
    """One search from one state: (the first landing's actions or None, rounds used)."""
    pr = policy_prior(model, pool, [arrival], device, horizon) if cfg.prior > 0 else None
    seed = request_seed(cfg.seed, arrival["course"], arrival["cp"], kind, attempt)
    f, r = search(pool, rules, [arrival], device, cfg, seed, pr, horizon)
    return (f[0][0] if f[0] else None), r[0]


BRAKE_BACK_TICKS = 6  # brake_plan holds "back" for 0 to this many ticks


def brake_plan(pool: CoursePool, rules: "Rules", arrival: dict, ticks: int, device):
    """Keys that stop the bot on its takeoff: "back" for the first k ticks (k = 0, 1, ... 6), then none, ``ticks``
    in all. The first sequence after which the exact physics has the bot standing still on the platform it
    stands on now (horizontal speed below 0.001), never on an obstacle; None if none does."""
    ks = list(range(min(BRAKE_BACK_TICKS, ticks) + 1))
    seqs = np.zeros((len(ks), ticks, 5), dtype=np.int64)
    seqs[:, :, 0] = 1  # move: none
    seqs[:, :, 1] = 1  # strafe: none
    seqs[:, :, 4] = YAW_STEPS_DEG.index(0.0)  # no turn
    for j, k in enumerate(ks):
        seqs[j, :k, 0] = 0  # back
    s = _state_rows([arrival] * len(ks), device)
    course = torch.full((len(ks),), arrival["course"], device=device)
    acts = torch.as_tensor(seqs, device=device)
    steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64, device=device) * (math.pi / 180)
    alive = torch.ones(len(ks), dtype=torch.bool, device=device)
    for t in range(ticks):
        y = s.yaw + steps[acts[:, t, 4]]
        s.yaw = torch.remainder(y + math.pi, 2 * math.pi) - math.pi
        s = physics.simulate(pool.world, course, s, action_to_controls(acts[:, t]))
        alive &= ~rules.on_obstacle(s, course)
    still = standing_on(pool, s, course)[:, arrival["cp"]] & (torch.linalg.norm(s.vel[:, [0, 2]], dim=-1) < 1e-3)
    ok = torch.nonzero(alive & still).flatten().tolist()
    return seqs[ok[0]] if ok else None


def horizons(cfg: "AssistConfig") -> dict:
    return {"takeoff": cfg.horizon, "retry": cfg.retry_horizon or cfg.horizon, "approach": cfg.approach}


BOUNCE_PLAN_HORIZON = 80  # a jump onto a slime pad: takeoff, fall, bounce and the flight to the exit
LADDER_PLAN_HORIZON = 64  # a ladder climb: 5 blocks take about 43 ticks from a standstill


def plan_target(course: C.Course, k: int) -> int:
    """The platform a plan from takeoff ``k`` ends on: the next one; for a jump onto a slime pad, the exit
    after the pad (how high the bounce goes depends on how the pad is hit, so both are one plan)."""
    sp = course.specs[k] if k < len(course.specs) else None
    return k + 2 if sp is not None and sp.bounce and k + 2 < course.n else k + 1


def plan_horizon(course: C.Course, k: int, kind: str, cfg: "AssistConfig") -> int:
    """Ticks a plan of ``kind`` for the jump from takeoff ``k`` may take: the configured horizon, at least
    ``BOUNCE_PLAN_HORIZON`` for slime jumps and ``LADDER_PLAN_HORIZON`` for ladder climbs."""
    h = horizons(cfg)[kind]
    sp = course.specs[k] if k < len(course.specs) else None
    if kind == "approach" or sp is None:
        return h
    if sp.bounce:
        return max(h, BOUNCE_PLAN_HORIZON)
    if sp.ladder:
        return max(h, LADDER_PLAN_HORIZON)
    return h


def specs_from_keys(keys: list[str], n_platforms: int) -> list[C.JumpSpec]:
    """A course file's jump specs, one per jump from platform k to k + 1 as ``courses.generate`` records them.

    Hand-designed courses (``courses.from_specs``) list only the specs they were built from; their runway
    steps and the exits after shaped targets and slime pads are put back."""
    specs = [spec_from_key(k) for k in keys]
    if len(specs) == n_platforms - 1:
        return specs
    out = []
    for sp in specs:
        if sp.runway:
            out.append(C.JumpSpec(1, 0, 0, (sp.runway, 1)))
        out.append(sp)
        if sp.block:
            out.append(C.exit_spec(sp.block))
        if sp.bounce:
            out.append(C.bounce_exit(sp))
    return out if len(out) == n_platforms - 1 else []  # no specs rather than specs of the wrong jumps


def courses_from_file(raw: list[dict]) -> list[C.Course]:
    """Courses of a course file (its "courses" list) with their jump specs, which ``Course.from_json`` leaves out."""
    out = []
    for d in raw:
        c = C.Course.from_json(d)
        if d.get("specs"):
            c.specs = specs_from_keys(d["specs"], c.n)
        out.append(c)
    return out


def assist_fields(course: C.Course, cfg: "AssistConfig") -> dict:
    """What the bots' trigger rules need of a course (bridge/assist.js): the planned takeoffs, the platform
    each plan ends on, and the retry horizon of each takeoff (for the stall check before a retry)."""
    ks = planned_takeoffs(course, cfg.jumps)
    return {"assist_takeoffs": ks, "assist_targets": [plan_target(course, k) for k in ks],
            "assist_retry_horizons": [plan_horizon(course, k, "retry", cfg) for k in ks]}


_KEY = re.compile(r"g(-?\d+)_dy(-?\d+)_l(-?\d+)_s(\d+)x(\d+)(?:_neo(\d)([RL])(?:w(\d+))?)?(?:_run(\d+))?"
                  r"(?:_hh(\d))?(?:_ice(\d))?(?:_b([a-z_]+?))?(?:_slime(\d+))?(_ladder)?$")


def spec_from_key(key: str) -> C.JumpSpec:
    """The JumpSpec a course file names by ``JumpSpec.key()``."""
    m = _KEY.match(key)
    if not m:
        raise ValueError(key)
    g, dy, lat, a, b, neo, side, wall, run, hh, ice, block, slime, ladder = m.groups()
    return C.JumpSpec(int(g), int(dy), int(lat), (int(a), int(b)), int(neo or 0), 1 if side != "L" else -1,
                      int(run or 0), int(wall or 1), int(hh or 0), int(ice or 0), block or "", int(slime or 0),
                      1 if ladder else 0)


def planned_takeoffs(course: C.Course, jumps: str) -> list[int]:
    """Where the assist plans: neo takeoffs; (``advanced``) also wide and momentum jumps; (``all``) every jump.

    A course read from a file needs its ``specs`` back first (``specs_from_keys``, as ``parkour.live`` does)."""
    if jumps == "neo":
        return neo_takeoffs(course)
    if jumps == "advanced":
        return advanced_takeoffs(course)
    if jumps == "all":
        return list(range(course.n - 1))
    raise ValueError(jumps)


def advanced_takeoffs(course: C.Course) -> list[int]:
    """Takeoffs of the jumps the advanced and extra levels added: neos, wide jumps (3 sideways) and run-up
    jumps (momentum, head hitters, ice).

    Neos by the wall rule (``neo_takeoffs``), the others by the course's jump
    specs (``specs[k]`` is the jump from platform k to k + 1)."""
    ks = set(neo_takeoffs(course))
    ks |= {k for k, sp in enumerate(course.specs) if abs(sp.lateral) == 3 or sp.runway > 0 or sp.neo}
    return sorted(k for k in ks if k + 1 < course.n)


def jump_kind(course: C.Course, k: int, neos: list) -> str:
    """The kind of the jump from platform k: neo, wide (3 sideways), momentum (run-up), run-up under a roof or
    on ice, onto a shaped block, off one (its exit), onto a slime pad, off a pad, a ladder climb, other, or basic."""
    if k in neos:
        return "neo"
    sp = course.specs[k] if k < len(course.specs) else None
    if sp is None:
        return "unknown"
    if sp.runway:
        return "momentum" if not (sp.ceiling or sp.ice) else "run-up (roof or ice)"
    if abs(sp.lateral) == 3:
        return "wide"
    if sp.block:
        return "onto a shaped block"
    if k in course.shapes:
        return "off a shaped block"
    if sp.bounce:
        return "onto a slime pad"
    if k in course.slime:
        return "off a slime pad"
    if sp.ladder:
        return "ladder climb"
    if sp.neo:
        return "other advanced"
    return "basic"


def neo_level() -> list[C.JumpSpec]:
    """The single-neo jump types of the advanced levels 6 and 7."""
    adv = C.load_levels(advanced=True)
    return [s for L in (6, 7) for s in adv[L] if s.neo == 1]


def random_neo_courses(family: int, n: int = 200, n_jumps: int = 10) -> list[C.Course]:
    """Courses of level-0-5 jumps where half the jumps are neos (generate's level mix with a neo-only level 6)."""
    levels = C.load_levels() + [neo_level()]
    rng = np.random.default_rng([2026, family, 6])
    return [C.generate(rng, levels, 6, n_jumps, seed=i) for i in range(n)]


def official_neo_courses(family: int) -> list[C.Course]:
    """The held-out level 6-8 courses (200 per level, 10 jumps) that contain a neo."""
    hard = held_out_courses(C.load_levels(advanced=True), 200, 10, family=family)[6 * 200:]
    return [c for c in hard if neo_takeoffs(c)]


def mixed_level() -> list[C.JumpSpec]:
    """Every special kind in one level, each group about equally likely: the single neos, the wide jumps and
    the momentum jumps of the advanced levels 6-8, and each extra level (head hitters, hard head hitters, ice
    run-ups, shaped blocks, slime bounces, ladder climbs). A group of n specs is repeated round(N / n) times,
    N the largest group, since ``courses.generate`` draws a level's specs uniformly."""
    adv = [s for L in C.load_levels(advanced=True)[6:] for s in L]
    groups = [[s for s in adv if s.neo == 1], [s for s in adv if abs(s.lateral) == 3 and not s.neo and not s.runway],
              [s for s in adv if s.runway and not s.neo]] + C.load_levels(extra=True)[6:]
    groups = [g for g in groups if g]
    most = max(len(g) for g in groups)
    return [s for g in groups for s in g * max(1, round(most / len(g)))]


def mixed_courses(family: int, n: int = 200, n_jumps: int = 10) -> list[C.Course]:
    """Courses of level-0-5 jumps where half the jumps are of the special kinds (``mixed_level``)."""
    levels = C.load_levels() + [mixed_level()]
    rng = np.random.default_rng([2026, family, 6])
    return [C.generate(rng, levels, 6, n_jumps, seed=i) for i in range(n)]


def course_set(name: str, family: int) -> list[C.Course]:
    if name == "official":
        return official_neo_courses(family)
    if name == "random":
        return random_neo_courses(family)
    if name == "mixed":
        return mixed_courses(family)
    raise ValueError(name)


class Rules:
    """``env.standing_on`` / ``env.on_obstacle`` for search rows that all play known courses.

    ``on_obstacle`` compares the feet with every obstacle slot of the course
    (256, mostly padding far outside the world); here only the slots that
    hold a cell in some course of the pool are compared, which gives the same
    answer (a padding slot can never be under the feet) at a fraction of the cost.
    """

    def __init__(self, pool: CoursePool):
        self.pool = pool
        used = (pool.extra[..., 0] > -10_000).sum(1)
        self.extra = pool.extra[:, :max(1, int(used.max()))].to(torch.float64)

    def on_obstacle(self, s: physics.PlayerState, course: torch.Tensor) -> torch.Tensor:
        e = self.extra[course]
        x, y, z = s.pos[:, 0:1], s.pos[:, 1:2], s.pos[:, 2:3]
        w = physics.HALF_WIDTH
        over = (x + w > e[..., 0]) & (x - w < e[..., 0] + 1) & (z + w > e[..., 2]) & (z - w < e[..., 2] + 1)
        level = (y - (e[..., 1] + 1)).abs() < STAND_EPS
        return s.on_ground & (over & level).any(dim=1)


def _state_rows(arrivals: list[dict], device) -> physics.PlayerState:
    t = lambda key, dt: torch.tensor([a[key] for a in arrivals], dtype=dt, device=device)  # noqa: E731
    return physics.PlayerState(t("pos", torch.float64), t("vel", torch.float64), t("yaw", torch.float64),
                               t("on_ground", torch.bool), t("jump_ticks", torch.int64), t("collided_h", torch.bool),
                               t("collided_v", torch.bool))


def policy_prior(model, pool: CoursePool, arrivals: list[dict], device, horizon: int) -> dict:
    """The policy's action probabilities along its own deterministic play from each arrival, (A, T, K) per head.

    ``neo_dagger.policy_prior`` with the horizon as a parameter."""
    obs_fn = Observer(device)
    s = _state_rows(arrivals, device)
    cidx = torch.tensor([a["course"] for a in arrivals], device=device)
    cp = torch.tensor([a["cp"] for a in arrivals], device=device)
    last = torch.tensor([a["last_action"] for a in arrivals], device=device)
    steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64, device=device) * (math.pi / 180)
    probs = {k: [] for k in HEADS}
    with torch.no_grad():
        for _ in range(horizon):
            o = obs_fn(pool, cidx, s, cp, last)
            logits = model.logits(model.norm(o))
            acts = []
            for k, lg in zip(HEADS, logits):
                probs[k].append(torch.softmax(lg.float(), -1))
                acts.append(lg.argmax(-1))
            a = torch.stack(acts, -1)
            y = s.yaw + steps[a[:, 4]]
            s.yaw = torch.remainder(y + math.pi, 2 * math.pi) - math.pi
            s = physics.simulate(pool.world, cidx, s, action_to_controls(a))
            k = torch.arange(C.MAX_PLATFORMS, device=device)
            cp = torch.maximum(cp, torch.where(standing_on(pool, s, cidx), k, torch.full_like(k, -1)).amax(1))
            last = a
    return {k: torch.stack(v, 1) for k, v in probs.items()}


def search(pool: CoursePool, rules: Rules, arrivals: list[dict], device, cfg: AssistConfig, seed: int,
           prior: dict | None = None, horizon: int = HORIZON) -> tuple[list, list]:
    """Batched cross-entropy search (as ``neo_dagger.search``), each arrival stopping once it has ``keep`` landings.

    An arrival may name a ``target`` platform further on than the next one
    (approach plans); the fall rule then uses the lowest top from ``cp`` to
    the target. Returns (found, rounds): per arrival a list of action
    sequences (each cut at its landing tick) and the number of search rounds
    it used.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    A, S, T = len(arrivals), cfg.samples, horizon
    steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64, device=device) * (math.pi / 180)
    init = _state_rows(arrivals, device)
    course = torch.tensor([a["course"] for a in arrivals], device=device)
    target = torch.tensor([a.get("target", a["cp"] + 1) for a in arrivals], device=device)
    tops = torch.stack([pool.top[a["course"], a["cp"]:a.get("target", a["cp"] + 1) + 1].min() for a in arrivals])
    goal = pool.centre[course, target]
    p = {"move": torch.tensor([0.05, 0.15, 0.8], device=device).repeat(A, T, 1),
         "strafe": torch.full((A, T, 3), 1 / 3, device=device),
         "jump": torch.tensor([0.7, 0.3], device=device).repeat(A, T, 1),
         "sprint": torch.tensor([0.2, 0.8], device=device).repeat(A, T, 1),
         "yaw": torch.tensor([0.05, 0.1, 0.15, 0.4, 0.15, 0.1, 0.05], device=device).repeat(A, T, 1)}
    if prior is not None and cfg.prior > 0:
        for key in p:
            p[key] = (cfg.prior * prior[key] + (1 - cfg.prior) * p[key]).clamp(min=0.01)
            p[key] = p[key] / p[key].sum(-1, keepdim=True)
    found = [[] for _ in range(A)]
    seen = [set() for _ in range(A)]
    rounds = [0] * A
    k_elite = max(1, int(S * ELITE))
    active = list(range(A))
    for it in range(cfg.iterations):
        act_idx = torch.tensor(active, device=device)
        a_n = len(active)
        n = a_n * S
        rep = lambda x: x[act_idx].repeat_interleave(S, dim=0)  # noqa: E731
        acts = {}
        for key, v in p.items():
            flat = v[act_idx].reshape(a_n * T, -1)
            draw = torch.multinomial(flat, S, replacement=True, generator=gen)
            acts[key] = draw.reshape(a_n, T, S).permute(0, 2, 1).reshape(n, T)
        s = physics.PlayerState(rep(init.pos), rep(init.vel), rep(init.yaw), rep(init.on_ground), rep(init.jump_ticks),
                                rep(init.collided_h), rep(init.collided_v))
        cidx, tgt, top_r, goal_r = rep(course), rep(target), rep(tops), rep(goal)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        landed = torch.zeros(n, dtype=torch.bool, device=device)
        land_tick = torch.full((n,), T, dtype=torch.int64, device=device)
        closest = torch.full((n,), 1e9, dtype=torch.float64, device=device)
        for t in range(T):
            y = s.yaw + steps[acts["yaw"][:, t]]
            s.yaw = torch.remainder(y + math.pi, 2 * math.pi) - math.pi
            a_t = torch.stack([acts[k][:, t] for k in HEADS], -1)
            s = physics.simulate(pool.world, cidx, s, action_to_controls(a_t))
            alive &= ~rules.on_obstacle(s, cidx) & (s.pos[:, 1] > top_r - 1.0)
            now = standing_on(pool, s, cidx).gather(1, tgt[:, None]).squeeze(1) & alive & ~landed
            land_tick = torch.where(now, torch.full_like(land_tick, t + 1), land_tick)
            landed |= now
            closest = torch.where(alive & ~landed, torch.minimum(closest, torch.linalg.norm(s.pos - goal_r, dim=-1)),
                                  closest)
        L = landed.cpu().numpy()
        if L.any():
            lt = land_tick.cpu().numpy()
            stack = torch.stack([acts[k] for k in HEADS], -1).cpu().numpy()
            for j in np.flatnonzero(L):
                a = active[j // S]
                if len(found[a]) >= cfg.keep:
                    continue
                seq = stack[j, :lt[j]]
                b = seq.tobytes()
                if b not in seen[a]:
                    seen[a].add(b)
                    found[a].append(seq)
        for a in active:
            rounds[a] = it + 1
        score = (landed.double() * 100.0 - torch.where(landed, torch.zeros_like(closest), closest)).reshape(a_n, S)
        top = torch.topk(score, k_elite, dim=1).indices
        for key, v in p.items():
            sel = acts[key].reshape(a_n, S, T).gather(1, top[:, :, None].expand(-1, -1, T))
            freq = torch.nn.functional.one_hot(sel, v.shape[-1]).float().mean(1)
            upd = (0.7 * freq + 0.3 * v[act_idx]).clamp(min=0.01)
            v[act_idx] = upd / upd.sum(-1, keepdim=True)
        active = [a for a in active if len(found[a]) < cfg.keep]
        if not active:
            break
    return found, rounds


def _arrival(env: ParkourEnv, i: int) -> dict:
    s = env.state
    return {"course": int(env.course[i]), "cp": int(env.cp[i]), "pos": s.pos[i].tolist(), "vel": s.vel[i].tolist(),
            "yaw": float(s.yaw[i]), "on_ground": bool(s.on_ground[i]), "jump_ticks": int(s.jump_ticks[i]),
            "collided_h": bool(s.collided_h[i]), "collided_v": bool(s.collided_v[i]),
            "last_action": env.last_action[i].tolist()}


def evaluate_assisted(model, pool: CoursePool, device, cfg: AssistConfig, stall_ticks: int = 200,
                      max_ticks: int = 4000, settle: int = 0, log=None, fall_rule: str = "local") -> dict:
    """First episode on every course of ``pool`` with the neo assist (``cfg.enabled``) or without it.

    With the assist off this is ``train.evaluate(..., per_course=True)`` course by course.
    """
    n = pool.size
    if cfg.enabled and cfg.prior > 0:
        assert model.obs_dim != OBS_DIM_EXT, "the search prior uses the 255-number observation"
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=stall_ticks, max_ticks=max_ticks, fall_rule=fall_rule,
                                     extended_obs=model.obs_dim == OBS_DIM_EXT), sampler=None)
    obs = env.reset_all(torch.arange(n, device=device))
    if settle:
        idle = torch.zeros(n, 6, dtype=torch.bool, device=device)
        for _ in range(settle):
            env.state = physics.simulate(pool.world, env.course, env.state, idle)
        env.dist = env._distance(env.state.pos, env.course, env.cp)
        env.reset_memory(torch.ones(n, dtype=torch.bool, device=device))
        obs = env.observe()
    rules = Rules(pool)
    takeoffs = [neo_takeoffs(c) for c in pool.courses]
    planned = [planned_takeoffs(c, cfg.jumps) for c in pool.courses]
    neo = torch.zeros(n, C.MAX_PLATFORMS, dtype=torch.bool, device=device)  # neo takeoffs (the statistics)
    plan_at = torch.zeros_like(neo)  # takeoffs where the search runs
    before = torch.zeros_like(neo)  # the platform before a planned takeoff, itself not one
    for i, ks in enumerate(takeoffs):
        neo[i, ks] = True
        plan_at[i, planned[i]] = True
        for k in planned[i]:
            if k >= 1 and k - 1 not in planned[i]:
                before[i, k - 1] = True
    hz = horizons(cfg)
    H = max(list(hz.values()) + [BOUNCE_PLAN_HORIZON, LADDER_PLAN_HORIZON, cfg.brake])
    retry_h = torch.full((n, C.MAX_PLATFORMS), hz["retry"], dtype=torch.int64, device=device)  # stall check of a retry
    for i, ks in enumerate(planned):
        for k in ks:
            retry_h[i, k] = plan_horizon(pool.courses[i], k, "retry", cfg)
    tries = torch.zeros(n, C.MAX_PLATFORMS, dtype=torch.int64, device=device)
    tried_before = torch.zeros_like(neo)
    last_try = torch.full((n,), -10**9, dtype=torch.int64, device=device)
    plan = torch.zeros(n, H, 5, dtype=torch.int64, device=device)
    plan_len = torch.zeros(n, dtype=torch.int64, device=device)
    plan_pos = torch.zeros(n, dtype=torch.int64, device=device)
    plan_t = torch.full((n,), -1, dtype=torch.int64, device=device)  # the platform a plan ends on
    braking = torch.zeros(n, dtype=torch.bool, device=device)  # the queued actions are a brake_plan
    rows = torch.arange(n, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    success = torch.zeros(n, dtype=torch.bool, device=device)
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    length = torch.zeros(n, dtype=torch.int64, device=device)
    prog = torch.zeros(n, dtype=torch.float64, device=device)
    reached = [set() for _ in range(n)]
    events = []  # one per search: course, kind, takeoff, target, tick, after_plan (a plan or brake ended this tick), ...
    plans = {"played": 0, "landed_as_planned": 0, "brakes": 0}
    search_seconds = 0.0
    with torch.no_grad():
        for t in range(max_ticks):
            a, _, _ = model.act(obs, deterministic=True)
            active = plan_pos < plan_len
            if bool(active.any()):
                queued = plan[rows, plan_pos.clamp(max=H - 1)]
                a = torch.where(active[:, None], queued, a)
                plan_pos = plan_pos + active.to(torch.int64)
            obs, _, _, _, _, info = env.step(a)
            done = info["done"]
            first = done & ~finished
            success |= first & info["success"]
            fell |= first & info["fell"]
            length = torch.where(first, info["ep_len"], length)
            prog = torch.where(first, info["progress_frac"], prog)
            ended = active & (plan_pos >= plan_len)  # a plan's last action was just played
            retry_now = ended & braking  # a brake just ended: retry at once
            braking &= ~ended
            if bool(ended.any()):
                for i in torch.nonzero(ended & ~retry_now).flatten().tolist():
                    plans["played"] += 1
                    ok = bool(info["success"][i]) if bool(first[i]) else int(env.cp[i]) >= int(plan_t[i])
                    plans["landed_as_planned"] += int(ok)
            finished |= done
            plan_len = torch.where(done, torch.zeros_like(plan_len), plan_len)
            cp = env.cp
            on = ~finished & env.state.on_ground
            for i in torch.nonzero(on & neo[rows, cp]).flatten().tolist():
                reached[i].add(int(cp[i]))
            if not cfg.enabled:
                if t % 25 == 0 and bool(finished.all()):
                    break
                continue
            free = on & (plan_pos >= plan_len)
            n_try = tries[rows, cp]
            kinds = {"takeoff": free & plan_at[rows, cp] & (n_try == 0),
                     "retry": (free & plan_at[rows, cp] & (n_try > 0) & (n_try <= cfg.retries)
                               & ((t - last_try >= cfg.retry_every) | retry_now) & (env.stall + retry_h[rows, cp] < stall_ticks)),
                     "approach": free & before[rows, cp] & ~tried_before[rows, cp] if cfg.approach else None}
            for kind, mask in kinds.items():
                want = [] if mask is None else torch.nonzero(mask).flatten().tolist()
                if not want:
                    continue
                t0 = time.time()
                arrivals = [_arrival(env, i) for i in want]
                horizon = []
                for i, arr in zip(want, arrivals):
                    k = arr["cp"] + 1 if kind == "approach" else arr["cp"]
                    arr["target"] = plan_target(pool.courses[i], k)
                    horizon.append(plan_horizon(pool.courses[i], k, kind, cfg))
                found, rounds = [], []
                if cfg.batch <= 1:
                    for i, arr, h in zip(want, arrivals, horizon):
                        attempt = 0 if kind == "approach" else int(tries[i, int(cp[i])])
                        seq, r = plan_once(model, pool, rules, arr, kind, attempt, cfg, h, device)
                        found.append([] if seq is None else [seq])
                        rounds.append(r)
                else:  # searches batched per tick (faster on a GPU); the bots cannot reproduce these seeds
                    found, rounds = [None] * len(arrivals), [None] * len(arrivals)
                    for h in sorted(set(horizon)):
                        idx = [j for j, hj in enumerate(horizon) if hj == h]
                        for b in range(0, len(idx), cfg.batch):
                            part = idx[b:b + cfg.batch]
                            arr_b = [arrivals[j] for j in part]
                            pr = policy_prior(model, pool, arr_b, device, h) if cfg.prior > 0 else None
                            seed = cfg.seed + 1009 * t + b if h == hz[kind] else cfg.seed + 1009 * t + b + 7919 * h
                            f, r = search(pool, rules, arr_b, device, cfg, seed, pr, h)
                            for j, fj, rj in zip(part, f, r):
                                found[j], rounds[j] = fj, rj
                dt = time.time() - t0
                search_seconds += dt
                for j, i in enumerate(want):
                    c = int(cp[i])
                    k = c + 1 if kind == "approach" else c
                    events.append({"course": i, "kind": kind, "takeoff": k, "target": arrivals[j]["target"], "tick": t + 1,
                                   "after_plan": bool(ended[i]), "found": bool(found[j]),
                                   "plan_ticks": len(found[j][0]) if found[j] else 0, "rounds": rounds[j], "braked": False})
                    if kind == "approach":
                        tried_before[i, c] = True
                    else:
                        tries[i, c] += 1
                        last_try[i] = t
                    if found[j]:
                        seq = torch.as_tensor(found[j][0], dtype=torch.int64, device=device)
                        plan[i, :len(seq)] = seq
                        plan_len[i] = len(seq)
                        plan_pos[i] = 0
                        plan_t[i] = arrivals[j]["target"]
                    elif cfg.brake and kind != "approach" and int(tries[i, c]) <= cfg.retries:
                        seq = brake_plan(pool, rules, arrivals[j], cfg.brake, device)
                        if seq is not None:
                            plan[i, :len(seq)] = torch.as_tensor(seq, device=device)
                            plan_len[i] = len(seq)
                            plan_pos[i] = 0
                            braking[i] = True
                            plans["brakes"] += 1
                            events[-1]["braked"] = True
                if log:
                    log({"tick": t + 1, "kind": kind, "searched": len(want), "found": sum(bool(f) for f in found),
                         "seconds": round(dt, 1), "unfinished": int((~finished).sum())})
            if t % 25 == 0 and bool(finished.all()):
                break
    lv = pool.level.cpu().numpy()
    s, f, ln, pr = (x.cpu().numpy() for x in (success, fell, length, prog))
    last = (pool.n - 1).cpu().numpy()
    cp_final = np.rint(pr * last).astype(int)
    per_course = []
    for i in range(n):
        ks = takeoffs[i]
        per_course.append({"success": bool(s[i]), "fell": bool(f[i]), "ticks": int(ln[i]), "progress": float(pr[i]),
                           "level": int(lv[i]), "neos": len(ks), "neos_reached": len(reached[i]),
                           "neos_passed": sum(1 for k in ks if k in reached[i] and (s[i] or cp_final[i] > k)),
                           "stopped_at": int(cp_final[i]) if not s[i] else None,
                           "stopped_at_neo": bool(not s[i] and int(cp_final[i]) in ks),
                           "stopped_at_kind": None if s[i] else jump_kind(pool.courses[i], int(cp_final[i]), ks)})
    return {"per_course": per_course, "events": events, "plans": plans, "search_seconds": round(search_seconds, 1)}


def summarize(res: dict) -> dict:
    """Counts for one run: courses completed, neo jumps reached and passed, by neos per course, and the searches."""
    pc = res["per_course"]
    out = {"courses": len(pc), "completed": sum(c["success"] for c in pc), "fell": sum(c["fell"] for c in pc),
           "neo_jumps": sum(c["neos"] for c in pc), "neo_jumps_reached": sum(c["neos_reached"] for c in pc),
           "neo_jumps_passed": sum(c["neos_passed"] for c in pc),
           "failed_at_a_neo": sum(c["stopped_at_neo"] for c in pc)}
    kinds = [c.get("stopped_at_kind") for c in pc if not c["success"] and c.get("stopped_at_kind")]
    if kinds:  # the jump each failed course stopped before
        out["failed_at"] = {k: kinds.count(k) for k in sorted(set(kinds))}
    by = {}
    for c in pc:
        key = str(c["neos"]) if c["neos"] < 5 else "5+"
        b = by.setdefault(key, {"courses": 0, "completed": 0})
        b["courses"] += 1
        b["completed"] += int(c["success"])
    out["by_neo_count"] = dict(sorted(by.items()))
    ev = res["events"]
    if ev:
        out["searches"] = {}
        for kind in ("takeoff", "retry", "approach"):
            e = [x for x in ev if x["kind"] == kind]
            if e:
                out["searches"][kind] = {"n": len(e), "found": sum(x["found"] for x in e)}
        out["plans_played"] = res["plans"]["played"]
        out["plans_landed_as_planned"] = res["plans"]["landed_as_planned"]
        if res["plans"].get("brakes"):
            out["brakes"] = res["plans"]["brakes"]
        out["search_seconds"] = res["search_seconds"]
        out["seconds_per_search"] = round(res["search_seconds"] / len(ev), 2)
    return out


def serve(model, courses: list[C.Course], cfg: AssistConfig, device, inp=None, out=None) -> None:
    """The bots' planner: one JSON request per line in, one plan per line out.

    Request: {"index": course index in ``courses``, "cp", "target", "kind",
    "attempt", "state": {"pos", "vel", "yaw", "onGround", "jumpTicks",
    "collidedH"}, "lastAction"}. Answer: {"plan": [[move, strafe, jump,
    sprint, yaw], ...] or null, "rounds", "brake": the brake_plan to play
    before the retry when there is no plan and ``brake`` is set (else null),
    "seconds"}. The search is the one ``evaluate_assisted`` runs with
    ``batch`` 1 (same seed for the same request).
    """
    inp = inp or sys.stdin
    out = out or sys.stdout
    pool = CoursePool(courses, device, C.GRID)
    rules = Rules(pool)
    with torch.no_grad():
        for line in inp:
            if not line.strip():
                continue
            req = json.loads(line)
            st = req["state"]
            arr = {"course": int(req["index"]), "cp": int(req["cp"]), "target": int(req["target"]),
                   "pos": [float(v) for v in st["pos"]], "vel": [float(v) for v in st["vel"]], "yaw": float(st["yaw"]),
                   "on_ground": bool(st["onGround"]), "jump_ticks": int(st["jumpTicks"]),
                   "collided_h": bool(st.get("collidedH", False)), "collided_v": False,
                   "last_action": [int(v) for v in req["lastAction"]]}
            t0 = time.time()
            k = arr["cp"] + 1 if req["kind"] == "approach" else arr["cp"]
            attempt = int(req.get("attempt", 0))
            seq, rounds = plan_once(model, pool, rules, arr, req["kind"], attempt, cfg,
                                    plan_horizon(courses[arr["course"]], k, req["kind"], cfg), device)
            brake = None
            if seq is None and cfg.brake and req["kind"] != "approach" and attempt + 1 <= cfg.retries:
                brake = brake_plan(pool, rules, arr, cfg.brake, device)
            out.write(json.dumps({"plan": None if seq is None else seq.tolist(), "rounds": rounds,
                                  "brake": None if brake is None else brake.tolist(),
                                  "seconds": round(time.time() - t0, 3)}) + "\n")
            out.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "results" / "final_policy.pt"))
    ap.add_argument("--set", choices=("official", "random", "mixed"), default="official")
    ap.add_argument("--family", type=int, default=VALIDATION_FAMILY)
    ap.add_argument("--off", action="store_true", help="the policy alone, for comparison")
    ap.add_argument("--courses", type=int, default=0, help="only the first N courses (quick checks)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--details", default="", help="also write every course's outcome and every search to this JSON file")
    ap.add_argument("--serve", action="store_true", help="plan for the bots: JSON lines on stdin/stdout (see serve)")
    ap.add_argument("--courses-file", default="", help="with --serve: the courses the bot plays")
    ap.add_argument("--device", default="", help="cpu or cuda (default: cuda if available); the bots' planner service "
                                                 "must use the device of the simulator run it is compared with")
    for k, v in asdict(AssistConfig()).items():
        if k != "enabled":
            ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    valid = {"official": (TEST_SEED_FAMILY, VALIDATION_FAMILY), "random": (RANDOM_TEST_FAMILY, RANDOM_VALIDATION_FAMILY),
             "mixed": (MIXED_TEST_FAMILY, MIXED_VALIDATION_FAMILY)}
    if a.family not in valid[a.set]:
        raise SystemExit(f"--set {a.set} has families {valid[a.set]} (test, validation)")
    cfg = AssistConfig(enabled=not a.off, **{k: getattr(a, k) for k in asdict(AssistConfig()) if k != "enabled"})
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.serve:
        courses = courses_from_file(json.loads(Path(a.courses_file).read_text())["courses"])
        serve(load_model(Path(a.checkpoint), dev), courses, cfg, dev)
        return
    courses = course_set(a.set, a.family)
    if a.courses:
        courses = courses[:a.courses]
    model = load_model(Path(a.checkpoint), dev)
    pool = CoursePool(courses, dev, C.GRID)
    t0 = time.time()
    res = evaluate_assisted(model, pool, dev, cfg, log=lambda d: print(json.dumps(d), flush=True))
    test = a.family in TEST_FAMILIES
    ck = Path(a.checkpoint).resolve()
    ck_name = ck.relative_to(ROOT).as_posix() if ck.is_relative_to(ROOT) else ck.name
    row = {"set": a.set, "family": a.family, "split": "test" if test else "validation",
           "checkpoint": ck_name, "checkpoint_sha1": hashlib.sha1(ck.read_bytes()).hexdigest()[:12], "assist": not a.off,
           "config": asdict(cfg), "tag": a.tag,
           "measured": f"simulator (float64 port of prismarine-physics), {dev.upper() if dev == 'cpu' else 'CUDA'}",
           **summarize(res), "seconds": round(time.time() - t0, 1)}
    if a.courses:
        row["first_courses_only"] = a.courses
    out = Path(a.out) if a.out else ROOT / "results" / "neo" / ("assist_test.jsonl" if test else "assist_validation.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    if a.details:
        Path(a.details).parent.mkdir(parents=True, exist_ok=True)
        Path(a.details).write_text(json.dumps({"row": row, "per_course": res["per_course"], "events": res["events"]}))
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()

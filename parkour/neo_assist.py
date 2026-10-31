"""Neo assist: an exact-model search at every neo takeoff (simulator only).

    python -m parkour.neo_assist --checkpoint results/final_policy.pt --set official --family 2
    python -m parkour.neo_assist --checkpoint results/final_policy.pt --set random --family 13 --prior 0.5
    python -m parkour.neo_assist --checkpoint results/final_policy.pt --set official --family 2 --off
    python -m parkour.neo_assist --checkpoint results/final_policy.pt --set single --family 40 --off

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
platform before a planned takeoff (not planned itself) or before a neo's
takeoff, for a plan of up to H ticks over both jumps, so the bot reaches
the takeoff in a state the neo can be made from; without a landing, the
platform's own takeoff search (``--jumps all``) runs in the same tick. With
``--jumps advanced`` the search also runs at wide (3 sideways) and run-up
jumps (momentum, head hitters, ice), with ``--jumps all`` at every jump.

Opt-in levers for the searches of a neo (at its takeoff, before it with
``--approach``, and its retries; all off at 0): ``--restarts N`` searches
again from the same state with a new seed when a search lands nothing, up to
N times; ``--neo-samples`` and ``--neo-iterations`` replace ``--samples`` and
``--iterations`` there; ``--approach-touch 1`` counts an approach plan only
if it stands on the neo's takeoff before it lands on the neo's target. They
act in ``plan_once``, so the bots' planner service uses them too (they need
``--batch 1``). Rows list them in their config only when they are set.

The searches run after each tick's step, so the first action comes from the
start state unsearched. The simulator's start is in the air, and the bot
lands before it can jump; the bots first stand 10 ticks on the start block
(``parkour.live.SETTLE_TICKS``; ``--settle 10`` starts the same way here),
and a first action that jumps is then the policy's own. With
``--plan-start 1`` the searches also run on the start state, before the
first tick's action (opt-in; from the simulator's own start it changes
nothing, since the bot is not yet on the ground).

The simulator and the bots resolve horizontal collisions x then z
(prismarine-physics); the game resolves the larger component first. Past a
wall's corner the two can end more than 0.25 blocks apart: the server then
logs "moved wrongly", and sends the bot back if its box started the tick
clear of every block. With ``--corner-limit L`` (opt-in; 0.25 is the
server's limit) a landing the search found is played again tick by tick in
both orders (``corner_gap``), and one with a tick more than L apart counts
as no landing: a neo's search restarts, an ordinary jump brakes and
retries. It is stricter than the server: it also drops ticks the server only
logs, and ticks it would not flag at all (the server orders the bot's
displacement, ``corner_gap`` the tick's velocity); HANDOFF.md section 3.

Every course's record says whether the bot did its neos as neos
(``neo_done``): it stood on each neo's takeoff before on any later platform,
then on the neo's target before on any later one, so a jump past the takeoff,
or a target reached from further on, does not count. Rows count the courses
completed that way (``completed_neo_done``) next to ``completed``.

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
  single    courses with exactly one neo (``single_neo_courses``): 320 per
            family, 10 jumps, nine from levels 0-5 and one of the 32
            single-neo types of levels 6 and 7, every type and every
            position equally often; families 41 (test) and 40 (validation),
            which nothing else uses and no training stream may contain.

Settings are chosen on the validation families; the test families are run
once with the chosen setting. Each run appends one JSON line to ``--out``
(default ``results/neo/assist_validation.jsonl`` or ``assist_test.jsonl``).
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import re
import sys
import time
from parkour.run_cache import append_jsonl_once, cache_lock, write_json_atomic
from dataclasses import asdict, dataclass, replace
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
from .train import TEST_SEED_FAMILY, VALIDATION_FAMILY, course_hash, held_out_courses

RANDOM_TEST_FAMILY = 12
RANDOM_VALIDATION_FAMILY = 13
MIXED_TEST_FAMILY = 14
MIXED_VALIDATION_FAMILY = 15
SINGLE_VALIDATION_FAMILY = 40  # the single-neo sets (families 18-39 are the routing work's)
SINGLE_TEST_FAMILY = 41
TEST_FAMILIES = (TEST_SEED_FAMILY, RANDOM_TEST_FAMILY, MIXED_TEST_FAMILY, SINGLE_TEST_FAMILY)
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
    # Opt-in levers for the searches of a neo (neo_request: its takeoff, approach and retries); 0 is off. Integers,
    # not bools: the command line and the bots' planner service pass every field as str(value).
    restarts: int = 0  # a search that lands nothing runs again from the same state with a new seed, this many times
    neo_samples: int = 0  # >0: replaces samples there
    neo_iterations: int = 0  # >0: replaces iterations there
    approach_touch: int = 0  # 1: an approach plan counts only if it stands on the neo's takeoff before it lands
    plan_start: int = 0  # 1: the searches also run on the start state, before the first tick's action
    corner_limit: float = 0.0  # >0: a landing whose plan has a tick this far apart in the game's collision order is dropped
    # The learned router's mode only (parkour.route_neo): 1: after a planned landing the router's next choice is planned
    # too (and so on while plans land), so the jumper never starts from a plan's landing state.
    plan_on: int = 0


NEO_LEVERS = ("restarts", "neo_samples", "neo_iterations", "approach_touch")
OPTIONAL = NEO_LEVERS + ("plan_start", "corner_limit", "plan_on")  # the fields a row's config lists only when they are set
KIND_CODE = {"takeoff": 0, "retry": 1, "approach": 2}


def request_seed(seed: int, course: int, cp: int, kind: str, attempt: int, restart: int = 0) -> int:
    """The search seed of one request (``restart`` r > 0: its r-th search again after none landed, ``restarts``): the
    same in the simulator and in the bots' planner service."""
    return seed * 1_000_003 + course * 4099 + cp * 67 + KIND_CODE[kind] * 7 + attempt + restart * 1_000_000_007


def config_meta(cfg: "AssistConfig") -> dict:
    """The config a run's row records: every field, the neo levers, ``plan_start`` and ``corner_limit`` only when set
    (rows from before they existed, and the cached parts of split runs, stay valid)."""
    return {k: v for k, v in asdict(cfg).items() if k not in OPTIONAL or v}


def neo_request(course: C.Course, cp: int, kind: str) -> bool:
    """Whether a request is a neo's: the takeoff it plans (``cp``, for an approach the next one) is a neo takeoff."""
    return (cp + 1 if kind == "approach" else cp) in neo_takeoffs(course)


def plan_once(model, pool: CoursePool, rules: "Rules", arrival: dict, kind: str, attempt: int, cfg: "AssistConfig",
              horizon: int, device, stats: dict | None = None):
    """One search from one state: (the first landing's actions or None, rounds used, restarts used).

    For a neo's request (``neo_request``) the neo levers apply: ``neo_samples`` and ``neo_iterations`` replace
    ``samples`` and ``iterations``, an approach plan must stand on the neo's takeoff before it lands
    (``approach_touch``), and while no search has landed, the search runs again from the same state with the next
    seed, up to ``restarts`` times. With ``corner_limit`` a landing whose ``corner_gap`` exceeds it counts as none
    (``stats["corner_dropped"]`` counts them). An arrival with a ``neo`` key says itself whether the request is a neo's
    (the router's blocked-path choices, ``parkour.route_neo``: always)."""
    neo = arrival["neo"] if "neo" in arrival else neo_request(pool.courses[arrival["course"]], arrival["cp"], kind)
    if neo:
        cfg = replace(cfg, samples=cfg.neo_samples or cfg.samples, iterations=cfg.neo_iterations or cfg.iterations)
        if kind == "approach" and cfg.approach_touch:
            arrival = dict(arrival, touch=arrival["cp"] + 1)
    pr = policy_prior(model, pool, [arrival], device, horizon) if cfg.prior > 0 else None
    rounds = 0
    for restart in range(1 + (cfg.restarts if neo else 0)):
        seed = request_seed(cfg.seed, arrival.get("seed_course", arrival["course"]), arrival["cp"], kind, attempt,
                            restart)
        f, r = search(pool, rules, [arrival], device, cfg, seed, pr, horizon)
        rounds += r[0]
        if f[0]:
            if cfg.corner_limit and corner_gap(pool, arrival, f[0][0], device) > cfg.corner_limit:
                if stats is not None:
                    stats["corner_dropped"] = stats.get("corner_dropped", 0) + 1
                continue
            return f[0][0], rounds, restart
    return None, rounds, restart


def corner_gap(pool: CoursePool, arrival: dict, seq, device) -> float:
    """The largest horizontal distance, over the ticks of a plan played from ``arrival``, between where a tick ends
    with prismarine-physics' collisions (x then z: the simulator and the bots) and where it ends with the game's
    order (the larger component first, ``physics.simulate(..., larger_first=True)``), both from the tick's start as
    the bots play it. A server logs "moved wrongly" when its own result is more than 0.25 blocks away, and sends
    the bot back if the bot's box started the tick clear of every block: past a wall's corner the two orders can end
    that far apart. The server orders the bot's displacement, this the tick's velocity, so a tick whose move already
    collided can be counted though the server sees nothing: a stricter check, never a looser one in the recorded
    runs."""
    s = _state_rows([arrival], device)
    course = torch.tensor([arrival["course"]], device=device)
    acts = torch.as_tensor(np.asarray(seq), dtype=torch.int64, device=device).reshape(-1, 5)
    steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64, device=device) * (math.pi / 180)
    gap = 0.0
    for t in range(len(acts)):
        y = s.yaw + steps[acts[t:t + 1, 4]]
        s.yaw = torch.remainder(y + math.pi, 2 * math.pi) - math.pi
        controls = action_to_controls(acts[t:t + 1])
        game = physics.simulate(pool.world, course, s, controls, larger_first=True)
        s = physics.simulate(pool.world, course, s, controls)
        gap = max(gap, float(torch.linalg.norm(s.pos[0, [0, 2]] - game.pos[0, [0, 2]])))
    return gap


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


def approach_platforms(course: C.Course, planned: list[int]) -> list[int]:
    """Platforms from which a plan over the next two jumps is tried first (``--approach``): the one before
    every planned takeoff that is not planned itself, and the one before every neo takeoff (with ``--jumps
    all`` every platform is planned, and the jump onto a neo's takeoff decides where the neo starts from)."""
    ps = set(planned)
    neos = set(neo_takeoffs(course))
    return sorted({k - 1 for k in ps if k >= 1 and (k - 1 not in ps or k in neos)})


def assist_fields(course: C.Course, cfg: "AssistConfig") -> dict:
    """What the bots' trigger rules need of a course (bridge/assist.js): the planned takeoffs, the platform
    each plan ends on, the retry horizon of each takeoff (for the stall check before a retry) and the
    platforms where an approach plan is tried."""
    ks = planned_takeoffs(course, cfg.jumps)
    return {"assist_takeoffs": ks, "assist_targets": [plan_target(course, k) for k in ks],
            "assist_retry_horizons": [plan_horizon(course, k, "retry", cfg) for k in ks],
            "assist_approach": approach_platforms(course, ks)}


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


def single_neo_types() -> list[tuple[int, C.JumpSpec]]:
    """(level, spec) of the 32 single-neo jump types in ``neo_level``'s order: level 6's as listed in
    configs/jump_levels_advanced.json, then level 7's."""
    adv = C.load_levels(advanced=True)
    return [(L, s) for L in (6, 7) for s in adv[L] if s.neo == 1]


def single_neo_ok(course: C.Course, k: int) -> bool:
    """Exactly one neo, at takeoff ``k``, by the wall rule (``neo_takeoffs``) and by the specs, and no wall cell
    inside the flight corridor of another jump (``courses._corridor``)."""
    if neo_takeoffs(course) != [k] or [j for j, sp in enumerate(course.specs) if sp.neo] != [k]:
        return False
    p = course.platforms
    cells = [tuple(int(v) for v in e) for e in course.extra]
    return not any(C._boxes_overlap(C._corridor(p[j], p[j + 1]), (x, x, y, y, z, z))
                   for j in range(course.n - 1) if j != k for x, y, z in cells)


SINGLE_MAX_DRAWS = 10_000  # draws of one single-neo course before giving up (the sets need at most a few dozen)


def single_neo_courses(family: int, n: int = 320, n_jumps: int = 10) -> list[C.Course]:
    """Courses with exactly one neo (``--set single``: families 40 validation, 41 test), by ``courses.generate_single_neo``.

    Course i has the single-neo type i mod 32 (``single_neo_types``) as jump (i mod 32 + i div 32) mod ``n_jumps``,
    so in 320 courses every type and every position comes equally often (each type once at every position). One
    random stream, ``[2026, family, 6]``, used in course order (the first courses of a longer set are the same
    courses), draws course i again until ``single_neo_ok``. Its level is its neo's own (6 or 7), its seed i.
    """
    types = single_neo_types()
    basic = C.load_levels()
    rng = np.random.default_rng([2026, family, 6])
    out = []
    for i in range(n):
        level, spec = types[i % len(types)]
        k = (i % len(types) + i // len(types)) % n_jumps
        for _ in range(SINGLE_MAX_DRAWS):
            c = C.generate_single_neo(rng, basic, spec, k, n_jumps, level=level, seed=i)
            if c is not None and single_neo_ok(c, k):
                break
        else:
            raise RuntimeError(f"no single-neo course {i} ({spec.key()} as jump {k}) in {SINGLE_MAX_DRAWS} draws")
        out.append(c)
    return out


def single_neo_hashes() -> set:
    """``train.course_hash`` of every course of both single-neo sets (320 each): no training course may equal one."""
    return {course_hash(c) for f in (SINGLE_VALIDATION_FAMILY, SINGLE_TEST_FAMILY) for c in single_neo_courses(f)}


def course_set(name: str, family: int) -> list[C.Course]:
    if name == "official":
        return official_neo_courses(family)
    if name == "random":
        return random_neo_courses(family)
    if name == "mixed":
        return mixed_courses(family)
    if name == "single":
        return single_neo_courses(family)
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
    the target, or the arrival's own ``floor`` (the router's jumps,
    ``parkour.route_neo``: the lower top of the two platforms). With a
    ``touch`` platform (``approach_touch``) a sequence
    lands only on the target after standing on that platform; until it has,
    its distance to that platform plus 1000 is its score's distance, so the
    search looks for sequences over it. Returns (found, rounds): per arrival
    a list of action sequences (each cut at its landing tick) and the number
    of search rounds it used.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    A, S, T = len(arrivals), cfg.samples, horizon
    steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64, device=device) * (math.pi / 180)
    init = _state_rows(arrivals, device)
    course = torch.tensor([a["course"] for a in arrivals], device=device)
    target = torch.tensor([a.get("target", a["cp"] + 1) for a in arrivals], device=device)
    tops = torch.stack([torch.tensor(float(a["floor"]), dtype=torch.float64, device=device) if "floor" in a else
                        pool.top[a["course"], a["cp"]:a.get("target", a["cp"] + 1) + 1].min() for a in arrivals])
    goal = pool.centre[course, target]
    touch = None  # the platform to stand on first (-1: none), and its centre
    if any(a.get("touch", -1) >= 0 for a in arrivals):
        touch = torch.tensor([a.get("touch", -1) for a in arrivals], device=device)
        via = pool.centre[course, touch.clamp(min=0)]
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
        if touch is not None:
            tch, via_r = rep(touch), rep(via)
            touched = tch < 0
        for t in range(T):
            y = s.yaw + steps[acts["yaw"][:, t]]
            s.yaw = torch.remainder(y + math.pi, 2 * math.pi) - math.pi
            a_t = torch.stack([acts[k][:, t] for k in HEADS], -1)
            s = physics.simulate(pool.world, cidx, s, action_to_controls(a_t))
            alive &= ~rules.on_obstacle(s, cidx) & (s.pos[:, 1] > top_r - 1.0)
            on = standing_on(pool, s, cidx)
            now = on.gather(1, tgt[:, None]).squeeze(1) & alive & ~landed
            dist = torch.linalg.norm(s.pos - goal_r, dim=-1)
            if touch is not None:
                touched |= on.gather(1, tch.clamp(min=0)[:, None]).squeeze(1) & alive
                now &= touched
                dist = torch.where(touched, dist, 1000.0 + torch.linalg.norm(s.pos - via_r, dim=-1))
            land_tick = torch.where(now, torch.full_like(land_tick, t + 1), land_tick)
            landed |= now
            closest = torch.where(alive & ~landed, torch.minimum(closest, dist), closest)
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
                      max_ticks: int = 4000, settle: int = 0, log=None, fall_rule: str = "local",
                      course_offset: int = 0) -> dict:
    """First episode on every course of ``pool`` with the neo assist (``cfg.enabled``) or without it.

    With the assist off this is ``train.evaluate(..., per_course=True)`` course by course. ``course_offset``:
    the index of the pool's first course in the whole course set, for a run in parts (``--part``): searches
    are seeded and events numbered by that index, as in one run over the whole set. ``neo_done`` per course:
    every neo was done as a neo, the checkpoint (the furthest platform stood on) took the value of its takeoff
    and then of its target, so the bot stood on the takeoff before on any later platform and then on the target
    before on any later one. ``settle``: ticks without keys on the start block first (the bots' start,
    ``parkour.live.SETTLE_TICKS``); a bot on the ground there is searched from before its first action only with
    ``cfg.plan_start``.
    """
    n = pool.size
    if cfg.enabled and cfg.plan_on:
        raise ValueError("plan_on is the learned router's (parkour.route_neo); with the course list use jumps 'all'")
    if cfg.enabled and cfg.prior > 0:
        assert model.obs_dim != OBS_DIM_EXT, "the search prior uses the 255-number observation"
    per_request = NEO_LEVERS + ("corner_limit",)
    if cfg.enabled and cfg.batch > 1 and any(getattr(cfg, k) for k in per_request):
        raise ValueError(f"{', '.join(per_request)} act per request (plan_once): they need batch 1")
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
    before = torch.zeros_like(neo)  # where an approach plan is tried (approach_platforms)
    for i, ks in enumerate(takeoffs):
        neo[i, ks] = True
        plan_at[i, planned[i]] = True
        before[i, approach_platforms(pool.courses[i], planned[i])] = True
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
    took = torch.zeros_like(neo)  # the values the first episode's checkpoint took (neo_done); it starts at 0
    took[:, 0] = True
    reached = [set() for _ in range(n)]
    events = []  # one per search: course, kind, takeoff, target, tick, after_plan (a plan or brake ended this tick), ...
    plans = {"played": 0, "landed_as_planned": 0, "brakes": 0}
    search_seconds = 0.0

    def searches(t: int, cp: torch.Tensor, on: torch.Tensor, ended: torch.Tensor, retry_now: torch.Tensor) -> None:
        """The searches the rules call for after tick t's step (t = -1: the start state, ``plan_start``); a plan
        found is played from the next tick on."""
        nonlocal search_seconds
        n_try = tries[rows, cp]
        for kind in ("approach", "takeoff", "retry"):  # an approach first; without its plan, the takeoff's own
            free = on & (plan_pos >= plan_len)  # search in the same tick (the bots do the same)
            if kind == "approach":
                mask = free & before[rows, cp] & ~tried_before[rows, cp] if cfg.approach else None
            elif kind == "takeoff":
                mask = free & plan_at[rows, cp] & (n_try == 0)
            else:
                mask = (free & plan_at[rows, cp] & (n_try > 0) & (n_try <= cfg.retries)
                        & ((t - last_try >= cfg.retry_every) | retry_now) & (env.stall + retry_h[rows, cp] < stall_ticks))
            want = [] if mask is None else torch.nonzero(mask).flatten().tolist()
            if not want:
                continue
            t0 = time.time()
            arrivals = [dict(_arrival(env, i), seed_course=course_offset + i) for i in want]
            horizon = []
            for i, arr in zip(want, arrivals):
                k = arr["cp"] + 1 if kind == "approach" else arr["cp"]
                arr["target"] = plan_target(pool.courses[i], k)
                horizon.append(plan_horizon(pool.courses[i], k, kind, cfg))
            found, rounds, restarts, dropped = [], [], [], []
            if cfg.batch <= 1:
                for i, arr, h in zip(want, arrivals, horizon):
                    attempt = 0 if kind == "approach" else int(tries[i, int(cp[i])])
                    st = {}
                    seq, r, rs = plan_once(model, pool, rules, arr, kind, attempt, cfg, h, device,
                                           **({"stats": st} if cfg.corner_limit else {}))
                    found.append([] if seq is None else [seq])
                    rounds.append(r)
                    restarts.append(rs)
                    dropped.append(st.get("corner_dropped", 0))
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
                events.append({"course": course_offset + i, "kind": kind, "takeoff": k, "target": arrivals[j]["target"],
                               "tick": t + 1,
                               "after_plan": bool(ended[i]), "found": bool(found[j]),
                               "plan_ticks": len(found[j][0]) if found[j] else 0, "rounds": rounds[j], "braked": False})
                if cfg.restarts:  # searches again after none landed (all rounds above count)
                    events[-1]["restarts"] = restarts[j]
                if cfg.corner_limit:  # landings dropped by the corner check
                    events[-1]["corner_dropped"] = dropped[j]
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

    with torch.no_grad():
        if cfg.enabled and cfg.plan_start:  # a start on the ground (settled): its searches before the first action
            on = env.state.on_ground.clone()
            for i in torch.nonzero(on & neo[rows, env.cp]).flatten().tolist():
                reached[i].add(int(env.cp[i]))
            none = torch.zeros(n, dtype=torch.bool, device=device)
            searches(-1, env.cp, on, none, none)
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
            took[rows, cp] = took[rows, cp] | ~finished  # (a finished episode's last value: from its progress)
            on = ~finished & env.state.on_ground
            for i in torch.nonzero(on & neo[rows, cp]).flatten().tolist():
                reached[i].add(int(cp[i]))
            if not cfg.enabled:
                if t % 25 == 0 and bool(finished.all()):
                    break
                continue
            searches(t, cp, on, ended, retry_now)
            if t % 25 == 0 and bool(finished.all()):
                break
    lv = pool.level.cpu().numpy()
    s, f, ln, pr = (x.cpu().numpy() for x in (success, fell, length, prog))
    last = (pool.n - 1).cpu().numpy()
    cp_final = np.rint(pr * last).astype(int)
    took = took.cpu().numpy()
    per_course = []
    for i in range(n):
        ks = takeoffs[i]
        cps = set(np.flatnonzero(took[i]).tolist()) | {int(cp_final[i])}
        per_course.append({"success": bool(s[i]), "fell": bool(f[i]), "ticks": int(ln[i]), "progress": float(pr[i]),
                           "level": int(lv[i]), "neos": len(ks), "neos_reached": len(reached[i]),
                           "neos_passed": sum(1 for k in ks if k in reached[i] and (s[i] or cp_final[i] > k)),
                           "neo_done": neo_done(ks, cps),
                           "stopped_at": int(cp_final[i]) if not s[i] else None,
                           "stopped_at_neo": bool(not s[i] and int(cp_final[i]) in ks),
                           "stopped_at_kind": None if s[i] else jump_kind(pool.courses[i], int(cp_final[i]), ks)})
    return {"per_course": per_course, "events": events, "plans": plans, "search_seconds": round(search_seconds, 1)}


def neo_done(takeoffs: list[int], values: set) -> bool:
    """Every neo done as a neo: the checkpoint took the value of each neo's takeoff and of its target (``values``:
    every value it took, the episode's last one included). It only grows, so the bot stood on the takeoff before on
    any later platform, then on the target before on any later one."""
    return all(k in values and k + 1 in values for k in takeoffs)


def summarize(res: dict) -> dict:
    """Counts for one run: courses completed (also with every neo done as a neo, ``neo_done``), neo jumps reached
    and passed, by neos per course and (records with ``neo_done``) by level, and the searches. Records from before
    ``neo_done`` existed give the summary they gave then, so a split run's cached parts republish the same row."""
    pc = res["per_course"]
    strict = all("neo_done" in c for c in pc)  # (not in the cached parts of runs from before it existed)
    done = lambda c: int(c["success"] and c["neo_done"])  # noqa: E731
    out = {"courses": len(pc), "completed": sum(c["success"] for c in pc)}
    if strict:
        out["completed_neo_done"] = sum(done(c) for c in pc)
    out.update({"fell": sum(c["fell"] for c in pc),
                "neo_jumps": sum(c["neos"] for c in pc), "neo_jumps_reached": sum(c["neos_reached"] for c in pc),
                "neo_jumps_passed": sum(c["neos_passed"] for c in pc),
                "failed_at_a_neo": sum(c["stopped_at_neo"] for c in pc)})
    kinds = [c.get("stopped_at_kind") for c in pc if not c["success"] and c.get("stopped_at_kind")]
    if kinds:  # the jump each failed course stopped before
        out["failed_at"] = {k: kinds.count(k) for k in sorted(set(kinds))}
    by, by_level = {}, {}
    for c in pc:
        key = str(c["neos"]) if c["neos"] < 5 else "5+"
        groups = [by.setdefault(key, {"courses": 0, "completed": 0})]
        if strict and "level" in c:  # (records with neo_done: summaries of older records stay as they were)
            groups.append(by_level.setdefault(c["level"], {"courses": 0, "completed": 0}))
        for b in groups:
            b["courses"] += 1
            b["completed"] += int(c["success"])
            if strict:
                b["completed_neo_done"] = b.get("completed_neo_done", 0) + done(c)
    out["by_neo_count"] = dict(sorted(by.items()))
    if by_level:
        out["by_level"] = {str(L): b for L, b in sorted(by_level.items())}
    ev = res["events"]
    if ev:
        out["searches"] = {}
        for kind in ("takeoff", "retry", "approach"):
            e = [x for x in ev if x["kind"] == kind]
            if e:
                out["searches"][kind] = {"n": len(e), "found": sum(x["found"] for x in e)}
        out["plans_played"] = res["plans"]["played"]
        out["plans_landed_as_planned"] = res["plans"]["landed_as_planned"]
        if any("corner_dropped" in x for x in ev):  # (runs with corner_limit only)
            out["corner_dropped"] = sum(x["corner_dropped"] for x in ev)
        if res["plans"].get("brakes"):
            out["brakes"] = res["plans"]["brakes"]
        out["search_seconds"] = res["search_seconds"]
        out["seconds_per_search"] = round(res["search_seconds"] / len(ev), 2)
    return out


def merge_results(parts: list[dict]) -> dict:
    """One result, as ``evaluate_assisted`` returns it, from the results of consecutive parts of a course set."""
    out = {"per_course": [], "events": [], "plans": {}, "search_seconds": 0.0}
    for p in parts:
        out["per_course"] += p["per_course"]
        out["events"] += p["events"]
        for k, v in p["plans"].items():
            out["plans"][k] = out["plans"].get(k, 0) + v
        out["search_seconds"] = round(out["search_seconds"] + p["search_seconds"], 1)
    return out


def part_bounds(n: int, k: int, parts: int) -> tuple[int, int]:
    """Courses [lo, hi) of part k (1-based) of ``parts`` contiguous parts of n courses."""
    if not 1 <= k <= parts <= n:
        raise ValueError(f"part {k} of {parts}")
    return round((k - 1) * n / parts), round(k * n / parts)


def serve(model, courses: list[C.Course], cfg: AssistConfig, device, inp=None, out=None) -> None:
    """The bots' planner: one JSON request per line in, one plan per line out.

    Request: {"index": course index in ``courses``, "cp", "target", "kind",
    "attempt", "state": {"pos", "vel", "yaw", "onGround", "jumpTicks",
    "collidedH"}, "lastAction"}. Answer: {"plan": [[move, strafe, jump,
    sprint, yaw], ...] or null, "rounds", "restarts" (only with ``restarts``
    set), "corner_dropped" (only with ``corner_limit`` set: landings the corner
    check dropped), "brake": the brake_plan to play before the retry when there is no
    plan and ``brake`` is set (else null), "seconds"}. The search is the one
    ``evaluate_assisted`` runs with ``batch`` 1 (``plan_once``: same seed and
    neo levers for the same request). A request with "route": true is the
    router's (``parkour.route_neo``: a blocked-path choice, the jump from
    platform "cp" to "target"): it is a neo's, its fall line is its "floor",
    and its horizon is the configured one for its kind, as there.
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
            if req.get("route"):  # the router's blocked-path choice (route_neo.route_arrival)
                arr.update(neo=True, floor=float(req["floor"]))
                horizon = horizons(cfg)[req["kind"]]
            else:
                horizon = plan_horizon(courses[arr["course"]], k, req["kind"], cfg)
            st = {}
            seq, rounds, restarts = plan_once(model, pool, rules, arr, req["kind"], attempt, cfg, horizon, device,
                                              **({"stats": st} if cfg.corner_limit else {}))
            brake = None
            if seq is None and cfg.brake and req["kind"] != "approach" and attempt + 1 <= cfg.retries:
                brake = brake_plan(pool, rules, arr, cfg.brake, device)
            out.write(json.dumps({"plan": None if seq is None else seq.tolist(), "rounds": rounds,
                                  **({"restarts": restarts} if cfg.restarts else {}),
                                  **({"corner_dropped": st.get("corner_dropped", 0)} if cfg.corner_limit else {}),
                                  "brake": None if brake is None else brake.tolist(),
                                  "seconds": round(time.time() - t0, 3)}) + "\n")
            out.flush()


PARTS_DIR = ROOT / "runs" / "assist_parts"  # parts of split runs (--part / --parts) until merged; --parts-dir


def _part_command(a, k: int, n: int) -> list[str]:
    """The command line of part k of n of the run that ``a`` describes."""
    cmd = [sys.executable, "-m", "parkour.neo_assist", "--set", a.set, "--family", str(a.family), "--checkpoint",
           a.checkpoint, "--tag", a.tag, "--name", a.name, "--parts-dir", a.parts_dir, "--part", f"{k}/{n}",
           "--threads", "1"]
    if a.off:
        cmd.append("--off")
    if a.courses:
        cmd += ["--courses", str(a.courses)]
    if a.settle:
        cmd += ["--settle", str(a.settle)]
    if a.device:
        cmd += ["--device", a.device]
    for key in asdict(AssistConfig()):
        if key != "enabled":
            cmd += ["--" + key.replace("_", "-"), str(getattr(a, key))]
    return cmd


def _cached_part(path, meta, bounds):
    part = json.loads(path.read_text())
    if (part.get('meta') != meta or part.get('courses') != list(bounds)
            or len(part.get('per_course', [])) != bounds[1] - bounds[0]):
        raise SystemExit(f'{path} was computed with other settings or courses; preserve it and use a new name')
    return part


def _run_parts(a, meta, count) -> list[dict]:
    """Run the parts of a split run that are not finished yet (at most --jobs at once); return all parts."""
    import subprocess
    folder = Path(a.parts_dir) / a.name
    files = [folder / f"{k}of{a.parts}.json" for k in range(1, a.parts + 1)]
    for k, f in enumerate(files, 1):
        if f.exists():
            _cached_part(f, meta, part_bounds(count, k, a.parts))
    todo = [k for k, f in enumerate(files, 1) if not f.exists()]
    running = []
    try:
        while todo or running:
            while todo and len(running) < a.jobs:
                k = todo.pop(0)
                log = open(folder / f"{k}of{a.parts}.log", "a")
                try:
                    proc = subprocess.Popen(_part_command(a, k, a.parts), cwd=str(ROOT), stdout=log,
                                            stderr=subprocess.STDOUT,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                except BaseException:
                    log.close()
                    raise
                running.append((k, proc, log))
            time.sleep(5)
            for item in [r for r in running if r[1].poll() is not None]:
                running.remove(item)
                item[2].close()
                if item[1].returncode != 0:
                    raise SystemExit(f"part {item[0]} of {a.parts} failed (see {folder / f'{item[0]}of{a.parts}.log'})")
    finally:
        for _, proc, log in running:
            if proc.poll() is None:
                if os.name == 'nt':
                    # Windows venv python.exe is a launcher; stop its child too.
                    subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
    return [_cached_part(f, meta, part_bounds(count, k, a.parts)) for k, f in enumerate(files, 1)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "results" / "final_policy.pt"))
    ap.add_argument("--set", choices=("official", "random", "mixed", "single"), default="official")
    ap.add_argument("--family", type=int, default=VALIDATION_FAMILY)
    ap.add_argument("--off", action="store_true", help="the policy alone, for comparison")
    ap.add_argument("--courses", type=int, default=0, help="only the first N courses (quick checks)")
    ap.add_argument("--settle", type=int, default=0, help="ticks without keys on the start block first, as the bots "
                                                          "start (parkour.live.SETTLE_TICKS, 10); in the row only when set")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--details", default="", help="also write every course's outcome and every search to this JSON file")
    ap.add_argument("--name", default="", help="a split run's name (--part, --parts): its parts are in PARTS_DIR/NAME")
    ap.add_argument("--part", default="", help="K/N: only the K-th of N contiguous parts of the course set, seeded as in the "
                                               "whole run; writes PARTS_DIR/NAME/KofN.json and no row")
    ap.add_argument("--parts", type=int, default=0, help="N: run the unfinished parts of NAME (--jobs at once, one thread "
                                                         "each), then merge them into one row; rerun it after an interruption")
    ap.add_argument("--parts-dir", default=str(PARTS_DIR), help="where split runs keep their parts, logs, locks and "
                                                                "merged row (default runs/assist_parts; e.g. a scratch "
                                                                "folder for a check run, with --out and --details)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--serve", action="store_true", help="plan for the bots: JSON lines on stdin/stdout (see serve)")
    ap.add_argument("--courses-file", default="", help="with --serve: the courses the bot plays")
    ap.add_argument("--realtime", action="store_true", help="with --serve: the live bot's planner (scripts/gpu_watchdog.py "
                                                            "never pauses it; nothing else changes)")
    ap.add_argument("--device", default="", help="cpu or cuda (default: cuda if available); the bots' planner service "
                                                 "must use the device of the simulator run it is compared with")
    for k, v in asdict(AssistConfig()).items():
        if k != "enabled":
            ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    valid = {"official": (TEST_SEED_FAMILY, VALIDATION_FAMILY), "random": (RANDOM_TEST_FAMILY, RANDOM_VALIDATION_FAMILY),
             "mixed": (MIXED_TEST_FAMILY, MIXED_VALIDATION_FAMILY),
             "single": (SINGLE_TEST_FAMILY, SINGLE_VALIDATION_FAMILY)}
    if a.family not in valid[a.set]:
        raise SystemExit(f"--set {a.set} has families {valid[a.set]} (test, validation)")
    cfg = AssistConfig(enabled=not a.off, **{k: getattr(a, k) for k in asdict(AssistConfig()) if k != "enabled"})
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.serve:
        courses = courses_from_file(json.loads(Path(a.courses_file).read_text())["courses"])
        serve(load_model(Path(a.checkpoint), dev), courses, cfg, dev)
        return
    if (a.part or a.parts) and not a.name:
        raise SystemExit("--part and --parts need --name")
    if (a.part or a.parts) and cfg.batch != 1:
        raise SystemExit("a run in parts needs batch 1 (batched searches are seeded by tick, not by course)")
    if a.parts < 0 or a.jobs < 1 or a.courses < 0 or (a.part and a.parts):
        raise SystemExit('Use nonnegative --parts/--courses, positive --jobs, and only one of --part/--parts')
    if a.part:
        try:
            k, n = (int(v) for v in a.part.split('/'))
        except ValueError as exc:
            raise SystemExit('--part must be K/N') from exc
        a.part = f'{k}/{n}'  # Equivalent spellings must own the same lock.
    a.parts_dir = os.path.abspath(a.parts_dir)  # (the parts' own processes start in ROOT)
    folder = Path(a.parts_dir) / a.name
    lock_name = 'manager.lock' if a.parts else 'part-' + a.part.replace('/', 'of') + '.lock'
    lock = cache_lock(folder / lock_name) if (a.part or a.parts) else nullcontext()
    with lock:
        _evaluate_cli(a, cfg, dev)


def _evaluate_cli(a, cfg, dev):
    courses = course_set(a.set, a.family)
    if a.courses:
        courses = courses[:a.courses]
    if a.parts:
        part_bounds(len(courses), 1, a.parts)
    test = a.family in TEST_FAMILIES
    ck = Path(a.checkpoint).resolve()
    ck_name = ck.relative_to(ROOT).as_posix() if ck.is_relative_to(ROOT) else ck.name
    meta = {"set": a.set, "family": a.family, "split": "test" if test else "validation",
            "checkpoint": ck_name, "checkpoint_sha1": hashlib.sha1(ck.read_bytes()).hexdigest()[:12], "assist": not a.off,
            "config": config_meta(cfg), "tag": a.tag,
            "measured": f"simulator (float64 port of prismarine-physics), {dev.upper() if dev == 'cpu' else 'CUDA'}"}
    if a.courses:
        meta["first_courses_only"] = a.courses
    if a.settle:
        meta["settle"] = a.settle
    if a.part or a.parts:
        meta['courses_sha256'] = hashlib.sha256(json.dumps([c.to_json() for c in courses], sort_keys=True).encode()).hexdigest()
    if a.parts:
        folder = Path(a.parts_dir) / a.name
        folder.mkdir(parents=True, exist_ok=True)
        merged = folder / "merged.json"
        if merged.exists():  # Validate before republishing cached results to any requested destination.
            previous = json.loads(merged.read_text())
            if any(previous.get(k) != v for k, v in meta.items()) or previous.get('parts') != a.parts:
                raise SystemExit(f'{merged} was computed with other settings; preserve it and use a new name')
            if not all((folder / f'{k}of{a.parts}.json').exists() for k in range(1, a.parts + 1)):
                raise SystemExit(f'{merged} exists but cached parts are missing; refusing to repeat evaluations')
        parts = _run_parts(a, meta, len(courses))
        if any(p["meta"] != meta for p in parts):
            raise SystemExit(f"the parts in {folder} were run with other settings than these")
        res = merge_results(parts)
        row = {**meta, **summarize(res), "parts": a.parts, "seconds": round(sum(p["seconds"] for p in parts), 1),
               'run_id': 'assist_parts/' + a.name}
    else:
        lo, hi = 0, len(courses)
        if a.part:
            k, n = (int(v) for v in a.part.split("/"))
            lo, hi = part_bounds(len(courses), k, n)
            part_file = Path(a.parts_dir) / a.name / f"{k}of{n}.json"
            if part_file.exists():
                _cached_part(part_file, meta, (lo, hi))
                print(f"{part_file} exists")
                return
        model = load_model(Path(a.checkpoint), dev)
        if hashlib.sha1(ck.read_bytes()).hexdigest()[:12] != meta['checkpoint_sha1']:
            raise SystemExit('Checkpoint changed while loading; use an immutable checkpoint copy')
        pool = CoursePool(courses[lo:hi], dev, C.GRID)
        t0 = time.time()
        res = evaluate_assisted(model, pool, dev, cfg, settle=a.settle, log=lambda d: print(json.dumps(d), flush=True),
                                course_offset=lo)
        if a.part:
            part_file.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(part_file, {"meta": meta, "courses": [lo, hi], **res, "seconds": round(time.time() - t0, 1)})
            return
        row = {**meta, **summarize(res), "seconds": round(time.time() - t0, 1)}
    out = Path(a.out) if a.out else ROOT / "results" / "neo" / ("assist_test.jsonl" if test else "assist_validation.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    details = Path(a.details) if a.details else (ROOT / "results" / "neo" / "assist_details" / f"{a.name}.json"
                                                 if a.parts else None)
    if details:
        details.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(details, {"row": row, "per_course": res["per_course"], "events": res["events"]})
    if a.parts:
        append_jsonl_once(out, row)
        write_json_atomic(merged, row)
    else:
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()

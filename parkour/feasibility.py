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
     python -m parkour.feasibility --beam  (writes configs/jump_beam.json: the
     same study with a head-hitter beam across the gap, 3 blocks above the
     takeoff surface; a jump type may carry a beam in obstacle courses only if
     its forgiveness with the beam stays within its own level)
     python -m parkour.feasibility --advanced  (writes
     configs/jump_feasibility_advanced.json and configs/jump_levels_advanced.json:
     neo, double and triple neo jumps with a curved strategy family, momentum
     jumps from a runway, and sideways offsets of 3)
     python -m parkour.feasibility --extra  (writes configs/jump_feasibility_extra.json
     and configs/jump_levels_extra.json: head hitters, runways under a 2-high
     roof, and ice runways, with a run-up strategy family, and jumps onto
     shaped blocks (slabs, soul sand, posts, ...); opt-in levels that
     load_levels(extra=True) appends)
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
from .courses import Course, JumpSpec, _beam, _platform_after, bounce_exit, exit_spec, shape_geometry, single_jump
from .env import CoursePool, on_obstacle, standing_on
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


def _evaluate_chunk(specs, device, beam=False, courses=None, land_on=1, horizon=HORIZON):
    courses = courses or [single_jump(s) for s in specs]
    if beam:
        for c in courses:
            c.extra = np.array(_beam(c.platforms[0], c.platforms[1], 0), dtype=np.int64).reshape(-1, 3)
    pool = CoursePool(courses, device, grid=(24, 16, 24))
    strategies = list(itertools.product(START_BACK, AIM_LATERAL, AIM_ALONG, JUMP_TICKS, SPRINT, RELEASE_AFTER))
    ns, nk = len(specs), len(strategies)
    n = ns * nk

    def col(i, dtype):
        return torch.tensor([s[i] if s[i] is not None else 10_000 for s in strategies], dtype=dtype,
                            device=device).repeat(ns)

    course = torch.arange(ns, device=device).repeat_interleave(nk)
    back, aim_lat, aim_along = (col(i, torch.float64) for i in range(3))
    # Momentum specs start at the back of their runway instead of the centre of a 1x1 block.
    back = back + torch.tensor([(sp.runway - 1) / 2 if sp.runway else 0.0 for sp in specs], dtype=torch.float64,
                               device=device).repeat_interleave(nk)
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
    alive = ~physics.inside_solid(pool.world, course, s.pos)  # never start inside a block
    low = pool.top[course][:, :land_on + 1].amin(dim=1)
    for t in range(horizon):
        c = torch.zeros(n, 6, dtype=torch.bool, device=device)
        held = t < release
        c[:, physics.FORWARD] = held
        c[:, physics.SPRINT] = held & sprint
        c[:, physics.JUMP] = jtick == t
        s.yaw = yaw
        s = physics.simulate(pool.world, course, s, c)
        landed |= standing_on(pool, s, course)[:, land_on] & alive
        # A strategy stops counting once it has fallen below the lower platform.
        alive &= s.pos[:, 1] > low - 1.0
    return landed.view(ns, nk).float().mean(dim=1).cpu().numpy()


NEO_BACK = (0.29, 0.0)
NEO_SHIFT = (0.0, 0.29)  # start shifted sideways toward the open side
NEO_ANGLE = (15.0, 30.0, 45.0, 60.0, 75.0, 90.0)  # takeoff yaw, degrees off the jump direction toward the open side
NEO_JUMP = range(0, 12)
NEO_DELAY = (0, 2, 4, 6, 9)  # ticks after the jump before turning back toward the target
NEO_TURN = (5.0, 10.0, 20.0, 40.0)  # degrees per tick
NEO_STRAFE = (False, True)


def _evaluate_neo_chunk(specs, device):
    """Curved strategies: face past the wall's edge, jump, then turn (and strafe) toward the target."""
    courses = [single_jump(s) for s in specs]
    pool = CoursePool(courses, device, grid=(24, 16, 24))
    strategies = list(itertools.product(NEO_BACK, NEO_ANGLE, NEO_JUMP, NEO_DELAY, NEO_TURN, NEO_STRAFE, NEO_SHIFT))
    ns, nk = len(specs), len(strategies)
    n = ns * nk

    def col(i, dtype):
        return torch.tensor([st[i] for st in strategies], dtype=dtype, device=device).repeat(ns)

    course = torch.arange(ns, device=device).repeat_interleave(nk)
    back, angle = col(0, torch.float64), col(1, torch.float64) * (math.pi / 180)
    jtick, delay = col(2, torch.int64), col(3, torch.int64)
    turn = col(4, torch.float64) * (math.pi / 180)
    strafe = col(5, torch.bool)
    shift = col(6, torch.float64)
    target = pool.centre[course, 1]
    start = pool.centre[course, 0].clone()
    start[:, 0] -= back
    start[:, 2] += torch.tensor([float(sp.side) for sp in specs], dtype=torch.float64,
                                device=device).repeat_interleave(nk) * shift
    # Heading +X has yaw -pi/2; turning toward +Z (right of +X) lowers the yaw.
    side = torch.tensor([float(sp.side) for sp in specs], dtype=torch.float64, device=device).repeat_interleave(nk)
    yaw = -math.pi / 2 - side * angle
    s = physics.PlayerState.at(start, yaw)
    landed = torch.zeros(n, dtype=torch.bool, device=device)
    alive = ~physics.inside_solid(pool.world, course, s.pos)  # never start inside a block
    low = torch.minimum(pool.top[course, 0], pool.top[course, 1])
    for t in range(HORIZON):
        steer = t >= jtick + delay
        to = target - s.pos
        want = torch.atan2(-to[:, 0], -to[:, 2])
        diff = torch.remainder(want - yaw + math.pi, 2 * math.pi) - math.pi
        yaw = torch.where(steer, yaw + torch.clamp(diff, -turn, turn), yaw)
        right = to[:, 0] * torch.cos(yaw) - to[:, 2] * torch.sin(yaw)  # target's offset along the right vector
        c = torch.zeros(n, 6, dtype=torch.bool, device=device)
        c[:, physics.FORWARD] = True
        c[:, physics.SPRINT] = True
        c[:, physics.JUMP] = jtick == t
        c[:, physics.RIGHT] = steer & strafe & (right > 0.05)
        c[:, physics.LEFT] = steer & strafe & (right < -0.05)
        s.yaw = yaw
        s = physics.simulate(pool.world, course, s, c)
        landed |= standing_on(pool, s, course)[:, 1] & alive
        alive &= s.pos[:, 1] > low - 1.0
    return landed.view(ns, nk).float().mean(dim=1).cpu().numpy()


def evaluate_neo(specs, device="cuda"):
    out = []
    for i in range(0, len(specs), 8):
        out.append(_evaluate_neo_chunk(specs[i:i + 8], device))
    return np.concatenate(out)


def advanced_specs():
    neo = [JumpSpec(g, dy, 0, (1, 1), neo=k, side=1, wall_at=w)
           for k in (1, 2, 3) for g in (1, 2, 3) for w in range(1, g + 1) for dy in (-1, 0, 1)]
    momentum = [JumpSpec(g, dy, lat, size, runway=r)
                for r in (2, 3, 4) for g in (4, 5, 6) for dy in (-1, 0, 1) for lat in (0, 1) for size in ((1, 1), (2, 2))]
    wide = [JumpSpec(g, dy, 3, size) for g in (0, 1, 2, 3) for dy in (-2, -1, 0, 1) for size in ((1, 1), (2, 2), (3, 3))]
    return neo, momentum, wide


def advanced_study(device):
    neo, momentum, wide = advanced_specs()
    t0 = time.time()
    f_neo = evaluate_neo(neo, device)
    f_mom = evaluate(momentum, device)
    f_wide = evaluate(wide, device)
    rows = []
    for kind, specs, frac in (("neo", neo, f_neo), ("momentum", momentum, f_mom), ("wide", wide, f_wide)):
        for sp, f in zip(specs, frac):
            rows.append({"kind": kind, "gap": sp.gap, "dy": sp.dy, "lateral": sp.lateral, "size": list(sp.size),
                         "neo": sp.neo, "wall_at": sp.wall_at, "runway": sp.runway, "forgiveness": round(float(f), 5)})

    def level_of(r):
        f = r["forgiveness"]
        if f <= 0:
            return None
        if r["kind"] == "neo":
            # Level by the wall: neo 6, double neo 7, triple neo 8; very tight ones one level up.
            return min(8, 5 + r["neo"] + (1 if f < 0.002 else 0))
        if r["kind"] == "momentum":
            return 8 if (r["gap"] >= 6 or f < 0.005) else (7 if (r["gap"] == 5 or r["dy"] > 0 or f < 0.02) else 6)
        return 7 if f < 0.02 else 6

    for r in rows:
        r["level"] = level_of(r)
    out = {"method": "single jumps from rest; straight strategies for momentum and wide jumps, curved ones for neos",
           "strategies": {"neo": len(NEO_BACK) * len(NEO_ANGLE) * len(NEO_JUMP) * len(NEO_DELAY) * len(NEO_TURN)
                          * len(NEO_STRAFE) * len(NEO_SHIFT),
                          "straight": len(START_BACK) * len(AIM_LATERAL) * len(AIM_ALONG) * len(JUMP_TICKS)
                          * len(SPRINT) * len(RELEASE_AFTER)},
           "seconds": round(time.time() - t0, 1), "specs": rows}
    (CONFIGS / "jump_feasibility_advanced.json").write_text(json.dumps(out, indent=1))
    levels = []
    for L in (6, 7, 8):
        members = []
        for r in rows:
            if r["level"] != L:
                continue
            base = {k: r[k] for k in ("gap", "dy", "lateral", "size", "neo", "runway", "wall_at")}
            if r["kind"] == "neo":
                members += [{**base, "side": 1}, {**base, "side": -1}]
            else:
                members.append({**base, "side": 1})
                if r["lateral"]:
                    members.append({**base, "lateral": -r["lateral"], "side": 1})
        if members:
            levels.append({"level": L, "specs": members})
    (CONFIGS / "jump_levels_advanced.json").write_text(json.dumps({"source": "parkour.feasibility --advanced",
                                                                   "levels": levels}, indent=1))
    for kind in ("neo", "momentum", "wide"):
        rk = [r for r in rows if r["kind"] == kind]
        print(kind, f"{sum(r['level'] is not None for r in rk)}/{len(rk)} possible")
    for L in levels:
        print("level", L["level"], len(L["specs"]), "specs")
    return out


# Run-up strategies (extra study): start at the back of the runway, sprint toward an aim point on
# the target, jump when the centre passes a point before (or past) the runway's end, optionally
# hopping on every landing before that (head-hitter momentum), optionally letting go after the jump.
RUN_AIM_LATERAL = (-0.3, -0.15, 0.0, 0.15, 0.3)
RUN_EDGE = (-0.28, -0.2, -0.1, 0.0, 0.15, 0.3, 0.5, 0.75, 1.0)  # blocks before the runway's end
RUN_HOP_FROM = (None, 0, 4)  # tick from which the bot jumps on every landing before the edge (None: never)
RUN_RELEASE = (None, 3)  # ticks after the final jump at which forward and sprint are released
RUN_HORIZON = 80


def extra_specs():
    roof = [JumpSpec(g, dy, lat, size, runway=r, ceiling=c)
            for c in (1, 2) for r in (2, 3, 4) for g in (1, 2, 3, 4) for dy in (-2, -1, 0) for lat in (0, 1)
            for size in ((1, 1), (2, 2))]
    ice = [JumpSpec(g, dy, lat, size, runway=r, ceiling=c, ice=1)
           for c in (0, 1) for r in (3, 4, 6) for g in (3, 4, 5) for dy in (-1, 0, 1) for lat in (0, 1)
           for size in ((1, 1), (2, 2)) if not (c and dy > 0)]
    return roof, ice


def _evaluate_runway_chunk(specs, device):
    courses = [single_jump(s) for s in specs]
    pool = CoursePool(courses, device, grid=(24, 16, 24))
    strategies = list(itertools.product(RUN_AIM_LATERAL, RUN_EDGE, RUN_HOP_FROM, RUN_RELEASE))
    ns, nk = len(specs), len(strategies)
    n = ns * nk
    course = torch.arange(ns, device=device).repeat_interleave(nk)
    col = lambda i, none: torch.tensor([st[i] if st[i] is not None else none for st in strategies],  # noqa: E731
                                       device=device).repeat(ns)
    aim_lat = col(0, 0).to(torch.float64)
    edge = col(1, 0).to(torch.float64)
    hop_from = col(2, 10_000)
    release = col(3, 10_000)
    run_end = pool.plat[course, 0, 1].to(torch.float64) + 1  # single_jump heads +X from its runway
    start = pool.centre[course, 0].clone()
    start[:, 0] = pool.plat[course, 0, 0].to(torch.float64) + 0.31  # back edge of the runway
    aim = pool.centre[course, 1].clone()
    aim[:, 2] += aim_lat
    d = aim - start
    yaw = torch.atan2(-d[:, 0], -d[:, 2])
    s = physics.PlayerState.at(start, yaw)
    landed = torch.zeros(n, dtype=torch.bool, device=device)
    alive = ~physics.inside_solid(pool.world, course, s.pos)
    jumped_at = torch.full((n,), 10_000, dtype=torch.int64, device=device)
    low = torch.minimum(pool.top[course, 0], pool.top[course, 1])
    for t in range(RUN_HORIZON):
        at_edge = (s.pos[:, 0] >= run_end - edge) & (jumped_at == 10_000)
        hop = (t >= hop_from) & (jumped_at == 10_000) & ~at_edge
        jump = s.on_ground & (at_edge | hop)
        jumped_at = torch.where(at_edge & s.on_ground & (jumped_at == 10_000), torch.full_like(jumped_at, t), jumped_at)
        held = t < jumped_at + release
        c = torch.zeros(n, 6, dtype=torch.bool, device=device)
        c[:, physics.FORWARD] = held
        c[:, physics.SPRINT] = held
        c[:, physics.JUMP] = jump
        s.yaw = yaw
        s = physics.simulate(pool.world, course, s, c)
        alive &= ~on_obstacle(pool, s, course) & (s.pos[:, 1] > low - 1.0)
        landed |= standing_on(pool, s, course)[:, 1] & alive
    return landed.view(ns, nk).float().mean(dim=1).cpu().numpy()


FLAT_SHAPES = ("bottom_slab", "soul_sand", "dirt_path", "enchanting_table", "daylight_detector", "bottom_trapdoor",
               "carpet")  # full footprint, top below a whole block
POST_SHAPES = ("fence_post", "wall_post", "end_rod")  # 1x1 only, narrow


def shape_specs():
    flat = [JumpSpec(g, dy, lat, size, block=b) for b in FLAT_SHAPES for g in (1, 2, 3) for dy in (-1, 0, 1)
            for lat in (0, 1) for size in ((1, 1), (2, 2))]
    post = [JumpSpec(g, dy, lat, (1, 1), block=b) for b in POST_SHAPES for g in (1, 2, 3)
            for dy in ((-2, -1, 0) if shape_geometry(b)[0] > 1 else (-1, 0)) for lat in (0, 1)]
    return flat + post


def exit_course(block: str, size: tuple) -> Course:
    """A shaped takeoff (``size``) and the fixed exit jump after it, heading +X, as in a course."""
    prev = np.array([4, 4 + size[0] - 1, 12, 12 + size[1] - 1, 8], dtype=np.int64)
    new = _platform_after(prev, 0, exit_spec(block))
    c = Course(np.stack([prev, new]), np.zeros(3), 0.0, specs=[exit_spec(block)], shapes={0: block})
    centres = c.centers()
    c.spawn = centres[0].copy()
    d = centres[1] - centres[0]
    c.spawn_yaw = float(math.atan2(-d[0], -d[2]))
    return c


def _evaluate_courses(courses, device):
    """Straight-strategy forgiveness of ready-made two-platform courses (heading +X)."""
    out = []
    for i in range(0, len(courses), CHUNK):
        part = courses[i:i + CHUNK]
        specs = [c.specs[0] for c in part]
        out.append(_evaluate_chunk(specs, device, courses=part))
    return np.concatenate(out)


BOUNCE_HORIZON = 80  # takeoff, fall to the pad, bounce and the flight to the exit


def bounce_specs():
    return [JumpSpec(g, dy, 0, size, bounce=b) for g in (1, 2, 3) for dy in (-2, -3, -4) for size in ((1, 1), (2, 2))
            for b in (1, 2, 3)]


def bounce_course(spec: JumpSpec) -> Course:
    """Takeoff (1x1 stone), the slime pad of ``spec`` and its exit, heading +X."""
    c = single_jump(spec)
    out = _platform_after(c.platforms[1], 0, bounce_exit(spec))
    c.platforms = np.concatenate([c.platforms, out[None, :]])
    return c


EXTRA_LEVELS = (("head hitter", lambda r: r["kind"] == "roof" and r["forgiveness"] >= 0.05),
                ("hard head hitter", lambda r: r["kind"] == "roof" and 0 < r["forgiveness"] < 0.05),
                ("ice run-up", lambda r: r["kind"] == "ice" and r["forgiveness"] > 0),
                ("shaped blocks", lambda r: r["kind"] == "shape" and r["forgiveness"] > 0 and r["exit_forgiveness"] >= 0.05),
                ("slime bounce", lambda r: r["kind"] == "bounce" and r["forgiveness"] > 0))


def extra_study(device):
    roof, ice = extra_specs()
    t0 = time.time()
    rows = []
    for kind, specs in (("roof", roof), ("ice", ice)):
        frac = np.concatenate([_evaluate_runway_chunk(specs[i:i + 12], device) for i in range(0, len(specs), 12)])
        for sp, f in zip(specs, frac):
            rows.append({"kind": kind, "gap": sp.gap, "dy": sp.dy, "lateral": sp.lateral, "size": list(sp.size),
                         "runway": sp.runway, "ceiling": sp.ceiling, "ice": sp.ice, "forgiveness": round(float(f), 5)})
    shaped = shape_specs()
    frac = evaluate(shaped, device)
    exits = {(b, sz): None for b in FLAT_SHAPES + POST_SHAPES for sz in ((1, 1), (2, 2))
             if not (b in POST_SHAPES and sz != (1, 1))}
    exit_f = _evaluate_courses([exit_course(b, sz) for b, sz in exits], device)
    exits = {k: round(float(f), 5) for k, f in zip(exits, exit_f)}
    for sp, f in zip(shaped, frac):
        rows.append({"kind": "shape", "gap": sp.gap, "dy": sp.dy, "lateral": sp.lateral, "size": list(sp.size),
                     "block": sp.block, "forgiveness": round(float(f), 5), "exit_forgiveness": exits[(sp.block, sp.size)]})
    bounce = bounce_specs()
    frac = np.concatenate([_evaluate_chunk([s.specs[0] for s in part], device, courses=part, land_on=2, horizon=BOUNCE_HORIZON)
                           for part in ([bounce_course(sp) for sp in bounce[i:i + CHUNK]]
                                        for i in range(0, len(bounce), CHUNK))])
    for sp, f in zip(bounce, frac):  # landing on the exit after the bounce
        rows.append({"kind": "bounce", "gap": sp.gap, "dy": sp.dy, "lateral": sp.lateral, "size": list(sp.size),
                     "bounce": sp.bounce, "forgiveness": round(float(f), 5)})
    out = {"method": "single run-up jumps from rest at the back of the runway; run-up strategies (jump at the "
                     "runway's edge, hops before it, release after). Shaped blocks: the straight strategies of the "
                     "basic study from a 1x1 stone block onto the shaped target, and from the shaped block over the "
                     "fixed exit jump (exit_forgiveness)",
           "strategies": len(RUN_AIM_LATERAL) * len(RUN_EDGE) * len(RUN_HOP_FROM) * len(RUN_RELEASE),
           "horizon_ticks": RUN_HORIZON, "seconds": round(time.time() - t0, 1), "specs": rows}
    (CONFIGS / "jump_feasibility_extra.json").write_text(json.dumps(out, indent=1))
    levels = []
    for name, rule in EXTRA_LEVELS:
        members = []
        for r in rows:
            if not rule(r):
                continue
            base = {k: r[k] for k in ("gap", "dy", "lateral", "size", "runway", "ceiling", "ice", "block", "bounce") if k in r}
            members.append({**base, "side": 1})
            if r["lateral"]:
                members.append({**base, "lateral": -r["lateral"], "side": 1})
        levels.append({"name": name, "specs": members})
    (CONFIGS / "jump_levels_extra.json").write_text(json.dumps({"source": "parkour.feasibility --extra",
                                                                "levels": levels}, indent=1))
    for kind in ("roof", "ice", "shape", "bounce"):
        rk = [r for r in rows if r["kind"] == kind]
        print(kind, f"{sum(r['forgiveness'] > 0 for r in rk)}/{len(rk)} possible")
    for L in levels:
        print(L["name"], len(L["specs"]), "specs")
    return out


def evaluate(specs, device="cuda", beam=False):
    out = []
    for i in range(0, len(specs), CHUNK):
        out.append(_evaluate_chunk(specs[i:i + CHUNK], device, beam))
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
    ap.add_argument("--beam", action="store_true", help="study jumps under a head-hitter beam")
    ap.add_argument("--advanced", action="store_true", help="study neo, momentum and wide jumps")
    ap.add_argument("--extra", action="store_true", help="study head hitters and ice run-ups")
    args = ap.parse_args(argv)
    if args.extra:
        return extra_study(args.device)
    if args.advanced:
        return advanced_study(args.device)
    if args.beam:
        return beam_study(args.device)
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



def beam_study(device):
    feas = json.loads((CONFIGS / "jump_feasibility.json").read_text())["specs"]
    rows = [r for r in feas if r["level"] is not None and r["gap"] >= 1]
    specs = [JumpSpec(r["gap"], r["dy"], r["lateral"], tuple(r["size"])) for r in rows]
    t0 = time.time()
    frac = evaluate(specs, device, beam=True)
    out = []
    for r, sp, f in zip(rows, specs, frac):
        ok = bool(f > 0 and f >= LEVEL_EDGES[r["level"]])
        for lat in ({r["lateral"], -r["lateral"]}):
            key = JumpSpec(sp.gap, sp.dy, lat, sp.size).key()
            out.append({"key": key, "level": r["level"], "forgiveness": r["forgiveness"],
                        "forgiveness_beam": round(float(f), 5), "beam_ok": ok})
    data = {"method": "as jump_feasibility.json, with a beam across the gap 3 blocks above the takeoff surface",
            "beam_height": 3, "seconds": round(time.time() - t0, 1), "specs": out}
    (CONFIGS / "jump_beam.json").write_text(json.dumps(data, indent=1))
    n_ok = sum(o["beam_ok"] for o in out)
    print(f"{len(out)} jump types with a gap, {n_ok} keep their level under a beam ({data['seconds']} s)")
    return data


if __name__ == "__main__":
    main()

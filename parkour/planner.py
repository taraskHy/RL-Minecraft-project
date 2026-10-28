"""Greedy route planner: the bot finds its own route from the blocks it sees.

The trained policy is told the platform it last stood on and the next three
(its platform slots). Here nothing tells it the course: after every landing
the planner looks at the solid blocks within ``radius`` of the player, groups
them into platforms, and picks a chain of up to three next platforms, which
fill the same slots. Only the goal's position is given (the beacon).

    python -m parkour.planner --checkpoint results/final_policy.pt [--obstacles] [--family 1]

Perception. A standable cell is a solid block with two air blocks above it.
With ``slab_only`` (the default) the block below must be air too, so the tops
of walls and pillars, which are columns, are not platforms. Standable cells
at the same height that touch side by side form one platform; it is used as
its bounding rectangle (x0, x1, z0, z1, y), like a course platform.

Choice. From the platform it stands on, the next one is the unvisited
platform that minimises ``distance - align * cos(angle to the goal)`` among
those within the jump limits (horizontal air gap up to ``max_gap``, height
change from ``dy_min`` to ``dy_max``). The rule repeats from the chosen
platform for the slots after it, and stops at the goal platform.

The same planner runs in the bot (bridge/planner.js); tests check that both
pick the same platforms.

With ``value_pick`` K > 0 (simulator only; the bot has no critic), the K
cheapest next platforms are each extended greedily into a chain, and the
chain whose slots give the highest value under the policy's own critic, at
the player's current state, is taken. ``drop`` adds a cost per block a jump
goes down; ``lookahead`` adds the cheapest onward jump. None of the three
helped on the validation courses
(results/planner/validation_tuning.jsonl); ``cost="distance", align=2.0``
did, and is the setting recommended for the final policy (report 7.11).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import courses as C
from . import physics
from .env import PLATFORM_SLOTS, STAND_EPS, CoursePool, EnvConfig, ParkourEnv

LOOKAHEAD = PLATFORM_SLOTS - 1
LOOKAHEAD_DEAD_END = 10.0  # cfg.lookahead: extra cost of a platform with no jump on from it
# From the feasibility study (configs/jump_feasibility.json, straight jumps from a 1x1 block):
# the longest possible air gap for each height change, and the best forgiveness at each gap.
MAX_GAP = {1: 3, 0: 4, -1: 4, -2: 4, -3: 5}
FORGIVENESS = {1: (0.593, 0.274, 0.124, 0.021), 0: (0.62, 0.62, 0.196, 0.078, 0.01),
               -1: (0.867, 0.876, 0.224, 0.104, 0.031), -2: (0.859, 0.878, 0.369, 0.13, 0.044),
               -3: (0.826, 0.859, 0.656, 0.156, 0.062, 0.021)}


@dataclass
class PlannerConfig:
    radius: int = 12      # blocks around the player's block (horizontal, square) that the bot sees
    below: int = 10       # and this many blocks below its feet, ``above`` above
    above: int = 4
    max_gap: float = 5.5  # largest horizontal air gap between two platforms (limits "flat")
    dy_min: int = -3
    dy_max: int = 1
    align: float = 3.0    # weight of the goal direction against the jump's cost (tuned on validation)
    slab_only: bool = True
    limits: str = "table"      # "table": gap up to MAX_GAP[dy] + slack; "flat": up to max_gap
    slack: float = 0.5
    cost: str = "difficulty"   # "difficulty": -log forgiveness of the jump; "distance": the air gap
    mark_passed: bool = True   # landing beyond the plan's next platform marks the ones skipped as passed
    clear_path: bool = True    # skip platforms whose straight flight path from here passes through blocks
    head_hitters: bool = True  # a slab 4 blocks above a platform it touches is a head hitter, not a target
    value_pick: int = 0        # >0: of this many cheapest next platforms, take the chain the critic values most
    drop: float = 0.0          # added cost per block a jump goes down (0: off, as in the bot)
    lookahead: bool = False    # rank next platforms by their cost plus the cheapest jump on from them


class Platform:
    __slots__ = ("cells", "x0", "x1", "z0", "z1", "y")

    def __init__(self, cells: list):
        self.cells = cells
        xs = [c[0] for c in cells]
        zs = [c[2] for c in cells]
        self.x0, self.x1, self.z0, self.z1, self.y = min(xs), max(xs), min(zs), max(zs), cells[0][1]

    def rect(self) -> tuple:
        return (self.x0, self.x1, self.z0, self.z1, self.y)

    def centre(self) -> tuple:
        return ((self.x0 + self.x1 + 1) / 2, self.y + 1, (self.z0 + self.z1 + 1) / 2)

    def contains(self, cell) -> bool:
        x, y, z = cell
        return y == self.y and self.x0 <= x <= self.x1 and self.z0 <= z <= self.z1


def perceive(is_solid, cx: int, fy: int, cz: int, cfg: PlannerConfig) -> list:
    """Platforms around block (cx, fy, cz), in scan order (y, then x, then z)."""
    def standable(x, y, z):
        if not is_solid(x, y, z) or is_solid(x, y + 1, z) or is_solid(x, y + 2, z):
            return False
        return not (cfg.slab_only and is_solid(x, y - 1, z))

    r = cfg.radius
    seen = set()
    out = []
    for y in range(fy - cfg.below, fy + cfg.above + 1):
        for x in range(cx - r, cx + r + 1):
            for z in range(cz - r, cz + r + 1):
                if (x, y, z) in seen or not standable(x, y, z):
                    continue
                cells = [(x, y, z)]
                seen.add((x, y, z))
                k = 0
                while k < len(cells):  # flood fill, same order as planner.js
                    a, _, b = cells[k]
                    k += 1
                    for nx, nz in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                        if abs(nx - cx) > r or abs(nz - cz) > r or (nx, y, nz) in seen:
                            continue
                        if standable(nx, y, nz):
                            seen.add((nx, y, nz))
                            cells.append((nx, y, nz))
                out.append(Platform(cells))
    return out


def _norm(a: float, b: float) -> float:
    # Written out (not math.hypot) so bridge/planner.js computes the same bits.
    return math.sqrt(a * a + b * b)


def path_clear(is_solid, a: Platform, b: Platform) -> bool:
    """No block on the straight line between the two platforms, at body height, outside both."""
    ac, bc = a.centre(), b.centre()
    y0, y1 = min(a.y, b.y) + 1, max(a.y, b.y) + 2
    n = max(1, math.ceil(_norm(bc[0] - ac[0], bc[2] - ac[2]) / 0.25))
    for k in range(1, n):
        t = k / n
        x = math.floor(ac[0] + t * (bc[0] - ac[0]))
        z = math.floor(ac[2] + t * (bc[2] - ac[2]))
        if (a.x0 <= x <= a.x1 and a.z0 <= z <= a.z1) or (b.x0 <= x <= b.x1 and b.z0 <= z <= b.z1):
            continue
        for y in range(y0, y1 + 1):
            if is_solid(x, y, z):
                return False
    return True


def head_hitter(p: Platform, platforms: list) -> bool:
    """A slab 4 blocks above another platform whose footprint it touches or overlaps (a beam
    over a jump; course platforms are at most 3 above the next one)."""
    for q in platforms:
        touch_x = max(0, p.x0 - (q.x1 + 1), q.x0 - (p.x1 + 1)) == 0
        touch_z = max(0, p.z0 - (q.z1 + 1), q.z0 - (p.z1 + 1)) == 0
        if p.y - q.y == 4 and touch_x and touch_z:
            return True
    return False


def options(platforms: list, frm: Platform, current: Platform, chain: list, visited: set, goal: tuple,
            cfg: PlannerConfig, is_solid=None) -> list:
    """(cost, platform) for every platform reachable from ``frm``, in ``platforms`` order."""
    fc = frm.centre()
    gx, gz = goal[0] - fc[0], goal[2] - fc[2]
    gn = _norm(gx, gz)
    out = []
    for p in platforms:
        if p is current or p in chain or any(c in visited for c in p.cells):
            continue
        dy = p.y - frm.y
        if dy < cfg.dy_min or dy > cfg.dy_max:
            continue
        dx = max(0, p.x0 - (frm.x1 + 1), frm.x0 - (p.x1 + 1))
        dz = max(0, p.z0 - (frm.z1 + 1), frm.z0 - (p.z1 + 1))
        d = _norm(dx, dz)
        if d > (MAX_GAP[dy] + cfg.slack if cfg.limits == "table" else cfg.max_gap):
            continue
        if cfg.clear_path and is_solid is not None and not path_clear(is_solid, frm, p):
            continue
        pc = p.centre()
        vx, vz = pc[0] - fc[0], pc[2] - fc[2]
        vn = _norm(vx, vz)
        cos = (vx * gx + vz * gz) / (vn * gn) if vn > 1e-9 and gn > 1e-9 else 0.0
        if cfg.cost == "difficulty":
            f = FORGIVENESS[dy]
            cost = -math.log(f[min(round(d), len(f) - 1)]) - cfg.align * cos
        else:
            cost = d - cfg.align * cos
        if cfg.drop and dy < 0:
            cost += cfg.drop * -dy
        out.append((cost, p))
    return out


def choose(platforms: list, current: Platform, visited: set, goal: tuple, cfg: PlannerConfig, is_solid=None,
           first: Platform | None = None) -> list:
    """Up to LOOKAHEAD platforms after ``current``: greedy, each from the one before (``first``, if
    given, is taken as the first)."""
    chain = []
    frm = current
    for _ in range(LOOKAHEAD):
        best, best_cost = None, math.inf
        if first is not None and not chain:
            best = first
        else:
            gcell = goal_cell(goal)
            for cost, p in options(platforms, frm, current, chain, visited, goal, cfg, is_solid):
                if cfg.lookahead and not p.contains(gcell):
                    onward = [c for c, _ in options(platforms, p, current, chain + [p], visited, goal, cfg, is_solid)]
                    cost += min(onward) if onward else LOOKAHEAD_DEAD_END
                if cost < best_cost - 1e-9:
                    best, best_cost = p, cost
        if best is None:
            break
        chain.append(best)
        if best.contains(goal_cell(goal)):
            break
        frm = best
    return chain


def goal_cell(goal: tuple) -> tuple:
    return (math.floor(goal[0]), round(goal[1]) - 1, math.floor(goal[2]))


def support_cells(pos, half_width: float = physics.HALF_WIDTH) -> list:
    """Cells under the player's footprint (x, then z), at the block row its feet stand on."""
    x, y, z = pos
    cy = math.floor(y + STAND_EPS) - 1
    return [(ix, cy, iz) for ix in range(math.floor(x - half_width), math.ceil(x + half_width))
            for iz in range(math.floor(z - half_width), math.ceil(z + half_width))]


class GreedyPlanner:
    """Planner state for one episode: visited cells, the platform stood on, the plan."""

    def __init__(self, is_solid, goal: tuple, cfg: PlannerConfig):
        self.is_solid = is_solid
        self.goal = tuple(float(v) for v in goal)
        self.cfg = cfg
        self.visited: set = set()
        self.current: Platform | None = None
        self.chain: list = []
        self.start_dist = None
        self.progress = 0.0
        self.events = []  # (tick, platform stood on, chosen chain) for the record
        self.scorer = None  # value_pick: (platform stood on, candidate chains) -> index of the chain to take

    def start(self, pos) -> None:
        cell = next((c for c in support_cells(pos) if self.is_solid(*c)), None)
        if cell is None:
            cell = (math.floor(pos[0]), math.floor(pos[1] + STAND_EPS) - 1, math.floor(pos[2]))
        self._replan(cell, pos, 0)

    def update(self, pos, on_ground: bool, tick: int) -> bool:
        """Re-plan after landing on a new platform. True when the plan changed."""
        if not on_ground or abs(pos[1] - round(pos[1])) > STAND_EPS:
            return False
        cells = [c for c in support_cells(pos) if self.is_solid(*c)]
        if not cells or (self.current is not None and any(self.current.contains(c) for c in cells)):
            return False
        return self._replan(cells[0], pos, tick)

    def _replan(self, cell, pos, tick) -> bool:
        cfg = self.cfg
        platforms = perceive(self.is_solid, math.floor(pos[0]), cell[1] + 1, math.floor(pos[2]), cfg)
        here = next((p for p in platforms if p.contains(cell)), None)
        if here is None:  # not a platform (an obstacle top): keep the old plan
            return False
        if cfg.mark_passed:
            for j, p in enumerate(self.chain):
                if p.contains(cell):
                    for q in self.chain[:j]:
                        self.visited.update(q.cells)
                    break
        self.current = here
        self.visited.update(here.cells)
        targets = [p for p in platforms if not head_hitter(p, platforms)] if cfg.head_hitters else platforms
        c = here.centre()
        dx, dy, dz = c[0] - self.goal[0], c[1] - self.goal[1], c[2] - self.goal[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if self.start_dist is None:
            self.start_dist = max(dist, 1e-9)
        self.progress = min(1.0, max(0.0, 1.0 - dist / self.start_dist))
        self.chain = choose(targets, here, self.visited, self.goal, cfg, self.is_solid)
        if cfg.value_pick > 0 and self.scorer is not None and self.chain:
            opts = options(targets, here, here, [], self.visited, self.goal, cfg, self.is_solid)
            firsts = [p for _, p in sorted(opts, key=lambda o: o[0])[:cfg.value_pick]]
            chains = [choose(targets, here, self.visited, self.goal, cfg, self.is_solid, first=p) for p in firsts]
            self.chain = chains[self.scorer(here, chains)]
        self.events.append([tick, list(here.rect()), [list(p.rect()) for p in self.chain]])
        return True

    def slots(self, chain: list | None = None):
        """Platforms for the policy's slots (current first), valid flags, goal flags; ``chain``
        replaces the plan."""
        ps = ([self.current] if self.current is not None else []) + (self.chain if chain is None else chain)
        gcell = goal_cell(self.goal)
        rects = [p.rect() for p in ps] + [(0, 0, 0, 0, 0)] * (PLATFORM_SLOTS - len(ps))
        valid = [True] * len(ps) + [False] * (PLATFORM_SLOTS - len(ps))
        is_goal = [p.contains(gcell) for p in ps] + [False] * (PLATFORM_SLOTS - len(ps))
        return rects, valid, is_goal


def evaluate_planned(model, pool: CoursePool, device, cfg: PlannerConfig, stall_ticks: int = 200,
                     max_ticks: int = 4000, settle: int = 0, record: bool = False, fall_rule: str = "local") -> dict:
    """First episode on every course of ``pool``, the policy's platform slots filled by the planner.

    Scoring (success, falls, stalls) still uses the true course, as in train.evaluate.
    """
    n = pool.size
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=stall_ticks, max_ticks=max_ticks, fall_rule=fall_rule), sampler=None)
    env.reset_all(torch.arange(n, device=device))
    if settle:
        idle = torch.zeros(n, 6, dtype=torch.bool, device=device)
        for _ in range(settle):
            env.state = physics.simulate(pool.world, env.course, env.state, idle)
        env.dist = env._distance(env.state.pos, env.course, env.cp)
        env.reset_memory(torch.ones(n, dtype=torch.bool, device=device))
    solid = pool.world.solid.cpu().numpy()
    gx, gy, gz = solid.shape[1:]
    goals = pool.centre[torch.arange(n, device=device), (pool.n - 1).clamp(min=0)].cpu().numpy()
    lo = 11  # the slots follow 6 proprioceptive numbers and the 5 of the previous action
    obs_now = env.observe() if cfg.value_pick else None  # refreshed every tick for the critic

    def scorer_for(i, pl):
        def score(here, chains):
            m = len(chains)
            r, v, g = zip(*(pl.slots(ch) for ch in chains))
            st = SimpleNamespace(pos=env.state.pos[i:i + 1].expand(m, 3), yaw=env.state.yaw[i:i + 1].expand(m))
            rows = obs_now[i:i + 1].repeat(m, 1)
            rows[:, lo:lo + PLATFORM_SLOTS * 11] = env.obs_fn.slots(
                st, torch.as_tensor(r, device=device), torch.as_tensor(v, device=device),
                torch.as_tensor(g, device=device)).to(rows.dtype)
            rows[:, -1] = pl.progress
            with torch.no_grad():
                return int(model.value(model.norm(rows)).argmax())
        return score

    planners = []
    for i in range(n):
        s = solid[i]

        def is_solid(x, y, z, s=s):
            return 0 <= x < gx and 0 <= y < gy and 0 <= z < gz and bool(s[x, y, z])
        pl = GreedyPlanner(is_solid, tuple(goals[i]), cfg)
        if cfg.value_pick:
            pl.scorer = scorer_for(i, pl)
        pl.start(tuple(env.state.pos[i].tolist()))
        planners.append(pl)

    rects = torch.zeros(n, PLATFORM_SLOTS, 5, dtype=torch.int64)
    valid = torch.zeros(n, PLATFORM_SLOTS, dtype=torch.bool)
    is_goal = torch.zeros(n, PLATFORM_SLOTS, dtype=torch.bool)
    progress = torch.zeros(n, dtype=torch.float64)

    def write(i):
        r, v, g = planners[i].slots()
        rects[i] = torch.as_tensor(r)
        valid[i] = torch.as_tensor(v)
        is_goal[i] = torch.as_tensor(g)
        progress[i] = planners[i].progress
    for i in range(n):
        write(i)
    dev = lambda t: t.to(device)  # noqa: E731
    R, V, G, P = dev(rects), dev(valid), dev(is_goal), dev(progress)

    def observe():
        obs = env.observe()
        obs[:, lo:lo + PLATFORM_SLOTS * 11] = env.obs_fn.slots(env.state, R, V, G).to(obs.dtype)
        obs[:, -1] = P.to(obs.dtype)
        return obs

    obs = observe()
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    success = torch.zeros(n, dtype=torch.bool, device=device)
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    length = torch.zeros(n, dtype=torch.int64, device=device)
    prog = torch.zeros(n, dtype=torch.float64, device=device)
    end_pos = torch.zeros(n, 3, dtype=torch.float64, device=device)
    with torch.no_grad():
        for t in range(max_ticks):
            a, _, _ = model.act(obs, deterministic=True)
            before = env.state.pos.clone()
            _, _, _, _, _, info = env.step(a)
            first = info["done"] & ~finished
            end_pos = torch.where(first[:, None], before, end_pos)
            success |= first & info["success"]
            fell |= first & info["fell"]
            length = torch.where(first, info["ep_len"], length)
            prog = torch.where(first, info["progress_frac"], prog)
            finished |= info["done"]
            if bool(finished.all()):
                break
            # Re-plan where a player still in its first episode landed on a new platform.
            pos = env.state.pos.cpu().numpy()
            grounded = (env.state.on_ground & ~finished).cpu().numpy()
            if cfg.value_pick:
                obs_now = env.observe()
            changed = False
            for i in np.flatnonzero(grounded):
                if planners[i].update(tuple(pos[i].tolist()), True, t + 1):
                    write(i)
                    changed = True
            if changed:
                R, V, G, P = dev(rects), dev(valid), dev(is_goal), dev(progress)
            obs = observe()
    lv = pool.level.cpu().numpy()
    s, f, ln, pr = (x.cpu().numpy() for x in (success, fell, length, prog))
    per_level = {}
    for L in sorted(set(lv.tolist())):
        m = lv == L
        per_level[int(L)] = {"success": float(s[m].mean()), "fell": float(f[m].mean()), "progress": float(pr[m].mean()), "n": int(m.sum())}
    out = {"success": float(s.mean()), "per_level": per_level,
           "per_course": {"success": s.tolist(), "fell": f.tolist(), "ticks": ln.tolist(), "progress": pr.tolist(),
                          "end_pos": end_pos.cpu().numpy().round(3).tolist()}}
    out["failure_modes"] = failure_modes(pool, planners, s)
    if record:
        out["plans"] = [p.events for p in planners]
    return out


def failure_modes(pool: CoursePool, planners: list, success) -> dict:
    """Why planned runs failed: the plan left the course, saw nothing, or the jump itself failed."""
    modes = {"success": 0, "wrong_platform": 0, "nothing_in_view": 0, "motor": 0}
    for i, pl in enumerate(planners):
        if success[i]:
            modes["success"] += 1
            continue
        plats = [tuple(int(v) for v in r) for r in pool.plat[i, :int(pool.n[i])].tolist()]
        kind = "motor"
        for _, here, chain in pl.events:
            if not chain:
                kind = "nothing_in_view"
                break
            k = plats.index(tuple(here)) if tuple(here) in plats else -1
            if k < 0 or k + 1 >= len(plats) or tuple(chain[0]) != plats[k + 1]:
                kind = "wrong_platform"
                break
        modes[kind] += 1
    return modes


def main(argv=None):
    from .live import load_model
    from .train import OBSTACLE_VALIDATION_FAMILY, evaluate, held_out_courses, obstacle_courses
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="results/final_policy.pt")
    ap.add_argument("--obstacles", action="store_true", help="held-out obstacle courses (decoy pillars, beams)")
    ap.add_argument("--family", type=int, default=1, help="1 test, 2 validation (tune on 2, report 1)")
    ap.add_argument("--per-level", type=int, default=200)
    ap.add_argument("--out", default="")
    ap.add_argument("--record", action="store_true", help="keep every plan and outcome in --out")
    for k, v in asdict(PlannerConfig()).items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v) if not isinstance(v, bool) else lambda s: s.lower() == "true", default=v)
    a = ap.parse_args(argv)
    cfg = PlannerConfig(**{k: getattr(a, k) for k in asdict(PlannerConfig())})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(a.checkpoint), dev)
    levels = C.load_levels()
    if a.obstacles:
        courses = obstacle_courses(levels, a.per_level, 10) if a.family == 1 else \
            held_out_courses(levels, a.per_level, 10, obstacle_prob=0.5, family=OBSTACLE_VALIDATION_FAMILY)
    else:
        courses = held_out_courses(levels, a.per_level, 10, family=a.family)
    pool = CoursePool(courses, dev, C.GRID)
    res = {"checkpoint": a.checkpoint, "planner": asdict(cfg), "obstacles": a.obstacles, "family": a.family,
           "courses": len(courses)}
    for rule in ("local", "remaining"):
        base = evaluate(model, pool, dev, 200, deterministic=True, fall_rule=rule)
        plan = evaluate_planned(model, pool, dev, cfg, fall_rule=rule, record=a.record)
        res[rule] = {"given_route": {"success": base["success"], "per_level": base["per_level"]},
                     "planner_route": {k: plan[k] for k in ("success", "per_level", "failure_modes")}}
        if a.record:
            res[rule]["plans"] = plan["plans"]
            res[rule]["per_course"] = plan["per_course"]
        print(json.dumps({"fall_rule": rule, "given": round(base["success"], 4), "planned": round(plan["success"], 4),
                          "per_level_given": {k: round(v["success"], 3) for k, v in base["per_level"].items()},
                          "per_level_planned": {k: round(v["success"], 3) for k, v in plan["per_level"].items()},
                          "failure_modes": plan["failure_modes"]}), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()

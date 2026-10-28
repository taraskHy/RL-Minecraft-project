"""Batched parkour environment on the GPU.

Each of N environments holds one course from a ``CoursePool`` and one player.
The agent acts every 50 ms game tick, exactly as a Mineflayer bot can, so the
same policy runs unchanged on a real server (bridge/parkour_bot.js).

Observation (all horizontal quantities in the player's frame: forward f and
right r; see ``OBS_LAYOUT``):
  proprio   v.f, v.r, v.y, on_ground, jump_ticks/10, collided_horizontally
  action    the previous action: move, strafe, jump, sprint, yaw step
  platforms the platform last stood on and the next three, each with
            valid, is_goal, four top-face corners (f, r), top height above feet
  scan      98 points ahead and around the player: floor height and ceiling
  goal      goal centre (f, r, up) and the fraction of platforms reached

Reward per tick (weights in ``RewardConfig``):
  progress  decrease of the remaining path length to the goal (distance to
            the next platform's top centre plus the chain of centre distances
            after it)
  landing   +landing for each new platform reached (counted jumps)
  goal      +goal when standing on the last platform (episode ends)
  fall      -fall when the feet drop 1 block below every nearby platform
            top (episode ends: there is no way back up)
  time      -time each tick
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from . import physics
from .courses import Course, MAX_PLATFORMS, path_remaining, shape_geometry

ACTION_DIMS = (3, 3, 2, 2, 7)  # move (back/none/forward), strafe (left/none/right), jump, sprint, yaw step
YAW_STEPS_DEG = (-20.0, -6.0, -1.5, 0.0, 1.5, 6.0, 20.0)
SCAN_FORWARD = tuple(-1.0 + 0.5 * i for i in range(14))
SCAN_LATERAL = tuple(-1.5 + 0.5 * i for i in range(7))
SCAN_FLOOR_LEVELS = (1, 0, -1, -2, -3, -4)  # cells relative to the feet cell, searched top-down
SCAN_CEILING_LEVELS = (2, 3)
NO_FLOOR = -5.0
PLATFORM_SLOTS = 4
FALL_MARGIN = 1.0
STAND_EPS = 1e-6

OBS_LAYOUT = (("proprio", 6), ("action", 5), ("platforms", PLATFORM_SLOTS * 11),
              ("scan", len(SCAN_FORWARD) * len(SCAN_LATERAL) * 2), ("goal", 3), ("progress", 1))
OBS_DIM = sum(n for _, n in OBS_LAYOUT)
# Extended observation (EnvConfig.extended_obs), for policies that find the route themselves:
# a coarse, wide block scan (one sample per block, so no 1x1 platform falls between samples)
# and the bot's memory of its last landing spots. Appended after the standard observation.
SCAN_FAR_FORWARD = tuple(float(i) for i in range(-4, 11))
SCAN_FAR_LATERAL = tuple(float(i) for i in range(-6, 7))
MEMORY_SLOTS = 3
OBS_LAYOUT_EXT = OBS_LAYOUT + (("farscan", len(SCAN_FAR_FORWARD) * len(SCAN_FAR_LATERAL) * 2), ("memory", MEMORY_SLOTS * 4))
OBS_DIM_EXT = sum(n for _, n in OBS_LAYOUT_EXT)


@dataclass
class RewardConfig:
    progress: float = 1.0
    landing: float = 1.0
    goal: float = 10.0
    fall: float = 5.0
    time: float = 0.01
    # Penalty for stalling (no new platform for stall_ticks). When set, a stall ends the
    # episode as a failure instead of a bootstrapped time limit, so refusing a hard jump
    # is no better than trying it and falling. 0 keeps the original behaviour.
    stall: float = 0.0


@dataclass
class EnvConfig:
    num_envs: int = 4096
    stall_ticks: int = 200  # truncate after this long without reaching a new platform
    max_ticks: int = 3000
    reward: RewardConfig = field(default_factory=RewardConfig)
    # "local": a fall is being more than FALL_MARGIN below platforms cp-1..cp+2 (training and all
    # results). "remaining": below every platform from cp-1 to the goal, so dropping onto a later,
    # lower platform (a shortcut a route planner may take) is not a fall. Evaluation only.
    fall_rule: str = "local"
    extended_obs: bool = False  # add OBS_LAYOUT_EXT's far scan and landing memory


MAX_EXTRA = 256


_SHAPE_TOPS = {}


def shape_tops(device) -> torch.Tensor:
    """(S,) highest box top of every shape id (1 for the full cube), for the scan on shaped pools."""
    if device not in _SHAPE_TOPS:
        names, boxes, counts = physics.shape_table()
        tops = torch.tensor([float(boxes[i, :int(counts[i]), 4].max()) for i in range(len(names))], dtype=torch.float64)
        _SHAPE_TOPS[device] = tops.to(device)
    return _SHAPE_TOPS[device]


class CoursePool:
    """Courses packed into tensors on one device."""

    def __init__(self, courses: list[Course], device, grid, ice: bool | None = None, shapes: bool | None = None):
        """``ice``: keep a slipperiness grid for ice platforms (default: if any course has ice; a pool
        that will receive ice courses through ``replace`` needs True). ``shapes``: the same for shaped
        platforms (``Course.shapes``: a collision shape grid and shape-aware tops and footprints)."""
        self.courses = courses
        self.device = device
        p = len(courses)
        gx, gy, gz = grid
        solid = torch.zeros(p, gx, gy, gz, dtype=torch.bool)
        plat = torch.zeros(p, MAX_PLATFORMS, 5, dtype=torch.int64)
        rem = torch.zeros(p, MAX_PLATFORMS, dtype=torch.float64)
        n = torch.zeros(p, dtype=torch.int64)
        spawn = torch.zeros(p, 3, dtype=torch.float64)
        spawn_yaw = torch.zeros(p, dtype=torch.float64)
        level = torch.zeros(p, dtype=torch.int64)
        for i, c in enumerate(courses):
            b = torch.as_tensor(c.blocks())
            solid[i, b[:, 0], b[:, 1], b[:, 2]] = True
            k = c.n
            plat[i, :k] = torch.as_tensor(c.platforms)
            rem[i, :k] = torch.as_tensor(path_remaining(c))
            n[i] = k
            spawn[i] = torch.as_tensor(c.spawn)
            spawn_yaw[i] = c.spawn_yaw
            level[i] = c.level
        self.slip = None
        if any(c.ice or c.slime for c in courses) if ice is None else ice:
            self.slip = torch.full(solid.shape, physics.DEFAULT_SLIPPERINESS, dtype=torch.float64)
            for i, c in enumerate(courses):
                self._set_ice(self.slip, i, c)
            self.slip = self.slip.to(device)
        self.slime = None
        if any(c.slime for c in courses) if ice is None else ice:  # slime blocks bounce (and are slippery)
            self.slime = torch.zeros(solid.shape, dtype=torch.bool)
            for i, c in enumerate(courses):
                self._set_slime(self.slime, i, c)
            self.slime = self.slime.to(device)
        self.shape_grid = None
        if any(c.shapes for c in courses) if shapes is None else shapes:
            self.shape_grid = torch.zeros(solid.shape, dtype=torch.uint8)
            for i, c in enumerate(courses):
                self._set_shapes(self.shape_grid, i, c)
            self.shape_grid = self.shape_grid.to(device)
        self.world = physics.World(solid.to(device), self.slip, self.shape_grid, self.slime)
        self.plat = plat.to(device)
        self.rem = rem.to(device)
        self.n = n.to(device)
        self.spawn = spawn.to(device)
        self.spawn_yaw = spawn_yaw.to(device)
        self.level = level.to(device)
        pf = self.plat.to(torch.float64)
        self.centre = torch.stack(((pf[..., 0] + pf[..., 1] + 1) / 2, pf[..., 4] + 1, (pf[..., 2] + pf[..., 3] + 1) / 2), -1)
        self.top = pf[..., 4] + 1
        # Standing area of each platform: x0, x1, z0, z1 (the block faces; a shape's own footprint).
        self.foot = torch.stack((pf[..., 0], pf[..., 1] + 1, pf[..., 2], pf[..., 3] + 1), -1)
        for i, c in enumerate(courses):
            self._set_geometry(i, c)
        self.size = p
        # Obstacle cells (walls, beams, pillars) per course, padded; standing on one ends the episode.
        self.extra = torch.full((p, MAX_EXTRA, 3), -10_000, dtype=torch.int64, device=device)
        for i, c in enumerate(courses):
            self._set_extra(i, c)
        self.version = 0  # bumped whenever a course is replaced

    @staticmethod
    def _set_ice(slip: torch.Tensor, index: int, course: Course) -> None:
        slip[index] = physics.DEFAULT_SLIPPERINESS
        cells = torch.as_tensor(course.ice_cells(), device=slip.device)
        if len(cells):
            slip[index, cells[:, 0], cells[:, 1], cells[:, 2]] = physics.ICE_SLIPPERINESS
        cells = torch.as_tensor(course.slime_cells(), device=slip.device)
        if len(cells):
            slip[index, cells[:, 0], cells[:, 1], cells[:, 2]] = physics.SLIME_SLIPPERINESS

    @staticmethod
    def _set_slime(grid: torch.Tensor, index: int, course: Course) -> None:
        grid[index] = False
        cells = torch.as_tensor(course.slime_cells(), device=grid.device)
        if len(cells):
            grid[index, cells[:, 0], cells[:, 1], cells[:, 2]] = True

    @staticmethod
    def _set_shapes(grid: torch.Tensor, index: int, course: Course) -> None:
        grid[index] = 0
        for k, name in course.shapes.items():
            x0, x1, z0, z1, y = (int(v) for v in course.platforms[k])
            grid[index, x0:x1 + 1, y, z0:z1 + 1] = physics.shape_id(name)

    def _set_geometry(self, index: int, course: Course) -> None:
        """Tops, footprints and centres of shaped platforms (others keep the whole-block values)."""
        for k, name in course.shapes.items():
            top, x0, x1, z0, z1 = shape_geometry(name)
            q = [float(v) for v in course.platforms[k]]
            self.top[index, k] = q[4] + top
            self.foot[index, k] = torch.tensor((q[0] + x0, q[1] + x1, q[2] + z0, q[3] + z1), dtype=torch.float64)
            self.centre[index, k] = torch.tensor(course.centers()[k], dtype=torch.float64)

    def _set_extra(self, index: int, course: Course) -> None:
        e = np.asarray(course.extra, dtype=np.int64).reshape(-1, 3)
        if len(e) > MAX_EXTRA:
            raise ValueError(f"course has {len(e)} obstacle cells; MAX_EXTRA is {MAX_EXTRA}")
        self.extra[index] = -10_000
        if len(e):
            self.extra[index, :len(e)] = torch.as_tensor(e, device=self.device)

    def replace(self, index: int, course: Course) -> None:
        """Swap one course in place (pool refresh)."""
        self.courses[index] = course
        solid = self.world.solid
        solid[index] = False
        b = torch.as_tensor(course.blocks(), device=self.device)
        solid[index, b[:, 0], b[:, 1], b[:, 2]] = True
        k = course.n
        self.plat[index] = 0
        self.plat[index, :k] = torch.as_tensor(course.platforms, device=self.device)
        self.rem[index] = 0
        self.rem[index, :k] = torch.as_tensor(path_remaining(course), device=self.device)
        self.n[index] = k
        self.spawn[index] = torch.as_tensor(course.spawn, device=self.device)
        self.spawn_yaw[index] = course.spawn_yaw
        self.level[index] = course.level
        pf = self.plat[index].to(torch.float64)
        self.centre[index] = torch.stack(((pf[:, 0] + pf[:, 1] + 1) / 2, pf[:, 4] + 1, (pf[:, 2] + pf[:, 3] + 1) / 2), -1)
        self.top[index] = pf[:, 4] + 1
        self.foot[index] = torch.stack((pf[:, 0], pf[:, 1] + 1, pf[:, 2], pf[:, 3] + 1), -1)
        if self.shape_grid is not None:
            self._set_shapes(self.shape_grid, index, course)  # the world reads this tensor's memory
        elif course.shapes:
            raise ValueError("this pool has no shape grid; build it with shapes=True")
        self._set_geometry(index, course)
        self._set_extra(index, course)
        if self.slip is not None:
            self._set_ice(self.slip, index, course)  # the world reads this tensor's memory
        elif course.ice or course.slime:
            raise ValueError("this pool has no ice grid; build it with ice=True")
        if self.slime is not None:
            self._set_slime(self.slime, index, course)
        elif course.slime:
            raise ValueError("this pool has no slime grid; build it with ice=True")
        self.version += 1


def action_to_controls(action: torch.Tensor) -> torch.Tensor:
    move, strafe, jump, sprint, _ = action.unbind(-1)
    return torch.stack((move == 2, move == 0, strafe == 0, strafe == 2, jump == 1, sprint == 1), -1)


def standing_on(pool: CoursePool, s: physics.PlayerState, course: torch.Tensor) -> torch.Tensor:
    """(N, K) bool: player stands on platform k (feet on its top face, footprint overlapping)."""
    f = pool.foot[course]
    x, y, z = s.pos[:, 0:1], s.pos[:, 1:2], s.pos[:, 2:3]
    w = physics.HALF_WIDTH
    over = ((x + w > f[..., 0]) & (x - w < f[..., 1]) & (z + w > f[..., 2]) & (z - w < f[..., 3]))
    level = (y - pool.top[course]).abs() < STAND_EPS
    k = torch.arange(MAX_PLATFORMS, device=s.pos.device)[None, :]
    return over & level & s.on_ground[:, None] & (k < pool.n[course][:, None])


def on_obstacle(pool: CoursePool, s: physics.PlayerState, course: torch.Tensor) -> torch.Tensor:
    """(N,) bool: on the ground with the feet on top of an obstacle cell."""
    e = pool.extra[course].to(torch.float64)  # (N, E, 3)
    x, y, z = s.pos[:, 0:1], s.pos[:, 1:2], s.pos[:, 2:3]
    w = physics.HALF_WIDTH
    over = (x + w > e[..., 0]) & (x - w < e[..., 0] + 1) & (z + w > e[..., 2]) & (z - w < e[..., 2] + 1)
    level = (y - (e[..., 1] + 1)).abs() < STAND_EPS
    return s.on_ground & (over & level).any(dim=1)


class Observer:
    """Builds observations; shared by the environment and the parity tests."""

    def __init__(self, device, extended: bool = False):
        def grid(fwd, lat):
            a = torch.tensor(fwd, dtype=torch.float64, device=device)
            b = torch.tensor(lat, dtype=torch.float64, device=device)
            aa, bb = torch.meshgrid(a, b, indexing="ij")
            return aa.reshape(-1), bb.reshape(-1)
        self.scan_a, self.scan_b = grid(SCAN_FORWARD, SCAN_LATERAL)
        self.extended = extended
        self.far_a, self.far_b = grid(SCAN_FAR_FORWARD, SCAN_FAR_LATERAL)
        self.floor_levels = torch.tensor(SCAN_FLOOR_LEVELS, dtype=torch.int64, device=device)
        self.ceiling_levels = torch.tensor(SCAN_CEILING_LEVELS, dtype=torch.int64, device=device)
        self.yaw_steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64, device=device)
        self.device = device

    def __call__(self, pool: CoursePool, course: torch.Tensor, s: physics.PlayerState, cp: torch.Tensor,
                 last_action: torch.Tensor, memory=None) -> torch.Tensor:
        n = course.shape[0]
        pos = s.pos
        sy, cy = torch.sin(s.yaw), torch.cos(s.yaw)
        fx, fz = -sy, -cy
        rx, rz = cy, -sy

        def rel(x, z):
            dx = x - pos[:, 0:1]
            dz = z - pos[:, 2:3]
            return dx * fx[:, None] + dz * fz[:, None], dx * rx[:, None] + dz * rz[:, None]

        v = s.vel
        proprio = torch.stack((
            (v[:, 0] * fx + v[:, 2] * fz) / 0.4, (v[:, 0] * rx + v[:, 2] * rz) / 0.4, v[:, 1] / 0.5,
            s.on_ground.to(torch.float64), s.jump_ticks.to(torch.float64) / 10, s.collided_h.to(torch.float64)), -1)

        la = last_action.to(torch.float64)
        act = torch.stack((la[:, 0] - 1, la[:, 1] - 1, la[:, 2], la[:, 3], self.yaw_steps[last_action[:, 4]] / 20), -1)

        # Platform slots: the one last stood on and the next three.
        j = cp[:, None] + torch.arange(PLATFORM_SLOTS, device=self.device)[None, :]
        nplat = pool.n[course]
        valid = j < nplat[:, None]
        jc = j.clamp(max=MAX_PLATFORMS - 1)
        p = pool.plat[course[:, None], jc]  # (N, S, 5)
        geometry = None
        if pool.shape_grid is not None:  # shaped platforms: their own footprints and tops
            geometry = (pool.foot[course[:, None], jc], pool.top[course[:, None], jc])
        slots = self.slots(s, p, valid, j == nplat[:, None] - 1, geometry)

        scan = self.scan(pool, course, s, self.scan_a, self.scan_b)

        gidx = (nplat - 1).clamp(min=0)
        g = pool.centre[course, gidx]
        gf, gr = rel(g[:, 0:1], g[:, 2:3])
        goal = torch.stack((gf[:, 0] / 20, gr[:, 0] / 20, (g[:, 1] - pos[:, 1]) / 10,
                            cp.to(torch.float64) / (nplat - 1).clamp(min=1).to(torch.float64)), -1)
        parts = [proprio, act, slots, scan, goal]
        if self.extended:
            far = self.scan(pool, course, s, self.far_a, self.far_b)
            mem, mvalid = memory
            mf, mr = rel(mem[..., 0], mem[..., 2])
            vf = mvalid.to(torch.float64)
            m = torch.stack((vf, mf / 4.0 * vf, mr / 4.0 * vf, (mem[..., 1] - pos[:, 1:2]) / 2.0 * vf), -1).reshape(n, -1)
            parts += [far, m]
        return torch.cat(parts, -1).to(torch.float32)

    def scan(self, pool: CoursePool, course: torch.Tensor, s: physics.PlayerState, a: torch.Tensor, b: torch.Tensor):
        """Height (top of the highest block within the floor levels) and ceiling at points (a ahead, b right)."""
        n = course.shape[0]
        pos = s.pos
        sy, cy = torch.sin(s.yaw), torch.cos(s.yaw)
        fx, fz = -sy, -cy
        rx, rz = cy, -sy
        px = pos[:, 0:1] + a * fx[:, None] + b * rx[:, None]
        pz = pos[:, 2:3] + a * fz[:, None] + b * rz[:, None]
        ix = torch.floor(px).to(torch.int64)
        iz = torch.floor(pz).to(torch.int64)
        fy = torch.floor(pos[:, 1] + STAND_EPS).to(torch.int64)
        levels = fy[:, None, None] + self.floor_levels[None, None, :]
        solid = pool.world.lookup(course[:, None, None], ix[..., None], levels, iz[..., None])  # (N, points, 6)
        found = solid.any(-1)
        first = torch.argmax(solid.to(torch.int8), dim=-1)  # first True in top-down order
        top_cell = fy[:, None] + self.floor_levels[first]
        if pool.world.shape_ids is None:
            h = torch.where(found, (top_cell + 1).to(torch.float64) - pos[:, 1:2], torch.full_like(px, NO_FLOOR))
        else:  # the top of the found block's shape (1 for a full block)
            off = shape_tops(pool.world.device)[pool.world.shapes(course[:, None], ix, top_cell, iz)]
            h = torch.where(found, (top_cell.to(torch.float64) + off) - pos[:, 1:2], torch.full_like(px, NO_FLOOR))
        clevels = fy[:, None, None] + self.ceiling_levels[None, None, :]
        ceiling = pool.world.lookup(course[:, None, None], ix[..., None], clevels, iz[..., None]).any(-1)
        return torch.stack((h / 3.0, ceiling.to(torch.float64)), -1).reshape(n, -1)

    def slots(self, s: physics.PlayerState, p: torch.Tensor, valid: torch.Tensor, is_goal: torch.Tensor,
              geometry=None) -> torch.Tensor:
        """(N, S * 11) slot features of platforms ``p`` (N, S, 5); also used by the route planner.

        ``geometry`` (footprints (N, S, 4), tops (N, S)) replaces the whole-block faces of ``p``
        (shaped platforms)."""
        n = p.shape[0]
        pos = s.pos
        sy, cy = torch.sin(s.yaw), torch.cos(s.yaw)
        fx, fz = -sy, -cy
        rx, rz = cy, -sy
        p = p.to(torch.float64)
        x0, x1, z0, z1 = p[..., 0], p[..., 1] + 1, p[..., 2], p[..., 3] + 1
        if geometry is not None:
            x0, x1, z0, z1 = geometry[0].unbind(-1)
        cxs = torch.stack((x0, x1, x1, x0), -1).reshape(n, -1)
        czs = torch.stack((z0, z0, z1, z1), -1).reshape(n, -1)
        dx = cxs - pos[:, 0:1]
        dz = czs - pos[:, 2:3]
        cf, cr = dx * fx[:, None] + dz * fz[:, None], dx * rx[:, None] + dz * rz[:, None]
        corners = torch.stack((cf, cr), -1).reshape(n, PLATFORM_SLOTS, 8) / 4.0
        dtop = ((p[..., 4] + 1) - pos[:, 1:2]) / 2.0 if geometry is None else (geometry[1] - pos[:, 1:2]) / 2.0
        vf = valid.to(torch.float64)
        return torch.cat((vf[..., None], (is_goal & valid).to(torch.float64)[..., None],
                          corners.clamp(-10, 10) * vf[..., None], (dtop * vf)[..., None]), -1).reshape(n, -1)


class LevelSampler:
    """Curriculum over difficulty levels.

    Levels up to the frontier are reviewed with weight 1; the level just above
    it gets weight ``edge_weight``. The frontier advances when the success rate
    on that edge level, over its last ``window`` episodes or more, reaches
    ``advance_at``. Counts arrive once per training iteration.
    """

    def __init__(self, n_levels: int, start: int = 0, edge_weight: float = 3.0, advance_at: float = 0.6,
                 window: int = 2000, fixed: list[float] | None = None):
        self.n_levels = n_levels
        self.frontier = start
        self.edge_weight = edge_weight
        self.advance_at = advance_at
        self.window = window
        self.fixed = fixed
        self.recent = [[] for _ in range(n_levels)]  # (episodes, successes) per update

    def probs(self) -> np.ndarray:
        if self.fixed is not None:
            w = np.array(self.fixed, dtype=np.float64)
        else:
            w = np.zeros(self.n_levels)
            w[: self.frontier + 1] = 1.0
            if self.frontier + 1 < self.n_levels:
                w[self.frontier + 1] = self.edge_weight
        return w / w.sum()

    def _rate(self, level: int):
        r = self.recent[level]
        n = sum(e for e, _ in r)
        return (sum(k for _, k in r) / n, n) if n else (None, 0)

    def update(self, episodes: np.ndarray, successes: np.ndarray) -> None:
        for level in range(self.n_levels):
            if episodes[level] > 0:
                r = self.recent[level]
                r.append((float(episodes[level]), float(successes[level])))
                while len(r) > 1 and sum(e for e, _ in r[1:]) >= self.window:
                    r.pop(0)
        edge = self.frontier + 1
        if self.fixed is None and edge < self.n_levels:
            rate, n = self._rate(edge)
            if n >= self.window and rate >= self.advance_at:
                self.frontier = edge

    def success_rates(self) -> list:
        return [self._rate(level)[0] for level in range(self.n_levels)]

    def state_dict(self) -> dict:
        return {"frontier": self.frontier, "recent": self.recent}

    def load_state_dict(self, d: dict) -> None:
        self.frontier = d["frontier"]
        self.recent = [list(map(tuple, r)) for r in d["recent"]]


# Per-level statistics the environment accumulates on the GPU between reads.
STATS = ("episodes", "success", "fell", "truncated", "ticks", "return", "progress")


class ParkourEnv:
    """Batched environment whose step makes no host synchronisation.

    Finished environments restart on a new course inside ``step``; the
    observation returned for them is the first of the new episode, and
    ``final_obs`` holds the last observation of the old one (for bootstrapping
    time-limit truncations).
    """

    def __init__(self, pool: CoursePool, config: EnvConfig, sampler: LevelSampler | None = None, seed: int = 0):
        self.pool = pool
        self.cfg = config
        self.device = pool.device
        self.n = config.num_envs
        self.sampler = sampler
        torch.manual_seed(seed)
        self.obs_fn = Observer(self.device, extended=config.extended_obs)
        self.yaw_steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64, device=self.device) * (math.pi / 180)
        self.no_action = torch.tensor([1, 1, 0, 0, 3], device=self.device)
        self.k_range = torch.arange(MAX_PLATFORMS, device=self.device)
        self.fall_range = torch.arange(-1, 3 if config.fall_rule == "local" else MAX_PLATFORMS, device=self.device)
        self.course = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.state = physics.PlayerState.at(torch.zeros(self.n, 3, dtype=torch.float64, device=self.device),
                                            torch.zeros(self.n, dtype=torch.float64, device=self.device))
        self.cp = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.ticks = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.stall = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.last_action = self.no_action.repeat(self.n, 1)
        self.dist = torch.zeros(self.n, dtype=torch.float64, device=self.device)
        self.ep_return = torch.zeros(self.n, dtype=torch.float64, device=self.device)
        # Landing memory (extended observation): the last MEMORY_SLOTS spots the player landed on,
        # newest first; a landing counts when it is at least 1 block from the newest one.
        self.mem = torch.zeros(self.n, MEMORY_SLOTS, 3, dtype=torch.float64, device=self.device)
        self.mem_valid = torch.zeros(self.n, MEMORY_SLOTS, dtype=torch.bool, device=self.device)
        self.n_levels = max(int(pool.level.max().item()) + 1, sampler.n_levels if sampler else 1)
        self.stats = torch.zeros(len(STATS), self.n_levels, dtype=torch.float64, device=self.device)
        self.level_probs = torch.ones(self.n_levels, dtype=torch.float32, device=self.device)
        self.rebuild_level_table()
        if sampler is not None:
            self.set_level_probs(sampler.probs())
        self.reset_all()

    # Course selection -------------------------------------------------------------------------------
    def rebuild_level_table(self) -> None:
        """Index courses by level; call after the pool changes."""
        lv = self.pool.level.cpu().numpy()
        groups = [np.nonzero(lv == level)[0] for level in range(self.n_levels)]
        m = max(1, max(len(g) for g in groups))
        table = np.zeros((self.n_levels, m), dtype=np.int64)
        for level, g in enumerate(groups):
            table[level, :len(g)] = g
        self.level_table = torch.as_tensor(table, device=self.device)
        self.level_count = torch.as_tensor([len(g) for g in groups], device=self.device)
        self.level_avail = self.level_count > 0

    def set_level_probs(self, probs) -> None:
        p = torch.as_tensor(np.asarray(probs, dtype=np.float32)[: self.n_levels], device=self.device)
        p = p * self.level_avail
        self.level_probs.copy_(p / p.sum())

    def _sample_courses(self, k: int) -> torch.Tensor:
        if self.sampler is None:
            return torch.randint(0, self.pool.size, (k,), device=self.device)
        lev = torch.multinomial(self.level_probs, k, replacement=True)
        cnt = self.level_count[lev]
        j = torch.minimum((torch.rand(k, device=self.device) * cnt).to(torch.int64), (cnt - 1).clamp(min=0))
        return self.level_table[lev, j]

    def _apply_reset(self, mask: torch.Tensor, courses: torch.Tensor) -> None:
        """Restart the rows in ``mask`` on ``courses`` (one entry per row)."""
        self.course = torch.where(mask, courses, self.course)
        fresh = physics.PlayerState.at(self.pool.spawn[self.course], self.pool.spawn_yaw[self.course])
        self.state.assign(mask, fresh)
        zero = torch.zeros_like(self.cp)
        self.cp = torch.where(mask, zero, self.cp)
        self.ticks = torch.where(mask, zero, self.ticks)
        self.stall = torch.where(mask, zero, self.stall)
        self.last_action = torch.where(mask[:, None], self.no_action[None, :], self.last_action)
        self.ep_return = torch.where(mask, torch.zeros_like(self.ep_return), self.ep_return)
        self.dist = torch.where(mask, self._distance(self.state.pos, self.course, self.cp), self.dist)
        self.reset_memory(mask)

    def reset_memory(self, mask: torch.Tensor) -> None:
        """Memory of the rows in ``mask``: only the current position (the start block)."""
        fresh = torch.zeros_like(self.mem)
        fresh[:, 0] = self.state.pos
        valid = torch.zeros_like(self.mem_valid)
        valid[:, 0] = True
        self.mem = torch.where(mask[:, None, None], fresh, self.mem)
        self.mem_valid = torch.where(mask[:, None], valid, self.mem_valid)

    def _remember_landings(self, prev_ground: torch.Tensor) -> None:
        pos = self.state.pos
        d = pos[:, [0, 2]] - self.mem[:, 0][:, [0, 2]]
        far = (d * d).sum(-1) >= 1.0
        store = self.state.on_ground & ~prev_ground & (far | ~self.mem_valid[:, 0])
        shifted = torch.cat((pos[:, None, :], self.mem[:, :-1]), 1)
        svalid = torch.cat((torch.ones_like(self.mem_valid[:, :1]), self.mem_valid[:, :-1]), 1)
        self.mem = torch.where(store[:, None, None], shifted, self.mem)
        self.mem_valid = torch.where(store[:, None], svalid, self.mem_valid)

    def reset_rows(self, mask: torch.Tensor) -> None:
        self._apply_reset(mask, self._sample_courses(self.n))

    def reset_all(self, courses: torch.Tensor | None = None) -> torch.Tensor:
        c = courses if courses is not None else self._sample_courses(self.n)
        self._apply_reset(torch.ones(self.n, dtype=torch.bool, device=self.device), c)
        return self.observe()

    def pop_stats(self) -> np.ndarray:
        """Per-level episode statistics since the last call, shape (len(STATS), levels)."""
        out = self.stats.cpu().numpy().copy()
        self.stats.zero_()
        return out

    # Helpers ------------------------------------------------------------------------------------------
    def _distance(self, pos, course, cp):
        last = self.pool.n[course] - 1
        nxt = torch.minimum((cp + 1).clamp(max=MAX_PLATFORMS - 1), last)
        c = self.pool.centre[course, nxt]
        return torch.linalg.norm(c - pos, dim=-1) + self.pool.rem[course, nxt]

    def standing_on(self, s: physics.PlayerState, course: torch.Tensor) -> torch.Tensor:
        return standing_on(self.pool, s, course)

    def observe(self) -> torch.Tensor:
        return self.obs_fn(self.pool, self.course, self.state, self.cp, self.last_action, (self.mem, self.mem_valid))

    # Step ---------------------------------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        cfg = self.cfg.reward
        s = self.state
        yaw = s.yaw + self.yaw_steps[action[:, 4]]
        s.yaw = torch.remainder(yaw + math.pi, 2 * math.pi) - math.pi
        prev_ground = s.on_ground
        new = physics.simulate(self.pool.world, self.course, s, action_to_controls(action))
        self.state = new
        self.last_action = action
        self.ticks = self.ticks + 1
        if self.cfg.extended_obs:
            self._remember_landings(prev_ground)

        stand = self.standing_on(new, self.course)
        k = self.k_range[None, :]
        ahead = stand & (k > self.cp[:, None])
        reached = torch.where(ahead, k, torch.full_like(k, -1)).amax(dim=1)
        new_cp = torch.maximum(self.cp, reached)
        gained = (new_cp - self.cp).to(torch.float64)
        last = self.pool.n[self.course] - 1
        success = new_cp >= last

        dist = self._distance(new.pos, self.course, new_cp)
        progress = self.dist - dist
        # Fall: feet more than FALL_MARGIN below every platform top from cp-1 to cp+2.
        j = new_cp[:, None] + self.fall_range[None, :]
        jv = (j >= 0) & (j <= last[:, None])
        tops = self.pool.top[self.course[:, None], j.clamp(0, MAX_PLATFORMS - 1)]
        tops = torch.where(jv, tops, torch.full_like(tops, math.inf))
        fell = (new.pos[:, 1] < tops.amin(dim=1) - FALL_MARGIN) & ~success
        # Standing on an obstacle (wall, beam, pillar) is not parkour: it ends the episode like a fall.
        fell = fell | (on_obstacle(self.pool, new, self.course) & ~success)

        self.stall = torch.where(gained > 0, torch.zeros_like(self.stall), self.stall + 1)
        truncated = ((self.stall >= self.cfg.stall_ticks) | (self.ticks >= self.cfg.max_ticks)) & ~success & ~fell
        stalled = truncated & (self.stall >= self.cfg.stall_ticks) & (cfg.stall > 0)
        terminal = success | fell | stalled
        reward = (cfg.progress * progress + cfg.landing * gained + cfg.goal * success.to(torch.float64)
                  - cfg.fall * fell.to(torch.float64) - cfg.stall * stalled.to(torch.float64) - cfg.time)
        self.cp = new_cp
        self.dist = dist
        self.ep_return = self.ep_return + reward

        done = terminal | truncated
        final_obs = self.observe()
        level = self.pool.level[self.course]
        frac = new_cp.to(torch.float64) / last.clamp(min=1).to(torch.float64)
        d = done.to(torch.float64)
        for i, v in enumerate((d, (success & done).to(torch.float64), fell.to(torch.float64),
                               truncated.to(torch.float64), d * self.ticks, d * self.ep_return, d * frac)):
            self.stats[i].index_add_(0, level, v)
        info = {"done": done, "success": success & done, "fell": fell, "truncated": truncated,
                "ep_len": self.ticks, "ep_return": self.ep_return, "level": level, "progress_frac": frac,
                "course": self.course}
        self._apply_reset(done, self._sample_courses(self.n))
        obs = self.observe()
        return obs, reward.to(torch.float32), terminal, truncated & ~stalled, final_obs, info

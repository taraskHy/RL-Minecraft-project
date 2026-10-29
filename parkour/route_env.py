"""Learned routing: a route policy picks the next platform, the frozen final policy jumps to it.

The mission (docs/MISSION_ROUTING.md): without the course list, choose the platforms that fill the
jumper's slots from what the bot can see, and beat the greedy planner (parkour/planner.py).

Perception is the planner's own, done on the GPU for every row at once. The generator's course
platforms are exactly what ``planner.perceive`` finds once the head-hitter rule drops the beams
(tests/test_route.py proves it on the validation courses), so each decision filters the course's
platforms by the planner's 25 x 25 window around the player's block (10 rows below the feet to 4
above), clips them to it, and applies the planner's jump limits (height change -3 to +1, air gap up
to ``MAX_GAP[dy] + slack``, a clear flight path), its visited and passed marks and its goal rule.

The route policy sees a set of tokens, one per platform in view, sorted by a geometric key (top
height, then x0, then z0: the planner's scan order), never by the course's order:

* geometry in a frame at the platform it plans from (``frm``), turned towards the goal: the centre,
  the four top-face corners, size, height change and air gap;
* the greedy planner's view of the platform: its cost, its rank among the selectable ones and
  whether it is the greedy choice;
* its distance to the goal, the goal flag, whether it reaches the window's edge, how many jumps lead
  on from it (within the limits, path not checked);
* role flags: selectable, the platform planned from, the one stood on, visited, already in the chain;
* global numbers repeated on every token: distance and height to the goal, the plan depth (0 for
  the decision, 1 and 2 for the slots after it), the number of selectable and visited platforms, the
  player's position and velocity at the decision (in the goal frame of the platform stood on), ticks
  since the last landing on a new platform and since the last decision.

The policy scores the selectable tokens (a small transformer over the set, ``route_policy``); the
chosen platform fills slot 1. Slots 2 and 3 are filled from it, either greedily (``fill="greedy"``,
the planner's own rule) or by the route policy's argmax with the chosen platform as ``frm``
(``fill="policy"``). The episode is judged by ``ParkourEnv`` on the true course, unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import physics
from .courses import MAX_PLATFORMS
from .env import PLATFORM_SLOTS, STAND_EPS, CoursePool, EnvConfig, ParkourEnv
from .planner import MAX_GAP, PlannerConfig

K = MAX_PLATFORMS
LOOK = PLATFORM_SLOTS - 1
OBS_SLOTS = slice(11, 11 + PLATFORM_SLOTS * 11)  # the jumper's platform slots in its observation
PROGRESS = 254  # the jumper's progress input (the last of the 255 numbers)
TOKEN_DIM = 45
NEG = -1e9


@dataclass
class RouteConfig:
    fill: str = "greedy"  # how slots 2 and 3 are filled: "greedy" (the planner's rule) or "policy"
    mark_passed: bool = True  # landing beyond the plan's next platform marks the ones skipped as passed


def standable_platforms(solid: np.ndarray, slab_only: bool = True) -> list:
    """Every platform of one course's block grid as planner.perceive would find it with an unlimited
    window: rects (x0, x1, z0, z1, y) of the side-connected standable cells, in scan order, and
    whether each fills its rect."""
    gx, gy, gz = solid.shape
    pad = np.zeros((gx, gy + 3, gz), dtype=bool)
    pad[:, 1:gy + 1] = solid
    st = pad[:, 1:gy + 1] & ~pad[:, 2:gy + 2] & ~pad[:, 3:gy + 3]
    if slab_only:
        st &= ~pad[:, 0:gy]
    seen = np.zeros_like(st)
    out = []
    for y in range(gy):
        xs, zs = np.nonzero(st[:, y, :])
        for x, z in zip(xs.tolist(), zs.tolist()):
            if seen[x, y, z]:
                continue
            cells = [(x, z)]
            seen[x, y, z] = True
            k = 0
            while k < len(cells):
                a, b = cells[k]
                k += 1
                for nx, nz in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                    if 0 <= nx < gx and 0 <= nz < gz and st[nx, y, nz] and not seen[nx, y, nz]:
                        seen[nx, y, nz] = True
                        cells.append((nx, nz))
            cx = [c[0] for c in cells]
            cz = [c[1] for c in cells]
            r = (min(cx), max(cx), min(cz), max(cz), y)
            out.append((r, len(cells) == (r[1] - r[0] + 1) * (r[3] - r[2] + 1)))
    return out


class RouteTables:
    """Per course of a pool: every standable platform (the course's own and obstacle tops the planner
    would see), padded to ``KT``, and the course index of each (-1: not a course platform)."""

    def __init__(self, pool: CoursePool, kt: int = K + 8):
        self.kt = kt
        p = pool.size
        self.rect = torch.zeros(p, kt, 5, dtype=torch.int64)
        self.n = torch.zeros(p, dtype=torch.int64)
        self.cidx = torch.full((p, kt), -1, dtype=torch.int64)
        self.irregular = 0  # platforms that do not fill their rect (clipping could split them)
        for i in range(p):
            self._set(pool, i)
        self.rect, self.n, self.cidx = (t.to(pool.device) for t in (self.rect, self.n, self.cidx))

    def _set(self, pool: CoursePool, i: int) -> None:
        plats = standable_platforms(pool.world.solid[i].cpu().numpy())
        if len(plats) > self.kt:
            raise ValueError(f"course {i} has {len(plats)} standable platforms; KT is {self.kt}")
        course = {tuple(int(v) for v in r): k for k, r in enumerate(pool.plat[i, :int(pool.n[i])].tolist())}
        self.rect[i] = 0
        self.cidx[i] = -1
        for t, (r, full) in enumerate(plats):
            self.rect[i, t] = torch.as_tensor(r)
            self.cidx[i, t] = course.get(r, -1)
            self.irregular += not full
        self.n[i] = len(plats)
        missing = len(course) - int((self.cidx[i] >= 0).sum())
        if missing:
            raise ValueError(f"course {i}: {missing} course platforms are not standable platforms")

    def replace(self, pool: CoursePool, i: int) -> None:
        self.replace_many(pool, [i])

    def replace_many(self, pool: CoursePool, idx: list) -> None:
        """Rebuild the rows of courses swapped in by a pool refresh."""
        dev = self.rect.device
        self.rect, self.n, self.cidx = self.rect.cpu(), self.n.cpu(), self.cidx.cpu()
        for i in idx:
            self._set(pool, i)
        self.rect, self.n, self.cidx = self.rect.to(dev), self.n.to(dev), self.cidx.to(dev)


def _max_gap_table(cfg: PlannerConfig, device) -> torch.Tensor:
    """Largest air gap for dy = -3..1 (index dy + 3)."""
    return torch.tensor([MAX_GAP[d] + cfg.slack if cfg.limits == "table" else cfg.max_gap for d in range(-3, 2)],
                        dtype=torch.float64, device=device)


def centres(r: torch.Tensor) -> torch.Tensor:
    """(..., 3) top-face centres of rects (..., 5) (x0, x1, z0, z1, y), as Platform.centre."""
    r = r.to(torch.float64)
    return torch.stack(((r[..., 0] + r[..., 1] + 1) / 2, r[..., 4] + 1, (r[..., 2] + r[..., 3] + 1) / 2), -1)


def air_gap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Horizontal air gap between rects a and b (broadcast), as planner.options."""
    zero = torch.zeros((), dtype=torch.int64, device=a.device)
    dx = torch.maximum(torch.maximum(zero, b[..., 0] - (a[..., 1] + 1)), a[..., 0] - (b[..., 1] + 1))
    dz = torch.maximum(torch.maximum(zero, b[..., 2] - (a[..., 3] + 1)), a[..., 2] - (b[..., 3] + 1))
    dx, dz = dx.to(torch.float64), dz.to(torch.float64)
    return torch.sqrt(dx * dx + dz * dz)


def contains(r: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Rect r (..., 5) contains cell (..., 3)."""
    return ((cell[..., 1] == r[..., 4]) & (r[..., 0] <= cell[..., 0]) & (cell[..., 0] <= r[..., 1])
            & (r[..., 2] <= cell[..., 2]) & (cell[..., 2] <= r[..., 3]))


def path_clear(world: physics.World, course: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
               check: torch.Tensor):
    """planner.path_clear for rect a (M, 1, 5) to every rect b (M, K, 5), where ``check`` (M, K); and
    whether a block sits 1 or 2 rows above that body-height band on the path (a head hitter)."""
    out = torch.ones(check.shape, dtype=torch.bool, device=check.device)
    head = torch.zeros_like(out)
    if not bool(check.any()):
        return out, head
    ac, bc = centres(a), centres(b)
    ddx, ddz = bc[..., 0] - ac[..., 0], bc[..., 2] - ac[..., 2]
    n = torch.clamp(torch.ceil(torch.sqrt(ddx * ddx + ddz * ddz) / 0.25), min=1).to(torch.int64)
    n = torch.where(check, n, torch.ones_like(n))
    nmax = int(n.max())
    if nmax <= 1:
        return out, head
    k = torch.arange(1, nmax, device=check.device, dtype=torch.int64)
    t = k.to(torch.float64)[None, None, :] / n.to(torch.float64)[..., None]  # (M, K, P)
    x = torch.floor(ac[..., 0:1] + t * (bc[..., 0:1] - ac[..., 0:1])).to(torch.int64)
    z = torch.floor(ac[..., 2:3] + t * (bc[..., 2:3] - ac[..., 2:3])).to(torch.int64)
    live = k[None, None, :] < n[..., None]

    def inside(r):
        return (r[..., 0:1] <= x) & (x <= r[..., 1:2]) & (r[..., 2:3] <= z) & (z <= r[..., 3:4])
    live = live & ~inside(a) & ~inside(b)
    y0 = torch.minimum(a[..., 4], b[..., 4]) + 1
    y1 = torch.maximum(a[..., 4], b[..., 4]) + 2
    blocked = torch.zeros_like(live)
    cidx = course[:, None, None].expand_as(x)
    for j in range(5):
        y = (y0 + j)[..., None].expand_as(x)
        ok = live & (y <= y1[..., None])
        blocked |= ok & world.lookup(cidx, x, y, z)
    above = torch.zeros_like(live)
    for j in (1, 2):
        above |= live & world.lookup(cidx, x, (y1 + j)[..., None].expand_as(x), z)
    return out & ~(blocked.any(-1) & check), above.any(-1) & check


class Scene:
    """The platforms in view of M deciding rows, in the planner's scan order."""

    def __init__(self, pool: CoursePool, tables: RouteTables, course: torch.Tensor, pos: torch.Tensor,
                 cell: torch.Tensor, cfg: PlannerConfig):
        dev = pool.device
        self.pool, self.course, self.cfg = pool, course, cfg
        m = course.shape[0]
        kt = self.kt = tables.kt
        plat = tables.rect[course]  # (M, KT, 5)
        n = tables.n[course]
        cx = torch.floor(pos[:, 0]).to(torch.int64)
        cz = torch.floor(pos[:, 2]).to(torch.int64)
        fy = cell[:, 1] + 1
        r = cfg.radius
        rect = plat.clone()
        rect[..., 0] = torch.maximum(plat[..., 0], (cx - r)[:, None])
        rect[..., 1] = torch.minimum(plat[..., 1], (cx + r)[:, None])
        rect[..., 2] = torch.maximum(plat[..., 2], (cz - r)[:, None])
        rect[..., 3] = torch.minimum(plat[..., 3], (cz + r)[:, None])
        kk = torch.arange(kt, device=dev)[None, :]
        seen = ((kk < n[:, None]) & (rect[..., 0] <= rect[..., 1]) & (rect[..., 2] <= rect[..., 3])
                & (plat[..., 4] >= (fy - cfg.below)[:, None]) & (plat[..., 4] <= (fy + cfg.above)[:, None]))
        inview = seen
        if cfg.head_hitters:  # planner.head_hitter: a slab 4 above a platform in view that it touches
            a, b = rect[:, :, None, :], rect[:, None, :, :]
            zero = torch.zeros((), dtype=torch.int64, device=dev)
            tx = torch.maximum(torch.maximum(zero, a[..., 0] - (b[..., 1] + 1)), b[..., 0] - (a[..., 1] + 1)) == 0
            tz = torch.maximum(torch.maximum(zero, a[..., 2] - (b[..., 3] + 1)), b[..., 2] - (a[..., 3] + 1)) == 0
            hh = ((a[..., 4] - b[..., 4] == 4) & tx & tz & seen[:, None, :]).any(-1)
            inview = seen & ~hh
        # Observable clipping: the rect reaches the window's edge (the bot cannot see past it).
        edge = inview & ((rect[..., 0] == (cx - r)[:, None]) | (rect[..., 1] == (cx + r)[:, None])
                         | (rect[..., 2] == (cz - r)[:, None]) | (rect[..., 3] == (cz + r)[:, None]))
        # Scan order (y, x0, z0); platforms out of view last. The sort never sees the course's order.
        big = 1 << 40
        key = torch.where(inview, (rect[..., 4] * 256 + rect[..., 0]) * 256 + rect[..., 2], torch.full_like(rect[..., 0], big))
        order = torch.argsort(key, dim=1, stable=True)
        self.order = order  # (M, KT) table index of each token
        g = lambda t: torch.gather(t, 1, order if t.dim() == 2 else order[..., None].expand(-1, -1, t.shape[-1]))  # noqa: E731
        self.rect, self.inview, self.edge = g(rect), g(inview), g(edge)
        self.cidx = torch.where(self.inview, g(tables.cidx[course]), torch.full_like(order, -1))  # privileged
        last = (pool.n[course] - 1).clamp(min=0)
        self.goal = pool.centre[course, last]  # (M, 3)
        gc = torch.stack((torch.floor(self.goal[:, 0]), torch.round(self.goal[:, 1]) - 1, torch.floor(self.goal[:, 2])), -1)
        self.is_goal = contains(self.rect, gc.to(torch.int64)[:, None, :]) & self.inview
        self.m = m
        self.gap_table = _max_gap_table(cfg, dev)

    def token_of(self, k: torch.Tensor) -> torch.Tensor:
        """Token position of course platform index k (M,) (-1: none or out of view)."""
        hit = (self.cidx == k[:, None]) & (k[:, None] >= 0)
        return torch.where(hit.any(1), hit.to(torch.int8).argmax(1), torch.full_like(k, -1))

    def options(self, frm: torch.Tensor, excluded: torch.Tensor):
        """Selectable tokens from token ``frm`` (M,) and the greedy cost of each (planner.options)."""
        cfg = self.cfg
        ar = torch.arange(self.m, device=frm.device)
        f = self.rect[ar, frm][:, None, :]  # (M, 1, 5)
        dy = self.rect[..., 4] - f[..., 4]
        d = air_gap(f, self.rect)
        lim = self.gap_table[(dy + 3).clamp(0, 4)]
        ok = self.inview & ~excluded & (dy >= cfg.dy_min) & (dy <= cfg.dy_max) & (d <= lim)
        ok[ar, frm] = False
        head = torch.zeros_like(ok)
        if cfg.clear_path:
            clear, head = path_clear(self.pool.world, self.course, f, self.rect, ok)
            ok = ok & clear
        fc, pc = centres(f), centres(self.rect)
        gx, gz = self.goal[:, None, 0] - fc[..., 0], self.goal[:, None, 2] - fc[..., 2]
        vx, vz = pc[..., 0] - fc[..., 0], pc[..., 2] - fc[..., 2]
        gn, vn = torch.sqrt(gx * gx + gz * gz), torch.sqrt(vx * vx + vz * vz)
        cos = torch.where((vn > 1e-9) & (gn > 1e-9), (vx * gx + vz * gz) / (vn * gn), torch.zeros_like(vn))
        cost = d - cfg.align * cos
        self.head = head & ok
        return ok, cost, d, dy, cos

    @staticmethod
    def greedy_pick(ok: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
        """First token (scan order) within 1e-9 of the cheapest selectable one; -1 if none."""
        c = torch.where(ok, cost, torch.full_like(cost, math.inf))
        best = c.min(1).values
        hit = ok & (c <= best[:, None] + 1e-9)
        return torch.where(ok.any(1), hit.to(torch.int8).argmax(1), torch.full_like(best, -1, dtype=torch.int64))

    def tokens(self, frm: torch.Tensor, here: torch.Tensor, excluded: torch.Tensor, visited: torch.Tensor,
               chain: torch.Tensor, depth: int, progress: torch.Tensor, player=None, timing=None):
        """Token features (M, K, TOKEN_DIM), selectable mask (M, K) and the greedy cost from ``frm``.

        ``player`` (pos (M, 3), velocity (M, 3)) at the decision (depth 0); ``timing`` (ticks since the
        last landing on a new platform, ticks since the last decision), both (M,)."""
        ok, cost, d, dy, cos = self.options(frm, excluded)
        dev = frm.device
        ar = torch.arange(self.m, device=dev)
        f = self.rect[ar, frm]
        fc = centres(f)  # (M, 3)
        gx, gz = self.goal[:, 0] - fc[:, 0], self.goal[:, 2] - fc[:, 2]
        gn = torch.sqrt(gx * gx + gz * gz)
        ux = torch.where(gn > 1e-9, gx / gn.clamp(min=1e-9), torch.ones_like(gn))
        uz = torch.where(gn > 1e-9, gz / gn.clamp(min=1e-9), torch.zeros_like(gn))
        vx, vz = -uz, ux

        def frame(x, z):  # world offsets from frm's centre -> (along the goal, to its side)
            dx, dz = x - fc[:, None, 0], z - fc[:, None, 2]
            return dx * ux[:, None] + dz * uz[:, None], dx * vx[:, None] + dz * vz[:, None]
        r = self.rect.to(torch.float64)
        pc = centres(self.rect)
        cu, cv = frame(pc[..., 0], pc[..., 2])
        xs = torch.stack((r[..., 0], r[..., 1] + 1, r[..., 1] + 1, r[..., 0]), -1)
        zs = torch.stack((r[..., 2], r[..., 2], r[..., 3] + 1, r[..., 3] + 1), -1)
        ku, kv = frame(xs.reshape(self.m, -1), zs.reshape(self.m, -1))
        ku, kv = ku.reshape(self.m, self.kt, 4), kv.reshape(self.m, self.kt, 4)
        sx, sz = r[..., 1] - r[..., 0] + 1, r[..., 3] - r[..., 2] + 1
        hx, hz = self.goal[:, None, 0] - pc[..., 0], self.goal[:, None, 2] - pc[..., 2]
        pg = torch.sqrt(hx * hx + hz * hz)
        pgy = self.goal[:, None, 1] - pc[..., 1]
        # Jumps on from each platform (limits only): to in-view platforms other than itself, frm and excluded ones.
        pair_d = air_gap(self.rect[:, :, None, :], self.rect[:, None, :, :])  # (M, K, K)
        pair_dy = self.rect[:, None, :, 4] - self.rect[:, :, None, 4]
        lim = self.gap_table[(pair_dy + 3).clamp(0, 4)]
        target_ok = self.inview & ~excluded
        target_ok = target_ok.clone()
        target_ok[ar, frm] = False
        eye = torch.eye(self.kt, dtype=torch.bool, device=dev)[None]
        onward = (pair_dy >= self.cfg.dy_min) & (pair_dy <= self.cfg.dy_max) & (pair_d <= lim) & target_ok[:, None, :] & ~eye
        n_on = onward.sum(-1).to(torch.float64)
        okf = ok.to(torch.float64)
        rank = torch.where(ok, cost, torch.full_like(cost, math.inf)).argsort(dim=1, stable=True).argsort(dim=1, stable=True).to(torch.float64)
        g = self.greedy_pick(ok, cost)
        is_greedy = torch.zeros_like(ok)
        is_greedy[ar[g >= 0], g[g >= 0]] = True
        role_frm = torch.zeros_like(ok)
        role_frm[ar, frm] = True
        role_here = torch.zeros_like(ok)
        hv = here >= 0
        role_here[ar[hv], here[hv]] = True
        n_sel = okf.sum(1)
        n_vis = (visited & self.inview).sum(1).to(torch.float64)
        dyf = (r[..., 4] - f[:, None, 4].to(torch.float64))
        zero = torch.zeros_like(gn)
        pu = pv = vu = vv = zero
        if player is not None and depth == 0:
            pos, vel = player
            dx, dz = pos[:, 0] - fc[:, 0], pos[:, 2] - fc[:, 2]
            pu, pv = (dx * ux + dz * uz) / 2, (dx * vx + dz * vz) / 2
            vu, vv = (vel[:, 0] * ux + vel[:, 2] * uz) / 0.4, (vel[:, 0] * vx + vel[:, 2] * vz) / 0.4
        t_new, t_dec = (zero, zero) if timing is None else (timing[0].to(torch.float64) / 200, timing[1].to(torch.float64) / 50)
        glob = torch.stack((gn / 20, (self.goal[:, 1] - fc[:, 1]) / 10, torch.full_like(gn, float(depth == 0)),
                            torch.full_like(gn, float(depth == 1)), torch.full_like(gn, float(depth == 2)),
                            n_sel / 8, n_vis / 10, progress.to(torch.float64), pu, pv, vu, vv,
                            t_new.clamp(max=3), t_dec.clamp(max=3)), -1)
        feats = torch.cat((
            torch.stack((cu / 8, cv / 8, dyf / 3, d.clamp(max=12) / 4, sx / 4, sz / 4, cos,
                         cost.clamp(-3, 12) / 4, torch.where(ok, rank, torch.full_like(rank, 16.0)) / 16,
                         is_greedy.to(torch.float64), pg / 20, pgy / 10, self.is_goal.to(torch.float64),
                         self.edge.to(torch.float64), n_on / 4, (n_on == 0).to(torch.float64),
                         okf, role_frm.to(torch.float64), role_here.to(torch.float64),
                         (visited & self.inview).to(torch.float64), chain.to(torch.float64),
                         ((dyf >= self.cfg.dy_min) & (dyf <= self.cfg.dy_max)).to(torch.float64),
                         self.head.to(torch.float64)), -1),
            ku / 8, kv / 8,
            glob[:, None, :].expand(-1, self.kt, -1)), -1)
        feats = feats * self.inview[..., None].to(torch.float64)
        assert feats.shape[-1] == TOKEN_DIM
        return feats.to(torch.float32), ok, cost


def support_cell(world: physics.World, course: torch.Tensor, pos: torch.Tensor):
    """The first solid cell under the player's footprint (x, then z) as planner.support_cells, and
    whether there is one. (M, 3), (M,)."""
    w = physics.HALF_WIDTH
    cy = torch.floor(pos[:, 1] + STAND_EPS).to(torch.int64) - 1
    xa = torch.floor(pos[:, 0] - w).to(torch.int64)
    xb = torch.ceil(pos[:, 0] + w).to(torch.int64)
    za = torch.floor(pos[:, 2] - w).to(torch.int64)
    zb = torch.ceil(pos[:, 2] + w).to(torch.int64)
    cells, found = [], []
    for i in range(2):
        for j in range(2):
            x, z = xa + i, za + j
            live = (x < xb) & (z < zb)
            cells.append(torch.stack((x, cy, z), -1))
            found.append(live & world.lookup(course, x, cy, z))
    cells = torch.stack(cells, 1)  # (M, 4, 3)
    found = torch.stack(found, 1)
    first = found.to(torch.int8).argmax(1)
    return cells[torch.arange(len(course), device=pos.device), first], found.any(1), cells, found


def next_on_course(cidx: torch.Tensor, ok: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Token of the selectable platform with the smallest course index above ``after`` (M,); -1: none.
    Privileged (the course's order): a training signal and a diagnostic, never an input."""
    ahead = ok & (cidx > after[:, None])
    key = torch.where(ahead, cidx, torch.full_like(cidx, 1 << 20))
    best = key.argmin(1)
    return torch.where(ahead.any(1), best, torch.full_like(best, -1))


class RouteRunner:
    """Route state of every row of a ParkourEnv, decisions on the GPU, and the jumper's observation.

    The same rules as planner.GreedyPlanner: a decision at the start of an episode and whenever a row
    stands (on the ground, feet on a whole block height) on a platform other than the one it last
    planned from; landing on a planned platform beyond the next marks the ones before it as passed.
    ``choose(feats, ok, inview)`` returns (token to take (M,), extra outputs); None: the greedy rule.
    """

    def __init__(self, env: ParkourEnv, tables: RouteTables, pcfg: PlannerConfig, rcfg: RouteConfig):
        self.env, self.tables, self.pcfg, self.rcfg = env, tables, pcfg, rcfg
        n, dev = env.n, env.device
        kt = tables.kt
        self.dev = dev
        self.visited = torch.zeros(n, kt, dtype=torch.bool, device=dev)  # table indices
        self.here_rect = torch.zeros(n, 5, dtype=torch.int64, device=dev)
        self.has_here = torch.zeros(n, dtype=torch.bool, device=dev)
        self.chain_tid = torch.full((n, LOOK), -1, dtype=torch.int64, device=dev)
        self.chain_rect = torch.zeros(n, LOOK, 5, dtype=torch.int64, device=dev)
        self.start_dist = torch.zeros(n, dtype=torch.float64, device=dev)
        self.progress = torch.zeros(n, dtype=torch.float64, device=dev)
        self.decisions = torch.zeros(n, dtype=torch.int64, device=dev)
        self.new_tick = torch.zeros(n, dtype=torch.int64, device=dev)  # env tick of the last landing on a new platform
        self.dec_tick = torch.zeros(n, dtype=torch.int64, device=dev)  # env tick of the last decision
        self.R = torch.zeros(n, PLATFORM_SLOTS, 5, dtype=torch.int64, device=dev)
        self.V = torch.zeros(n, PLATFORM_SLOTS, dtype=torch.bool, device=dev)
        self.G = torch.zeros(n, PLATFORM_SLOTS, dtype=torch.bool, device=dev)

    def reset(self, m: torch.Tensor) -> None:
        """Forget the route of rows (bool mask) whose episode restarted."""
        self.visited[m] = False
        self.has_here[m] = False
        self.chain_tid[m] = -1
        self.chain_rect[m] = 0
        self.start_dist[m] = 0
        self.progress[m] = 0
        self.decisions[m] = 0
        self.new_tick[m] = 0
        self.dec_tick[m] = 0
        self.R[m] = 0
        self.V[m] = False
        self.G[m] = False

    def triggers(self, active: torch.Tensor):
        """Rows (bool mask) that re-plan after this tick, and every row's support cell."""
        s = self.env.state
        cell, found, cells, solid = support_cell(self.env.pool.world, self.env.course, s.pos)
        flat = (s.pos[:, 1] - torch.round(s.pos[:, 1])).abs() <= STAND_EPS
        on_here = (contains(self.here_rect[:, None, :], cells) & solid).any(1) & self.has_here
        return active & s.on_ground & flat & found & ~on_here, cell

    def start_cells(self) -> torch.Tensor:
        """GreedyPlanner.start: the first solid support cell, else the cell under the feet."""
        s = self.env.state
        cell, found, _, _ = support_cell(self.env.pool.world, self.env.course, s.pos)
        fallback = torch.stack((torch.floor(s.pos[:, 0]), torch.floor(s.pos[:, 1] + STAND_EPS) - 1,
                                torch.floor(s.pos[:, 2])), -1).to(torch.int64)
        return torch.where(found[:, None], cell, fallback)

    def decide(self, rows: torch.Tensor, cell: torch.Tensor, choose=None, fill_choose=None):
        """Re-plan the rows (index tensor) standing on ``cell`` (N, 3). Returns a dict about the rows
        that re-planned, or None."""
        if rows.numel() == 0:
            return None
        env, dev = self.env, self.dev
        course = env.course[rows]
        cl = cell[rows]
        sc = Scene(env.pool, self.tables, course, env.state.pos[rows], cl, self.pcfg)
        hit = contains(sc.rect, cl[:, None, :]) & sc.inview
        has = hit.any(1)
        if not bool(has.all()):  # not on a platform (an obstacle top): keep the old plan
            keep = has.nonzero().squeeze(1)
            return self.decide(rows[keep], cell, choose, fill_choose) if keep.numel() else None
        m = rows.numel()
        ar = torch.arange(m, device=dev)
        here = hit.to(torch.int8).argmax(1)
        tid = sc.order
        if self.rcfg.mark_passed:
            inchain = contains(self.chain_rect[rows], cl[:, None, :]) & (self.chain_tid[rows] >= 0)  # (M, LOOK)
            j = torch.where(inchain.any(1), inchain.to(torch.int8).argmax(1), torch.full_like(here, -1))
            for q in range(LOOK - 1):
                mark = (j > q) & (self.chain_tid[rows, q] >= 0)
                self.visited[rows[mark], self.chain_tid[rows[mark], q]] = True
        now = env.ticks[rows]
        fresh = ~self.visited[rows, tid[ar, here]]
        timing = (now - self.new_tick[rows], now - self.dec_tick[rows])
        self.new_tick[rows] = torch.where(fresh, now, self.new_tick[rows])
        self.dec_tick[rows] = now
        self.visited[rows, tid[ar, here]] = True
        here_rect = sc.rect[ar, here]
        self.here_rect[rows] = here_rect
        self.has_here[rows] = True
        dist = torch.linalg.norm(centres(here_rect) - sc.goal, dim=-1)
        first = self.decisions[rows] == 0
        self.start_dist[rows] = torch.where(first, dist.clamp(min=1e-9), self.start_dist[rows])
        prog = (1.0 - dist / self.start_dist[rows]).clamp(0.0, 1.0)
        self.progress[rows] = prog
        self.decisions[rows] += 1
        visited = torch.gather(self.visited[rows], 1, tid)  # token order
        excluded = visited.clone()
        excluded[ar, here] = True
        chain = torch.zeros_like(excluded)
        feats, ok, cost = sc.tokens(here, here, excluded, visited, chain, 0, prog,
                                    (env.state.pos[rows], env.state.vel[rows]), timing)
        any_ok = ok.any(1)
        out = None
        if choose is None:
            act = Scene.greedy_pick(ok, cost)
        elif choose == "oracle":  # diagnostic only: the course's next platform (privileged)
            act = next_on_course(sc.cidx, ok, env.cp[rows])
        else:
            act, out = choose(feats, ok, sc.inview)
            act = torch.where(any_ok, act, torch.full_like(act, -1))
        picks = [act]
        prev = act
        for depth in (1, 2):
            live = (prev >= 0) & ~sc.is_goal[ar, prev.clamp(min=0)]
            if not bool(live.any()):
                prev = torch.full_like(act, -1)
                picks.append(prev)
                continue
            chain[ar[prev >= 0], prev[prev >= 0]] = True
            ex = excluded | chain
            frm = prev.clamp(min=0)
            if fill_choose == "oracle":
                okd, *_ = sc.options(frm, ex)
                nxt = next_on_course(sc.cidx, okd, sc.cidx[ar, frm])
            elif self.rcfg.fill == "greedy" or fill_choose is None:
                okd, costd, *_ = sc.options(frm, ex)
                nxt = Scene.greedy_pick(okd, costd)
            else:
                fd, okd, _ = sc.tokens(frm, here, ex, visited, chain, depth, prog, None, timing)
                nxt = fill_choose(fd, okd, sc.inview)
                nxt = torch.where(okd.any(1), nxt, torch.full_like(nxt, -1))
            prev = torch.where(live, nxt, torch.full_like(nxt, -1))
            picks.append(prev)
        picks = torch.stack(picks, 1)  # (M, LOOK) tokens, -1 = none
        pv = picks >= 0
        pc = picks.clamp(min=0)
        crect = torch.where(pv[..., None], sc.rect[ar[:, None], pc], torch.zeros_like(sc.rect[:, :LOOK]))
        self.chain_rect[rows] = crect
        self.chain_tid[rows] = torch.where(pv, tid[ar[:, None], pc], torch.full_like(pc, -1))
        self.R[rows, 0] = here_rect
        self.R[rows, 1:] = crect
        self.V[rows, 0] = True
        self.V[rows, 1:] = pv
        self.G[rows, 0] = sc.is_goal[ar, here]
        self.G[rows, 1:] = sc.is_goal[ar[:, None], pc] & pv
        return {"rows": rows, "feats": feats, "ok": ok, "inview": sc.inview, "action": act, "any_ok": any_ok,
                "out": out, "cidx": sc.cidx, "here": here, "picks": picks, "rect": sc.rect}

    def observe(self) -> torch.Tensor:
        env = self.env
        obs = env.observe()
        obs[:, OBS_SLOTS] = env.obs_fn.slots(env.state, self.R, self.V, self.G).to(obs.dtype)
        obs[:, PROGRESS] = self.progress.to(obs.dtype)
        return obs


def evaluate_routed(jumper, pool: CoursePool, tables: RouteTables, choose=None, fill_choose=None,
                    pcfg: PlannerConfig | None = None, rcfg: RouteConfig | None = None, stall_ticks: int = 200,
                    max_ticks: int = 4000, fall_rule: str = "local", record: bool = False, settle: int = 0) -> dict:
    """First episode on every course, the jumper's slots filled by the router (``choose=None``: the
    greedy planner's choice, which must reproduce parkour.planner.evaluate_planned)."""
    pcfg = pcfg or PlannerConfig()
    rcfg = rcfg or RouteConfig()
    dev = pool.device
    n = pool.size
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=stall_ticks, max_ticks=max_ticks, fall_rule=fall_rule),
                     sampler=None)
    env.reset_all(torch.arange(n, device=dev))
    if settle:
        idle = torch.zeros(n, 6, dtype=torch.bool, device=dev)
        for _ in range(settle):
            env.state = physics.simulate(pool.world, env.course, env.state, idle)
        env.dist = env._distance(env.state.pos, env.course, env.cp)
        env.reset_memory(torch.ones(n, dtype=torch.bool, device=dev))
    run = RouteRunner(env, tables, pcfg, rcfg)
    events = [[] for _ in range(n)] if record else None

    def note(res, t):
        if res is None or not record:
            return
        rects = res["rect"].tolist()
        for i, r in enumerate(res["rows"].tolist()):
            here = rects[i][res["here"][i].item()]
            events[r].append([t, here, [rects[i][p] for p in res["picks"][i].tolist() if p >= 0]])
    with torch.no_grad():
        note(run.decide(torch.arange(n, device=dev), run.start_cells(), choose, fill_choose), 0)
        obs = run.observe()
        finished = torch.zeros(n, dtype=torch.bool, device=dev)
        success = torch.zeros(n, dtype=torch.bool, device=dev)
        fell = torch.zeros(n, dtype=torch.bool, device=dev)
        length = torch.zeros(n, dtype=torch.int64, device=dev)
        for t in range(max_ticks):
            a, _, _ = jumper.act(obs, deterministic=True)
            _, _, _, _, _, info = env.step(a)
            first = info["done"] & ~finished
            success |= first & info["success"]
            fell |= first & info["fell"]
            length = torch.where(first, info["ep_len"], length)
            finished |= info["done"]
            if bool(finished.all()):
                break
            trig, cell = run.triggers(~finished)
            note(run.decide(trig.nonzero().squeeze(1), cell, choose, fill_choose), t + 1)
            obs = run.observe()
    lv = pool.level.cpu().numpy()
    s, f, ln = (x.cpu().numpy() for x in (success, fell, length))
    per_level = {}
    for L in sorted(set(lv.tolist())):
        msk = lv == L
        per_level[int(L)] = {"success": float(s[msk].mean()), "fell": float(f[msk].mean()), "n": int(msk.sum())}
    out = {"success": float(s.mean()), "per_level": per_level,
           "per_course": {"success": s.tolist(), "fell": f.tolist(), "ticks": ln.tolist()}}
    if record:
        out["events"] = events
    return out

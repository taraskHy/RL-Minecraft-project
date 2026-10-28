"""Batched Minecraft player physics, ported from prismarine-physics 1.11.1.

prismarine-physics is the movement engine that Mineflayer runs on the client
side. A Mineflayer bot moves exactly as that engine computes and the server
accepts the result, so a faithful port of it is a faithful simulator of the
bot in real Minecraft. This port covers what parkour courses use: air and full
solid cubes with default slipperiness, and two opt-in extensions that change
nothing unless passed: per-cell slipperiness (ice) and per-cell collision
shapes (slabs, stairs, soul sand, carpets, trapdoors, fence and wall posts,
..., ``World(shape=...)`` with the boxes of configs/block_shapes.json), and
slime bounces (``World(slime=...)``, with slipperiness 0.8 in ``slip``), and
climbing ladders and vines (``World(climb=...)``). Water, lava, honey,
cobwebs, scaffolding, sneaking, effects and elytra are not ported.

Every expression keeps the operand order of the JavaScript source, in float64,
so trajectories match the original to rounding error (see
tests/test_physics_parity.py, which runs the real engine through Node).

One deliberate change, made identically in the bot (bridge/physics_patch.js):
block collisions use vanilla Minecraft's 1e-7 tolerance (VoxelShape.collide).
Without it, a player resting exactly against a block face can read as
overlapping it after rounding (32.3 - 0.3 = 31.999999999999996), the engine
then stops blocking that block, and the player slips into its corner.

Coordinates follow Mineflayer: yaw 0 faces north (-Z), yaw increases
counter-clockwise seen from above (pi/2 faces west, -X). The forward vector is
(-sin yaw, 0, -cos yaw) and the right vector is (cos yaw, 0, -sin yaw).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache

import torch

from .paths import CONFIGS

# Constants as prismarine-physics evaluates them (JavaScript doubles).
GRAVITY = 0.08
AIRDRAG = 0.9800000190734863  # Math.fround(1 - 0.02)
JUMP_VELOCITY = 0.41999998688697815  # Math.fround(0.42)
SPRINT_JUMP_BOOST = 0.2
STEP_HEIGHT = 0.6
NEGLIGIBLE_VELOCITY = 0.003
HALF_WIDTH = 0.3
PLAYER_HEIGHT = 1.8
AIRBORNE_INERTIA = 0.91
AIRBORNE_ACCELERATION = 0.02
AIR_SPRINT_ACCELERATION = AIRBORNE_ACCELERATION + AIRBORNE_ACCELERATION * 0.3
AUTOJUMP_COOLDOWN = 10
DEFAULT_SLIPPERINESS = 0.6
ICE_SLIPPERINESS = 0.98  # ice and packed ice (blue ice: 0.989); only with World(slip=...)
SLIME_SLIPPERINESS = 0.8  # slime blocks also bounce (World(slime=...))
LADDER_MAX_SPEED = 0.15  # in a ladder or vine cell (World(climb=...))
LADDER_CLIMB_SPEED = 0.2
GROUND_INERTIA = 0.6 * 0.91  # default slipperiness * 0.91
_INERTIA_CUBED = GROUND_INERTIA * GROUND_INERTIA * GROUND_INERTIA
WALK_ACCELERATION = 0.1 * (0.1627714 / _INERTIA_CUBED)
SPRINT_ACCELERATION = (0.1 + 0.1 * 0.3) * (0.1627714 / _INERTIA_CUBED)

# Collision window in cells. It covers any query box whose horizontal moves
# stay below 2.4 blocks per tick and vertical moves below 3.2 blocks per tick.
WINDOW = (4, 7, 4)

# Vanilla's collision tolerance: faces closer than this count as touching, not overlapping.
COLLISION_EPSILON = 1e-7

# Control channels, in this order, in every controls tensor.
FORWARD, BACK, LEFT, RIGHT, JUMP, SPRINT = range(6)
N_CONTROLS = 6


@lru_cache(maxsize=None)
def shape_table() -> tuple:
    """(names, boxes (S, K, 6) float64, counts (S,)) of configs/block_shapes.json; shape id = index, 0 = full cube.

    The boxes are prismarine-block's for Minecraft 1.21.4 (bridge/export_shapes.js writes the file
    and checks it); K is the most boxes any shape has, shorter lists are padded."""
    data = json.loads((CONFIGS / "block_shapes.json").read_text())["shapes"]
    k = max(len(d["boxes"]) for d in data)
    boxes = torch.zeros(len(data), k, 6, dtype=torch.float64)
    for i, d in enumerate(data):
        if d["boxes"]:
            boxes[i, :len(d["boxes"])] = torch.tensor(d["boxes"], dtype=torch.float64)
    counts = torch.tensor([len(d["boxes"]) for d in data], dtype=torch.int64)
    return tuple(d["name"] for d in data), boxes, counts


def shape_id(name: str) -> int:
    return shape_table()[0].index(name)


class World:
    """Solid-cube occupancy for a pool of courses on one device.

    ``solid`` has shape (P, GX, GY, GZ). Cell (i, j, k) of course p is the
    block occupying [i, i+1) x [j, j+1) x [k, k+1). Everything outside the grid
    is air.

    ``slip`` (optional, same shape, float) is each cell's slipperiness, read
    under the player's feet for ground friction as prismarine-physics does
    (ice 0.98). Without it every block has the default 0.6 and the simulation
    is exactly the one all results so far were measured with.

    ``shape`` (optional, same shape, uint8) is each solid cell's collision
    shape id (``shape_table``; 0, the full cube, everywhere else). Without it
    every solid cell is a full cube, through the original code path.
    """

    def __init__(self, solid: torch.Tensor, slip: torch.Tensor | None = None, shape: torch.Tensor | None = None,
                 slime: torch.Tensor | None = None, climb: torch.Tensor | None = None):
        if solid.dim() != 4:
            raise ValueError("solid must have shape (P, GX, GY, GZ)")
        self.solid = solid.to(torch.bool).contiguous()
        self.shape = tuple(solid.shape[1:])
        self.flat = self.solid.view(-1)
        self.device = solid.device
        if slip is not None and tuple(slip.shape) != tuple(solid.shape):
            raise ValueError("slip must have the shape of solid")
        self.slip = None if slip is None else slip.to(torch.float64).contiguous().view(-1)
        if slime is not None and tuple(slime.shape) != tuple(solid.shape):
            raise ValueError("slime must have the shape of solid")
        self.slime = None if slime is None else slime.to(torch.bool).contiguous().view(-1)
        if climb is not None and tuple(climb.shape) != tuple(solid.shape):
            raise ValueError("climb must have the shape of solid")
        # Ladders and vines: cells the player climbs in (a ladder is also solid with its thin box; a vine is not).
        self.climb = None if climb is None else climb.to(torch.bool).contiguous().view(-1)
        self.shape_ids = None
        if shape is not None:
            if tuple(shape.shape) != tuple(solid.shape):
                raise ValueError("shape must have the shape of solid")
            _, boxes, counts = shape_table()
            self.shape_ids = shape.to(torch.uint8).contiguous().view(-1)
            self.boxes = boxes.to(self.device)
            self.box_counts = counts.to(self.device)

    def lookup(self, course, ix, iy, iz):
        """Solid flags for integer cells; arguments broadcast together."""
        gx, gy, gz = self.shape
        inside = (ix >= 0) & (ix < gx) & (iy >= 0) & (iy < gy) & (iz >= 0) & (iz < gz)
        index = ((course * gx + ix.clamp(0, gx - 1)) * gy + iy.clamp(0, gy - 1)) * gz + iz.clamp(0, gz - 1)
        return self.flat[index] & inside

    def slime_at(self, course, ix, iy, iz):
        """Slime flags of integer cells (False outside the grid); needs ``slime``."""
        return self._flag(self.slime, course, ix, iy, iz)

    def climb_at(self, course, pos):
        """(N,) bool: the cell of the feet (pos floored) is a ladder or vine; needs ``climb``."""
        f = torch.floor(pos).to(torch.int64)
        return self._flag(self.climb, course, f[:, 0], f[:, 1], f[:, 2])

    def _flag(self, flat, course, ix, iy, iz):
        gx, gy, gz = self.shape
        inside = (ix >= 0) & (ix < gx) & (iy >= 0) & (iy < gy) & (iz >= 0) & (iz < gz)
        index = ((course * gx + ix.clamp(0, gx - 1)) * gy + iy.clamp(0, gy - 1)) * gz + iz.clamp(0, gz - 1)
        return flat[index] & inside

    def shapes(self, course, ix, iy, iz):
        """Shape ids (int64) of integer cells, 0 outside the grid; needs ``shape``."""
        gx, gy, gz = self.shape
        index = ((course * gx + ix.clamp(0, gx - 1)) * gy + iy.clamp(0, gy - 1)) * gz + iz.clamp(0, gz - 1)
        return self.shape_ids[index].to(torch.int64)

    def cell_boxes(self, course, ix, iy, iz, solid):
        """Absolute collision boxes of the cells (lo (..., K, 3), hi (..., K, 3), mask (..., K)).

        Box = cell + shape box, added in the order prismarine's AABB.offset adds them
        (exact here: every coordinate is a multiple of 1/64)."""
        ids = self.shapes(course, ix, iy, iz)
        b = self.boxes[ids]  # (..., K, 6)
        k = torch.arange(self.boxes.shape[1], device=b.device)
        mask = solid[..., None] & (k < self.box_counts[ids][..., None])
        cell = torch.stack(torch.broadcast_tensors(ix, iy, iz), -1).to(torch.float64)[..., None, :]
        return b[..., :3] + cell, b[..., 3:] + cell, mask

    def slipperiness(self, course, ix, iy, iz):
        """Slipperiness of integer cells (the default outside the grid); needs ``slip``."""
        gx, gy, gz = self.shape
        inside = (ix >= 0) & (ix < gx) & (iy >= 0) & (iy < gy) & (iz >= 0) & (iz < gz)
        index = ((course * gx + ix.clamp(0, gx - 1)) * gy + iy.clamp(0, gy - 1)) * gz + iz.clamp(0, gz - 1)
        values = self.slip[index]
        return torch.where(inside, values, torch.full_like(values, DEFAULT_SLIPPERINESS))


@dataclass
class PlayerState:
    pos: torch.Tensor  # (N, 3) float64, feet centre
    vel: torch.Tensor  # (N, 3) float64, blocks per tick
    yaw: torch.Tensor  # (N,) float64, radians
    on_ground: torch.Tensor  # (N,) bool
    jump_ticks: torch.Tensor  # (N,) int64
    collided_h: torch.Tensor  # (N,) bool
    collided_v: torch.Tensor  # (N,) bool

    @classmethod
    def at(cls, pos: torch.Tensor, yaw: torch.Tensor) -> "PlayerState":
        n = pos.shape[0]
        dev = pos.device
        return cls(
            pos=pos.to(torch.float64).clone(),
            vel=torch.zeros(n, 3, dtype=torch.float64, device=dev),
            yaw=yaw.to(torch.float64).clone(),
            on_ground=torch.zeros(n, dtype=torch.bool, device=dev),
            jump_ticks=torch.zeros(n, dtype=torch.int64, device=dev),
            collided_h=torch.zeros(n, dtype=torch.bool, device=dev),
            collided_v=torch.zeros(n, dtype=torch.bool, device=dev),
        )

    def clone(self) -> "PlayerState":
        return PlayerState(*(t.clone() for t in (self.pos, self.vel, self.yaw, self.on_ground,
                                                  self.jump_ticks, self.collided_h, self.collided_v)))

    def assign(self, mask: torch.Tensor, other: "PlayerState") -> None:
        """Copy ``other`` into the rows selected by the boolean ``mask``."""
        for name in ("pos", "vel", "yaw", "on_ground", "jump_ticks", "collided_h", "collided_v"):
            mine = getattr(self, name)
            theirs = getattr(other, name)
            m = mask.view(-1, *([1] * (mine.dim() - 1)))
            setattr(self, name, torch.where(m, theirs, mine))


def player_bb(pos: torch.Tensor) -> torch.Tensor:
    """AABB(-w, 0, -w, w, h, w).offset(pos): columns minX minY minZ maxX maxY maxZ."""
    x, y, z = pos.unbind(-1)
    w = HALF_WIDTH
    return torch.stack(((-w) + x, 0.0 + y, (-w) + z, w + x, PLAYER_HEIGHT + y, w + z), dim=-1)


def _extend(bb: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    neg = d < 0
    zero = torch.zeros_like(d)
    return torch.cat((bb[:, :3] + torch.where(neg, d, zero), bb[:, 3:] + torch.where(neg, zero, d)), dim=-1)


def _offset_axis(bb: torch.Tensor, axis: int, d: torch.Tensor) -> torch.Tensor:
    out = bb.clone()
    out[:, axis] = bb[:, axis] + d
    out[:, axis + 3] = bb[:, axis + 3] + d
    return out


class _Cells:
    """Solid cells that getSurroundingBBs would return for a query box."""

    def __init__(self, world: World, course: torch.Tensor, query: torch.Tensor):
        wx, wy, wz = WINDOW
        dev = query.device
        base = torch.floor(query[:, :3]).to(torch.int64)
        base[:, 1] -= 1
        top = torch.floor(query[:, 3:]).to(torch.int64)
        ix = base[:, 0:1] + torch.arange(wx, device=dev)
        iy = base[:, 1:2] + torch.arange(wy, device=dev)
        iz = base[:, 2:3] + torch.arange(wz, device=dev)
        solid = world.lookup(course.view(-1, 1, 1, 1), ix[:, :, None, None], iy[:, None, :, None], iz[:, None, None, :])
        valid = ((ix <= top[:, 0:1])[:, :, None, None] & (iy <= top[:, 1:2])[:, None, :, None]
                 & (iz <= top[:, 2:3])[:, None, None, :])
        self.solid = solid & valid
        self.overflow = (top - base + 1 > torch.tensor(WINDOW, device=dev)).any(dim=-1)
        self.cx = ix.to(torch.float64)
        self.cy = iy.to(torch.float64)
        self.cz = iz.to(torch.float64)
        self.shaped = world.shape_ids is not None
        if self.shaped:
            # Every box of every cell in the window: (N, wx, wy, wz, K).
            self.lo, self.hi, self.bmask = world.cell_boxes(
                course.view(-1, 1, 1, 1), ix[:, :, None, None], iy[:, None, :, None], iz[:, None, None, :], self.solid)

    def offset(self, bb: torch.Tensor, d: torch.Tensor, axis: int) -> torch.Tensor:
        """Sequential AABB.computeOffset{X,Y,Z} over all cells, as a reduction, with vanilla's tolerance."""
        if self.shaped:
            return self._offset_boxes(bb, d, axis)
        eps = COLLISION_EPSILON
        coords = (self.cx, self.cy, self.cz)
        overlaps = []
        for other in range(3):
            if other == axis:
                continue
            c = coords[other]
            overlaps.append((other, (bb[:, other + 3, None] > c + eps) & (bb[:, other, None] < c + 1 - eps)))
        mask = self.solid
        for other, ov in overlaps:
            shape = [-1, 1, 1, 1]
            shape[1 + other] = ov.shape[1]
            mask = mask & ov.view(shape)
        dims = tuple(i for i in (1, 2, 3) if i != 1 + axis)
        line = mask.any(dim=dims)  # (N, window along axis)
        c = coords[axis]
        lo = bb[:, axis, None]
        hi = bb[:, axis + 3, None]
        inf = torch.full_like(c, math.inf)
        up = torch.where(line & (hi <= c + eps), (c - hi).clamp(min=0.0), inf).amin(dim=1)
        down = torch.where(line & (lo >= c + 1 - eps), ((c + 1) - lo).clamp(max=0.0), -inf).amax(dim=1)
        # A block that limits the move to under the tolerance stops it entirely.
        up = torch.where(up < eps, torch.zeros_like(up), up)
        down = torch.where(down > -eps, torch.zeros_like(down), down)
        return torch.where(d > 0, torch.where(up < d, up, d), torch.where(d < 0, torch.where(down > d, down, d), d))

    def _offset_boxes(self, bb: torch.Tensor, d: torch.Tensor, axis: int) -> torch.Tensor:
        """``offset`` with every box of every cell (``World(shape=...)``); for full cubes the same numbers."""
        eps = COLLISION_EPSILON
        col = (lambda v: v.view(-1, 1, 1, 1, 1))  # noqa: E731
        mask = self.bmask
        for other in range(3):
            if other != axis:
                mask = mask & (col(bb[:, other + 3]) > self.lo[..., other] + eps) & (col(bb[:, other]) < self.hi[..., other] - eps)
        lo_a, hi_a = self.lo[..., axis], self.hi[..., axis]
        bmin, bmax = col(bb[:, axis]), col(bb[:, axis + 3])
        inf = torch.full_like(lo_a, math.inf)
        up = torch.where(mask & (bmax <= lo_a + eps), (lo_a - bmax).clamp(min=0.0), inf).flatten(1).amin(dim=1)
        down = torch.where(mask & (bmin >= hi_a - eps), (hi_a - bmin).clamp(max=0.0), -inf).flatten(1).amax(dim=1)
        up = torch.where(up < eps, torch.zeros_like(up), up)
        down = torch.where(down > -eps, torch.zeros_like(down), down)
        return torch.where(d > 0, torch.where(up < d, up, d), torch.where(d < 0, torch.where(down > d, down, d), d))


def _move(world: World, course: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor,
          on_ground: torch.Tensor, check: bool):
    """moveEntity without webs, sneaking or special blocks."""
    dx, dy, dz = vel.unbind(-1)
    bb0 = player_bb(pos)
    cells = _Cells(world, course, _extend(bb0, vel))
    cy = cells.offset(bb0, dy, 1)
    bb = _offset_axis(bb0, 1, cy)
    cx = cells.offset(bb, dx, 0)
    bb = _offset_axis(bb, 0, cx)
    cz = cells.offset(bb, dz, 2)
    bb = _offset_axis(bb, 2, cz)

    # Step-up attempt (stepHeight 0.6), computed for every row and then selected.
    step_y = torch.full_like(dy, STEP_HEIGHT)
    zero = torch.zeros_like(dy)
    try_step = (on_ground | ((cy != dy) & (dy < 0))) & ((cx != dx) | (cz != dz))
    cells2 = _Cells(world, course, _extend(bb0, torch.stack((dx, step_y, dz), -1)))
    bb_xz = _extend(bb0, torch.stack((cx, zero, cz), -1))
    sy1 = cells2.offset(bb_xz, step_y, 1)
    sy2 = cells2.offset(bb0, step_y, 1)
    b1 = _offset_axis(bb0, 1, sy1)
    b2 = _offset_axis(bb0, 1, sy2)
    sx1 = cells2.offset(b1, dx, 0)
    sx2 = cells2.offset(b2, dx, 0)
    b1 = _offset_axis(b1, 0, sx1)
    b2 = _offset_axis(b2, 0, sx2)
    sz1 = cells2.offset(b1, dz, 2)
    sz2 = cells2.offset(b2, dz, 2)
    b1 = _offset_axis(b1, 2, sz1)
    b2 = _offset_axis(b2, 2, sz2)
    pick1 = (sx1 * sx1 + sz1 * sz1) > (sx2 * sx2 + sz2 * sz2)
    sx = torch.where(pick1, sx1, sx2)
    sy = torch.where(pick1, -sy1, -sy2)
    sz = torch.where(pick1, sz1, sz2)
    sbb = torch.where(pick1[:, None], b1, b2)
    sy = cells2.offset(sbb, sy, 1)
    sbb = _offset_axis(sbb, 1, sy)
    keep = (cx * cx + cz * cz) >= (sx * sx + sz * sz)
    use_step = try_step & ~keep

    fx = torch.where(use_step, sx, cx)
    fy = torch.where(use_step, sy, cy)
    fz = torch.where(use_step, sz, cz)
    fbb = torch.where(use_step[:, None], sbb, bb)

    if check:
        bad = cells.overflow | (cells2.overflow & try_step)
        if bool(bad.any()):
            raise RuntimeError("collision window too small for this move")

    new_pos = torch.stack((fbb[:, 0] + HALF_WIDTH, fbb[:, 1], fbb[:, 2] + HALF_WIDTH), dim=-1)
    hit_x = fx != dx
    hit_y = fy != dy
    hit_z = fz != dz
    collided_h = hit_x | hit_z
    collided_v = hit_y
    new_on_ground = collided_v & (dy < 0)
    vy_after = torch.where(hit_y, zero, dy)
    if world.slime is not None:
        # A vertical collision with slime 0.2 below the feet reverses the velocity (no sneaking here).
        bounce = hit_y & world.slime_at(course, torch.floor(new_pos[:, 0]).to(torch.int64),
                                        torch.floor(new_pos[:, 1] - 0.2).to(torch.int64),
                                        torch.floor(new_pos[:, 2]).to(torch.int64))
        vy_after = torch.where(bounce, -dy, vy_after)
    new_vel = torch.stack((torch.where(hit_x, zero, dx), vy_after, torch.where(hit_z, zero, dz)), dim=-1)
    return new_pos, new_vel, new_on_ground, collided_h, collided_v


def simulate(world: World, course: torch.Tensor, state: PlayerState, controls: torch.Tensor,
             check: bool = False) -> PlayerState:
    """One 50 ms tick of physics.simulatePlayer for every row.

    ``controls`` is a bool tensor (N, 6) in the order FORWARD, BACK, LEFT,
    RIGHT, JUMP, SPRINT. The yaw in ``state`` is the yaw for this tick.
    Returns a new state; the input is not modified.
    """
    vel = state.vel
    vel = torch.where(vel.abs() < NEGLIGIBLE_VELOCITY, torch.zeros_like(vel), vel)
    forward_c, back_c, left_c, right_c, jump, sprint = controls.unbind(-1)

    jt = state.jump_ticks
    jt = torch.where(jump & (jt > 0), jt - 1, jt)
    do_jump = jump & state.on_ground & (jt == 0)
    yaw2 = math.pi - state.yaw
    sin2 = torch.sin(yaw2)
    cos2 = torch.cos(yaw2)
    vx, vy, vz = vel.unbind(-1)
    vy = torch.where(do_jump, torch.full_like(vy, JUMP_VELOCITY), vy)
    boost = do_jump & sprint
    vx = torch.where(boost, vx - sin2 * SPRINT_JUMP_BOOST, vx)
    vz = torch.where(boost, vz + cos2 * SPRINT_JUMP_BOOST, vz)
    jt = torch.where(do_jump, torch.full_like(jt, AUTOJUMP_COOLDOWN), jt)
    jt = torch.where(jump, jt, torch.zeros_like(jt))

    strafe = (right_c.to(torch.float64) - left_c.to(torch.float64)) * 0.98
    forward = (forward_c.to(torch.float64) - back_c.to(torch.float64)) * 0.98

    ground = state.on_ground

    def const(value):
        return torch.full_like(vx, value)

    accel = torch.where(ground,
                        torch.where(sprint, const(SPRINT_ACCELERATION), const(WALK_ACCELERATION)),
                        torch.where(sprint, const(AIR_SPRINT_ACCELERATION), const(AIRBORNE_ACCELERATION)))
    inertia = torch.where(ground, const(GROUND_INERTIA), const(AIRBORNE_INERTIA))
    if world.slip is not None:
        # The block under the feet centre (pos.offset(0, -1, 0)) sets the ground friction.
        p = state.pos
        under = world.slipperiness(course, torch.floor(p[:, 0]).to(torch.int64),
                                   torch.floor(p[:, 1] - 1).to(torch.int64), torch.floor(p[:, 2]).to(torch.int64))
        g_inertia = under * 0.91
        g_speed = torch.where(sprint, const(0.1 + 0.1 * 0.3), const(0.1))
        g_accel = g_speed * (0.1627714 / (g_inertia * g_inertia * g_inertia))
        accel = torch.where(ground, g_accel, accel)
        inertia = torch.where(ground, g_inertia, inertia)

    speed = torch.sqrt(strafe * strafe + forward * forward)
    k = accel / torch.clamp(speed, min=1.0)
    st = strafe * k
    fw = forward * k
    heading = speed >= 0.01
    vx = torch.where(heading, vx - (st * cos2 + fw * sin2), vx)
    vz = torch.where(heading, vz + (fw * cos2 - st * sin2), vz)
    if world.climb is not None:  # on a ladder or vine: slow sideways, fall at most 0.15 per tick (no sneaking)
        on = world.climb_at(course, state.pos)
        m = LADDER_MAX_SPEED
        vx = torch.where(on, torch.maximum(const(-m), torch.minimum(vx, const(m))), vx)
        vz = torch.where(on, torch.maximum(const(-m), torch.minimum(vz, const(m))), vz)
        vy = torch.where(on, torch.maximum(vy, const(-m)), vy)

    pos, mvel, on_ground, collided_h, collided_v = _move(
        world, course, state.pos, torch.stack((vx, vy, vz), -1), state.on_ground, check)

    mx, my, mz = mvel.unbind(-1)
    if world.climb is not None:  # pushing against something, or jumping (1.14+), while on a ladder climbs it
        my = torch.where(world.climb_at(course, pos) & (collided_h | jump), const(LADDER_CLIMB_SPEED), my)
    my = (my - GRAVITY) * AIRDRAG
    mx = mx * inertia
    mz = mz * inertia
    return PlayerState(pos=pos, vel=torch.stack((mx, my, mz), -1), yaw=state.yaw.clone(),
                       on_ground=on_ground, jump_ticks=jt, collided_h=collided_h, collided_v=collided_v)


def forward_vector(yaw: torch.Tensor) -> torch.Tensor:
    return torch.stack((-torch.sin(yaw), torch.zeros_like(yaw), -torch.cos(yaw)), -1)


def right_vector(yaw: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.cos(yaw), torch.zeros_like(yaw), -torch.sin(yaw)), -1)


def inside_solid(world: World, course: torch.Tensor, pos: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """(N,) bool: the player's box overlaps a solid cell by more than ``eps`` on every axis.

    The collision code only stops movement into blocks; a player that starts
    inside one can walk out through it, as in Minecraft. Searches that choose
    start positions use this to reject them.
    """
    w = HALF_WIDTH
    x, y, z = pos.unbind(-1)
    if world.shape_ids is not None:
        return _inside_boxes(world, course, pos, eps)
    bx = torch.floor(x - w).to(torch.int64)[:, None] + torch.arange(2, device=pos.device)
    by = torch.floor(y).to(torch.int64)[:, None] + torch.arange(3, device=pos.device)
    bz = torch.floor(z - w).to(torch.int64)[:, None] + torch.arange(2, device=pos.device)
    solid = world.lookup(course.view(-1, 1, 1, 1), bx[:, :, None, None], by[:, None, :, None], bz[:, None, None, :])
    fx, fy, fz = bx.to(pos.dtype), by.to(pos.dtype), bz.to(pos.dtype)
    ox = ((x[:, None] + w) > fx + eps) & ((x[:, None] - w) < fx + 1 - eps)
    oy = ((y[:, None] + PLAYER_HEIGHT) > fy + eps) & (y[:, None] < fy + 1 - eps)
    oz = ((z[:, None] + w) > fz + eps) & ((z[:, None] - w) < fz + 1 - eps)
    hit = solid & ox[:, :, None, None] & oy[:, None, :, None] & oz[:, None, None, :]
    return hit.flatten(1).any(dim=1)


def _inside_boxes(world: World, course: torch.Tensor, pos: torch.Tensor, eps: float) -> torch.Tensor:
    """``inside_solid`` with collision shapes (the row below too: posts are 1.5 blocks tall)."""
    w = HALF_WIDTH
    x, y, z = pos.unbind(-1)
    dev = pos.device
    bx = (torch.floor(x - w).to(torch.int64)[:, None] + torch.arange(2, device=dev))[:, :, None, None]
    by = (torch.floor(y).to(torch.int64)[:, None] - 1 + torch.arange(4, device=dev))[:, None, :, None]
    bz = (torch.floor(z - w).to(torch.int64)[:, None] + torch.arange(2, device=dev))[:, None, None, :]
    c = course.view(-1, 1, 1, 1)
    lo, hi, mask = world.cell_boxes(c, bx, by, bz, world.lookup(c, bx, by, bz))
    col = (lambda v: v.view(-1, 1, 1, 1, 1))  # noqa: E731
    hit = (mask & (col(x + w) > lo[..., 0] + eps) & (col(x - w) < hi[..., 0] - eps)
           & (col(y + PLAYER_HEIGHT) > lo[..., 1] + eps) & (col(y) < hi[..., 1] - eps)
           & (col(z + w) > lo[..., 2] + eps) & (col(z - w) < hi[..., 2] - eps))
    return hit.flatten(1).any(dim=1)

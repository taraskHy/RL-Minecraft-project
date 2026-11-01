"""Water pieces for courses (opt-in; HANDOFF.md section 3, 1 Nov): a pool between two platforms.

The simulator moves a player in water exactly as prismarine-physics does (``physics.World(water=...)``, checked
against the real engine: ``tests/test_physics_parity.py`` ``test_water_cpu``). A piece is one jump of a course, from
a takeoff platform over (or down into) a pool to the exit platform on the pool's far rim:

    takeoff    gap    near wall | water ... water | exit (the far wall's top row, and one row beyond it)
    -------           ----------+-----------------+------
                                |      basin      |

The water fills a stone basin up to the rim row, so in the real game it has no air beside it or below it and does
not flow: every cell is a still source. The player gets out over the far rim by the engine's own jump out of the
water (blocked sideways with room above: 0.3 upward). Three kinds, by their numbers (``WaterSpec``):

  drop    a pool breaking a drop: the takeoff 3-8 rows above the water, the pool 6-8 long and 2-3 deep
  column  a column to swim up: a narrow pool (1 wide, 5 long) 4-6 deep that the player sinks into and swims up out
          of (a shaft entered at its foot would need an opening there, which in the real game lets the water out)
  swim    a gap to swim across: a long shallow channel (5-8 long, 2 deep), entered from the rim's height or 1 above

Every pool is longer than a jump from the takeoff reaches (the final policy, which never saw water, clears none
dry: ``tests/test_water.py``), so a piece is passed only through the water.

``water_piece`` builds the cells of one piece after a takeoff platform; ``water_course`` makes a short course of
ordinary jumps with water pieces in it. Courses keep the water in ``Course.water`` (x, y, z, level 0) and the basin
in ``Course.extra``; in water the environment's fall and obstacle rules do not apply (``env.in_water``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import courses as C


@dataclass(frozen=True)
class WaterSpec:
    kind: str  # "drop", "column" or "swim"
    length: int  # water cells along the heading
    width: int  # water cells across
    depth: int  # water rows
    drop: int  # rows from the takeoff's top down to the water surface's row (0: level with the rim)
    gap: int = 1  # air cells between the takeoff and the near wall

    def key(self) -> str:
        return f"water_{self.kind}_l{self.length}w{self.width}d{self.depth}_drop{self.drop}_g{self.gap}"


def water_specs() -> list[WaterSpec]:
    """Every water piece the sets draw from: pools breaking a drop, columns to swim up, gaps to swim across."""
    out = []
    for drop in (3, 5, 8):
        for length in (6, 8):
            for depth in (2, 3):
                out.append(WaterSpec("drop", length, 3, depth, drop))
    for depth in (4, 5, 6):
        for drop in (1, 3):
            out.append(WaterSpec("column", 5, 1, depth, drop))
    for length in (5, 6, 8):
        for drop in (0, 1):
            out.append(WaterSpec("swim", length, 2, 2, drop))
    return out


def water_piece(prev: np.ndarray, heading: int, spec: WaterSpec) -> tuple[np.ndarray, list, list]:
    """(exit platform, basin cells, water cells) of ``spec`` after takeoff ``prev`` (x0, x1, z0, z1, y) going
    ``heading``. The water surface row is ``prev``'s row minus ``drop``; the exit is the far wall's top row and the
    row's next cell beyond it (2 long), as wide as the pool; the basin's floor is under the water and its walls,
    and the walls rise to the surface row on all four sides."""
    hx, hz = C.HEADINGS[heading]
    x0, x1, z0, z1, y = (int(v) for v in prev)
    rim = y - spec.drop  # the water's top row (its surface at rim + 8/9) and the walls' top row
    bottom = rim - spec.depth + 1  # the lowest water row
    floor = bottom - 1
    # Along the heading: near wall a, water a+1 .. a+L, far wall a+L+1 (the exit's first cell), then one more.
    if hx != 0:
        lo, hi = z0, z0 + spec.width - 1  # (the pool's side starts at the takeoff's)
        if hx > 0:
            near = x1 + spec.gap + 1
            along = [near + i for i in range(spec.length + 3)]
        else:
            near = x0 - spec.gap - 1
            along = [near - i for i in range(spec.length + 3)]
        cross = list(range(lo - 1, hi + 2))  # the side walls one cell out
        wcells = [(along[1 + i], yy, zz) for i in range(spec.length) for zz in range(lo, hi + 1)
                  for yy in range(bottom, rim + 1)]
        exit_lo, exit_hi = sorted((along[spec.length + 1], along[spec.length + 2]))
        new = np.array([exit_lo, exit_hi, lo, hi, rim], dtype=np.int64)
        cells = [(xx, floor, zz) for xx in along[:spec.length + 2] for zz in cross]  # the floor
        for yy in range(bottom, rim + 1):
            cells += [(along[0], yy, zz) for zz in cross]  # the near wall
            cells += [(along[spec.length + 1], yy, zz) for zz in cross if not (yy == rim and lo <= zz <= hi)]
            cells += [(along[1 + i], yy, zz) for i in range(spec.length) for zz in (lo - 1, hi + 1)]
    else:
        lo, hi = x0, x0 + spec.width - 1
        if hz > 0:
            near = z1 + spec.gap + 1
            along = [near + i for i in range(spec.length + 3)]
        else:
            near = z0 - spec.gap - 1
            along = [near - i for i in range(spec.length + 3)]
        cross = list(range(lo - 1, hi + 2))
        wcells = [(xx, yy, along[1 + i]) for i in range(spec.length) for xx in range(lo, hi + 1)
                  for yy in range(bottom, rim + 1)]
        exit_lo, exit_hi = sorted((along[spec.length + 1], along[spec.length + 2]))
        new = np.array([lo, hi, exit_lo, exit_hi, rim], dtype=np.int64)
        cells = [(xx, floor, zz) for zz in along[:spec.length + 2] for xx in cross]
        for yy in range(bottom, rim + 1):
            cells += [(xx, yy, along[0]) for xx in cross]
            cells += [(xx, yy, along[spec.length + 1]) for xx in cross if not (yy == rim and lo <= xx <= hi)]
            cells += [(xx, yy, along[1 + i]) for i in range(spec.length) for xx in (lo - 1, hi + 1)]
    water = [(x, yy, z, 0) for x, yy, z in wcells]
    return new, sorted(set(cells)), water


def _clear(cells: list, water: list, platforms: list, new: np.ndarray, grid=C.GRID) -> bool:
    """The piece's cells inside the grid and clear of every older platform but the takeoff (with its headroom)."""
    gx, gy, gz = grid
    pts = list(cells) + [(x, y, z) for x, y, z, _ in water]
    if any(not (0 <= x < gx and 1 <= y < gy - C.HEADROOM and 0 <= z < gz) for x, y, z in pts):
        return False
    occupied = set()
    for x0, x1, z0, z1, y in (np.asarray(p).tolist() for p in platforms[:-1]):
        for xx in range(x0 - 1, x1 + 2):
            for zz in range(z0 - 1, z1 + 2):
                for yy in range(y - 1, y + C.HEADROOM + 1):
                    occupied.add((xx, yy, zz))
    x0, x1, z0, z1, y = (int(v) for v in platforms[-1])  # the takeoff: its own column and headroom only
    for xx in range(x0, x1 + 1):
        for zz in range(z0, z1 + 1):
            for yy in range(y, y + C.HEADROOM + 1):
                occupied.add((xx, yy, zz))
    return not any(p in occupied for p in pts)


def water_course(rng: np.random.Generator, basic: list[list[C.JumpSpec]], pieces: dict, n_specs: int = 8,
                 grid=C.GRID, turn_prob: float = 0.25, yaw_noise: float = math.radians(60), level: int = 0,
                 seed: int = 0) -> C.Course | None:
    """A course of ``n_specs`` jumps whose jumps at the positions of ``pieces`` (position -> ``WaterSpec``) are
    water pieces and the others ordinary jumps of ``basic`` (as ``parkour.slime.generate_fixed`` draws them).
    None if a jump does not fit in 30 attempts."""
    gx, gy, gz = grid
    s = int(rng.choice([1, 2, 3], p=[0.4, 0.3, 0.3]))
    heading = int(rng.integers(4))
    hx, hz = C.HEADINGS[heading]
    y0 = int(rng.integers(16, gy - 8))  # high enough for a drop into a deep pool
    if hx > 0:
        x0 = int(rng.integers(3, 10))
    elif hx < 0:
        x0 = int(rng.integers(gx - 10 - s, gx - 3 - s))
    else:
        x0 = int(rng.integers(8, gx - 8 - s))
    if hz > 0:
        z0 = int(rng.integers(3, 10))
    elif hz < 0:
        z0 = int(rng.integers(gz - 10 - s, gz - 3 - s))
    else:
        z0 = int(rng.integers(8, gz - 8 - s))
    platforms = [np.array([x0, x0 + s - 1, z0, z0 + s - 1, y0], dtype=np.int64)]
    specs, extras, water = [], [], []
    for j in range(n_specs):
        placed = False
        for attempt in range(30):
            h = heading
            if rng.random() < turn_prob or attempt >= 15:
                h = (heading + int(rng.choice([1, 3]))) % 4
            if j in pieces:
                new, cells, wcells = water_piece(platforms[-1], h, pieces[j])
                # The exit as any platform (older obstacles out of its way), the basin clear of older platforms
                # and out of every earlier jump's flight corridor.
                if not C._fits(new, platforms, grid, extras=extras) or not _clear(cells, wcells, platforms, new, grid):
                    continue
                if any(C._boxes_overlap(C._corridor(platforms[i], platforms[i + 1]), (x, x, y, y, z, z))
                       for i in range(len(platforms) - 1) for x, y, z in cells):
                    continue
                platforms.append(new)
                extras.extend(cells)
                water.extend(wcells)
                specs.append(C.JumpSpec(pieces[j].gap + pieces[j].length + 1, int(new[4] - platforms[-2][4]), 0,
                                        (2, 1)))
            else:
                lv = int(rng.integers(len(basic)))
                spec = basic[lv][int(rng.integers(len(basic[lv])))]
                new = C._platform_after(platforms[-1], h, spec)
                # (a pool's exit stands on its far wall: cells right under the takeoff are in no jump's way)
                t0x, t1x, t0z, t1z, ty = (int(v) for v in platforms[-1])
                ex = [e for e in extras if not (t0x <= e[0] <= t1x and t0z <= e[2] <= t1z and e[1] < ty)]
                if not C._fits(new, platforms, grid, extras=ex):
                    continue
                wet = {(x, y, z) for x, y, z, _ in water}
                if any(C._boxes_overlap(C._corridor(platforms[-1], new), (x, x, y, y, z, z)) for x, y, z in wet):
                    continue
                platforms.append(new)
                specs.append(spec)
            heading = h
            placed = True
            break
        if not placed:
            return None
    c = C.Course(np.stack(platforms), np.zeros(3), 0.0, level=level, seed=seed, specs=specs,
                 extra=np.array(sorted(set(extras)), dtype=np.int64).reshape(-1, 3),
                 water=np.array(water, dtype=np.int64).reshape(-1, 4))
    if len(c.extra) > C.MAX_EXTRA:
        return None
    centres = c.centers()
    c.spawn = centres[0].copy()
    d = centres[1] - centres[0]
    c.spawn_yaw = float(math.atan2(-d[0], -d[2]) + rng.uniform(-yaw_noise, yaw_noise))
    return c

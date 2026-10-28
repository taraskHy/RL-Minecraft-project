"""Procedural parkour courses.

A course is a chain of rectangular one-block-thick platforms floating in the
void. The agent starts on platform 0 and must land on every platform in turn
until it stands on the last one (the goal). Platforms are axis-aligned boxes
of stone; nothing else is solid.

Each jump between consecutive platforms is described by a ``JumpSpec``:

* ``gap``: air blocks between the two platforms along the heading,
* ``dy``: target top minus takeoff top (-3 .. +1),
* ``lateral``: sideways offset of the target's centre line (blocks),
* ``size``: the target platform's extent (along, across) the heading.

Difficulty levels group jump specs by how forgiving they are (see
``configs/jump_levels.json``, produced by ``parkour.feasibility``).

Optional obstacles (``obstacle_prob`` > 0; off by default) add blocks that
are not platforms: a *beam* across the gap three blocks above the takeoff
surface, which the player's head hits at the top of a jump (a "head hitter";
placed only on jump types that stay possible with it, see
``configs/jump_beam.json``), and decoy *pillars* beside a landing platform.

Advanced jumps (levels 6 and up, ``configs/jump_levels_advanced.json``):

* *neo*: a 2-high wall fills the first gap cell, so the player must curve
  around its edge in the air. ``neo`` = 1 is a wall flush with the platforms'
  edge on the open side ``side`` (a neo), 2 sticks out one more block (a double
  neo), 3 two more (a triple neo). On the other side the wall runs 3 blocks
  past the platforms.
* *momentum*: ``runway`` > 0 first places a runway (``runway`` long, 1 wide)
  one block ahead, so the jump itself starts with speed (e.g. a 4-block gap,
  a "5 block jump").

Extra jumps (opt-in levels after the others, ``configs/jump_levels_extra.json``;
they change nothing unless loaded with ``load_levels(extra=True)``):

* *head hitter*: ``ceiling`` 1 puts a ceiling 2 blocks above a runway's
  surface over every runway block but the first, so the run-up and the
  takeoff are under a 2-high roof (the head hits it at once, which caps the
  jump's rise at 0.2 blocks); ``ceiling`` 2 continues the roof over the gap
  (a tunnel). The first runway block stays open so the jump onto the runway
  can land.
* *ice*: ``ice`` 1 makes the runway ice (slipperiness 0.98: slow to speed
  up, faster at the end).
* *shaped blocks*: ``block`` names the target's block shape (a bottom slab,
  soul sand, a fence post, ...; configs/block_shapes.json), so its top is not
  a whole block (``Course.shapes``). A fixed easy exit jump (1-block gap, to a
  2x2 stone platform level with the shape's block row, or one row up after a
  post) follows, so every other jump still starts from stone.
* *slime bounce*: ``bounce`` > 0 makes the target a slime block (a pad, below
  the takeoff); the player bounces off it to an exit platform ``bounce``
  blocks above the pad and 2 blocks further on (``Course.slime``).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .paths import CONFIGS

HEADINGS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # +X, +Z, -X, -Z
GRID = (64, 32, 64)
MAX_PLATFORMS = 16
HEADROOM = 4  # air kept above every platform and along every flight path
MAX_EXTRA = 256  # obstacle cells a course may have (env.MAX_EXTRA); checked for head-hitter roofs


@dataclass(frozen=True)
class JumpSpec:
    gap: int
    dy: int
    lateral: int
    size: tuple = (1, 1)
    neo: int = 0  # 0: no wall; 1 neo, 2 double neo, 3 triple neo
    side: int = 1  # neo: +1 go around on the right of the heading, -1 on the left
    runway: int = 0  # momentum: length of a runway placed before the jump
    wall_at: int = 1  # neo: which gap cell (1 = right after the takeoff) holds the wall
    ceiling: int = 0  # head hitter (runway jumps): 1 roof over the runway after its first block, 2 also over the gap
    ice: int = 0  # 1: the runway is ice
    block: str = ""  # shaped target: a shape name of configs/block_shapes.json ("" = stone)
    bounce: int = 0  # >0: the target is a slime pad; the exit platform after it is this many blocks above it

    def key(self) -> str:
        k = f"g{self.gap}_dy{self.dy}_l{self.lateral}_s{self.size[0]}x{self.size[1]}"
        if self.neo:
            k += f"_neo{self.neo}{'R' if self.side > 0 else 'L'}"
            if self.wall_at != 1:
                k += f"w{self.wall_at}"
        if self.runway:
            k += f"_run{self.runway}"
        if self.ceiling:
            k += f"_hh{self.ceiling}"
        if self.ice:
            k += f"_ice{self.ice}"
        if self.block:
            k += f"_b{self.block}"
        if self.bounce:
            k += f"_slime{self.bounce}"
        return k


def _spec_from(d: dict) -> "JumpSpec":
    return JumpSpec(d["gap"], d["dy"], d["lateral"], tuple(d["size"]), d.get("neo", 0), d.get("side", 1),
                    d.get("runway", 0), d.get("wall_at", 1), d.get("ceiling", 0), d.get("ice", 0), d.get("block", ""),
                    d.get("bounce", 0))


@lru_cache(maxsize=None)
def shape_geometry(name: str) -> tuple:
    """(top, x0, x1, z0, z1) of a block shape inside its cell: the highest box top and the boxes' footprint."""
    for d in json.loads((CONFIGS / "block_shapes.json").read_text())["shapes"]:
        if d["name"] == name:
            b = d["boxes"]
            return (max(x[4] for x in b), min(x[0] for x in b), max(x[3] for x in b), min(x[2] for x in b),
                    max(x[5] for x in b))
    raise KeyError(name)


def exit_spec(block: str) -> "JumpSpec":
    """The fixed jump off a shaped platform: 1-block gap to 2x2 stone, one row up after a post (top above 1)."""
    return JumpSpec(1, 1 if shape_geometry(block)[0] > 1 else 0, 0, (2, 2))


def bounce_exit(spec: "JumpSpec") -> "JumpSpec":
    """The jump off a slime pad: a 2-block gap to 2x2 stone ``spec.bounce`` blocks above the pad."""
    return JumpSpec(2, spec.bounce, 0, (2, 2))


@dataclass
class Course:
    platforms: np.ndarray  # (K, 5) int64: x0, x1, z0, z1, y (inclusive cell ranges; top face at y + 1)
    spawn: np.ndarray  # (3,) float64 feet position
    spawn_yaw: float
    level: int = 0
    seed: int = 0
    specs: list = field(default_factory=list)
    name: str = ""
    extra: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))  # obstacle cells
    ice: list = field(default_factory=list)  # indices of platforms made of ice (optional; slipperiness 0.98)
    shapes: dict = field(default_factory=dict)  # platform index -> block shape name (optional; default stone)
    slime: list = field(default_factory=list)  # indices of platforms made of slime blocks (optional; they bounce)

    @property
    def n(self) -> int:
        return len(self.platforms)

    def blocks(self) -> np.ndarray:
        cells = []
        for x0, x1, z0, z1, y in self.platforms:
            xs, zs = np.meshgrid(np.arange(x0, x1 + 1), np.arange(z0, z1 + 1), indexing="ij")
            cells.append(np.stack((xs.ravel(), np.full(xs.size, y), zs.ravel()), -1))
        if len(self.extra):
            cells.append(np.asarray(self.extra, dtype=np.int64).reshape(-1, 3))
        return np.concatenate(cells).astype(np.int64)

    def ice_cells(self) -> np.ndarray:
        """(M, 3) cells of the ice platforms."""
        cells = [np.zeros((0, 3), dtype=np.int64)]
        for k in self.ice:
            x0, x1, z0, z1, y = (int(v) for v in self.platforms[k])
            xs, zs = np.meshgrid(np.arange(x0, x1 + 1), np.arange(z0, z1 + 1), indexing="ij")
            cells.append(np.stack((xs.ravel(), np.full(xs.size, y), zs.ravel()), -1))
        return np.concatenate(cells).astype(np.int64)

    def slime_cells(self) -> np.ndarray:
        """(M, 3) cells of the slime platforms."""
        cells = [np.zeros((0, 3), dtype=np.int64)]
        for k in self.slime:
            x0, x1, z0, z1, y = (int(v) for v in self.platforms[k])
            xs, zs = np.meshgrid(np.arange(x0, x1 + 1), np.arange(z0, z1 + 1), indexing="ij")
            cells.append(np.stack((xs.ravel(), np.full(xs.size, y), zs.ravel()), -1))
        return np.concatenate(cells).astype(np.int64)

    def centers(self) -> np.ndarray:
        p = self.platforms.astype(np.float64)
        c = np.stack(((p[:, 0] + p[:, 1] + 1) / 2, p[:, 4] + 1, (p[:, 2] + p[:, 3] + 1) / 2), -1)
        for k, name in self.shapes.items():  # a shaped platform: the centre of its footprint, on its top
            top, x0, x1, z0, z1 = shape_geometry(name)
            q = self.platforms[k]
            c[k] = ((q[0] + x0 + q[1] + x1) / 2, q[4] + top, (q[2] + z0 + q[3] + z1) / 2)
        return c

    def to_json(self) -> dict:
        d = {"platforms": self.platforms.tolist(), "spawn": self.spawn.tolist(), "spawn_yaw": self.spawn_yaw,
             "level": self.level, "seed": self.seed, "specs": [s.key() for s in self.specs], "name": self.name,
             "extra": np.asarray(self.extra).reshape(-1, 3).tolist()}
        if self.ice:  # only when used, so the files of courses without ice are unchanged
            d["ice"] = [int(k) for k in self.ice]
        if self.shapes:
            d["shapes"] = {str(k): v for k, v in sorted(self.shapes.items())}
        if self.slime:
            d["slime"] = [int(k) for k in self.slime]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Course":
        return cls(np.array(d["platforms"], dtype=np.int64), np.array(d["spawn"], dtype=np.float64),
                   float(d["spawn_yaw"]), int(d.get("level", 0)), int(d.get("seed", 0)), name=d.get("name", ""),
                   extra=np.array(d.get("extra", []), dtype=np.int64).reshape(-1, 3), ice=[int(k) for k in d.get("ice", [])],
                   shapes={int(k): v for k, v in d.get("shapes", {}).items()}, slime=[int(k) for k in d.get("slime", [])])


def load_levels(path: Path | None = None, advanced: bool = False, extra: bool = False) -> list[list[JumpSpec]]:
    """Jump specs per difficulty level, from the feasibility study.

    With ``advanced``, the neo and momentum levels from
    ``configs/jump_levels_advanced.json`` follow the six basic levels. With
    ``extra``, the head-hitter and ice levels of ``configs/jump_levels_extra.json``
    follow those (so their level numbers depend on ``advanced``).
    """
    path = path or CONFIGS / "jump_levels.json"
    data = json.loads(Path(path).read_text())
    levels = [[_spec_from(s) for s in level["specs"]] for level in data["levels"]]
    if advanced:
        adv = json.loads((CONFIGS / "jump_levels_advanced.json").read_text())
        levels += [[_spec_from(s) for s in level["specs"]] for level in adv["levels"]]
    if extra:
        ext = json.loads((CONFIGS / "jump_levels_extra.json").read_text())
        levels += [[_spec_from(s) for s in level["specs"]] for level in ext["levels"]]
    return levels


def _platform_after(prev: np.ndarray, heading: int, spec: JumpSpec) -> np.ndarray:
    """Place the target platform for ``spec`` after ``prev`` going ``heading``."""
    hx, hz = HEADINGS[heading]
    x0, x1, z0, z1, y = (int(v) for v in prev)
    along, across = spec.size
    if hx != 0:
        # Lateral axis is Z. Right of +X is +Z; right of -X is -Z.
        if hx > 0:
            nx0 = x1 + spec.gap + 1
            nx1 = nx0 + along - 1
        else:
            nx1 = x0 - spec.gap - 1
            nx0 = nx1 - along + 1
        centre = (z0 + z1 + 1) / 2 + hx * spec.lateral
        nz0 = int(math.floor(centre - across / 2 + 0.5))
        nz1 = nz0 + across - 1
    else:
        # Lateral axis is X. Right of +Z is -X; right of -Z is +X.
        if hz > 0:
            nz0 = z1 + spec.gap + 1
            nz1 = nz0 + along - 1
        else:
            nz1 = z0 - spec.gap - 1
            nz0 = nz1 - along + 1
        centre = (x0 + x1 + 1) / 2 - hz * spec.lateral
        nx0 = int(math.floor(centre - across / 2 + 0.5))
        nx1 = nx0 + across - 1
    return np.array([nx0, nx1, nz0, nz1, y + spec.dy], dtype=np.int64)


def _box_of(p: np.ndarray, up: int = HEADROOM) -> tuple:
    return (p[0], p[1], p[4], p[4] + up, p[2], p[3])


def _boxes_overlap(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3] and a[4] <= b[5] and b[4] <= a[5]


def _fits(new: np.ndarray, platforms: list, grid=GRID, margin: int = 2, extras=()) -> bool:
    gx, gy, gz = grid
    if new[0] < margin or new[1] > gx - 1 - margin or new[2] < margin or new[3] > gz - 1 - margin:
        return False
    if new[4] < 3 or new[4] > gy - 1 - HEADROOM - 1:
        return False
    prev = platforms[-1]
    # The flight corridor between prev and new must be clear of other platforms.
    corridor = (min(prev[0], new[0]), max(prev[1], new[1]), min(prev[4], new[4]) - 1,
                max(prev[4], new[4]) + HEADROOM, min(prev[2], new[2]), max(prev[3], new[3]))
    # Keep one air block between the new platform and every older one, and its headroom free.
    grown = (new[0] - 1, new[1] + 1, new[4] - HEADROOM, new[4] + HEADROOM, new[2] - 1, new[3] + 1)
    for i, p in enumerate(platforms):
        pb = (p[0], p[1], p[4], p[4], p[2], p[3])
        if _boxes_overlap(grown, pb) and i != len(platforms) - 1:
            return False
        if i < len(platforms) - 1 and _boxes_overlap(corridor, pb):
            return False
    # The new platform must not sit inside prev's headroom or touch it at the same height.
    if _boxes_overlap((new[0], new[1], new[4], new[4], new[2], new[3]), _box_of(prev)):
        return False
    # Obstacles already placed stay out of the new platform's surroundings and flight path.
    for x, y, z in extras:
        cell = (x, x, y, y, z, z)
        if _boxes_overlap(grown, cell) or _boxes_overlap(corridor, cell):
            return False
    return True


def _corridor(a: np.ndarray, b: np.ndarray) -> tuple:
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[4], b[4]) - 1, max(a[4], b[4]) + HEADROOM,
            min(a[2], b[2]), max(a[3], b[3]))


def _neo_wall(prev: np.ndarray, new: np.ndarray, heading: int, spec: JumpSpec, reach: int = 3,
              top: int | None = None) -> list:
    """Cells of a neo wall: gap cell ``spec.wall_at``, from the takeoff surface up.

    It is 2 high above the takeoff surface, or reaches 2 above block row ``top``
    if that is higher, so no nearby platform lets the player land on its top.
    """
    hx, hz = HEADINGS[heading]
    over = spec.neo - 1  # blocks the wall sticks out past the platforms on the open side
    ys = tuple(range(int(prev[4]) + 1, max(int(prev[4]), int(top if top is not None else prev[4])) + 3))
    if hx != 0:
        along = prev[1] + spec.wall_at if hx > 0 else prev[0] - spec.wall_at
        lo, hi = min(prev[2], new[2]), max(prev[3], new[3])
        open_dir = spec.side * hx  # right of +X is +Z
        z0, z1 = (lo - reach, hi + over) if open_dir > 0 else (lo - over, hi + reach)
        return [(int(along), y, int(z)) for z in range(z0, z1 + 1) for y in ys]
    along = prev[3] + spec.wall_at if hz > 0 else prev[2] - spec.wall_at
    lo, hi = min(prev[0], new[0]), max(prev[1], new[1])
    open_dir = -spec.side * hz  # right of +Z is -X
    x0, x1 = (lo - reach, hi + over) if open_dir > 0 else (lo - over, hi + reach)
    return [(int(x), y, int(along)) for x in range(x0, x1 + 1) for y in ys]


def _place(platforms: list, extras: list, heading: int, spec: JumpSpec, grid=GRID, wide_neo: bool = False):
    """Platforms and wall cells for ``spec`` after ``platforms[-1]``, or None if it does not fit.

    With ``wide_neo`` a neo may also follow a takeoff wider than one block, as long as the
    landing reaches the wall's open end (otherwise it lies two or more lanes behind the wall,
    which no search could land).
    """
    prev = platforms[-1]
    added = []
    if spec.neo:
        # A simplification of the generator, not a parkour rule: by default neos are only placed
        # after a takeoff one block wide across the jump. The wall spans the takeoff's whole
        # width, so from a wider takeoff the landing could end up two lanes from the wall's open
        # end, which is impossible (like a double neo). ``wide_neo`` allows wide takeoffs when
        # the landing reaches the open end (parkour.neo_demos finds such 2x2-takeoff neos).
        hx, _ = HEADINGS[heading]
        across = (prev[3] - prev[2] + 1) if hx != 0 else (prev[1] - prev[0] + 1)
        if across != 1 and not wide_neo:
            return None
    if spec.runway:
        run = _platform_after(prev, heading, JumpSpec(1, 0, 0, (spec.runway, 1)))
        if not _fits(run, platforms, grid, extras=extras):
            return None
        added.append(run)
        prev = run
    base = JumpSpec(spec.gap, spec.dy, spec.lateral, spec.size)
    new = _platform_after(prev, heading, base)
    if not _fits(new, platforms + added, grid, extras=extras):
        return None
    added.append(new)
    cells = []
    if spec.neo and wide_neo:
        hx, hz = HEADINGS[heading]
        if hx != 0:
            open_dir = spec.side * hx  # +1: the wall's open end is towards +z
            lo, hi = min(prev[2], new[2]), max(prev[3], new[3])
            if (open_dir > 0 and new[3] != hi) or (open_dir < 0 and new[2] != lo):
                return None
        else:
            open_dir = -spec.side * hz  # +1: towards +x
            lo, hi = min(prev[0], new[0]), max(prev[1], new[1])
            if (open_dir > 0 and new[1] != hi) or (open_dir < 0 and new[0] != lo):
                return None
    if spec.block or spec.bounce:
        out = _platform_after(new, heading, exit_spec(spec.block) if spec.block else bounce_exit(spec))
        if not _fits(out, platforms + added, grid, extras=extras):
            return None
        added.append(out)
    if spec.ceiling:
        cells = _ceiling(prev, new, heading, spec)
        allp = platforms + added
        if len(extras) + len(cells) > MAX_EXTRA or not _roof_clear(cells, allp, len(allp) - 2, grid):
            return None
    if spec.neo:
        # Taller than every platform from two before the takeoff to the target: none can reach its top.
        top = max(int(p[4]) for p in (platforms[-3:] + added))
        cells = _neo_wall(prev, new, heading, spec, top=top)
        allp = platforms + added
        k = len(allp) - 1
        if not _clear_of(cells, allp, (k - 1, k), grid):
            return None
        if len(platforms) >= 2:
            # The wall must not stand in the way of the jump onto the takeoff platform.
            arrive = _corridor(platforms[-2], platforms[-1])
            if any(_boxes_overlap(arrive, (x, x, y, y, z, z)) for x, y, z in cells):
                return None
    return added, cells


def _beam(prev: np.ndarray, new: np.ndarray, heading: int, height: int = 3) -> list:
    """Cells of a beam across the gap, ``height`` blocks above the takeoff surface."""
    hx, hz = HEADINGS[heading]
    y = int(prev[4]) + 1 + height
    if hx != 0:
        lo, hi = (prev[1] + 1, new[0] - 1) if hx > 0 else (new[1] + 1, prev[0] - 1)
        across = range(min(prev[2], new[2]), max(prev[3], new[3]) + 1)
        return [(int(x), y, int(z)) for x in range(lo, hi + 1) for z in across]
    lo, hi = (prev[3] + 1, new[2] - 1) if hz > 0 else (new[3] + 1, prev[2] - 1)
    across = range(min(prev[0], new[0]), max(prev[1], new[1]) + 1)
    return [(int(x), y, int(z)) for z in range(lo, hi + 1) for x in across]


def _ceiling(run: np.ndarray, new: np.ndarray, heading: int, spec: JumpSpec) -> list:
    """Roof cells of a head-hitter runway: 2 blocks of air above its surface over every runway block
    but the first (where the jump onto the runway lands), and with ``spec.ceiling`` 2 over the gap too."""
    hx, hz = HEADINGS[heading]
    y = int(run[4]) + 3
    x0, x1, z0, z1 = (int(v) for v in run[:4])
    if hx > 0:
        cells = [(x, y, z) for x in range(x0 + 1, x1 + 1) for z in range(z0, z1 + 1)]
    elif hx < 0:
        cells = [(x, y, z) for x in range(x0, x1) for z in range(z0, z1 + 1)]
    elif hz > 0:
        cells = [(x, y, z) for z in range(z0 + 1, z1 + 1) for x in range(x0, x1 + 1)]
    else:
        cells = [(x, y, z) for z in range(z0, z1) for x in range(x0, x1 + 1)]
    if spec.ceiling >= 2:
        cells += _beam(run, new, heading, height=2)
    return cells


def _roof_clear(cells: list, platforms: list, run_k: int, grid=GRID) -> bool:
    """Roof cells inside the grid and off every platform but their own runway ``run_k``: the target
    (``run_k`` + 1) keeps its column of headroom free, all others also a one-block margin."""
    gx, gy, gz = grid
    for x, y, z in cells:
        if not (0 <= x < gx and 0 <= y < gy and 0 <= z < gz):
            return False
        for i, p in enumerate(platforms):
            if i == run_k:
                continue
            if i == run_k + 1:
                box = (p[0], p[1], p[4], p[4] + HEADROOM, p[2], p[3])
            else:
                box = (p[0] - 1, p[1] + 1, p[4] - 1, p[4] + HEADROOM, p[2] - 1, p[3] + 1)
            if _boxes_overlap(box, (x, x, y, y, z, z)):
                return False
    return True


def _pillar(new: np.ndarray, heading: int, side: int) -> list:
    """A 4-high decoy pillar two blocks to one side of a platform, level with its middle."""
    hx, hz = HEADINGS[heading]
    y0 = int(new[4]) - 1
    if hx != 0:
        z = new[3] + 2 if side * hx > 0 else new[2] - 2
        x = (new[0] + new[1]) // 2
    else:
        x = new[0] - 2 if side * hz > 0 else new[1] + 2
        z = (new[2] + new[3]) // 2
    return [(int(x), y, int(z)) for y in range(y0, y0 + 4)]


def _clear_of(cells: list, platforms: list, skip: tuple, grid=GRID) -> bool:
    """Obstacle cells inside the grid and away from every platform and its headroom.

    Platforms in ``skip`` (the two ends of a beam's jump) only keep their own
    column of headroom free; all others also keep a one-block margin.
    """
    gx, gy, gz = grid
    for x, y, z in cells:
        if not (0 <= x < gx and 0 <= y < gy and 0 <= z < gz):
            return False
        for i, p in enumerate(platforms):
            if i in skip:
                box = (p[0], p[1], p[4], p[4] + HEADROOM, p[2], p[3])
            else:
                box = (p[0] - 1, p[1] + 1, p[4] - 1, p[4] + HEADROOM, p[2] - 1, p[3] + 1)
            if _boxes_overlap(box, (x, x, y, y, z, z)):
                return False
    return True


def load_beam_ok(path: Path | None = None) -> set:
    path = path or CONFIGS / "jump_beam.json"
    return {r["key"] for r in json.loads(Path(path).read_text())["specs"] if r["beam_ok"]}


def generate(rng: np.random.Generator, levels: list[list[JumpSpec]], level: int, n_jumps: int,
             grid=GRID, turn_prob: float = 0.25, start_size: int | None = None, seed: int = 0,
             yaw_noise: float = math.radians(60), obstacle_prob: float = 0.0, beam_ok: set | None = None,
             wide_neo: bool = False, ice_prob: float = 0.0) -> Course:
    """One random course whose jumps come from ``levels[:level + 1]``.

    Half of the jumps come from ``level`` itself and half from easier levels,
    so every course mixes new and familiar jumps. With ``ice_prob`` each
    platform after the start is ice with that chance (drawn last, so the
    layout is the one the same random numbers give without ice).
    """
    gx, gy, gz = grid
    s = start_size or int(rng.choice([1, 2, 3], p=[0.4, 0.3, 0.3]))
    heading = int(rng.integers(4))
    hx, hz = HEADINGS[heading]
    # Start near the side opposite the heading so the course has room to run.
    y0 = int(rng.integers(10, gy - 14))
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
    specs = []
    extras: list = []
    ice_runways: list = []
    shaped: dict = {}
    pads: list = []
    if obstacle_prob > 0 and beam_ok is None:
        beam_ok = load_beam_ok()
    for _ in range(n_jumps):
        if len(platforms) >= MAX_PLATFORMS:  # runways add platforms; only reachable with momentum specs
            break
        placed = False
        for attempt in range(30):
            lv = level if (rng.random() < 0.5 or level == 0) else int(rng.integers(0, level))
            spec = levels[lv][int(rng.integers(len(levels[lv])))]
            h = heading
            if rng.random() < turn_prob or attempt >= 15:
                h = (heading + int(rng.choice([1, 3]))) % 4
            if spec.neo or spec.runway or spec.block or spec.bounce:
                if len(platforms) + 1 + (1 if spec.runway else 0) + (1 if spec.block or spec.bounce else 0) > MAX_PLATFORMS:
                    continue
                placed_parts = _place(platforms, extras, h, spec, grid, wide_neo=wide_neo)
                if placed_parts is None:
                    continue
                new_platforms, wall = placed_parts
                if spec.runway:
                    specs.append(JumpSpec(1, 0, 0, (spec.runway, 1)))
                if spec.ice:
                    ice_runways.append(len(platforms))
                if spec.block:
                    shaped[len(platforms)] = spec.block
                if spec.bounce:
                    pads.append(len(platforms))
                platforms.extend(new_platforms)
                extras.extend(wall)
                specs.append(spec)
                if spec.block:
                    specs.append(exit_spec(spec.block))
                if spec.bounce:
                    specs.append(bounce_exit(spec))
                heading = h
                placed = True
                break
            new = _platform_after(platforms[-1], h, spec)
            if _fits(new, platforms, grid, extras=extras):
                platforms.append(new)
                specs.append(spec)
                # Obstacles draw random numbers only when enabled, so default courses are unchanged.
                if obstacle_prob > 0 and rng.random() < obstacle_prob:
                    k = len(platforms) - 1
                    if rng.random() < 0.6 and spec.gap >= 1 and spec.key() in beam_ok:
                        cells = _beam(platforms[-2], new, h)
                        skip = (k - 1, k)
                    else:
                        cells = _pillar(new, h, int(rng.choice([-1, 1])))
                        skip = ()
                    if _clear_of(cells, platforms, skip, grid):
                        extras.extend(cells)
                heading = h
                placed = True
                break
        if not placed:
            break
    plats = np.stack(platforms)
    c = Course(plats, np.zeros(3), 0.0, level=level, seed=seed, specs=specs,
               extra=np.array(extras, dtype=np.int64).reshape(-1, 3), ice=ice_runways, shapes=shaped, slime=pads)
    centres = c.centers()
    c.spawn = centres[0].copy()
    d = centres[1] - centres[0]
    yaw_to = math.atan2(-d[0], -d[2])  # Mineflayer yaw that faces direction d
    c.spawn_yaw = float(yaw_to + rng.uniform(-yaw_noise, yaw_noise))
    if ice_prob > 0:
        c.ice = sorted(set(c.ice) | {k for k in range(1, c.n) if rng.random() < ice_prob})
    return c


def single_jump(spec: JumpSpec, takeoff: tuple = (1, 1), grid=(24, 16, 24), heading: int = 0) -> Course:
    """A two-platform course for one jump, used by the feasibility study.

    A momentum spec takes off from its runway (``spec.runway`` x 1); a neo spec
    gets its wall.
    """
    gx, gy, gz = grid
    hx, hz = HEADINGS[heading]
    a, b = (spec.runway, 1) if spec.runway else takeoff
    y0 = 8
    x0, z0 = 4, gz // 2 - b // 2
    prev = np.array([x0, x0 + a - 1, z0, z0 + b - 1, y0], dtype=np.int64)
    new = _platform_after(prev, heading, JumpSpec(spec.gap, spec.dy, spec.lateral, spec.size))
    plats = np.stack([prev, new])
    extra = _neo_wall(prev, new, heading, spec) if spec.neo else []
    if spec.ceiling:
        extra = extra + _ceiling(prev, new, heading, spec)
    c = Course(plats, np.zeros(3), 0.0, specs=[spec], extra=np.array(extra, dtype=np.int64).reshape(-1, 3),
               ice=[0] if spec.ice else [], shapes={1: spec.block} if spec.block else {}, slime=[1] if spec.bounce else [])
    centres = c.centers()
    c.spawn = centres[0].copy()
    d = centres[1] - centres[0]
    c.spawn_yaw = float(math.atan2(-d[0], -d[2]))
    return c


def from_specs(steps: list, start=(4, 12, 30), start_size: int = 3, heading: int = 0, name: str = "") -> Course:
    """A hand-designed course: ``steps`` is a list of (JumpSpec, turn) with turn -1, 0 or +1.

    Its level is 0 (levels index generated difficulty); ``name`` identifies it.
    """
    x, y, z = start
    platforms = [np.array([x, x + start_size - 1, z, z + start_size - 1, y], dtype=np.int64)]
    extras: list = []
    ice: list = []
    shaped: dict = {}
    for spec, turn in steps:
        heading = (heading + turn) % 4
        if spec.neo or spec.runway or spec.block:
            placed = _place(platforms, extras, heading, spec, wide_neo=True)
            if placed is None:
                raise ValueError(f"{spec.key()} does not fit after platform {len(platforms) - 1}")
            if spec.ice:
                ice.append(len(platforms))
            if spec.block:
                shaped[len(platforms)] = spec.block
            platforms.extend(placed[0])
            extras.extend(placed[1])
        else:
            platforms.append(_platform_after(platforms[-1], heading, spec))
    c = Course(np.stack(platforms), np.zeros(3), 0.0, level=0, specs=[s for s, _ in steps], name=name,
               extra=np.array(extras, dtype=np.int64).reshape(-1, 3), ice=ice, shapes=shaped)
    centres = c.centers()
    c.spawn = centres[0].copy()
    d = centres[1] - centres[0]
    c.spawn_yaw = float(math.atan2(-d[0], -d[2]))
    return c


def path_remaining(course: Course) -> np.ndarray:
    """Distance from each platform centre to the goal along the chain of centres."""
    c = course.centers()
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    rem = np.zeros(len(c))
    rem[:-1] = np.cumsum(seg[::-1])[::-1]
    return rem

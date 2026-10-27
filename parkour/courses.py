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
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .paths import CONFIGS

HEADINGS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # +X, +Z, -X, -Z
GRID = (64, 32, 64)
MAX_PLATFORMS = 16
HEADROOM = 4  # air kept above every platform and along every flight path


@dataclass(frozen=True)
class JumpSpec:
    gap: int
    dy: int
    lateral: int
    size: tuple = (1, 1)

    def key(self) -> str:
        return f"g{self.gap}_dy{self.dy}_l{self.lateral}_s{self.size[0]}x{self.size[1]}"


@dataclass
class Course:
    platforms: np.ndarray  # (K, 5) int64: x0, x1, z0, z1, y (inclusive cell ranges; top face at y + 1)
    spawn: np.ndarray  # (3,) float64 feet position
    spawn_yaw: float
    level: int = 0
    seed: int = 0
    specs: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.platforms)

    def blocks(self) -> np.ndarray:
        cells = []
        for x0, x1, z0, z1, y in self.platforms:
            xs, zs = np.meshgrid(np.arange(x0, x1 + 1), np.arange(z0, z1 + 1), indexing="ij")
            cells.append(np.stack((xs.ravel(), np.full(xs.size, y), zs.ravel()), -1))
        return np.concatenate(cells).astype(np.int64)

    def centers(self) -> np.ndarray:
        p = self.platforms.astype(np.float64)
        return np.stack(((p[:, 0] + p[:, 1] + 1) / 2, p[:, 4] + 1, (p[:, 2] + p[:, 3] + 1) / 2), -1)

    def to_json(self) -> dict:
        return {"platforms": self.platforms.tolist(), "spawn": self.spawn.tolist(), "spawn_yaw": self.spawn_yaw,
                "level": self.level, "seed": self.seed, "specs": [s.key() for s in self.specs]}

    @classmethod
    def from_json(cls, d: dict) -> "Course":
        return cls(np.array(d["platforms"], dtype=np.int64), np.array(d["spawn"], dtype=np.float64),
                   float(d["spawn_yaw"]), int(d.get("level", 0)), int(d.get("seed", 0)))


def load_levels(path: Path | None = None) -> list[list[JumpSpec]]:
    """Jump specs per difficulty level, from the feasibility study."""
    path = path or CONFIGS / "jump_levels.json"
    data = json.loads(Path(path).read_text())
    return [[JumpSpec(s["gap"], s["dy"], s["lateral"], tuple(s["size"])) for s in level["specs"]]
            for level in data["levels"]]


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


def _fits(new: np.ndarray, platforms: list, grid=GRID, margin: int = 2) -> bool:
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
    return True


def generate(rng: np.random.Generator, levels: list[list[JumpSpec]], level: int, n_jumps: int,
             grid=GRID, turn_prob: float = 0.25, start_size: int | None = None, seed: int = 0,
             yaw_noise: float = math.radians(60)) -> Course:
    """One random course whose jumps come from ``levels[:level + 1]``.

    Half of the jumps come from ``level`` itself and half from easier levels,
    so every course mixes new and familiar jumps.
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
    for _ in range(n_jumps):
        placed = False
        for attempt in range(30):
            lv = level if (rng.random() < 0.5 or level == 0) else int(rng.integers(0, level))
            spec = levels[lv][int(rng.integers(len(levels[lv])))]
            h = heading
            if rng.random() < turn_prob or attempt >= 15:
                h = (heading + int(rng.choice([1, 3]))) % 4
            new = _platform_after(platforms[-1], h, spec)
            if _fits(new, platforms, grid):
                platforms.append(new)
                specs.append(spec)
                heading = h
                placed = True
                break
        if not placed:
            break
    plats = np.stack(platforms)
    c = Course(plats, np.zeros(3), 0.0, level=level, seed=seed, specs=specs)
    centres = c.centers()
    c.spawn = centres[0].copy()
    d = centres[1] - centres[0]
    yaw_to = math.atan2(-d[0], -d[2])  # Mineflayer yaw that faces direction d
    c.spawn_yaw = float(yaw_to + rng.uniform(-yaw_noise, yaw_noise))
    return c


def single_jump(spec: JumpSpec, takeoff: tuple = (1, 1), grid=(24, 16, 24), heading: int = 0) -> Course:
    """A two-platform course for one jump, used by the feasibility study."""
    gx, gy, gz = grid
    hx, hz = HEADINGS[heading]
    a, b = takeoff
    y0 = 8
    x0, z0 = 4, gz // 2 - b // 2
    prev = np.array([x0, x0 + a - 1, z0, z0 + b - 1, y0], dtype=np.int64)
    new = _platform_after(prev, heading, spec)
    plats = np.stack([prev, new])
    c = Course(plats, np.zeros(3), 0.0, specs=[spec])
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

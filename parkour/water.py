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

The held-out water sets (``water_set``; HANDOFF.md section 3, 1 Nov 21:20): families 100 (validation) and 101
(test), 288 courses of 8 jumps, 2 of them water pieces (``water_plan``); training courses with one piece on
families 102-107 (``water_training_courses``). ``water_takeoffs`` finds a course's water jumps, ``water_cps`` and
``water_blocks`` hand them to the water skill (``neo_assist --skill-blocks water``, ``neo_skill bank --water``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

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


WATER_VALIDATION_FAMILY = 100  # the water sets (HANDOFF.md section 3, 1 Nov); 96 and 99 are only tests' streams
WATER_TEST_FAMILY = 101
WATER_TRAIN_FAMILIES = tuple(range(102, 108))  # training courses with one water piece; nothing else uses them
WATER_PATTERNS = ((1, 4), (2, 5), (3, 6), (1, 6), (2, 7), (4, 7))  # spec positions of a set course's 2 pieces:
# never the first jump (the skill starts on the platform before the takeoff), never two in a row
WATER_SET_SIZE = 288  # every kind 24 times
MAX_DRAWS = 10_000


def water_takeoffs(course: C.Course) -> list[int]:
    """The takeoffs k of the course's water jumps (``water_piece``): water in the exit's own row (the surface's
    row, which is the exit's) inside the box of platforms k and k+1."""
    if not len(course.water):
        return []
    w = np.asarray(course.water).reshape(-1, 4)
    p = np.asarray(course.platforms)
    out = []
    for k in range(course.n - 1):
        a, b = p[k], p[k + 1]
        if ((w[:, 0] >= min(a[0], b[0])) & (w[:, 0] <= max(a[1], b[1])) & (w[:, 2] >= min(a[2], b[2]))
                & (w[:, 2] <= max(a[3], b[3])) & (w[:, 1] == b[4])).any():
            out.append(k)
    return out


def water_cps(course: C.Course) -> list[int]:
    """The checkpoints at which the water skill plays (the course-list mode, as ``slime.slime_cps``): for each
    water jump from takeoff t, t-1 and t, so it plays from the first ground contact on the platform before the
    takeoff until the bot first stands on the exit."""
    return sorted({k for t in water_takeoffs(course) for k in (t - 1, t) if k >= 0})


def water_blocks(course: C.Course) -> list[tuple[int, int]]:
    """(k, first) of each water jump's block for a bank (``neo_skill.build_bank``): k = t-1, the jump onto the
    takeoff t (the segment rule's takeoff: the bot must stand on t, k+1, before the exit, so the episode ends on
    the exit, k+2), and the platform the skill starts on (k)."""
    return [(t - 1, t - 1) for t in water_takeoffs(course) if t >= 1]


def water_ok(course: C.Course, pieces: dict, n_specs: int = 8) -> bool:
    """The water jumps exactly at the positions of ``pieces`` (spec position j: takeoff j), and no other special
    jump: no wall in any jump's way (``neo_dagger.neo_takeoffs``), no slime, shaped block or ladder."""
    from .neo_dagger import neo_takeoffs
    return (len(course.specs) == n_specs and water_takeoffs(course) == sorted(pieces) and not course.slime
            and not course.shapes and not len(course.ladders) and not neo_takeoffs(course))


def water_plan(i: int) -> dict:
    """Course i of the water sets: 2 pieces at ``WATER_PATTERNS[(i div 12) mod 6]``, kinds (2i + j) mod 24, so
    every 12 courses hold every kind once and every 72 every kind in every pattern."""
    kinds = water_specs()
    where = WATER_PATTERNS[(i // (len(kinds) // 2)) % len(WATER_PATTERNS)]
    return {p: kinds[(2 * i + j) % len(kinds)] for j, p in enumerate(where)}


def water_set(family: int, n: int = WATER_SET_SIZE) -> list[C.Course]:
    """The water set of ``family``: 8 jump specs, 2 of them water pieces (``water_plan``), the other jumps of
    levels 0-5; one random stream ``[2026, family, 6]`` in course order (the first courses of a longer set are the
    same courses), a course drawn again until its pieces are where the plan puts them (``water_ok``)."""
    basic = C.load_levels()
    rng = np.random.default_rng([2026, family, 6])
    out = []
    for i in range(n):
        plan = water_plan(i)
        for _ in range(MAX_DRAWS):
            c = water_course(rng, basic, plan, 8, level=6, seed=i)
            if c is not None and water_ok(c, plan):
                break
        else:
            raise RuntimeError(f"no water course {i} of family {family} in {MAX_DRAWS} draws")
        out.append(c)
    return out


@lru_cache(maxsize=None)
def held_water_hashes() -> frozenset:
    """``train.course_hash`` of every course of the water sets: no training course may equal one."""
    from .train import course_hash
    return frozenset(course_hash(c) for f in (WATER_VALIDATION_FAMILY, WATER_TEST_FAMILY) for c in water_set(f))


def water_training_courses(families=WATER_TRAIN_FAMILIES, per_family: int = 4032) -> tuple[list[C.Course], list[int]]:
    """Courses of 8 jump specs with one water piece (course i: kind i mod 24 at spec position 1 + (i div 24) mod 7,
    the other jumps of levels 0-5; the sets' generator), none equal to a course of the water sets; each course's
    level set to its kind (0-23)."""
    from .train import course_hash
    held = held_water_hashes()
    kinds = water_specs()
    basic = C.load_levels()
    out, types = [], []
    for f in families:
        if f in (WATER_VALIDATION_FAMILY, WATER_TEST_FAMILY):
            raise ValueError(f"family {f} is a held-out set")
        rng = np.random.default_rng([2026, f, 6])
        for i in range(per_family):
            t = i % len(kinds)
            plan = {1 + (i // len(kinds)) % 7: kinds[t]}
            for _ in range(MAX_DRAWS):
                c = water_course(rng, basic, plan, 8, level=t, seed=i)
                if c is not None and water_ok(c, plan):
                    break
            else:
                continue
            if course_hash(c) in held:
                continue
            out.append(c)
            types.append(t)
    return out, types

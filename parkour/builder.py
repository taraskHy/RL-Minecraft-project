"""Course builder: a course built by hand in the game becomes a course the bots play (HANDOFF.md section 3, 1 Nov).

    python -m parkour.builder --viewer YourMinecraftName     the local server, the builder bot, the networks
    python -m parkour.builder --serve                        the converter alone (JSON lines; the bot starts it)

Ethan joins the local server (``parkour.server``, 127.0.0.1:25570, Minecraft Java 1.21.4) and types ``!build``:
the bot gives him creative mode and a kit of only the blocks the simulator models (``KIT``). He builds a course,
stands on its first block and types ``!start``, on its last and types ``!goal``, then ``!run``. Without chat (turned
off in the game or by the account) the panel's buttons beside the spawn give the same commands
(``bridge/builder_panel.js``), and marker blocks mark the course (``MARKERS``: a gold block the start, a diamond
block the goal, emerald blocks platforms the route must pass; stone to the simulator). The builder bot
(``bridge/builder_bot.js``) scans every block around the build with ``bot.blockAt`` and sends them here;
``scan_to_course`` refuses blocks the simulator does not model, naming each one and where it is, and otherwise
returns the course: platforms in route order, obstacle cells, ice, slime, shaped platforms and ladders, in course
grid coordinates, with the world position of grid cell (0, 0, 0). The bot plays it with the final policy and the
neo skill on each neo's block (``neo_assist.skill_cps``), with the learned router choosing the platforms (no course
list; only courses without shaped blocks, which the route planner does not see) or with the course list, and says
in chat how it went while Ethan spectates.

What a build may hold, as in the generated courses:

- Platforms: rectangles one block thick with two blocks of air above them, of stone, ice, slime or one shaped
  block (``configs/block_shapes.json``: slabs, stairs, trapdoors, carpets, fence and wall posts, iron bars, end
  rods, ...), at least one block of air apart. A shaped block may stand on another block; a full block on another
  full block is a wall, unless a ladder beside it makes it a ladder climb's target.
- Obstacles (walls, neo walls, head-hitter roofs): stone cells that are not platforms, and platforms that are
  not on the route from the start to the goal.
- Ladders, on the face of a stone wall.

The build: the blocks linked to the start's block, each within ``REACH`` (12 blocks across, as far as the bot sees
around it, and 24 up or down) of another; anything farther is no part of it and is left out, so the bot scans a box
wide enough for any build, and a
build that reaches the box's faces is too large. Command blocks count as stone, buttons and signs as air (they have
no collision), so the spawn pad and the builder's panel beside it are walls of a build near them, as the bot sees
them.

The route: the start's platform, then the longest chain of platforms (each a jump, ``_reach``, from the last) that
ends on the goal's; ties go to the chain that turns least, then to the shorter jumps. At most 16 platforms
(``courses.MAX_PLATFORMS``) and 256 obstacle cells (``env.MAX_EXTRA``), inside the 64 x 32 x 64 grid with two
blocks to spare on every side but below, where the lowest platform is on row 3 (``FLOOR``, as in a generated
course) and the lowest block on row 1 or higher. The learned router plays only routes of jumps it offers
(``router_why``).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import courses as C
from .paths import CONFIGS

KIT = ("stone", "gold_block", "diamond_block", "emerald_block", "slime_block", "ice", "ladder", "stone_slab", "stone_stairs",
       "oak_trapdoor", "white_carpet", "oak_fence", "cobblestone_wall", "iron_bars", "end_rod")  # what !build gives (64 of
# each, in this order: the first nine fill the hotbar, so the marker blocks are at hand)
# Marker blocks: where the start, the goal and the platforms the route must pass are, for a player who cannot chat
# !start / !goal / !via (bridge/builder_bot.js finds them). Full blocks with stone's friction, so stone to the simulator.
MARKERS = {"gold_block": "start", "diamond_block": "goal", "emerald_block": "via"}
STONE_LIKE = ("stone", "command_block", "chain_command_block", "repeating_command_block")  # full blocks, stone's friction
IGNORED = ("air", "cave_air", "void_air", "structure_void")  # no collision (the bots put structure void under carpets)
NO_COLLISION = ("_button", "_sign")  # name endings of blocks with no collision box: air to the simulator
# A block of the build is at most this far (x, y, z) from another: what the bot sees around it while it plays (the
# learned router looks 12 blocks around, PlannerConfig.radius), and any drop the grid holds.
REACH = (12, 24, 12)
MARGIN = 2  # free grid cells on every side of the build (courses._fits keeps two)
FLOOR = 3  # the lowest platform's row in the grid (courses._fits: y >= 3)
MAX_CANDIDATES = 40  # platforms the route search considers
MAX_EXPANSIONS = 400_000  # route search steps before it settles for the best chain found
DEMO_FAMILIES = (80, 81, 82)  # the courses !demo builds (demo_courses); nothing else uses these families


class BuildError(ValueError):
    """A build the simulator cannot model, with the blocks at fault: ``refused`` [[x, y, z, name, why], ...]."""

    def __init__(self, message: str, refused=None):
        super().__init__(message)
        self.refused = refused or []


@lru_cache(maxsize=None)
def _shapes() -> tuple:
    return tuple(json.loads((CONFIGS / "block_shapes.json").read_text())["shapes"])


def _same_boxes(a, b, tol: float = 1e-9) -> bool:
    key = lambda q: tuple(round(v, 6) for v in q)  # noqa: E731
    a, b = sorted(a, key=key), sorted(b, key=key)
    return len(a) == len(b) and all(len(p) == len(q) and all(abs(u - v) < tol for u, v in zip(p, q))
                                    for p, q in zip(a, b))


def classify(name: str, props: dict, boxes: list) -> tuple[str, object]:
    """(kind, detail) of one scanned block: ("full", None) stone, ("ice", None), ("slime", None), ("ladder",
    facing index), ("shape", shape name) for a block state whose collision boxes are a modelled shape's, or
    ("air", None). Raises ``BuildError`` for a block or a block state the simulator does not model."""
    name = name.removeprefix("minecraft:")
    if name in IGNORED or name.endswith(NO_COLLISION):
        return "air", None
    if name in STONE_LIKE or name in MARKERS:
        return "full", None
    if name == "ice":
        return "ice", None
    if name == "slime_block":
        return "slime", None
    if name == "ladder":
        facing = str(props.get("facing", ""))
        if facing not in C.LADDER_FACING:
            raise BuildError(f"a ladder facing {facing!r}")
        return "ladder", C.LADDER_FACING.index(facing)
    candidates = [s for s in _shapes() if s["block"] == name or (name.endswith("_carpet") and s["block"].endswith("_carpet"))]
    if not candidates:
        raise BuildError("not a block the simulator models")
    for s in candidates:
        if s["name"] != "full" and _same_boxes(boxes, s["boxes"]):
            return "shape", s["name"]
    if name == "stone_slab" and props.get("type") == "double":
        return "full", None  # a double slab is a full block
    raise BuildError("this state of it is not modelled (a connected fence, wall or bars, an open trapdoor, a corner stair, "
                     "a turned rod)")


def _reach(a: np.ndarray, b: np.ndarray, top_a: float, top_b: float, ladder_b: bool, slime_a: bool) -> float | None:
    """The horizontal gap of a jump from platform a to b (air cells between them, as a diagonal distance), or None
    if no jump makes it: at most 5 blocks across (6 when it drops two or more), and at most 1.25 up, except onto a
    ladder climb's target (up a ladder) or off slime (a bounce)."""
    dx = max(0, int(b[0]) - int(a[1]) - 1, int(a[0]) - int(b[1]) - 1)
    dz = max(0, int(b[2]) - int(a[3]) - 1, int(a[2]) - int(b[3]) - 1)
    gap = math.hypot(dx, dz)
    up = top_b - top_a
    if up > 1.25 + 1e-9 and not ladder_b and not slime_a:
        return None
    if gap > (6 if up <= -2 else 5):
        return None
    return gap


def _linked(points: np.ndarray, first: int, reach=REACH) -> np.ndarray:
    """Which of ``points`` (N x 3 cells) a chain of points, each within ``reach`` (x, y, z) of the next, links to
    ``points[first]``."""
    keep = np.zeros(len(points), dtype=bool)
    keep[first] = True
    frontier, lim = np.array([first]), np.asarray(reach)
    while len(frontier):
        rest = np.flatnonzero(~keep)
        if not len(rest):
            break
        near, pr = np.zeros(len(rest), dtype=bool), points[rest]
        for k in range(0, len(frontier), 256):
            near |= (np.abs(pr[None, :, :] - points[frontier[k:k + 256]][:, None, :]) <= lim).all(axis=2).any(axis=0)
        frontier = rest[near]
        keep[frontier] = True
    return keep


def _build_of(blocks: list, start, goal, via, box, warnings: list) -> list:
    """The scanned blocks that make up the build: those linked (``REACH``) to the start's, the goal's or a via
    block (if these are apart, the route search says no jump joins them). Raises ``BuildError`` if the build
    reaches the faces of the scanned ``box`` [x0, y0, z0, x1, y1, z1] (inclusive), so that part may lie outside."""
    solid = [b for b in blocks if str(b[3]).removeprefix("minecraft:") not in IGNORED]
    pts = np.array([[int(b[0]), int(b[1]), int(b[2])] for b in solid], dtype=np.int64).reshape(-1, 3)
    where = {tuple(p): i for i, p in enumerate(pts.tolist())}
    marked = [where[p] for p in (tuple(int(v) for v in pos) for pos in [start, goal, *via]) if p in where]
    if tuple(int(v) for v in start) not in where:
        return blocks  # (no block at the start: scan_to_course says so)
    keep = np.zeros(len(pts), dtype=bool)
    for i in marked:
        if not keep[i]:
            keep |= _linked(pts, i)
    if (~keep).sum():
        warnings.append(f"{int((~keep).sum())} block(s) farther from the build than {REACH[0]} across or {REACH[1]} up "
                        "or down were left out")
    if box is not None:
        lo, hi = np.asarray(box[:3]), np.asarray(box[3:])
        edge = np.flatnonzero(keep & ((pts == lo) | (pts == hi)).any(axis=1))
        if len(edge):
            x, y, z, bname = solid[edge[0]][:4]
            raise BuildError(f"the build reaches the edge of the scanned box at {x} {y} {z}: it is larger than a course "
                             f"can be ({C.GRID[0] - 2 * MARGIN} blocks across, {C.GRID[1]} rows)",
                             [[int(x), int(y), int(z), str(bname).removeprefix("minecraft:"), "at the edge of the scanned box"]])
    return [b for b, k in zip(solid, keep) if k]


def _lift_markers(blocks: list, marks: list, lifted: list) -> tuple[list, list]:
    """A marker block set on top of a platform's block marks that block (a platform of any kind): the marks move
    down onto it and the markers leave the build (the bot lifts them off before it plays). ``lifted`` gets them
    as [x, y, z, name]."""
    at = {(int(b[0]), int(b[1]), int(b[2])): str(b[3]).removeprefix("minecraft:") for b in blocks}
    moved, gone = [], set()
    for pos in marks:
        p = tuple(int(v) for v in pos)
        below = (p[0], p[1] - 1, p[2])
        if at.get(p) in MARKERS and below in at and at[below] not in IGNORED and not at[below].endswith(NO_COLLISION):
            if p not in gone:
                gone.add(p)
                lifted.append([*p, at[p]])
            p = below
        moved.append(p)
    return [b for b in blocks if (int(b[0]), int(b[1]), int(b[2])) not in gone], moved


def scan_to_course(blocks: list, start, goal, name: str = "build", via=(), box=None, ordered: bool = True,
                   lifted: list | None = None) -> tuple[C.Course, tuple[int, int, int], list[str]]:
    """The course of a scanned build (``blocks``: [x, y, z, name, props, boxes] in world coordinates; ``start``,
    ``goal`` and the optional ``via``: blocks the player stood on, or marker blocks; ``box``: the box the bot
    scanned, if it says), its origin (the world position of grid cell 0, 0, 0) and warnings. The route passes the
    via platforms in the given order (!via), or in any order if not ``ordered`` (emerald markers). Marker blocks set
    on top of a platform go to ``lifted`` (``_lift_markers``). Raises ``BuildError``."""
    warnings: list = []
    blocks, (start, goal, *via) = _lift_markers(blocks, [start, goal, *via], [] if lifted is None else lifted)
    blocks = _build_of(blocks, start, goal, via, box, warnings)
    cells: dict = {}
    refused = []
    for x, y, z, bname, props, boxes in blocks:
        try:
            kind, detail = classify(bname, props or {}, boxes or [])
        except BuildError as e:
            refused.append([int(x), int(y), int(z), bname.removeprefix("minecraft:"), str(e)])
            continue
        if kind != "air":
            cells[(int(x), int(y), int(z))] = (kind, detail)
    if refused:
        names = sorted({r[3] for r in refused})
        raise BuildError(f"{len(refused)} block(s) the simulator does not model: {', '.join(names)}", refused)
    solid = {p for p, (k, _) in cells.items() if k != "ladder"}
    ladders = {p: d for p, (k, d) in cells.items() if k == "ladder"}

    def platform_cell(p) -> bool:
        x, y, z = p
        kind = cells[p][0]
        if kind == "ladder" or (x, y + 1, z) in cells or (x, y + 2, z) in cells:
            return False
        if (x, y - 1, z) not in solid or kind == "shape":
            return True
        return any((x + dx, y, z + dz) in ladders for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    tops = {p for p in solid if platform_cell(p)}
    # Candidates: 4-connected cells of one kind (stone, ice, slime or one shape) at one height.
    seen, comps = set(), []
    for p in sorted(tops):
        if p in seen:
            continue
        kind = cells[p]
        comp, todo = [], [p]
        seen.add(p)
        while todo:
            q = todo.pop()
            comp.append(q)
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r = (q[0] + dx, q[1], q[2] + dz)
                if r in tops and r not in seen and cells[r] == kind:
                    seen.add(r)
                    todo.append(r)
        comps.append((comp, kind))
    # A candidate over another (some cell straight above one of its cells) up to HEADROOM rows higher is a roof, and
    # one exactly HEADROOM rows higher within a block of it across is a beam over a gap: a generated course keeps
    # that space above every platform free (courses._fits), and no jump climbs or drops HEADROOM rows. One that is
    # not a filled rectangle is a wall too.
    over = [{(q[0], q[2]) for q in comp} for comp, _ in comps]
    near = [{(q[0] + dx, q[2] + dz) for q in comp for dx in (-1, 0, 1) for dz in (-1, 0, 1)} for comp, _ in comps]
    plats = []
    for i, (comp, kind) in enumerate(comps):
        y = comp[0][1]
        if any((0 < y - other[0][1] <= C.HEADROOM and over[i] & over[j])
               or (y - other[0][1] == C.HEADROOM and near[i] & over[j]) for j, (other, _) in enumerate(comps)):
            continue
        xs, zs = [q[0] for q in comp], [q[2] for q in comp]
        rect = (min(xs), max(xs), min(zs), max(zs), y)
        if len(comp) != (rect[1] - rect[0] + 1) * (rect[3] - rect[2] + 1):
            warnings.append(f"the slab at {comp[0][0]} {y} {comp[0][2]} is not a rectangle: it counts as a wall")
            for pos, what in ((start, "start"), (goal, "goal")):
                if tuple(int(v) for v in pos) in comp:
                    raise BuildError(f"the {what} is on a slab that is not a rectangle", [[*comp[0], kind[0], "not a rectangle"]])
            continue
        plats.append((rect, kind))
    for i, (a, _) in enumerate(plats):  # one block of air between platforms (none touch in a generated course)
        for b, _ in plats[i + 1:]:
            if (a[4] == b[4] and a[0] - 1 <= b[1] and b[0] - 1 <= a[1] and a[2] - 1 <= b[3] and b[2] - 1 <= a[3]):
                raise BuildError(f"the platforms at {a[0]} {a[4]} {a[2]} and {b[0]} {b[4]} {b[2]} touch; leave a block "
                                 "of air between platforms")

    def find(pos, what):
        x, y, z = (int(v) for v in pos)
        for i, (r, _) in enumerate(plats):
            if r[0] <= x <= r[1] and r[2] <= z <= r[3] and r[4] == y:
                return i
        raise BuildError(f"no platform at the {what} ({x} {y} {z}): its mark (the marker block, or the block stood on for "
                         f"!{what}) must be a block of a platform, one block thick with two of air above")

    s, g = find(start, "start"), find(goal, "goal")
    if s == g:
        raise BuildError("the start and the goal are the same platform")
    vias = [find(v, "via") for v in via]
    if not ordered:  # emerald markers: one platform may hold several, and one on the start's or goal's means nothing
        vias = [v for v in dict.fromkeys(vias) if v not in (s, g)]
    if len(set(vias)) != len(vias) or {s, g} & set(vias):
        raise BuildError("a !via platform is given twice, or is the start or the goal")
    order = {v: k for k, v in enumerate(vias)}  # the route passes them in this order (if ordered)
    if len(plats) > MAX_CANDIDATES:
        raise BuildError(f"{len(plats)} platforms; the builder takes at most {MAX_CANDIDATES}")

    def top(i):
        r, (kind, detail) = plats[i]
        return r[4] + (C.shape_geometry(detail)[0] if kind == "shape" else 1.0)

    def ladder_target(i):
        r = plats[i][0]
        return any((x + dx, r[4], z + dz) in ladders for x in range(r[0], r[1] + 1) for z in range(r[2], r[3] + 1)
                   for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    n = len(plats)
    arr = [np.array(r, dtype=np.int64) for r, _ in plats]
    centre = [((r[0] + r[1] + 1) / 2, (r[2] + r[3] + 1) / 2) for r, _ in plats]
    edges = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j:
                gap = _reach(arr[i], arr[j], top(i), top(j), ladder_target(j), plats[i][1][0] == "slime")
                if gap is not None:
                    edges[i].append((gap, j))
        edges[i].sort()

    def turn(a, b, c) -> float:
        """The angle (degrees) between the jump a -> b and the jump b -> c, seen from above."""
        u = (centre[b][0] - centre[a][0], centre[b][1] - centre[a][1])
        v = (centre[c][0] - centre[b][0], centre[c][1] - centre[b][1])
        nu, nv = math.hypot(*u), math.hypot(*v)
        if nu == 0 or nv == 0:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))))

    # The route: the longest simple chain of jumps from the start's platform to the goal's (through the !via
    # platforms in their order); of those, the one that turns least (a generated course rarely doubles back), then
    # the one with the shorter jumps.
    best = None
    steps = 0
    path, used = [s], {s}

    def dfs(i, turning, length, passed):
        nonlocal best, steps
        steps += 1
        if steps > MAX_EXPANSIONS:
            return
        if i == g:
            key = (len(path), -round(turning, 6), -length)
            if passed == len(vias) and (best is None or key > best[0]):
                best = (key, list(path))
            return
        for gap, j in edges[i]:
            if j not in used and (j not in order or not ordered or order[j] == passed):
                used.add(j)
                path.append(j)
                dfs(j, turning + (turn(path[-3], i, j) if len(path) >= 3 else 0.0), length + gap,
                    passed + (j in order))
                path.pop()
                used.discard(j)

    dfs(s, 0.0, 0.0, 0)
    if best is None:
        raise BuildError("no chain of jumps leads from the start to the goal" + (" through the !via platforms in order"
                         if vias else "") + " (at most 5 blocks across, 1 up, except up a ladder or off slime)")
    route = best[1]
    if len(route) > C.MAX_PLATFORMS:
        raise BuildError(f"the route has {len(route)} platforms; a course has at most {C.MAX_PLATFORMS}")
    if steps > MAX_EXPANSIONS:
        warnings.append("the route search stopped early; the route may skip platforms")
    off = [i for i in range(n) if i not in route]
    if off:
        warnings.append(f"{len(off)} platform(s) off the route count as walls")
    on_route = {(x, plats[i][0][4], z) for i in route for x in range(plats[i][0][0], plats[i][0][1] + 1)
                for z in range(plats[i][0][2], plats[i][0][3] + 1)}
    extra = sorted(p for p in solid if p not in on_route)
    bad = [[*p, cells[p][1] or cells[p][0], "not a platform: walls and roofs must be stone"]
           for p in extra if cells[p][0] != "full"]
    if bad:
        raise BuildError(f"{len(bad)} block(s) of ice, slime or a shaped block are neither a platform nor stone; "
                         "walls and roofs must be stone", [[x, y, z, str(k), w] for x, y, z, k, w in bad])
    for (x, y, z), f in ladders.items():  # a ladder hangs on the block behind it (the side it faces away from)
        bx, bz = {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0)}[f]
        if (x + bx, y, z + bz) not in solid:
            raise BuildError(f"the ladder at {x} {y} {z} has no block behind it", [[x, y, z, "ladder", "nothing behind it"]])
    # Grid coordinates: the build's lowest corner two cells in from the grid's x and z edges, its lowest platform on
    # row 3 (lower if a block lies below it: the lowest block on row 1), every block two rows below the grid's top.
    every = list(cells)
    lo = [min(p[a] for p in every) for a in range(3)]
    low_platform = min(plats[i][0][4] for i in route)
    origin = (lo[0] - MARGIN, min(lo[1] - 1, low_platform - FLOOR), lo[2] - MARGIN)
    hi = [max(p[a] for p in every) - origin[a] for a in range(3)]
    gx, gy, gz = C.GRID
    if hi[0] > gx - 1 - MARGIN or hi[2] > gz - 1 - MARGIN:
        raise BuildError(f"the build is too large: at most {gx - 2 * MARGIN} x {gz - 2 * MARGIN} blocks across")
    top_platform = max(plats[i][0][4] for i in route) - origin[1]
    if hi[1] > gy - 1 - MARGIN or top_platform > gy - 2 - C.HEADROOM:
        raise BuildError(f"the build is too tall for the simulator's {gy}-row grid: its platforms at most "
                         f"{gy - 2 - C.HEADROOM - FLOOR} rows apart, and its highest block at most "
                         f"{gy - 1 - MARGIN - (low_platform - origin[1])} rows above its lowest platform")
    if len(extra) > 256:
        raise BuildError(f"{len(extra)} wall cells; a course has at most 256")
    ox, oy, oz = origin
    platforms = np.array([[r[0] - ox, r[1] - ox, r[2] - oz, r[3] - oz, r[4] - oy] for r in (plats[i][0] for i in route)],
                         dtype=np.int64)
    kinds = [plats[i][1] for i in route]
    course = C.Course(platforms, np.zeros(3), 0.0, name=name,
                      extra=np.array([[x - ox, y - oy, z - oz] for x, y, z in extra], dtype=np.int64).reshape(-1, 3),
                      ice=[k for k, (kind, _) in enumerate(kinds) if kind == "ice"],
                      slime=[k for k, (kind, _) in enumerate(kinds) if kind == "slime"],
                      shapes={k: d for k, (kind, d) in enumerate(kinds) if kind == "shape"},
                      ladders=np.array([[x - ox, y - oy, z - oz, f] for (x, y, z), f in sorted(ladders.items())],
                                       dtype=np.int64).reshape(-1, 4))
    centres = course.centers()
    course.spawn = centres[0].copy()
    d = centres[1] - centres[0]
    course.spawn_yaw = float(math.atan2(-d[0], -d[2]))  # Mineflayer yaw that faces the next platform
    return course, origin, warnings


def course_blocks(course: C.Course, origin=(0, 0, 0)) -> list:
    """The blocks a scan of ``course`` built at ``origin`` sees ([x, y, z, name, props, boxes], world coordinates),
    as ``bridge/parkour_bot.js`` builds a course: stone platforms and walls, ice, slime, shaped platforms (structure
    void under a carpet), ladders. For tests and the bot's self-check."""
    ox, oy, oz = origin
    by_name = {s["name"]: s for s in _shapes()}
    full = [[0, 0, 0, 1, 1, 1]]
    out = []
    for k, (x0, x1, z0, z1, y) in enumerate(np.asarray(course.platforms).tolist()):
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                if k in course.shapes:
                    s = by_name[course.shapes[k]]
                    out.append([x + ox, y + oy, z + oz, s["block"], dict(s["properties"]), s["boxes"]])
                    if s["block"].endswith("_carpet"):
                        out.append([x + ox, y - 1 + oy, z + oz, "structure_void", {}, []])
                else:
                    name = "ice" if k in course.ice else "slime_block" if k in course.slime else "stone"
                    out.append([x + ox, y + oy, z + oz, name, {}, full])
    for x, y, z in np.asarray(course.extra).reshape(-1, 3).tolist():
        out.append([x + ox, y + oy, z + oz, "stone", {}, full])
    for x, y, z, f in np.asarray(course.ladders).reshape(-1, 4).tolist():
        s = by_name["ladder_" + C.LADDER_FACING[f]]
        out.append([x + ox, y + oy, z + oz, "ladder", dict(s["properties"]), s["boxes"]])
    return out


def same_build(course: C.Course, origin, other: C.Course, other_origin) -> bool:
    """Whether two courses are the same blocks in the world: platforms in order, walls, ice, slime, shapes and
    ladders (each at its own origin)."""
    def cells(c, o):
        p = np.asarray(c.platforms).reshape(-1, 5) + np.array([o[0], o[0], o[2], o[2], o[1]])
        ex = sorted(map(tuple, (np.asarray(c.extra).reshape(-1, 3) + np.array(o)).tolist()))
        ld = sorted(map(tuple, (np.asarray(c.ladders).reshape(-1, 4) + np.array([*o, 0])).tolist()))
        return p.tolist(), ex, ld, sorted(c.ice), sorted(c.slime), {int(k): v for k, v in c.shapes.items()}
    return cells(course, origin) == cells(other, other_origin)


def demo_courses() -> list[C.Course]:
    """The courses ``!demo`` builds, for a check of the whole builder without a builder: 8 with two neos, 4 with one,
    and one each of the shaped-block, ladder, slime and ice levels, from ``DEMO_FAMILIES``."""
    from .neo_assist import single_neo_courses, two_neo_courses
    out = two_neo_courses(DEMO_FAMILIES[0], 8) + single_neo_courses(DEMO_FAMILIES[1], 4)
    levels = C.load_levels()
    extra = {L["name"]: [C._spec_from(d) for d in L["specs"]]
             for L in json.loads((CONFIGS / "jump_levels_extra.json").read_text())["levels"]}
    rng = np.random.default_rng([2026, DEMO_FAMILIES[2], 6])
    for name in ("shaped blocks", "ladder climb", "slime bounce", "ice run-up"):
        out.append(C.generate(rng, levels + [extra[name]], 6, 10, seed=len(out)))
    return out


def router_why(course: C.Course) -> str:
    """Why the learned router cannot play the course's route ('' if it can): it does not see shaped platforms
    (bridge/policy.js), climbs no ladder, and offers only jumps that change height by ``dy_min`` to ``dy_max`` with an
    air gap up to ``MAX_GAP[dy] + slack`` (``planner.options``, ``bridge/router.js``)."""
    from .planner import MAX_GAP, PlannerConfig
    cfg = PlannerConfig()
    if course.shapes:
        return "shaped blocks"
    if len(course.ladders):
        return "a ladder climb"
    p = np.asarray(course.platforms).tolist()
    for k, (a, b) in enumerate(zip(p, p[1:]), start=1):
        dy = b[4] - a[4]
        gap = math.hypot(max(0, b[0] - a[1] - 1, a[0] - b[1] - 1), max(0, b[2] - a[3] - 1, a[2] - b[3] - 1))
        if dy > cfg.dy_max:
            return f"jump {k} goes {dy} up"
        if dy < cfg.dy_min:
            return f"jump {k} drops {-dy}"
        if gap > MAX_GAP[dy] + cfg.slack:
            return f"jump {k} is {gap:.1f} blocks across ({dy:+d} in height)"
    return ""


def describe(course: C.Course) -> dict:
    """What the bot says about a converted course, and the course-list mode's skill checkpoints."""
    from .neo_assist import skill_cps
    from .neo_dagger import neo_takeoffs
    neos = neo_takeoffs(course)
    why = router_why(course)
    return {"platforms": course.n, "jumps": course.n - 1, "neos": neos, "skill_cps": skill_cps(course),
            "ice": len(course.ice), "slime": len(course.slime), "shapes": len(course.shapes),
            "ladders": int(len(course.ladders)), "walls": int(len(course.extra)),
            "router_ok": not why, "router_why": why}


def convert(request: dict) -> dict:
    """One converter request (``serve``): {"blocks", "start", "goal"[, "via", "name", "box", "ordered"]} ->
    {"course", "origin", "about", "warnings", "lifted"} or {"error", "refused"}."""
    lifted: list = []
    try:
        course, origin, warnings = scan_to_course(request["blocks"], request["start"], request["goal"],
                                                  request.get("name", "build"), request.get("via", ()),
                                                  request.get("box"), request.get("ordered", True), lifted)
    except BuildError as e:
        return {"error": str(e), "refused": e.refused[:20]}
    about = describe(course)
    d = course.to_json()
    d["skill_cps"] = about["skill_cps"]
    out = {"course": d, "origin": list(origin), "about": about, "warnings": warnings, "lifted": lifted}
    if "expect" in request:  # (!demo: the course the bot built)
        out["matches_expected"] = same_build(course, origin, C.Course.from_json(request["expect"]["course"]),
                                             request["expect"]["origin"])
    return out


class Checker:
    """The simulator's view of a run in the game (``check``): the game's ticks replayed (``parkour.live.replay``,
    from the episode's first recorded state) and the same networks' own run from the course's spawn after the
    bots' 10 settle ticks (as ``parkour.live``'s simulator re-run), with the course list or the learned router."""

    def __init__(self, checkpoint: str, skill: str = "", router: str = "", stall: int = 200):
        self.paths = (checkpoint, skill, router)
        self.stall = stall
        self.nets = None

    def _load(self):
        import torch
        from .live import load_model
        checkpoint, skill, router = self.paths
        model = load_model(Path(checkpoint), "cpu")
        sk = load_model(Path(skill), "cpu") if skill else None
        rt = fill = None
        if router:
            from .route_policy import load_router
            from .route_train import router_choosers
            rt = router_choosers(load_router(Path(router), "cpu"), True)
            fill = torch.load(router, map_location="cpu", weights_only=False)["config"].get("fill", "greedy")
        self.nets = (model, sk, rt, fill)

    def check(self, req: dict) -> dict:
        from .env import CoursePool
        from .live import SETTLE_TICKS, replay
        from .neo_assist import AssistConfig
        if self.nets is None:
            self._load()
        model, sk, rt, fill = self.nets
        course = C.Course.from_json(req["course"])
        rep = replay([course], [dict(req["episode"], index=0)])[0]
        pool = CoursePool([course], "cpu", C.GRID)
        if req.get("mode") == "router":
            if rt is None:
                raise ValueError("no router loaded")
            from .planner import PlannerConfig
            from .route_env import RouteConfig, RouteTables
            from .route_neo import evaluate_routed_neo
            pc = evaluate_routed_neo(model, pool, RouteTables(pool), rt[0], rt[1], PlannerConfig(),
                                     RouteConfig(fill=fill, blocked="keep"), AssistConfig(enabled=False), self.stall,
                                     settle=SETTLE_TICKS, skill=sk)["per_course"][0]
        else:
            from .neo_assist import evaluate_assisted
            pc = evaluate_assisted(model, pool, "cpu", AssistConfig(enabled=False), stall_ticks=self.stall,
                                   settle=SETTLE_TICKS, skill=sk)["per_course"][0]
        return {"max_pos_err": rep["max_pos_err"], "first_divergence_tick": rep.get("first_divergence_tick"),
                "replayed_ticks": rep["ticks"], "sim_success": bool(pc["success"]), "sim_ticks": int(pc["ticks"])}


def serve(inp=None, out=None, checker: Checker | None = None) -> None:
    """JSON lines: one request per line on stdin, one answer per line on stdout (``bridge/builder_bot.js``): a
    build to convert, or {"check": {"course", "episode", "mode"}} for the simulator's view of a run."""
    inp, out = inp or sys.stdin, out or sys.stdout
    for line in inp:
        if line.strip():
            try:
                req = json.loads(line)
                if "check" in req:
                    if checker is None:
                        raise ValueError("no networks loaded for checks")
                    reply = checker.check(req["check"])
                else:
                    reply = convert(req)
            except Exception as e:  # a malformed request: say so and keep serving
                reply = {"error": f"converter error: {e}", "refused": []}
            out.write(json.dumps(reply) + "\n")
            out.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="the converter alone, JSON lines on stdin/stdout")
    ap.add_argument("--checkpoint", default="", help="with --serve: the final policy, for the simulator checks")
    ap.add_argument("--skill", default="", help="with --serve: the neo skill")
    ap.add_argument("--router", default="", help="with --serve: the learned router")
    a, rest = ap.parse_known_args(argv)
    if a.serve:
        serve(checker=Checker(a.checkpoint, a.skill, a.router) if a.checkpoint else None)
        return
    from .build_session import run  # the server, the bot and the networks (it takes --checkpoint, --skill, --router too)
    run(sys.argv[1:] if argv is None else list(argv))


if __name__ == "__main__":
    main()

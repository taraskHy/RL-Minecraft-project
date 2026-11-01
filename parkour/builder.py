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

The route: the start's platform, then the longest chain of platforms (each a jump, ``_reach``, from the last) that
ends on the goal's, ties to the shorter jumps. At most 16 platforms (``courses.MAX_PLATFORMS``) and 256 obstacle
cells (``env.MAX_EXTRA``), inside the 64 x 32 x 64 grid with two blocks to spare on every side.
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

KIT = ("stone", "ice", "slime_block", "ladder", "stone_slab", "stone_stairs", "oak_trapdoor", "white_carpet",
       "oak_fence", "cobblestone_wall", "iron_bars", "end_rod", "gold_block", "diamond_block", "emerald_block")  # what !build gives (64 of each)
# Marker blocks: where the start, the goal and the platforms the route must pass are, for a player who cannot chat
# !start / !goal / !via (bridge/builder_bot.js finds them). Full blocks with stone's friction, so stone to the simulator.
MARKERS = {"gold_block": "start", "diamond_block": "goal", "emerald_block": "via"}
IGNORED = ("air", "cave_air", "void_air", "structure_void")  # no collision (the bots put structure void under carpets)
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
    if name in IGNORED:
        return "air", None
    if name == "stone" or name in MARKERS:
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


def scan_to_course(blocks: list, start, goal, name: str = "build", via=()) -> tuple[C.Course, tuple[int, int, int], list[str]]:
    """The course of a scanned build (``blocks``: [x, y, z, name, props, boxes] in world coordinates; ``start``,
    ``goal`` and the optional ``via`` (in order): blocks the player stood on), its origin (the world position of
    grid cell 0, 0, 0) and warnings. Raises ``BuildError``."""
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
    warnings = []
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
        raise BuildError(f"no platform at the {what} ({x} {y} {z}); stand on it and say !{what} again")

    s, g = find(start, "start"), find(goal, "goal")
    if s == g:
        raise BuildError("the start and the goal are the same platform")
    vias = [find(v, "via") for v in via]
    if len(set(vias)) != len(vias) or {s, g} & set(vias):
        raise BuildError("a !via platform is given twice, or is the start or the goal")
    order = {v: k for k, v in enumerate(vias)}  # the route passes them in this order
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
            if j not in used and (j not in order or order[j] == passed):
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
    # Grid coordinates: the build's lowest corner two cells in from the grid's edges, its lowest platform on row 3.
    every = list(cells)
    lo = [min(p[a] for p in every) for a in range(3)]
    low_platform = min(plats[i][0][4] for i in route)
    origin = (lo[0] - MARGIN, min(lo[1], low_platform - FLOOR) - 1, lo[2] - MARGIN)
    hi = [max(p[a] for p in every) - origin[a] for a in range(3)]
    gx, gy, gz = C.GRID
    if hi[0] > gx - 1 - MARGIN or hi[2] > gz - 1 - MARGIN or max(plats[i][0][4] for i in route) - origin[1] > gy - 2 - C.HEADROOM:
        raise BuildError(f"the build is too large: at most {gx - 2 * MARGIN - 1} x {gz - 2 * MARGIN - 1} blocks across "
                         f"and {gy - C.HEADROOM - FLOOR - 3} high")
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


def describe(course: C.Course) -> dict:
    """What the bot says about a converted course, and the course-list mode's skill checkpoints."""
    from .neo_assist import skill_cps
    from .neo_dagger import neo_takeoffs
    neos = neo_takeoffs(course)
    return {"platforms": course.n, "jumps": course.n - 1, "neos": neos, "skill_cps": skill_cps(course),
            "ice": len(course.ice), "slime": len(course.slime), "shapes": len(course.shapes),
            "ladders": int(len(course.ladders)), "walls": int(len(course.extra)),
            "router_ok": not course.shapes}  # (bridge/policy.js: the route planner does not see shaped platforms)


def convert(request: dict) -> dict:
    """One converter request (``serve``): {"blocks", "start", "goal"[, "via", "name"]} -> {"course", "origin",
    "about", "warnings"} or {"error", "refused"}."""
    try:
        course, origin, warnings = scan_to_course(request["blocks"], request["start"], request["goal"],
                                                  request.get("name", "build"), request.get("via", ()))
    except BuildError as e:
        return {"error": str(e), "refused": e.refused[:20]}
    about = describe(course)
    d = course.to_json()
    d["skill_cps"] = about["skill_cps"]
    out = {"course": d, "origin": list(origin), "about": about, "warnings": warnings}
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
    from .build_session import run  # the server, the bot and the networks
    run(rest)


if __name__ == "__main__":
    main()

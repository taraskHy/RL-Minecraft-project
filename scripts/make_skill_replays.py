"""Replays of the one network on fresh courses of every skill added after levels 0-5.

    python scripts/make_skill_replays.py courses OUT.json
    python scripts/make_skill_replays.py page OUT.html RUN_DIR [RUN_DIR ...]

The user asked on 29 October 2026 for replays of head hitters, slime, ice and
every other new kind. The policy is the shipped one network,
results/combined_policy.pt (the distilled student ``combined-distill``,
iteration 899; provenance in results/combined_policy_provenance.json). The
skills are the ones added after levels 0-5 (results/final_policy.pt covers
levels 0-5 and obstacles).

The first 45 courses, fixed on 29 October 2026 before any outcome was seen:

* extra kinds, global levels 9-14 (9 head hitter, 10 hard head hitter,
  11 ice run-up, 12 shaped blocks, 13 slime, 14 ladders): seeds 0-3 of each
  level, seed family 42;
* the advanced levels, global levels 6-8 (neos around a wall, runway or
  momentum jumps, jumps 3 blocks sideways): seeds 0-2 of each level, seed
  family 42 (kind token "neo", course ids neo-<level>-<seed>, as in the
  course file; only some of these courses have a neo, see "own jumps" below);
* ice courses, global levels 15-20 (basic levels 0-5 with ice on 30% of the
  platforms, SKILL_ICE_PROB): seeds 0-1 of each level, seed family 43.

These 45 courses, no obstacles, are built exactly as parkour.train.skill_sets
builds a group's held-out standard set (the group's level table from
skill_tables / SKILL_GROUPS, held_out_courses with the family, the same lo..hi
filter and relabel to global level ids; ice through the basic table with
ice_prob=SKILL_ICE_PROB and level + ICE_GROUP[0]), only from seed family 42 or
43 instead of the validation or test families. n_jumps is 10: the generator
draws up to 10 jumps and stops at 16 platforms (courses.MAX_PLATFORMS) or when
no jump fits in 30 tries, and it adds a step onto the runway before every
runway jump and an exit jump after every shaped block and slime pad. So a
course has 1 to 10 drawn jumps and 1 to 15 platform-to-platform jumps.

The 20 one-type courses, added the same day at the user's request (a replay
of every trained block type, fence posts and wall posts included), after the
first 45 had been played in offline physics and the first 14 in real
Minecraft, and before any of the 20 was played:

* one block type per course, global level 12 (the shaped-blocks level, local
  level 3 of configs/jump_levels_extra.json): for each of its 10 types, k = 0..9
  in the order bottom_slab, soul_sand, dirt_path, enchanting_table,
  daylight_detector, bottom_trapdoor, carpet, fence_post, wall_post, end_rod
  (checked against the set of "block" values in that level's specs), seeds 0
  and 1 from the stream numpy default_rng([2026, 44, k]): courses.generate on
  the extra group's table, on a copy whose shaped level keeps only the specs
  of that type, so every shaped jump on the course uses that one type;
  n_jumps 10, no obstacles, level label 12. Kind "shaped: <type>". The shapes
  of configs/block_shapes.json that no training level uses (top slab, top
  trapdoor, iron bars, stairs) are left out; family 44 with k = 10..13 is
  scripts/make_untrained_shape_courses.py's untrained-shape set (separate
  streams, written after this set).

65 courses in all. Families 42, 43 and 44 (k = 0..9) had never been used (0-5,
7, 11-17 and 99 were in use; 18-39 are the routing work's, 40-41 the
single-neo sets), so no policy was trained, chosen or tested on these courses.
``courses`` checks that the same construction with the validation families
gives skill_sets("validation")'s courses, and writes {"courses": [...]} for
``parkour.live --courses-file`` (Course.to_json plus each course's kind,
family and id for the page). The first 45 were written first to
results/games/skill-replays-courses.json (kept as it is); the file the bots
play is results/games/skill-replays-courses-65.json, whose first 45 courses
are those.

Recordings of the 65 courses (each in its own new folder; keep these folders as they are):

    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m parkour.live --offline --checkpoint results/combined_policy.pt
        --courses-file results/games/skill-replays-courses-65.json --out results/games/skill-replays-offline-65
    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m parkour.live --checkpoint results/combined_policy.pt
        --courses-file results/games/skill-replays-courses-65.json --out results/games/skill-replays-live-65

The earlier folders results/games/skill-replays-offline (the first 45
courses, complete) and results/games/skill-replays-live (real Minecraft,
stopped at course 14: a carpet with air below it breaks in vanilla Minecraft;
no summary.json) were played from skill-replays-courses.json and are kept as
evidence only. The page is built from the two -65 runs.

``page`` turns the runs into results/skill_replays.html
(scripts/skill_replays_template.html): every course of the set from every
run, in the frame format of the showcase replay export
(exports/advanced-ice-replays/README.md on branch
codex/advanced-ice-replay-export-20261029, columns x y z vx vy vz yaw
on_ground jump_ticks platform_reached): frames[0] is the recorded start state
(the episode's init), frames[t + 1] the state after actions[t] (traj[t];
bridge/policy.js Episode: the first action is chosen on the start state,
every later tick records the state and then acts, and the tick that ends the
episode acts no more, so ticks = len(traj) = len(actions); parkour/live.py
replay() re-simulates the same way), yaw accumulated from the start yaw by
each action's YAW_STEPS_DEG turn, deg * pi / 180, and wrapped as
bridge/policy.js wrapYaw does after every turn (the bots' order), and the
platform reached per tick by bridge/policy.js's standing rule (on the ground,
within 1e-6 of a platform's top and overlapping its footprint by the 0.3
half-width, a shaped platform's own top and footprint; the highest index
counts and the reached index never decreases). Positions and velocities are
rounded to 6 decimals in the page. Every block the physics has is drawn:
stone, ice and slime platforms (course.ice / course.slime are platform
indices), shaped platforms box by box from configs/block_shapes.json,
ladders from their ladder_<facing> boxes, and the extra cells, each with the
role the generator gave it (parkour.courses _ceiling: head-hitter roof,
_neo_wall: neo wall, _ladder: the wall under a ladder target), found by
rebuilding those cells from the course's own jump specs. Real Minecraft also
has a structure_void block (no collision box) under every carpet platform
(bridge/parkour_bot.js) where the simulator and the offline bot have air; it
is not drawn.

Per course the page also gets each jump's leg kind ("drawn" from the level
tables, or added by the generator: "runway", the step onto the next jump's
runway; "exit", the fixed jump off a shaped block; "bounce_exit", the jump off
a slime pad; each checked against the key the generator gives it), the jump
that placed every extra cell, and the kind's own jumps: the jumps of the
kind's own feature (head hitter _hh, ice run-up _ice1, shaped _b, slime
_slime, ladder _ladder; the advanced levels: _neo, runway jumps _run, and
jumps 3 blocks sideways; ice courses: every jump onto or off an ice platform).
An own jump counts as reached when the bot stood on its takeoff (for a runway
jump, the runway) and as passed when it reached the next platform (after a
shaped block or a slime pad, the exit platform after it), from the recorded
platform reached. The builder checks that every course has exactly one
episode per run, frames = ticks + 1, the replayed platform equals the bot's,
every result equals summary.json, every extra cell has exactly one role and
one jump, and the embedded JSON cannot end the page's <script> element. These
are post-training recordings on fresh showcase courses, not benchmark results.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour import courses as C  # noqa: E402
from parkour.courses import shape_geometry  # noqa: E402
from parkour.env import YAW_STEPS_DEG  # noqa: E402
from parkour.paths import CONFIGS  # noqa: E402
from parkour.train import (ICE_GROUP, ICE_VALIDATION_FAMILY, SKILL_GROUPS, SKILL_ICE_PROB, VALIDATION_FAMILY,  # noqa: E402
                           held_out_courses, skill_sets, skill_tables)

TEMPLATE = ROOT / "scripts" / "skill_replays_template.html"
POLICY = "results/combined_policy.pt"
PROVENANCE = "results/combined_policy_provenance.json"
COURSES_FILE = "results/games/skill-replays-courses-65.json"  # the 65 courses the bots play
FIRST_COURSES_FILE = "results/games/skill-replays-courses.json"  # the first 45, written before the 20 were added
N_JUMPS = 10
FAMILY = 42  # extra and advanced groups (never used before 29 Oct 2026)
ICE_FAMILY = 43  # ice courses
SHAPE_FAMILY = 44  # one block type per course: stream default_rng([2026, 44, k]) for type k
SHAPED_LEVEL = 12  # global id of the shaped-blocks level
SHAPE_TYPES = ("bottom_slab", "soul_sand", "dirt_path", "enchanting_table", "daylight_detector", "bottom_trapdoor",
               "carpet", "fence_post", "wall_post", "end_rod")  # k = 0..9
SHAPE_SEEDS = 2
# (token, label, group, global levels, seeds per level, family, what is new on these courses)
KINDS = (
    ("head-hitter", "Head hitter", "extra", (9,), 4, FAMILY,
     "a run-up under a roof two blocks above the runway: a jump under it rises only 0.2 blocks"),
    ("hard-head-hitter", "Hard head hitter", "extra", (10,), 4, FAMILY,
     "head-hitter run-ups to longer jumps, some with the roof continued over the gap"),
    ("ice-run-up", "Ice run-up", "extra", (11,), 4, FAMILY,
     "the runway before the jump is ice (slipperiness 0.98: slow to speed up, faster at the end)"),
    ("shaped", "Shaped blocks", "extra", (12,), 4, FAMILY,
     "landing on a block that is not a whole cube (slab, soul sand, fence post, end rod, ...)"),
    ("slime", "Slime", "extra", (13,), 4, FAMILY,
     "dropping onto a slime pad and bouncing up to an exit platform"),
    ("ladder", "Ladders", "extra", (14,), 4, FAMILY,
     "stepping onto a ladder on a wall and climbing to the target on top"),
    ("neo", "Advanced levels", "advanced", (6, 7, 8), 3, FAMILY,
     "levels 6-8: neos around a wall, runway (momentum) jumps and jumps 3 blocks sideways, mixed with easier "
     "jumps; the course ids say neo, the course file's token for these levels"),
    ("ice", "Ice courses", "ice", tuple(range(ICE_GROUP[0], ICE_GROUP[1] + 1)), 2, ICE_FAMILY,
     "basic levels 0-5 with ice on 30% of the platforms after the start"),
)
# The kinds' own jumps: feature token -> (one, many, what it is). own_jumps finds them in the jump specs.
FEATURES = {
    "hh": ("head-hitter jump", "head-hitter jumps", "the jump after a run-up under a roof (spec _hh)"),
    "ice_run": ("jump off an ice runway", "jumps off an ice runway", "the jump after a run-up on ice (spec _ice1)"),
    "shaped": ("shaped-block landing", "shaped-block landings", "landing on a shaped block (spec _b) and the exit jump off it"),
    "slime": ("slime bounce", "slime bounces", "the drop onto a slime pad (spec _slime) and the bounce to the exit platform"),
    "ladder": ("ladder climb", "ladder climbs", "stepping onto a ladder and climbing to the target on top (spec _ladder)"),
    "neo": ("neo jump", "neo jumps", "a jump around the edge of a wall (spec _neo)"),
    "run": ("runway jump", "runway jumps", "a momentum jump off a runway (spec _run, no roof, no ice)"),
    "side3": ("jump 3 sideways", "jumps 3 sideways", "a plain jump 3 blocks sideways (level 6; levels 0-5 go up to 2)"),
    "ice": ("jump onto or off ice", "jumps onto or off ice", "a jump whose takeoff or landing platform is ice"),
}
KIND_FEATURES = {"head-hitter": ("hh",), "hard-head-hitter": ("hh",), "ice-run-up": ("ice_run",), "shaped": ("shaped",),
                 "slime": ("slime",), "ladder": ("ladder",), "neo": ("neo", "run", "side3"), "ice": ("ice",)}
TYPE_FEATURES = ("shaped",)  # the one-type courses
SHAPE_BLOCK_NAMES = {  # how the page names each type: (name, "lands on ..."); configs/block_shapes.json has the block state
    "bottom_slab": ("bottom slab", "a bottom stone slab"), "soul_sand": ("soul sand", "soul sand"),
    "dirt_path": ("dirt path", "a dirt path block"), "enchanting_table": ("enchanting table", "an enchanting table"),
    "daylight_detector": ("daylight detector", "a daylight detector"),
    "bottom_trapdoor": ("bottom trapdoor", "a closed oak trapdoor in the bottom half"), "carpet": ("carpet", "a white carpet"),
    "fence_post": ("fence post", "a lone oak fence post"), "wall_post": ("wall post", "a lone cobblestone wall post"),
    "end_rod": ("end rod", "an upright end rod")}
RUN_KINDS = {  # summary.json "mode" prefix -> (run token, label)
    "live": ("live", "Real Minecraft"),
    "offline": ("offline", "Offline physics"),
}
HALF_WIDTH = 0.3  # bridge/policy.js
STAND_EPS = 1e-6
FRAME_COLUMNS = ["x", "y", "z", "vx", "vy", "vz", "yaw", "on_ground", "jump_ticks", "platform_reached"]


# ---------- the course set ----------
def group_courses(group: str, per_level: int, family: int) -> list:
    """One group's held-out standard set as parkour.train.skill_sets builds it, from seed family ``family``."""
    table = skill_tables([group])[group]
    _, lo, hi, first = SKILL_GROUPS[group]
    out = [c for c in held_out_courses(table, per_level, N_JUMPS, family=family) if lo <= c.level <= hi]
    for c in out:
        c.level = first + c.level - lo
    return out


def ice_courses(per_level: int, family: int) -> list:
    """The ice group as skill_sets builds it: the basic table with ice on SKILL_ICE_PROB of the platforms."""
    out = held_out_courses(skill_tables(["basic"])["basic"], per_level, N_JUMPS, family=family, ice_prob=SKILL_ICE_PROB)
    for c in out:
        c.level = ICE_GROUP[0] + c.level
    return out


def check_construction() -> None:
    """With the validation families, the construction above gives skill_sets("validation")'s standard courses."""
    ref = skill_sets("validation", 2, N_JUMPS)
    same = lambda a, b: [c.to_json() for c in a] == [c.to_json() for c in b]  # noqa: E731
    for g in ("advanced", "extra"):
        assert same(group_courses(g, 2, VALIDATION_FAMILY), ref[g]["standard"]), g
    assert same(ice_courses(2, ICE_VALIDATION_FAMILY), ref["ice"]["standard"]), "ice"


def course_id(token: str, levels: tuple, level: int, seed: int) -> str:
    return f"{token}-{seed}" if len(levels) == 1 else f"{token}-{level}-{seed}"


def shaped_local_level() -> int:
    _, lo, _, first = SKILL_GROUPS["extra"]
    return SHAPED_LEVEL - first + lo  # 9 in the extra group's table (basic 0-5, then the extra levels)


def shape_types() -> dict:
    """The block types of the shaped-blocks level: SHAPE_TYPES checked against the "block" values of its specs.

    If they differed, the specs' set would be used, in SHAPE_TYPES's order with the specs' other types appended
    (in the order they first appear); the result says what differed. No other level of any table uses a block."""
    table = skill_tables(["extra"])["extra"]
    specs = table[shaped_local_level()]
    counts = Counter(s.block for s in specs)
    order = list(dict.fromkeys(s.block for s in specs))
    assert "" not in counts, "every spec of the shaped-blocks level names a block"
    every = C.load_levels(advanced=True, extra=True)
    used = {L: sorted({s.block for s in lv if s.block}) for L, lv in enumerate(every) if any(s.block for s in lv)}
    assert list(used) == [SHAPED_LEVEL], f"blocks outside the shaped-blocks level: {used}"
    types = [t for t in SHAPE_TYPES if t in counts] + [t for t in order if t not in SHAPE_TYPES]
    return {"types": types, "spec_counts": {t: counts[t] for t in types}, "specs": len(specs),
            "missing_from_specs": [t for t in SHAPE_TYPES if t not in counts],
            "added_from_specs": [t for t in order if t not in SHAPE_TYPES]}


def type_courses() -> list:
    """[(type, Course)]: SHAPE_SEEDS courses per block type, every shaped jump of one type (level label 12)."""
    table = skill_tables(["extra"])["extra"]
    local = shaped_local_level()
    out = []
    for k, name in enumerate(shape_types()["types"]):
        levels = list(table)  # a copy of the table: configs/ stays as it is
        levels[local] = [s for s in table[local] if s.block == name]
        rng = np.random.default_rng([2026, SHAPE_FAMILY, k])
        for i in range(SHAPE_SEEDS):  # as held_out_courses: one stream, the seeds in turn
            c = C.generate(rng, levels, local, N_JUMPS, seed=i, obstacle_prob=0.0, ice_prob=0.0)
            c.level = SHAPED_LEVEL
            out.append((name, c))
    return out


def course_set() -> list[dict]:
    """The 65 courses in the fixed order (kinds as in KINDS, then level, then seed; then the one-type shaped
    courses by type and seed), as course-file dicts."""
    out = []
    for token, _, group, levels, seeds, family, _ in KINDS:
        cs = ice_courses(seeds, family) if group == "ice" else group_courses(group, seeds, family)
        for L in levels:
            mine = [c for c in cs if c.level == L]
            assert [c.seed for c in mine] == list(range(seeds)), (token, L)
            for c in mine:
                assert len(c.specs) == c.n - 1, token  # one spec per jump, as parkour.live's specs_from_keys expects
                d = c.to_json()
                d.update(kind=token, family=family, id=course_id(token, levels, L, c.seed))
                out.append(d)
    assert len(out) == 45
    for name, c in type_courses():
        assert len(c.specs) == c.n - 1, name
        assert set(c.shapes.values()) <= {name}, f"a {name} course holds other shapes: {c.shapes}"
        if not c.shapes:  # possible in principle (no jump drawn from the shaped level); reported, the set stays fixed
            print(f"note: shaped-{name}-{c.seed} has no shaped platform", file=sys.stderr)
        d = c.to_json()
        d.update(kind=f"shaped: {name}", family=SHAPE_FAMILY, id=f"shaped-{name}-{c.seed}")
        out.append(d)
    assert len(out) == 65 and len({d["id"] for d in out}) == 65
    return out


def bare(d: dict) -> dict:
    """A course-file dict without the page's own keys: what parkour.live writes to its courses.json."""
    return {k: v for k, v in d.items() if k not in ("kind", "family", "id")}


def check_first_file(cs: list) -> None:
    """The 45-course file written first holds exactly the first 45 courses of the set."""
    first = ROOT / FIRST_COURSES_FILE
    if first.exists():
        assert json.loads(first.read_text(encoding="utf-8"))["courses"] == cs[:45], f"{FIRST_COURSES_FILE} differs"


def cmd_courses(a) -> None:
    check_construction()
    cs = course_set()
    check_first_file(cs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; the course file is written once")
    a.out.write_text(json.dumps({"courses": cs}), encoding="utf-8")
    # The file must read back as parkour.live reads it, with the same courses.
    from parkour.neo_assist import courses_from_file
    back = [c.to_json() for c in courses_from_file(json.loads(a.out.read_text(encoding="utf-8"))["courses"])]
    assert back == [bare(d) for d in cs], "the course file does not read back through courses_from_file"
    rows = {}
    for d in cs:
        r = rows.setdefault(d["kind"], {"courses": 0, "levels": set(), "family": d["family"], "platforms": []})
        r["courses"] += 1
        r["levels"].add(d["level"])
        r["platforms"].append(len(d["platforms"]))
    print(json.dumps({"out": str(a.out), "courses": len(cs), "sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
                      "shape_types": shape_types(),
                      "shapes_per_course": {d["id"]: d.get("shapes", {}) for d in cs if d.get("shapes")},
                      "kinds": {k: {**v, "levels": sorted(v["levels"])} for k, v in rows.items()}}, indent=1))


# ---------- the page ----------
def wrap(yaw: float) -> float:
    """bridge/policy.js wrapYaw (the same as torch.remainder(yaw + pi, 2 pi) - pi)."""
    return (yaw + math.pi) % (2 * math.pi) - math.pi


def platform_geometry(course: dict) -> list:
    """bridge/policy.js platformGeometry: standing area and top [x0, x1, z0, z1, top] of every platform."""
    shapes = course.get("shapes") or {}
    out = []
    for k, (x0, x1, z0, z1, y) in enumerate(course["platforms"]):
        name = shapes.get(str(k))
        if not name:
            out.append((x0, x1 + 1, z0, z1 + 1, y + 1))
        else:
            top, sx0, sx1, sz0, sz1 = shape_geometry(name)
            out.append((x0 + sx0, x1 + sx1, z0 + sz0, z1 + sz1, y + top))
    return out


def standing_on(row, geom) -> int:
    """bridge/policy.js standingOn: the highest platform the player stands on, or -1."""
    x, y, z, on = row[0], row[1], row[2], row[6]
    if not on:
        return -1
    best = -1
    for k, (x0, x1, z0, z1, top) in enumerate(geom):
        if x + HALF_WIDTH > x0 and x - HALF_WIDTH < x1 and z + HALF_WIDTH > z0 and z - HALF_WIDTH < z1 and abs(y - top) < STAND_EPS:
            best = k
    return best


def frames_of(ep: dict, course: dict) -> list:
    """[x, y, z, vx, vy, vz, yaw, on_ground, jump_ticks, platform_reached] for the start state and every tick."""
    init, traj, acts = ep["init"], ep["traj"], ep["actions"]
    assert len(traj) == ep["ticks"] == len(acts), "one action per recorded tick (the last one ended the episode)"
    geom = platform_geometry(course)
    yaw, cp = init["yaw"], 0
    frames = [[*init["pos"], *init["vel"], yaw, int(bool(init["onGround"])), int(init["jumpTicks"]), 0]]
    for row, a in zip(traj, acts):
        yaw = wrap(yaw + YAW_STEPS_DEG[a[4]] * math.pi / 180)  # the bots' order: deg * PI / 180
        k = standing_on(row, geom)
        if k > cp:
            cp = k
        frames.append([*row[:6], yaw, int(row[6]), int(row[7]), cp])
    assert cp == ep["cp"], f"replayed platform {cp}, the bot recorded {ep['cp']}"
    assert (ep["result"] == "success") == (cp == len(course["platforms"]) - 1)
    return frames


def rounded(frames: list) -> list:
    return [[round(v, 6) for v in f[:7]] + f[7:] for f in frames]


def read_episodes(run: Path) -> list:
    if (run / "episodes.jsonl").exists():
        text = (run / "episodes.jsonl").read_text(encoding="utf-8")
    else:
        text = gzip.decompress((run / "episodes.jsonl.gz").read_bytes()).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def type_kinds() -> list:
    """The kinds of the one-type courses, in KINDS's shape: (token, label, group, levels, seeds, family, about)."""
    out = []
    for t in shape_types()["types"]:
        top, x0, x1, z0, z1 = shape_geometry(t)
        name, phrase = SHAPE_BLOCK_NAMES.get(t, (t.replace("_", " "), t.replace("_", " ")))
        out.append((f"shaped: {t}", f"{name[0].upper()}{name[1:]} only", "extra", (SHAPED_LEVEL,), SHAPE_SEEDS, SHAPE_FAMILY,
                    f"every shaped jump on this course lands on {phrase} (its top {top:g} blocks above the floor of "
                    f"its cell, footprint {x1 - x0:g} x {z1 - z0:g} blocks), then jumps off it"))
    return out


def course_specs(d: dict) -> list:
    """The course's JumpSpecs, one per jump from platform j to j + 1 (parkour.live reads the file the same way)."""
    from parkour.neo_assist import specs_from_keys
    specs = specs_from_keys(d["specs"], len(d["platforms"]))
    assert len(specs) == len(d["platforms"]) - 1 and [s.key() for s in specs] == d["specs"], d["id"]
    return specs


def leg_kinds(d: dict) -> list:
    """Per jump: "drawn" (drawn from the level tables) or added by courses.generate: "runway" (the step onto the
    runway of the next jump, JumpSpec(1, 0, 0, (runway, 1))), "exit" (exit_spec after a shaped block) or
    "bounce_exit" (bounce_exit after a slime pad). generate appends a draw's jumps in that order, so the jump
    before a runway jump is its runway step and the jump after a shaped or slime jump is its exit."""
    specs = course_specs(d)
    out = []
    for j, s in enumerate(specs):
        if j + 1 < len(specs) and specs[j + 1].runway:
            assert s.key() == C.JumpSpec(1, 0, 0, (specs[j + 1].runway, 1)).key(), (d["id"], j)
            out.append("runway")
        elif j and specs[j - 1].block:
            assert s.key() == C.exit_spec(specs[j - 1].block).key(), (d["id"], j)
            out.append("exit")
        elif j and specs[j - 1].bounce:
            assert s.key() == C.bounce_exit(specs[j - 1]).key(), (d["id"], j)
            out.append("bounce_exit")
        else:
            out.append("drawn")
    drawn = out.count("drawn")
    assert 1 <= drawn <= N_JUMPS, (d["id"], drawn)
    if drawn < N_JUMPS:  # generate stops early only at MAX_PLATFORMS or after 30 failed tries (then fewer platforms)
        assert len(d["platforms"]) <= C.MAX_PLATFORMS, d["id"]
    return out


def own_jumps(d: dict, features: tuple) -> list:
    """[[feature, j, through]]: the jumps of course ``d`` with one of ``features`` (FEATURES), in course order.

    Jump j goes from platform j to j + 1. It counts as reached when the platform reached is at least j (for a
    runway jump, j is the runway) and as passed when it is at least through + 1: through is j, or j + 1 for a
    shaped block or a slime pad (their exit jump, which generate adds, is part of it)."""
    specs, legs = course_specs(d), leg_kinds(d)
    ice = set(d.get("ice", []))
    out = []
    for j, s in enumerate(specs):
        has = {
            "hh": bool(s.ceiling), "ice_run": bool(s.ice), "shaped": bool(s.block), "slime": bool(s.bounce),
            "ladder": bool(s.ladder), "neo": bool(s.neo), "run": bool(s.runway) and not s.ceiling and not s.ice,
            "side3": legs[j] == "drawn" and abs(s.lateral) == 3 and not (s.neo or s.runway or s.block or s.bounce or s.ladder),
            "ice": j in ice or j + 1 in ice}
        for f in features:
            if has[f]:
                out.append([f, j, j + 1 if f in ("shaped", "slime") else j])
    return out


def extra_roles(d: dict) -> tuple:
    """(roles, jumps): the role of every extra cell of course ``d``, as parkour.courses placed it, and the jump
    whose spec placed it.

    "roof": a head-hitter roof (_ceiling), "neo": a neo wall (_neo_wall), "ladder": the wall under a ladder target
    (_ladder). Every jump spec with a roof, a neo wall or a ladder rebuilds its cells with the generator's own
    helper from the course's platforms (spec j is the jump from platform j to j + 1, after its runway if it has
    one); the heading is the one of the four whose cells are all extra cells of the course (for a ladder, whose
    ladder rows are also the course's). The cells must cover the extra cells exactly: these courses have no
    obstacles (obstacle_prob 0), so no cell may be left over, and no cell may come from two jumps."""
    plats = [np.array(p, dtype=np.int64) for p in d["platforms"]]
    specs = course_specs(d)
    extra = [tuple(int(v) for v in e) for e in d["extra"]]
    have = set(extra)
    ladder_rows = {tuple(int(v) for v in r) for r in d.get("ladders", [])}
    role: dict = {}
    owner: dict = {}
    for j, s in enumerate(specs):
        if not (s.ceiling or s.neo or s.ladder):
            continue
        prev, new = plats[j], plats[j + 1]
        found = set()
        for h in range(4):
            if s.ceiling:
                kind, cells = "roof", C._ceiling(prev, new, h, s)
            elif s.ladder:
                wall, rows = C._ladder(prev, new, h)
                if not {tuple(int(v) for v in r) for r in rows} <= ladder_rows:
                    continue
                kind, cells = "ladder", wall
            else:  # the wall reaches 2 above the highest platform from two before the takeoff (its runway's) to the target
                top = max(int(p[4]) for p in plats[max(0, j - (3 if s.runway else 2)):j + 2])
                kind, cells = "neo", C._neo_wall(prev, new, h, s, top=top)
            cells = frozenset(tuple(int(v) for v in c) for c in cells)
            if cells and cells <= have:
                found.add((kind, cells))
        assert len(found) == 1, f"{d['id']}: jump {j} ({s.key()}) matches {len(found)} placements"
        kind, cells = found.pop()
        for c in cells:
            assert role.setdefault(c, kind) == kind, f"{d['id']}: cell {c} is both {role[c]} and {kind}"
            assert owner.setdefault(c, j) == j, f"{d['id']}: cell {c} comes from jumps {owner[c]} and {j}"
    out = [role.get(c, "other") for c in extra]
    assert "other" not in out, f"{d['id']}: extra cells no jump explains: {[c for c in extra if c not in role]}"
    return out, [owner[c] for c in extra]


def collision_boxes(c: dict, shapes: dict) -> list:
    """Every collision box of a page course in world blocks [x0, y0, z0, x1, y1, z1]: platforms (a shaped one box
    by box), extra cells (whole blocks) and ladders, as the page draws them."""
    out = []
    sh = c.get("shapes") or {}
    for k, (x0, x1, z0, z1, y) in enumerate(c["platforms"]):
        bl = shapes[sh[str(k)]]["boxes"] if str(k) in sh else [[0, 0, 0, 1, 1, 1]]
        out += [[x + b[0], y + b[1], z + b[2], x + b[3], y + b[4], z + b[5]] for x in range(x0, x1 + 1) for z in range(z0, z1 + 1) for b in bl]
    out += [[x, y, z, x + 1, y + 1, z + 1] for x, y, z in c["extra"]]
    for x, y, z, f in c.get("ladders", []):
        out += [[x + b[0], y + b[1], z + b[2], x + b[3], y + b[4], z + b[5]] for b in shapes["ladder_" + ("north", "east", "south", "west")[f]]["boxes"]]
    return out


def past_edge(courses_out: list, episodes: list, shapes: dict) -> dict:
    """Grounded frames whose hitbox (0.6 x 0.6 around x, z) is over no box top at its feet height: the physics moves
    vertically first (at the previous tick's x and z) and then sideways, so a tick can land and end past the edge.
    The page counts the same from the embedded frames."""
    boxes = {c["id"]: collision_boxes(c, shapes) for c in courses_out}
    over = lambda B, x, y, z: any(abs(b[4] - y) < STAND_EPS and x + HALF_WIDTH > b[0] and x - HALF_WIDTH < b[3]  # noqa: E731
                                  and z + HALF_WIDTH > b[2] and z - HALF_WIDTH < b[5] for b in B)
    grounded = past = landed_before = 0
    for e in episodes:
        B, F = boxes[e["course"]], e["frames"]
        for i, f in enumerate(F):
            if f[7]:
                grounded += 1
                if not over(B, f[0], f[1], f[2]):
                    past += 1
                    landed_before += bool(i) and over(B, F[i - 1][0], f[1], F[i - 1][2])
    assert landed_before == past, "a grounded frame past an edge that the previous tick's x and z do not explain"
    return {"grounded": grounded, "past_an_edge": past, "over_a_block_at_the_previous_ticks_x_z": landed_before}


def cmd_page(a) -> None:
    check_construction()
    cs = course_set()
    by_id = {d["id"]: d for d in cs}
    course_file = ROOT / COURSES_FILE
    assert json.loads(course_file.read_text(encoding="utf-8"))["courses"] == cs, f"{COURSES_FILE} is not the fixed set"
    prov = json.loads((ROOT / PROVENANCE).read_text(encoding="utf-8"))
    policy_sha = sha256(ROOT / POLICY)
    assert policy_sha == prov["export_sha256"], "results/combined_policy.pt is not the file its provenance names"
    kinds = list(KINDS) + type_kinds()
    assert {d["kind"] for d in cs} == {k[0] for k in kinds}
    for t, _, _, lv, n, f, _ in kinds:  # every kind's courses: its levels, its family, seeds 0..n-1 per level
        mine = [d for d in cs if d["kind"] == t]
        assert len(mine) == len(lv) * n and {d["family"] for d in mine} == {f} and {d["level"] for d in mine} == set(lv), t
        assert sorted((d["level"], d["seed"]) for d in mine) == [(L, i) for L in lv for i in range(n)], t

    runs, episodes, seen = [], [], set()
    sim_by_course = {}
    for run_dir in a.runs:
        run_dir = run_dir.resolve()
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        token, label = next(v for k, v in RUN_KINDS.items() if summary["mode"].startswith(k))
        assert token not in seen, f"two {label} runs"
        seen.add(token)
        assert Path(summary["checkpoint"]).as_posix().endswith(POLICY), summary["checkpoint"]
        run_courses = json.loads((run_dir / "courses.json").read_text(encoding="utf-8"))["courses"]
        assert run_courses == [bare(d) for d in cs], f"{run_dir} did not play the fixed course set"
        eps = read_episodes(run_dir)
        rows = summary["episodes"]
        # One episode per course, in course order, and the summary's rows are the same episodes.
        assert [e["index"] for e in eps] == list(range(len(cs))), f"{run_dir}: not exactly one episode per course"
        assert [r["index"] for r in rows] == [e["index"] for e in eps] and summary["courses"] == len(cs) == len(rows)
        assert summary["success"] == sum(r["result"] == "success" for r in rows) / len(rows)
        for e, r in zip(eps, rows):
            d = cs[e["index"]]
            assert e.get("mode") == token and not e.get("assist") and not e.get("plans")
            assert (r["result"], r["platform"], r["ticks"], r["n"], r["level"]) == (e["result"], e["cp"], e["ticks"], e["n"], e["level"])
            assert e["success"] == (e["result"] == "success") and e["result"] in ("success", "fell", "truncated")
            assert e["n"] == len(d["platforms"]) and e["level"] == d["level"] and e["seed"] == d["seed"]
            frames = frames_of(e, d)
            assert len(frames) == e["ticks"] + 1 == len(e["actions"]) + 1
            sim = (bool(r["sim_success"]), int(r["sim_ticks"]))
            assert sim_by_course.setdefault(d["id"], sim) == sim, f"{d['id']}: the simulator differs between runs"
            corr = r.get("server_corrections")
            assert corr == e.get("server_corrections") and (corr is None) == (token != "live"), (d["id"], corr)
            episodes.append({
                "id": f"{d['id']}-{token}", "course": d["id"], "run": token, "result": e["result"],
                "success": e["result"] == "success", "platform_reached": e["cp"], "ticks": e["ticks"],
                "duration_seconds": e["ticks"] / 20, "sim_success": sim[0], "sim_ticks": sim[1],
                "replay_max_pos_err": r["replay_max_pos_err"], "replay_first_divergence": r["replay_first_divergence"],
                "server_corrections": corr,
                "wall_seconds": round(e["wall_seconds"], 2) if e.get("wall_seconds") is not None else None,
                "actions": e["actions"], "frames": rounded(frames)})
        runs.append({"run": token, "label": label, "dir": rel(run_dir), "mode": summary["mode"],
                     "checkpoint": summary["checkpoint"], "wall_seconds": summary["wall_seconds"],
                     "courses": summary["courses"], "success": summary["success"],
                     "sim_success_same_courses": summary["sim_success_same_courses"],
                     "outcome_agreement": summary["outcome_agreement"], "replay_max_pos_err": summary["replay_max_pos_err"],
                     "replay_exact_episodes": summary["replay_exact_episodes"],
                     "episodes_sha256": sha256(run_dir / "episodes.jsonl") if (run_dir / "episodes.jsonl").exists() else None,
                     "summary_sha256": sha256(run_dir / "summary.json")})
    order = {"live": 0, "offline": 1}
    runs.sort(key=lambda r: order[r["run"]])
    episodes.sort(key=lambda e: (order[e["run"]], list(by_id).index(e["course"])))

    all_shapes = {s["name"]: s for s in json.loads((CONFIGS / "block_shapes.json").read_text(encoding="utf-8"))["shapes"]}
    used = {n for d in cs for n in (d.get("shapes") or {}).values()} | {
        "ladder_" + ("north", "east", "south", "west")[f] for d in cs for *_, f in d.get("ladders", [])}
    st = shape_types()
    assert set(st["types"]) == {n for d in cs for n in (d.get("shapes") or {}).values()}, "a trained type is on no course"
    for t in st["types"]:  # the page's edge-to-edge gaps take every footprint as a square centred in its cell
        _, x0, x1, z0, z1 = shape_geometry(t)
        assert abs(x0 - z0) < 1e-12 and abs(x1 - z1) < 1e-12 and abs(x0 - (1 - x1)) < 1e-12, t
    kind_features = {t: (TYPE_FEATURES if t.startswith("shaped: ") else KIND_FEATURES[t]) for t, *_ in kinds}
    courses_out = []
    for d in cs:
        c = {k: d[k] for k in ("id", "kind", "level", "family", "seed", "platforms", "spawn", "spawn_yaw", "specs", "extra")}
        c |= {k: d[k] for k in ("ice", "shapes", "slime", "ladders") if k in d}
        c["extra_roles"], c["extra_jumps"] = extra_roles(d)
        c["legs"] = leg_kinds(d)
        c["own_jumps"] = own_jumps(d, kind_features[d["kind"]])
        if d["kind"].startswith("shaped: "):  # every shaped jump of a one-type course is of its type
            assert [f for f, *_ in c["own_jumps"]].count("shaped") == len(d.get("shapes") or {}), d["id"]
        courses_out.append(c)
    payload = {
        "schema": "parkour-skill-replays-v1",
        "tick_hz": 20,
        "frame_columns": FRAME_COLUMNS,
        "action_columns": ["move", "strafe", "jump", "sprint", "turn_index"],
        "yaw_steps_degrees": list(YAW_STEPS_DEG),
        "policy": {"file": POLICY, "sha256": policy_sha, "provenance": PROVENANCE, "run": prov["run"],
                   "iteration": prov["iteration"], "source": prov["source"], "selection": prov["selection"]},
        "course_set": {"file": COURSES_FILE, "sha256": sha256(course_file), "families": [FAMILY, ICE_FAMILY, SHAPE_FAMILY],
                       "jumps": N_JUMPS, "max_platforms": C.MAX_PLATFORMS, "first_courses": 45,
                       "fixed": ("the first 45 courses on 29 October 2026 before any outcome was seen; the 20 one-type courses "
                                 "the same day at the user's request, after the first 45 had been played in offline physics "
                                 "and the first 14 in real Minecraft, and before any of the 20 was played"),
                       "builder": "scripts/make_skill_replays.py"},
        "features": {t: {"one": one, "many": many, "about": about} for t, (one, many, about) in FEATURES.items()},
        "kinds": [{"token": t, "label": lab, "group": g, "levels": list(lv), "seeds": n, "family": f, "about": about,
                   "features": list(kind_features[t]),
                   "set": "type" if t.startswith("shaped: ") else "skill", **({"block_type": t[8:]} if t.startswith("shaped: ") else {})}
                  for t, lab, g, lv, n, f, about in kinds],
        "block_types": [{"name": t, "label": SHAPE_BLOCK_NAMES.get(t, (t,))[0], "block": all_shapes[t]["block"],
                         "state": all_shapes[t]["properties"], "spec_count": st["spec_counts"][t],
                         "top": shape_geometry(t)[0]} for t in st["types"]],
        "shaped_level": {"global": SHAPED_LEVEL, "local_in_extra_file": shaped_local_level() - SKILL_GROUPS["extra"][1],
                         "specs": st["specs"]},
        "shapes": {k: {"block": all_shapes[k]["block"], "boxes": all_shapes[k]["boxes"]} for k in sorted(used)},
        "runs": runs,
        "courses": courses_out,
        "episodes": episodes,
    }
    for e in episodes:
        assert len(e["frames"]) == e["ticks"] + 1 == len(e["actions"]) + 1, e["id"]
    assert len(episodes) == len(cs) * len(runs) and len({e["id"] for e in episodes}) == len(episodes)
    assert {(e["course"], e["run"]) for e in episodes} == {(d["id"], r["run"]) for d in cs for r in runs}
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    assert "</" not in data and "<!--" not in data  # safe inside the page's <script> element
    template = TEMPLATE.read_text(encoding="utf-8")
    assert template.count("__REPLAY_DATA__") == 1
    html = template.replace("__REPLAY_DATA__", data)
    assert len(html.encode("utf-8")) < 16_000_000, "the page must stay under 16 MB"
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:  # the same bytes on every platform
        f.write(html)

    # Per kind and per block type: real Minecraft, offline and simulator completions (summary.json, as embedded).
    made = {(e["course"], e["run"]): e["success"] for e in episodes}
    sim = {e["course"]: e["sim_success"] for e in episodes}
    tok = [r["run"] for r in runs]
    count = lambda ids: {"courses": len(ids), **{r: sum(made[i, r] for i in ids) for r in tok},  # noqa: E731
                         "sim": sum(sim[i] for i in ids)}
    per_kind = {t: count([d["id"] for d in cs if d["kind"] == t]) for t, *_ in kinds}
    per_type = {}
    for t in st["types"]:
        ids = [d["id"] for d in cs if t in (d.get("shapes") or {}).values()]
        per_type[t] = {"platforms": sum(list((d.get("shapes") or {}).values()).count(t) for d in cs), **count(ids),
                       "own_courses": count([d["id"] for d in cs if d["kind"] == f"shaped: {t}"])}
    corr = {e["course"]: e["server_corrections"] for e in episodes if e["run"] == "live"}
    # Each kind's own jumps: on the courses, and reached / passed per recording (from the recorded platform reached).
    reach = {(e["course"], e["run"]): e["platform_reached"] for e in episodes}
    own = {}
    for t, *_ in kinds:
        for f in kind_features[t]:
            js = [(c["id"], j, th) for c in courses_out if c["kind"] == t for g, j, th in c["own_jumps"] if g == f]
            own[f"{t} / {f}"] = {"on_courses": len(js), "courses_with_one": len({i for i, *_ in js}),
                                 **{r: {"reached": sum(reach[i, r] >= j for i, j, _ in js),
                                        "passed": sum(reach[i, r] >= th + 1 for i, _, th in js)} for r in tok}}
    legs = Counter(k for c in courses_out for k in c["legs"])
    drawn = [c["legs"].count("drawn") for c in courses_out]
    print(json.dumps({"out": str(a.out), "bytes": a.out.stat().st_size, "episodes": len(episodes), "runs": tok,
                      "frames": sum(len(e["frames"]) for e in episodes),
                      "jumps": {"legs": dict(legs), "drawn_per_course": [min(drawn), max(drawn)],
                                "legs_per_course": [min(len(c["legs"]) for c in courses_out), max(len(c["legs"]) for c in courses_out)],
                                "courses_with_fewer_than_n_jumps_drawn": sum(n < N_JUMPS for n in drawn),
                                "courses_at_max_platforms": sum(len(c["platforms"]) == C.MAX_PLATFORMS for c in courses_out)},
                      "own_jumps": own,
                      "grounded_frames_past_an_edge": past_edge(courses_out, episodes, payload["shapes"]),
                      "all": count(list(by_id)), "per_kind": per_kind, "per_type": per_type,
                      "disagreements": [i for i in by_id if len({made[i, r] for r in tok} | {sim[i]}) > 1],
                      "server_corrections": {"total": sum(corr.values()), "nonzero": {k: v for k, v in corr.items() if v}} if corr else None,
                      "replay_max_pos_err": {r["run"]: r["replay_max_pos_err"] for r in runs},
                      "replay_exact_episodes": {r["run"]: r["replay_exact_episodes"] for r in runs}}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("courses", help="write the fixed course set for parkour.live --courses-file")
    c.add_argument("out", type=Path)
    p = sub.add_parser("page", help="build the replay page from parkour.live run folders")
    p.add_argument("out", type=Path)
    p.add_argument("runs", type=Path, nargs="+")
    a = ap.parse_args()
    cmd_courses(a) if a.cmd == "courses" else cmd_page(a)


if __name__ == "__main__":
    main()

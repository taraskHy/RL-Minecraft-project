"""Replays on fresh courses: the one network on every skill added after levels 0-5, on courses with one neo each
(the planner plays those too) and on block shapes that no training level uses.

    python scripts/make_skill_replays.py courses OUT.json
    python scripts/make_skill_replays.py page OUT.html --set COURSES RUN_DIR [RUN_DIR ...] [--set COURSES RUN_DIR ...]

The page, from the repository root (one command line; results/games/ holds every course file and every run folder
it reads, episodes as episodes.jsonl.gz, so a fresh clone rebuilds it; the runs/ folders of the same names hold the
same files with episodes.jsonl unpacked, and either can be given):

    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/make_skill_replays.py page results/skill_replays.html
        --set results/games/skill-replays-courses-65.json
              results/games/skill-replays-live-65 results/games/skill-replays-offline-65
        --set results/games/skill-replays-courses-neo.json
              results/games/skill-replays-neo-network-live results/games/skill-replays-neo-network-offline
              results/games/skill-replays-neo-planner-live results/games/skill-replays-neo-planner-offline
        --set results/games/skill-replays-courses-untrained.json
              results/games/skill-replays-untrained-live results/games/skill-replays-untrained-offline

Each --set is a course file followed by the run folders that played it, in any order. The builder recognises the
set by its courses and checks the file against the set's own construction (below), so a file that differs from it
stops the build.

The user asked on 29 October 2026 for replays of head hitters, slime, ice and
every other new kind. The policy is the shipped one network,
results/combined_policy.pt (the distilled student ``combined-distill``,
iteration 899; provenance in results/combined_policy_provenance.json; its
config's skills are basic, advanced and extra). The skills are the ones added
after levels 0-5, which the one network covers and results/final_policy.pt
(levels 0-5 and obstacles only) does not.

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

Two sets were added the same evening, after the user saw that no replay shows
a neo done (the nine advanced-level courses hold three neos, and the one
network fell at each) and asked for stairs; each was fixed before any of its
outcomes was seen:

* one neo per course: results/games/skill-replays-courses-neo.json, written by
  scripts/make_neo_showcase_courses.py (parkour.neo_assist.single_neo_courses
  with its own seed family 45, never used before: of its first 32 courses,
  which hold each of the 32 single-neo types of levels 6 and 7 once, the
  even-numbered ones, so 16 types, each the neo of one course; the other nine
  jumps are ordinary jumps of levels 0-5). Each course names its neo:
  neo_jump (the takeoff platform; the neo is the jump from it to the next
  one), neo_type (the index in single_neo_types) and neo_spec. Kind
  "neo-single", course ids neo-single-<i>.
* block shapes no training level uses: results/games/skill-replays-courses-untrained.json,
  written by scripts/make_untrained_shape_courses.py: stairs, top slab, top
  trapdoor and iron bars, two courses each, from the streams
  default_rng([2026, 44, k]) with k = 10..13 (continuing the one-type courses'
  k = 0..9 in family 44): courses.generate on the extra group's table, its
  shaped-blocks level keeping only the dirt-path specs (the end-rod specs for
  iron bars) with the block swapped, every stairs platform turned to rise in
  the direction of travel. Kind "untrained: <shape>", course ids
  untrained-<shape>-<seed>; based_on names the trained type whose jumps were used.

The one network plays both sets; on the one-neo courses the planner plays too:
results/final_policy.pt with parkour.neo_assist's search, set to plan every
jump (parkour.live --assist: when the bot first stands on a takeoff it pauses,
a cross-entropy search over the bot's discrete actions, the policy's action
set (with the setting's prior 0 the policy is not consulted), in the
simulator's physics from the bot's exact state, looks for a sequence that
lands on the next platform, the bot checks it with prismarine-physics on its
own world and plays it; when a takeoff's first search finds none it brakes to
a stop and searches once more, and when that one finds none too the policy
plays the jump; the policy plays every tick without a plan, the first of each
course included). The planner's policy is results/final_policy.pt as saved
from runs/obstacles-g995/checkpoints/best.pt at iteration 699 without the
optimizer state (so the two files' hashes differ): the builder reads the
iteration and the source from the checkpoint and, where runs/ has that file,
checks that the weights are the same. The setting is the one chosen on the
single-neo validation set (family 40, results/neo/assist_validation.jsonl,
where results/final_policy.pt alone, row "single V0: final policy alone",
completed none of the courses):

    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m parkour.live [--offline] --checkpoint results/combined_policy.pt
        --courses-file results/games/skill-replays-courses-neo.json --out runs/skill-replays-neo-network-<live|offline>
    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m parkour.live [--offline] --checkpoint results/final_policy.pt
        --courses-file results/games/skill-replays-courses-neo.json --out runs/skill-replays-neo-planner-<live|offline>
        --assist '{"jumps": "all", "retries": 1, "brake": 14, "retry_horizon": 48, "restarts": 3}'
    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m parkour.live [--offline] --checkpoint results/combined_policy.pt
        --courses-file results/games/skill-replays-courses-untrained.json --out runs/skill-replays-untrained-<live|offline>

(each summary.json holds the planner's whole config, which the builder checks
against that --assist JSON with AssistConfig's other defaults and batch 1;
the folders are archived under results/games/ with the same names). The
planner is not the one network: the search, with an exact model of the
game's physics, plans the jumps, and results/final_policy.pt plays every move
without a plan; it measures what the bot does when it plans its jumps.

``page`` turns the runs into results/skill_replays.html
(scripts/skill_replays_template.html). A recording is a place (real Minecraft
or offline physics) and a player: "one network" (results/combined_policy.pt
alone) or "planner" (results/final_policy.pt with the search); every course
has the one network's two recordings, the one-neo courses also the planner's
two. Every course of every set from every run, in the frame format of the
showcase replay export
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
jumps 3 blocks sideways; ice courses: every jump onto or off an ice platform;
the one-neo courses: the neo; the never-trained shapes: the landings on the
shape). An own jump counts as reached when the bot stood on its takeoff (for a
runway jump, the runway) and as passed when it reached the next platform
(after a shaped block or a slime pad, the exit platform after it), from the
recorded platform reached. A neo counts as done when the platform reached
took the value of the neo's takeoff and then of its target (the bot stood on
the takeoff before on any later platform, then on the target before on any
later one: the rule of parkour.neo_assist.neo_done); the page counts that from
the embedded frames.

The planner's records hold its searches, not its plans: bridge/policy.js
Episode.record writes "assist", one entry per request of bridge/assist.js
(the tick, the takeoff, the kind: takeoff or retry, whether the search found a
sequence, whether prismarine-physics verified it, whether it was played,
whether the bot braked, the search rounds and seconds), and the actions,
which do not say where they came from. The ticks a plan played follow from
the rules that made them: parkour.neo_assist.search cuts every sequence at the
tick it first stands on its target, and the episode takes the next request on
the tick its queue empties; a brake is brake_plan's keys ("back" for up to 6
ticks, then none; ``brake`` ticks in all) and the retry comes on the tick it
ends. Only a takeoff's first search brakes when it finds nothing (bridge/assist.js
and parkour.neo_assist brake while the attempt is within ``retries``, 1 here);
when the second search finds nothing too, the policy plays the jump. So a
played plan's actions are the recorded actions from its search's tick to the
first tick on which the bot stands on the plan's target (the target from the
run's courses.json: assist_targets), a brake's the next ``brake`` ticks, and
every other tick's action is the policy's. The builder checks it: every played
plan reaches its target within the plan's horizon, the next search comes on
exactly the tick the plan ends (or the plan ends the episode on the goal), a
search brakes exactly when it is a takeoff's first and found nothing, every
brake's actions have brake_plan's form and its retry comes on the tick it ends,
no search falls inside a plan or a brake, the searches add up to summary.json's
counts, and the run's assist fields equal parkour.neo_assist.assist_fields for
its config.

The builder checks that every course has exactly one episode per run, frames =
ticks + 1, the replayed platform equals the bot's, every result equals
summary.json, the simulator's result for a course is the same in every
recording of the same player (the one network's simulator run plays the
network; the planner's, parkour.neo_assist.evaluate_assisted, plays the
planner), every extra cell has exactly one role and one jump, and the embedded
JSON cannot end the page's <script> element. These are post-training
recordings on fresh showcase courses, not benchmark results.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
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
PLANNER_POLICY = "results/final_policy.pt"  # the planner's policy: the final policy (levels 0-5 and obstacles)
PLANNER_EXPORT = "results/final_policy.json"  # its bot export; meta.checkpoint names its source
PLANNER_SETTING = {"jumps": "all", "retries": 1, "brake": 14, "retry_horizon": 48, "restarts": 3}  # the --assist JSON
ASSIST_VALIDATION = "results/neo/assist_validation.jsonl"  # where the setting was chosen (single-neo, family 40)
COURSES_FILE = "results/games/skill-replays-courses-65.json"  # the 65 courses the bots play
FIRST_COURSES_FILE = "results/games/skill-replays-courses.json"  # the first 45, written before the 20 were added
N_JUMPS = 10
FAMILY = 42  # extra and advanced groups (never used before 29 Oct 2026)
ICE_FAMILY = 43  # ice courses
SHAPE_FAMILY = 44  # one block type per course: stream default_rng([2026, 44, k]) for type k
NEO_FAMILY = 45  # one neo per course (scripts/make_neo_showcase_courses.py)
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
NEO_KIND = ("neo-single", "Neo jumps, one per course")  # the one-neo set's kind token and label
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
TYPE_FEATURES = ("shaped",)  # the one-type and the never-trained courses
NEO_FEATURES = ("neo",)  # the one-neo courses
SHAPE_BLOCK_NAMES = {  # how the page names each type: (name, "lands on ..."); configs/block_shapes.json has the block state
    "bottom_slab": ("bottom slab", "a bottom stone slab"), "soul_sand": ("soul sand", "soul sand"),
    "dirt_path": ("dirt path", "a dirt path block"), "enchanting_table": ("enchanting table", "an enchanting table"),
    "daylight_detector": ("daylight detector", "a daylight detector"),
    "bottom_trapdoor": ("bottom trapdoor", "a closed oak trapdoor in the bottom half"), "carpet": ("carpet", "a white carpet"),
    "fence_post": ("fence post", "a lone oak fence post"), "wall_post": ("wall post", "a lone cobblestone wall post"),
    "end_rod": ("end rod", "an upright end rod"),
    # the shapes no training level uses (scripts/make_untrained_shape_courses.py); stairs: every facing is one type
    "stairs": ("stairs", "stone stairs, a half-block slab with a step on half of it, turned to rise the way the bot travels"),
    "top_slab": ("top slab", "a stone slab in the top half of its cell"),
    "top_trapdoor": ("top trapdoor", "a closed oak trapdoor in the top half of its cell"),
    "iron_bars": ("iron bars", "a lone iron bars post")}
RUN_KINDS = {  # summary.json "mode" prefix -> (place token, label)
    "live": ("live", "Real Minecraft"),
    "offline": ("offline", "Offline physics"),
}
PLAYERS = {"network": POLICY, "planner": PLANNER_POLICY}  # player -> the checkpoint its runs name
SET_ORDER = ("skills", "neo", "untrained")  # the page's order of the course sets
REC_ORDER = ("live", "offline", "planner-live", "planner-offline")  # recordings: place, or planner-place
PAGE_KEYS = ("kind", "family", "id", "neo_type", "neo_jump", "neo_spec", "based_on")  # the course files' own keys
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
    return {k: v for k, v in d.items() if k not in PAGE_KEYS}


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


# ---------- the newer sets: one neo per course, and the shapes no training level uses ----------
def script(name: str):
    """A course generator of scripts/ (the newer sets), imported from its file."""
    spec = importlib.util.spec_from_file_location(f"_skill_replays_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_of(raw: list) -> str:
    """Which set a course file holds, by its courses' kinds."""
    kinds = {d.get("kind", "") for d in raw}
    if kinds == {NEO_KIND[0]}:
        return "neo"
    if kinds and all(k.startswith("untrained: ") for k in kinds):
        return "untrained"
    return "skills"


def fixed_courses(name: str) -> list:
    """The set's courses from its own construction (course-file dicts, as the file holds them)."""
    if name == "skills":
        check_construction()
        out = course_set()
    elif name == "neo":
        out = script("make_neo_showcase_courses").courses()
    else:
        out = script("make_untrained_shape_courses").courses()
    return json.loads(json.dumps(out))  # as read back from a file (string keys)


def untrained_table() -> list:
    """(k, shape, trained type whose jumps it takes) of scripts/make_untrained_shape_courses.py."""
    return list(script("make_untrained_shape_courses").UNTRAINED)


def neo_kind(cs: list) -> tuple:
    """The one-neo set's kind, in KINDS's shape (seeds None: the seeds are the courses' own), its text from the courses."""
    from parkour.neo_assist import single_neo_types
    types = single_neo_types()
    pos = sorted({d["neo_jump"] for d in cs})
    lv = tuple(sorted({d["level"] for d in cs}))
    about = (f"exactly one neo per course, a jump around the edge of a wall: {len({d['neo_type'] for d in cs})} of the "
             f"{len(types)} single-neo jump types of levels {' and '.join(map(str, sorted({L for L, _ in types})))}, "
             f"one per course, as jump {', '.join(str(k) for k in pos[:-1])} or {pos[-1]} (counting from 0); "
             f"the other jumps are ordinary jumps of levels 0-5")
    return (NEO_KIND[0], NEO_KIND[1], "neo", lv, None, NEO_FAMILY, about)


def untrained_kinds() -> list:
    """The kinds of the never-trained shapes, in KINDS's shape."""
    out = []
    for _, shape, base in untrained_table():
        top, x0, x1, z0, z1 = shape_geometry("stairs_north" if shape == "stairs" else shape)
        btop = shape_geometry(base)[0]
        name, phrase = SHAPE_BLOCK_NAMES[shape]
        out.append((f"untrained: {shape}", f"Never trained: {name}", "untrained", (SHAPED_LEVEL,), SHAPE_SEEDS, SHAPE_FAMILY,
                    f"every shaped jump on this course lands on {phrase} (its top {blocks(top)} above the floor of its "
                    f"cell, footprint {x1 - x0:g} x {z1 - z0:g} blocks), a shape no training level uses: the jumps are "
                    f"the {SHAPE_BLOCK_NAMES[base][0]} jumps of the shaped-blocks level (top {btop:g}) with the block "
                    f"swapped; then it jumps off"))
    return out


def shapes_of_type(t: str, all_shapes: dict) -> list:
    """The configs/block_shapes.json shapes of a page block type: stairs is every stairs facing, others one shape."""
    return sorted(n for n in all_shapes if n.startswith("stairs_")) if t == "stairs" else [t]


def check_neo_courses(cs: list) -> None:
    """Every one-neo course holds exactly its named neo: the wall rule finds one neo, at neo_jump, the spec there is
    neo_spec, the only spec with a neo, and neo_type names that spec (and the course's level) in single_neo_types."""
    from parkour.neo_assist import courses_from_file, single_neo_types
    from parkour.neo_dagger import neo_takeoffs
    types = single_neo_types()
    for d, c in zip(cs, courses_from_file([bare(d) for d in cs])):
        k = d["neo_jump"]
        assert neo_takeoffs(c) == [k], (d["id"], neo_takeoffs(c))
        assert [j for j, s in enumerate(c.specs) if s.neo] == [k] and c.specs[k].key() == d["neo_spec"], d["id"]
        assert types[d["neo_type"]][1].key() == d["neo_spec"] and types[d["neo_type"]][0] == d["level"], d["id"]
        assert d["family"] == NEO_FAMILY and d["id"] == f"{NEO_KIND[0]}-{d['seed']}", d["id"]


def check_untrained_courses(cs: list, all_shapes: dict) -> None:
    """Every never-trained course holds only its kind's shape (stairs: any facing), at least once; family 44."""
    for d in cs:
        t = d["kind"][len("untrained: "):]
        on = set((d.get("shapes") or {}).values())
        assert on and on <= set(shapes_of_type(t, all_shapes)), (d["id"], on)
        assert d["family"] == SHAPE_FAMILY and d["level"] == SHAPED_LEVEL and d["id"] == f"untrained-{t}-{d['seed']}", d["id"]


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


def stands(row, q) -> bool:
    """bridge/policy.js standingOn's test for one platform (q: its platform_geometry row); row: x y z vx vy vz on_ground."""
    x, y, z, on = row[0], row[1], row[2], row[6]
    return bool(on) and x + HALF_WIDTH > q[0] and x - HALF_WIDTH < q[1] and z + HALF_WIDTH > q[2] and z - HALF_WIDTH < q[3] \
        and abs(y - q[4]) < STAND_EPS


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


def episodes_bytes(run: Path) -> bytes:
    """The run's episodes.jsonl, from the file or from its gzip archive (the same bytes)."""
    if (run / "episodes.jsonl").exists():
        return (run / "episodes.jsonl").read_bytes()
    return gzip.decompress((run / "episodes.jsonl.gz").read_bytes())


def read_episodes(run: Path) -> list:
    return [json.loads(line) for line in episodes_bytes(run).decode("utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def blocks(v: float) -> str:
    return f"{v:g} block" + ("" if v == 1 else "s")


def type_kinds() -> list:
    """The kinds of the one-type courses, in KINDS's shape: (token, label, group, levels, seeds, family, about)."""
    out = []
    for t in shape_types()["types"]:
        top, x0, x1, z0, z1 = shape_geometry(t)
        name, phrase = SHAPE_BLOCK_NAMES.get(t, (t.replace("_", " "), t.replace("_", " ")))
        out.append((f"shaped: {t}", f"{name[0].upper()}{name[1:]} only", "extra", (SHAPED_LEVEL,), SHAPE_SEEDS, SHAPE_FAMILY,
                    f"every shaped jump on this course lands on {phrase} (its top {blocks(top)} above the floor of "
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


def neo_done(frames: list, k: int) -> bool:
    """parkour.neo_assist.neo_done for one neo at takeoff k: the platform reached took the values k and k + 1."""
    values = {f[9] for f in frames}
    return k in values and k + 1 in values


# ---------- the planner's searches and the ticks its plans played ----------
def plan_control(e: dict, d: dict, course, run_course: dict, cfg) -> tuple:
    """(searches, control) of one planner episode: the search log as the page shows it, and [[start, end, kind,
    search]] for the ticks whose actions came from a played plan ("plan") or a brake ("brake"): actions[start:end],
    search being the index of the search that gave them. Every other tick's action is the policy's.

    A played plan ends on the first tick its target is stood on (parkour.neo_assist.search cuts it there); the
    next search must then come on exactly that tick, or the plan ended the episode on the goal. A brake is
    ``cfg.brake`` ticks of brake_plan's keys ("back" for 0 to 6 ticks, then none) and its retry comes when it ends."""
    from parkour.neo_assist import BRAKE_BACK_TICKS, plan_horizon
    geom = platform_geometry(d)
    traj, acts, log = e["traj"], e["actions"], e["assist"]
    target = dict(zip(run_course["assist_takeoffs"], run_course["assist_targets"]))
    searches, control = [], []
    for n, x in enumerate(log):
        T = x["tick"]
        assert x["kind"] in ("takeoff", "retry") and not x.get("error"), (d["id"], x)  # no approach plans in this setting
        assert x["cp"] == x["takeoff"] and x["takeoff"] in target, (d["id"], x)
        assert 1 <= T <= e["ticks"] and (not control or T >= control[-1][1]), f"{d['id']}: a search inside a plan or a brake"
        assert x["played"] == (x["found"] and x["verified"]) and not (x["played"] and x["braked"]), (d["id"], x)
        # Only a takeoff's first search brakes, when it finds nothing (retries 1); the page's text says so.
        assert x["braked"] == (x["kind"] == "takeoff" and not x["found"]), (d["id"], x)
        nxt = log[n + 1] if n + 1 < len(log) else None
        tg = target[x["takeoff"]]
        if x["played"]:
            end = next((t for t in range(T + 1, e["ticks"] + 1) if stands(traj[t - 1], geom[tg])), None)
            assert end is not None, f"{d['id']}: the plan of search {n} never stands on its target {tg}"
            assert end - T <= plan_horizon(course, x["takeoff"], x["kind"], cfg), (d["id"], n, end - T)
            assert (nxt["tick"] == end) if nxt else (end == e["ticks"] and e["result"] == "success"), \
                f"{d['id']}: search {n}'s plan ends at tick {end}, the next search is at {nxt and nxt['tick']}"
            control.append([T, end, "plan", n])
        elif x["braked"]:
            end = T + cfg.brake
            b = acts[T:end]
            back = sum(1 for a in b if a[0] == 0)
            assert len(b) == cfg.brake and back <= BRAKE_BACK_TICKS and all(a[0] == 0 for a in b[:back]), (d["id"], n)
            assert all(a[0] == 1 and a[1:] == [1, 0, 0, YAW_STEPS_DEG.index(0.0)] for a in b[back:]), (d["id"], n)
            assert all(a[1:] == [1, 0, 0, YAW_STEPS_DEG.index(0.0)] for a in b), (d["id"], n)
            assert nxt and nxt["kind"] == "retry" and nxt["takeoff"] == x["takeoff"] and nxt["tick"] == end, (d["id"], n)
            control.append([T, end, "brake", n])
        searches.append({"kind": x["kind"], "takeoff": x["takeoff"], "target": tg, "tick": T, "found": x["found"],
                         "verified": x["verified"], "played": x["played"], "braked": x["braked"], "rounds": x["rounds"],
                         "seconds": x["seconds"]})
    return searches, control


def assist_setting(summaries: list) -> dict:
    """The planner's config: the same in every planner run, the --assist JSON of the docstring with AssistConfig's
    other defaults (batch 1, as parkour.live sets it), its row of the single-neo validation set and the row of the
    planner's policy alone there, if there (each measured on this results/final_policy.pt: its sha1)."""
    from parkour.neo_assist import AssistConfig, config_meta
    cfgs = [s["assist"]["config"] for s in summaries]
    full = lambda c: asdict(AssistConfig(**c))  # noqa: E731  (fields added after a recording, as plan_start: their defaults)
    assert all(full(c) == full(cfgs[0]) for c in cfgs), "the planner runs differ in their config"
    assert full(cfgs[0]) == asdict(AssistConfig(**{"batch": 1, **PLANNER_SETTING})), cfgs[0]
    cfg = AssistConfig(**cfgs[0])
    rows = [json.loads(line) for line in (ROOT / ASSIST_VALIDATION).read_text(encoding="utf-8").splitlines() if line.strip()]
    single = [r for r in rows if r.get("set") == "single" and r.get("split") == "validation" and r.get("checkpoint") == PLANNER_POLICY
              and not r.get("settle") and not r.get("first_courses_only")]  # the simulator's own start, every course
    mine = [r for r in single if r.get("assist") and r.get("config") == config_meta(cfg)]
    alone = [r for r in single if r.get("assist") is False]
    assert len(mine) <= 1 and len(alone) <= 1, "two validation rows with the planner's setting, or of its policy alone"
    sha1 = hashlib.sha1((ROOT / PLANNER_POLICY).read_bytes()).hexdigest()
    for r in mine + alone:
        assert sha1.startswith(r.get("checkpoint_sha1", "")), f"{ASSIST_VALIDATION} row {r['tag']!r} measured another file"
    chosen = solo = None
    if mine:
        r = mine[0]
        chosen = {"file": ASSIST_VALIDATION, "tag": r["tag"], "set": r["set"], "family": r["family"], "measured": r["measured"],
                  "courses": r["courses"], "completed": r["completed"], "completed_neo_done": r.get("completed_neo_done")}
    if alone:
        r = alone[0]
        solo = {"file": ASSIST_VALIDATION, "tag": r["tag"], "set": r["set"], "family": r["family"], "measured": r["measured"],
                "courses": r["courses"], "completed": r["completed"], "neo_jumps": r["neo_jumps"],
                "neo_jumps_reached": r["neo_jumps_reached"], "neo_jumps_passed": r["neo_jumps_passed"]}
    return {"config": cfgs[0], "cfg": cfg, "chosen": chosen, "alone": solo}


def checkpoint(rel_path: str) -> dict:
    import torch
    return torch.load(ROOT / rel_path, map_location="cpu", weights_only=False)


def planner_policy() -> dict:
    """The planner's policy file: its iteration and its source as the checkpoint records them (results/final_policy.pt
    is runs/obstacles-g995/checkpoints/best.pt's weights without the optimizer state, so the two hashes differ).
    results/final_policy.json's meta names the same source; where runs/ holds the source, its weights and iteration
    are the same. The page calls it the final policy for levels 0-5 and obstacles: its config has no advanced or
    extra levels and no skills, and obstacles on."""
    import torch
    ck = checkpoint(PLANNER_POLICY)
    src = str(ck["source"])  # "path (notes)"
    path, _, note = src.partition(" (")
    note = note[:-1] if note.endswith(")") else note
    meta = json.loads((ROOT / PLANNER_EXPORT).read_text(encoding="utf-8")).get("meta", {})
    assert meta.get("checkpoint") == path, f"{PLANNER_EXPORT} names {meta.get('checkpoint')}, the checkpoint {path}"
    cfg = ck["config"] if isinstance(ck["config"], dict) else vars(ck["config"])
    assert not cfg.get("advanced") and not cfg.get("extra") and not cfg.get("skills") and cfg.get("obstacle_prob", 0) > 0, cfg
    if (ROOT / path).exists():  # the training PC; a fresh clone has no runs/
        src_ck = checkpoint(path)
        assert src_ck["iteration"] == ck["iteration"] and set(src_ck["model"]) == set(ck["model"]) and all(
            torch.equal(src_ck["model"][k], ck["model"][k]) for k in ck["model"]), f"{PLANNER_POLICY} is not {path}'s weights"
    return {"iteration": int(ck["iteration"]), "source": path, "source_note": note}


def check_network_config() -> None:
    """The page says the one network covers levels 0-5 and the skills added after them: its config's skills are
    basic, advanced and extra, with ice on (the ice courses are the basic levels with ice)."""
    cfg = checkpoint(POLICY)["config"]
    cfg = cfg if isinstance(cfg, dict) else vars(cfg)
    assert set(str(cfg.get("skills", "")).split(",")) == {"basic", "advanced", "extra"} and cfg.get("ice_prob", 0) > 0, cfg


def read_run(run_dir: Path, set_name: str, cs: list, courses: list, setting: dict | None) -> tuple:
    """(run, episodes) of one run folder of a set, every check above applied."""
    from parkour.neo_assist import assist_fields
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    where, label = next(v for k, v in RUN_KINDS.items() if summary["mode"].startswith(k))
    player = "planner" if "assist" in summary else "network"
    rec = where if player == "network" else f"planner-{where}"
    assert Path(summary["checkpoint"]).as_posix().endswith(PLAYERS[player]), summary["checkpoint"]
    run_courses = json.loads((run_dir / "courses.json").read_text(encoding="utf-8"))["courses"]
    if player == "network":
        assert run_courses == [bare(d) for d in cs], f"{run_dir} did not play the fixed course set"
    else:  # the planner's courses.json adds each course's assist fields (parkour.live: assist_fields)
        assert [{k: v for k, v in rc.items() if not k.startswith("assist_")} for rc in run_courses] == [bare(d) for d in cs], \
            f"{run_dir} did not play the fixed course set"
        for rc, c in zip(run_courses, courses):
            af = assist_fields(c, setting["cfg"])
            assert {k: v for k, v in rc.items() if k.startswith("assist_")} == af, f"{run_dir}: assist fields differ"
    eps = read_episodes(run_dir)
    rows = summary["episodes"]
    # One episode per course, in course order, and the summary's rows are the same episodes.
    assert [e["index"] for e in eps] == list(range(len(cs))), f"{run_dir}: not exactly one episode per course"
    assert [r["index"] for r in rows] == [e["index"] for e in eps] and summary["courses"] == len(cs) == len(rows)
    assert summary["success"] == sum(r["result"] == "success" for r in rows) / len(rows)
    out = []
    for e, r in zip(eps, rows):
        d = cs[e["index"]]
        assert e.get("mode") == where and not e.get("plans")
        assert (e.get("assist") is not None) == (player == "planner") and (player == "planner" or not e.get("assist"))
        assert (r["result"], r["platform"], r["ticks"], r["n"], r["level"]) == (e["result"], e["cp"], e["ticks"], e["n"], e["level"])
        assert e["success"] == (e["result"] == "success") and e["result"] in ("success", "fell", "truncated")
        assert e["n"] == len(d["platforms"]) and e["level"] == d["level"] and e["seed"] == d["seed"]
        frames = frames_of(e, d)
        assert len(frames) == e["ticks"] + 1 == len(e["actions"]) + 1
        corr = r.get("server_corrections")
        assert corr == e.get("server_corrections") and (corr is None) == (where != "live"), (d["id"], corr)
        ep = {"id": f"{d['id']}-{rec}", "course": d["id"], "run": rec, "where": where, "player": player, "set": set_name,
              "result": e["result"], "success": e["result"] == "success", "platform_reached": e["cp"], "ticks": e["ticks"],
              "duration_seconds": e["ticks"] / 20, "sim_success": bool(r["sim_success"]), "sim_ticks": int(r["sim_ticks"]),
              "replay_max_pos_err": r["replay_max_pos_err"], "replay_first_divergence": r["replay_first_divergence"],
              "server_corrections": corr,
              "wall_seconds": round(e["wall_seconds"], 2) if e.get("wall_seconds") is not None else None}
        if player == "planner":
            ep["searches"], ep["control"] = plan_control(e, d, courses[e["index"]], run_courses[e["index"]], setting["cfg"])
        ep |= {"actions": e["actions"], "frames": rounded(frames)}
        ep["_neo_done"] = neo_done(frames, d["neo_jump"]) if "neo_jump" in d else None  # for the printed counts only
        out.append(ep)
    run = {"run": rec, "where": where, "player": player, "set": set_name, "label": label, "dir": rel(run_dir),
           "mode": summary["mode"], "checkpoint": summary["checkpoint"], "wall_seconds": summary["wall_seconds"],
           "courses": summary["courses"], "success": summary["success"],
           "sim_success_same_courses": summary["sim_success_same_courses"],
           "outcome_agreement": summary["outcome_agreement"], "replay_max_pos_err": summary["replay_max_pos_err"],
           "replay_exact_episodes": summary["replay_exact_episodes"],
           "episodes_sha256": hashlib.sha256(episodes_bytes(run_dir)).hexdigest(),  # of episodes.jsonl (unpacked)
           "summary_sha256": sha256(run_dir / "summary.json")}
    if player == "planner":
        a = summary["assist"]
        log = [x for e in eps for x in e["assist"]]
        # The searches recorded per episode add up to the summary's counts.
        assert (a["requests"], a["found"], a["verified"], a["played"], a["braked"], a["planner_errors"]) == (
            len(log), sum(bool(x["found"]) for x in log), sum(bool(x["verified"]) for x in log),
            sum(bool(x["played"]) for x in log), sum(bool(x["braked"]) for x in log), sum(bool(x.get("error")) for x in log))
        assert a["planner_errors"] == 0, f"{run_dir}: the planner service failed; not a valid planner run"
        run["assist"] = {k: a[k] for k in ("requests", "found", "verified", "played", "braked", "planner_errors", "planner_seconds")}
    return run, out


def cmd_page(a) -> None:
    by_set: dict = {}
    for spec in a.sets:
        if len(spec) < 2:
            raise SystemExit("--set takes a course file and at least one run folder")
        course_file, run_dirs = Path(spec[0]), [Path(p) for p in spec[1:]]
        raw = json.loads(course_file.read_text(encoding="utf-8"))["courses"]
        name = set_of(raw)
        assert name not in by_set, f"two course files of the {name} set"
        cs = fixed_courses(name)
        assert raw == cs, f"{course_file} is not the fixed {name} set"
        by_set[name] = {"file": course_file, "courses": cs, "runs": run_dirs}
    names = [n for n in SET_ORDER if n in by_set]
    cs = [d for n in names for d in by_set[n]["courses"]]
    by_id = {d["id"]: d for d in cs}
    assert len(by_id) == len(cs), "a course id in two sets"
    prov = json.loads((ROOT / PROVENANCE).read_text(encoding="utf-8"))
    policy_sha = sha256(ROOT / POLICY)
    assert policy_sha == prov["export_sha256"], "results/combined_policy.pt is not the file its provenance names"
    all_shapes = {s["name"]: s for s in json.loads((CONFIGS / "block_shapes.json").read_text(encoding="utf-8"))["shapes"]}

    kinds = []
    if "skills" in by_set:
        kinds += list(KINDS) + type_kinds()
    if "neo" in by_set:
        check_neo_courses(by_set["neo"]["courses"])
        kinds.append(neo_kind(by_set["neo"]["courses"]))
    if "untrained" in by_set:
        check_untrained_courses(by_set["untrained"]["courses"], all_shapes)
        kinds += untrained_kinds()
    assert {d["kind"] for d in cs} == {k[0] for k in kinds}
    for t, _, _, lv, n, f, _ in kinds:  # every kind's courses: its levels, its family, seeds 0..n-1 per level
        mine = [d for d in cs if d["kind"] == t]
        assert {d["family"] for d in mine} == {f} and {d["level"] for d in mine} == set(lv), t
        if n is not None:
            assert len(mine) == len(lv) * n, t
            assert sorted((d["level"], d["seed"]) for d in mine) == [(L, i) for L in lv for i in range(n)], t
    set_kind = {t: g for t, _, g, *_ in kinds}
    kind_set = lambda t: "type" if t.startswith("shaped: ") else set_kind[t] if set_kind[t] in ("neo", "untrained") else "skill"  # noqa: E731

    from parkour.neo_assist import courses_from_file
    planner_summaries = [json.loads((r / "summary.json").read_text(encoding="utf-8")) for n in names for r in by_set[n]["runs"]]
    planner_summaries = [s for s in planner_summaries if "assist" in s]
    setting = assist_setting(planner_summaries) if planner_summaries else None
    runs, episodes, seen = [], [], set()
    sim_by_course = {}
    for name in names:
        s = by_set[name]
        course_objs = courses_from_file([bare(d) for d in s["courses"]])
        for run_dir in s["runs"]:
            run, eps = read_run(run_dir.resolve(), name, s["courses"], course_objs, setting)
            assert (name, run["run"]) not in seen, f"two {run['run']} runs of the {name} set"
            seen.add((name, run["run"]))
            assert run["player"] == "network" or name == "neo", f"the planner plays only the one-neo set ({run_dir})"
            for e in eps:  # the simulator's result for a course: the same in every recording of the same player
                sim = (e["sim_success"], e["sim_ticks"])
                assert sim_by_course.setdefault((e["course"], e["player"]), sim) == sim, \
                    f"{e['course']}: the simulator differs between the {e['player']}'s runs"
            runs.append(run)
            episodes += eps
    order = {r: i for i, r in enumerate(REC_ORDER)}
    set_order = {n: i for i, n in enumerate(SET_ORDER)}
    runs.sort(key=lambda r: (order[r["run"]], set_order[r["set"]]))
    ids = list(by_id)
    episodes.sort(key=lambda e: (order[e["run"]], ids.index(e["course"])))
    for name in names:  # every course of a set has the network's recordings in both places
        have = {r["run"] for r in runs if r["set"] == name}
        assert {"live", "offline"} <= have, f"the {name} set lacks a recording of the one network: {have}"

    used = {n for d in cs for n in (d.get("shapes") or {}).values()} | {
        "ladder_" + ("north", "east", "south", "west")[f] for d in cs for *_, f in d.get("ladders", [])}
    block_types = []
    if "skills" in by_set:
        st = shape_types()
        skill_cs = by_set["skills"]["courses"]
        assert set(st["types"]) == {n for d in skill_cs for n in (d.get("shapes") or {}).values()}, "a trained type is on no course"
        block_types += [{"name": t, "label": SHAPE_BLOCK_NAMES.get(t, (t,))[0], "block": all_shapes[t]["block"],
                         "state": all_shapes[t]["properties"], "spec_count": st["spec_counts"][t], "trained": True,
                         "shapes": [t], "top": shape_geometry(t)[0]} for t in st["types"]]
    if "untrained" in by_set:
        every = C.load_levels(advanced=True, extra=True)
        for _, t, base in untrained_table():
            shapes = shapes_of_type(t, all_shapes)
            n_specs = sum(1 for lv in every for s in lv if s.block in shapes)
            assert n_specs == 0, f"{t} is in a training level"
            state = {k: v for k, v in all_shapes[shapes[0]]["properties"].items() if k != "facing"}
            block_types.append({"name": t, "label": SHAPE_BLOCK_NAMES[t][0], "block": all_shapes[shapes[0]]["block"],
                                "state": state, "spec_count": n_specs, "trained": False, "based_on": base,
                                "shapes": [n for n in shapes if n in used] or shapes[:1],
                                "top": shape_geometry(shapes[0])[0]})
    for bt in block_types:  # the page's edge-to-edge gaps take every footprint as a square centred in its cell
        for sname in bt["shapes"]:
            _, x0, x1, z0, z1 = shape_geometry(sname)
            assert abs(x0 - z0) < 1e-12 and abs(x1 - z1) < 1e-12 and abs(x0 - (1 - x1)) < 1e-12, sname
    type_of = {sname: bt["name"] for bt in block_types for sname in bt["shapes"]}
    assert {n for d in cs for n in (d.get("shapes") or {}).values()} <= set(type_of), "a shape of no block type"

    kind_features = {t: (TYPE_FEATURES if kind_set(t) in ("type", "untrained") else NEO_FEATURES if kind_set(t) == "neo"
                         else KIND_FEATURES[t]) for t, *_ in kinds}
    set_of_course = {d["id"]: n for n in names for d in by_set[n]["courses"]}
    courses_out = []
    for d in cs:
        c = {k: d[k] for k in ("id", "kind", "level", "family", "seed", "platforms", "spawn", "spawn_yaw", "specs", "extra")}
        c |= {k: d[k] for k in ("ice", "shapes", "slime", "ladders", "based_on") if k in d}
        c["set"] = set_of_course[d["id"]]
        if "neo_jump" in d:
            c["neo"] = {"jump": d["neo_jump"], "type": d["neo_type"], "spec": d["neo_spec"]}
        c["extra_roles"], c["extra_jumps"] = extra_roles(d)
        c["legs"] = leg_kinds(d)
        c["own_jumps"] = own_jumps(d, kind_features[d["kind"]])
        if kind_set(d["kind"]) in ("type", "untrained"):  # every shaped jump of these courses is of the kind's type
            assert [f for f, *_ in c["own_jumps"]].count("shaped") == len(d.get("shapes") or {}), d["id"]
        if "neo_jump" in d:  # the one neo is the kind's own jump
            assert [o[1] for o in c["own_jumps"]] == [d["neo_jump"]], d["id"]
        courses_out.append(c)

    from parkour.neo_assist import single_neo_types
    check_network_config()
    players = {"network": {"label": "one network", "file": POLICY, "sha256": policy_sha, "provenance": PROVENANCE,
                           "run": prov["run"], "iteration": prov["iteration"], "source": prov["source"],
                           "selection": prov["selection"]}}
    if setting:
        players["planner"] = {"label": "planner", "file": PLANNER_POLICY, "sha256": sha256(ROOT / PLANNER_POLICY),
                              "export": PLANNER_EXPORT, **planner_policy(),
                              "assist": setting["config"], "assist_json": PLANNER_SETTING, "chosen": setting["chosen"],
                              "alone": setting["alone"]}
    sets_out = []
    for name in names:
        s = by_set[name]
        # jumps: what the generators ask for (n_jumps 10 in all three builders); max_platforms: where generate stops
        row = {"id": name, "file": rel(s["file"]), "sha256": sha256(s["file"]), "courses": len(s["courses"]),
               "jumps": N_JUMPS, "max_platforms": C.MAX_PLATFORMS}
        assert all(len(d["specs"]) <= N_JUMPS + sum(lk != "drawn" for lk in leg_kinds(d)) for d in s["courses"]), name
        if name == "skills":
            row |= {"families": [FAMILY, ICE_FAMILY, SHAPE_FAMILY],
                    "first_courses": 45, "builder": "scripts/make_skill_replays.py",
                    "fixed": ("the first 45 courses on 29 October 2026 before any outcome was seen; the 20 one-type courses "
                              "the same day at the user's request, after the first 45 had been played in offline physics "
                              "and the first 14 in real Minecraft, and before any of the 20 was played")}
        elif name == "neo":
            row |= {"families": [NEO_FAMILY], "builder": "scripts/make_neo_showcase_courses.py",
                    "single_neo_types": len(single_neo_types()),
                    "fixed": ("on 29 October 2026, after the user saw that no replay shows a neo done, before any outcome "
                              "on these courses was seen")}
        else:
            row |= {"families": [SHAPE_FAMILY], "k": [k for k, *_ in untrained_table()],
                    "builder": "scripts/make_untrained_shape_courses.py",
                    "fixed": "on 29 October 2026, when the user asked for stairs, before any outcome on these courses was seen"}
        sets_out.append(row)
    recordings = [{"id": r, "where": r.split("-")[-1], "player": "planner" if r.startswith("planner-") else "network"}
                  for r in REC_ORDER if any(x["run"] == r for x in runs)]
    payload = {
        "schema": "parkour-skill-replays-v2",
        "tick_hz": 20,
        "frame_columns": FRAME_COLUMNS,
        "action_columns": ["move", "strafe", "jump", "sprint", "turn_index"],
        "yaw_steps_degrees": list(YAW_STEPS_DEG),
        "players": players,
        "recordings": recordings,
        "course_sets": sets_out,
        "features": {t: {"one": one, "many": many, "about": about} for t, (one, many, about) in FEATURES.items()},
        "kinds": [{"token": t, "label": lab, "group": g, "levels": list(lv), "seeds": n, "family": f, "about": about,
                   "features": list(kind_features[t]), "set": kind_set(t),
                   **({"block_type": t.split(": ", 1)[1]} if kind_set(t) in ("type", "untrained") else {})}
                  for t, lab, g, lv, n, f, about in kinds],
        "block_types": block_types,
        "shaped_level": {"global": SHAPED_LEVEL, "local_in_extra_file": shaped_local_level() - SKILL_GROUPS["extra"][1],
                         "specs": shape_types()["specs"]},
        "shapes": {k: {"block": all_shapes[k]["block"], "boxes": all_shapes[k]["boxes"]} for k in sorted(used)},
        "runs": runs,
        "courses": courses_out,
        "episodes": [{k: v for k, v in e.items() if not k.startswith("_")} for e in episodes],
    }
    for e in episodes:
        assert len(e["frames"]) == e["ticks"] + 1 == len(e["actions"]) + 1, e["id"]
    runs_of = {n: [r["run"] for r in runs if r["set"] == n] for n in names}
    assert len(episodes) == sum(len(by_set[n]["courses"]) * len(runs_of[n]) for n in names)
    assert len({e["id"] for e in episodes}) == len(episodes)
    assert {(e["course"], e["run"]) for e in episodes} == {(d["id"], r) for n in names for d in by_set[n]["courses"] for r in runs_of[n]}
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    assert "</" not in data and "<!--" not in data  # safe inside the page's <script> element
    template = TEMPLATE.read_text(encoding="utf-8")
    assert template.count("__REPLAY_DATA__") == 1
    html = template.replace("__REPLAY_DATA__", data)
    assert len(html.encode("utf-8")) < 16_000_000, "the page must stay under 16 MB"
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:  # the same bytes on every platform
        f.write(html)

    # Per kind and player, per block type, per recording: real Minecraft, offline and simulator completions
    # (summary.json, as embedded), and the one-neo courses' neos done (from the frames).
    made = {(e["course"], e["run"]): e["success"] for e in episodes}
    sim = {(e["course"], e["player"]): e["sim_success"] for e in episodes}
    recs_of = {d["id"]: [e["run"] for e in episodes if e["course"] == d["id"]] for d in cs}

    def count(ids, player="network"):
        rs = [r for r in REC_ORDER if r.startswith("planner-") == (player == "planner")]
        ids = [i for i in ids if any(r in recs_of[i] for r in rs)]
        return {"courses": len(ids), **{r: sum(made[i, r] for i in ids) for r in rs if any(r in recs_of[i] for i in ids)},
                "sim": sum(sim[i, player] for i in ids)}
    per_kind = {}
    for t, *_ in kinds:
        ids_t = [d["id"] for d in cs if d["kind"] == t]
        per_kind[t] = count(ids_t) if kind_set(t) != "neo" else {"one network": count(ids_t), "planner": count(ids_t, "planner")}
    per_type = {}
    for bt in block_types:
        ids_t = [d["id"] for d in cs if set((d.get("shapes") or {}).values()) & set(bt["shapes"])]
        per_type[bt["name"]] = {"trained": bt["trained"], "spec_count": bt["spec_count"],
                                "platforms": sum(1 for d in cs for n in (d.get("shapes") or {}).values() if n in bt["shapes"]),
                                **count(ids_t),
                                "own_courses": count([d["id"] for d in cs if d["kind"] in (f"shaped: {bt['name']}", f"untrained: {bt['name']}")])}
    per_recording = {r["run"] + " / " + r["set"]: {"dir": r["dir"], "courses": r["courses"],
                                                   "completed": sum(made[e["course"], r["run"]] for e in episodes
                                                                    if e["run"] == r["run"] and e["set"] == r["set"]),
                                                   "summary_success": r["success"], "sim": round(r["sim_success_same_courses"] * r["courses"])}
                     for r in runs}
    neo_eps = [e for e in episodes if e["_neo_done"] is not None]

    def by_a_plan(e: dict) -> bool:
        """A planner's neo done by a played plan: the plan of a search on the neo's takeoff ends on the tick the bot
        first stands on its target (the platform reached first takes the target's value)."""
        k = by_id[e["course"]]["neo_jump"]
        land = next(i for i, f in enumerate(e["frames"]) if f[9] == k + 1)
        return any(kind == "plan" and e["searches"][n]["takeoff"] == k and end == land for _, end, kind, n in e["control"])
    neos = {r: {"courses": len(es), "done": sum(e["_neo_done"] for e in es), "completed": sum(e["success"] for e in es),
                "completed_neo_done": sum(e["success"] and e["_neo_done"] for e in es),
                **({"done_by_a_plan": sum(e["_neo_done"] and by_a_plan(e) for e in es)} if r.startswith("planner-") else {})}
            for r in REC_ORDER if (es := [e for e in neo_eps if e["run"] == r])}
    plans = {}
    for e in episodes:
        if e["player"] != "planner":
            continue
        p = plans.setdefault(e["run"], {"searches": 0, "plans_played": 0, "brakes": 0, "plan_ticks": 0, "brake_ticks": 0, "ticks": 0})
        p["searches"] += len(e["searches"])
        p["plans_played"] += sum(c[2] == "plan" for c in e["control"])
        p["brakes"] += sum(c[2] == "brake" for c in e["control"])
        p["plan_ticks"] += sum(c[1] - c[0] for c in e["control"] if c[2] == "plan")
        p["brake_ticks"] += sum(c[1] - c[0] for c in e["control"] if c[2] == "brake")
        p["ticks"] += e["ticks"]
    for p in plans.values():
        p["policy_ticks"] = p["ticks"] - p["plan_ticks"] - p["brake_ticks"]
    reach = {(e["course"], e["run"]): e["platform_reached"] for e in episodes}
    own = {}
    for t, *_ in kinds:
        for f in kind_features[t]:
            js = [(c["id"], j, th) for c in courses_out if c["kind"] == t for g, j, th in c["own_jumps"] if g == f]
            rs = sorted({r for i, *_ in js for r in recs_of[i]}, key=REC_ORDER.index)
            own[f"{t} / {f}"] = {"on_courses": len(js), "courses_with_one": len({i for i, *_ in js}),
                                 **{r: {"reached": sum(reach[i, r] >= j for i, j, _ in js),
                                        "passed": sum(reach[i, r] >= th + 1 for i, _, th in js)} for r in rs}}
    legs = Counter(k for c in courses_out for k in c["legs"])
    drawn = [c["legs"].count("drawn") for c in courses_out]
    corr = {(e["course"], e["run"]): e["server_corrections"] for e in episodes if e["where"] == "live"}
    same_actions = {}
    for pl in ("network", "planner"):
        pairs = [(e, next((x for x in episodes if x["course"] == e["course"] and x["player"] == pl and x["where"] == "offline"), None))
                 for e in episodes if e["player"] == pl and e["where"] == "live"]
        if pairs:
            same_actions[pl] = f"{sum(o is not None and o['actions'] == e['actions'] for e, o in pairs)} of {len(pairs)} courses"
    print(json.dumps({"out": str(a.out), "bytes": a.out.stat().st_size, "episodes": len(episodes),
                      "recordings": [r["id"] for r in recordings], "sets": {n: len(by_set[n]["courses"]) for n in names},
                      "frames": sum(len(e["frames"]) for e in episodes),
                      "jumps": {"legs": dict(legs), "drawn_per_course": [min(drawn), max(drawn)],
                                "legs_per_course": [min(len(c["legs"]) for c in courses_out), max(len(c["legs"]) for c in courses_out)],
                                "courses_with_fewer_than_n_jumps_drawn": sum(n < N_JUMPS for n in drawn),
                                "courses_at_max_platforms": sum(len(c["platforms"]) == C.MAX_PLATFORMS for c in courses_out)},
                      "own_jumps": own,
                      "grounded_frames_past_an_edge": past_edge(courses_out, episodes, payload["shapes"]),
                      "all_one_network": count(ids), "per_recording": per_recording,
                      "per_kind": per_kind, "per_type": per_type, "neos_done": neos, "planner_ticks": plans,
                      "same_actions_live_and_offline": same_actions,
                      "planner_setting": {k: v for k, v in (setting or {}).items() if k != "cfg"} or None,
                      "planner_policy": {k: players["planner"][k] for k in ("file", "sha256", "iteration", "source", "source_note")}
                      if setting else None,
                      "disagreements": [f"{i} / {p}" for i in ids for p in ("network", "planner") if (i, p) in sim
                                        and len({made[i, r] for r in recs_of[i] if r.startswith("planner-") == (p == "planner")} | {sim[i, p]}) > 1],
                      "server_corrections": {"total": sum(corr.values()), "nonzero": {f"{k[0]} / {k[1]}": v for k, v in corr.items() if v}} if corr else None,
                      "replay_max_pos_err": {f"{r['run']} / {r['set']}": r["replay_max_pos_err"] for r in runs},
                      "replay_exact_episodes": {f"{r['run']} / {r['set']}": r["replay_exact_episodes"] for r in runs}}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("courses", help="write the fixed course set for parkour.live --courses-file")
    c.add_argument("out", type=Path)
    p = sub.add_parser("page", help="build the replay page from course files and parkour.live run folders")
    p.add_argument("out", type=Path)
    p.add_argument("--set", dest="sets", action="append", nargs="+", required=True, metavar="PATH",
                   help="a course file, then the run folders that played it (repeat for each set)")
    a = ap.parse_args()
    cmd_courses(a) if a.cmd == "courses" else cmd_page(a)


if __name__ == "__main__":
    main()

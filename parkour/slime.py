"""Slime-heavy and post-heavy course sets (opt-in; HANDOFF.md section 3, 1 Nov).

The final policy lands 39.9% of the 41 single slime-bounce kinds and never 22 of them (``parkour.jumpeval --extra``,
``results/extra/jump_eval_final_policy.json``): an exit one block above the pad it makes, two or three it does
not. Its observation does not tell a slime pad from stone. These sets put many slime bounces, or many narrow
posts (fence and wall posts, iron bars, end rods) mixed with slime, in one course:

  slime  ``slime_courses``: 8 jump specs, 4 of them slime bounces (a pad and its exit: two jumps each), so 12
         jumps and 13 platforms; course i's bounces are the kinds (4i + j) mod 41, at the spec positions
         ``SLIME_PATTERNS[i mod 8]``; the other jumps of levels 0-5. Families 55 (validation) and 56 (test).
  posts  ``posts_courses``: 8 jump specs, 4 of them a narrow post (a post and its exit: two jumps each; the
         post kinds of the shaped-block level and the iron-bars kinds, ``post_kinds``) and 2 slime bounces, so
         14 jumps and 15 platforms; families 57 (validation) and 58 (test).

Each set is drawn like the single-neo sets: one random stream ``[2026, family, 6]`` in course order (the first
courses of a longer set are the same courses), a course drawn again until its special jumps are where the plan
puts them (``fixed_ok``).
"""
from __future__ import annotations

import json
import math

import numpy as np

from . import courses as C
from .paths import CONFIGS

SLIME_VALIDATION_FAMILY = 55
SLIME_TEST_FAMILY = 56
POSTS_VALIDATION_FAMILY = 57
POSTS_TEST_FAMILY = 58
SLIME_TRAIN_FAMILIES = tuple(range(84, 90))  # training courses of these sets' generator; nothing else uses them
SLIME_PATTERNS = ((1, 3, 5, 7), (0, 2, 4, 6), (0, 1, 4, 5), (2, 3, 6, 7), (0, 1, 2, 3), (4, 5, 6, 7), (1, 2, 5, 6),
                  (0, 3, 4, 7))  # spec positions of the 4 special jumps: apart, in pairs, four in a row
POSTS_PATTERNS = ((0, 2, 4, 6), (1, 3, 5, 7), (0, 1, 4, 5), (2, 3, 6, 7))  # the 4 posts; slime at 2 of the others
MAX_DRAWS = 10_000


def _extra_levels() -> dict:
    return {L["name"]: [C._spec_from(d) for d in L["specs"]]
            for L in json.loads((CONFIGS / "jump_levels_extra.json").read_text())["levels"]}


def slime_kinds() -> list[C.JumpSpec]:
    """The 41 slime-bounce kinds of the extra levels, in their order."""
    return list(_extra_levels()["slime bounce"])


def post_kinds() -> list[C.JumpSpec]:
    """The narrow posts of the shaped-block level (fence posts, wall posts, end rods), then the iron-bars kinds
    (``configs/jump_levels_posts.json``, ``parkour.feasibility --posts``) if that file exists."""
    out = [s for s in _extra_levels()["shaped blocks"] if s.block in ("fence_post", "wall_post", "end_rod")]
    path = CONFIGS / "jump_levels_posts.json"
    if path.exists():
        out += [C._spec_from(d) for L in json.loads(path.read_text())["levels"] for d in L["specs"]]
    return out


def generate_fixed(rng: np.random.Generator, basic: list[list[C.JumpSpec]], fixed: dict, n_specs: int = 8,
                   grid=C.GRID, turn_prob: float = 0.25, yaw_noise: float = math.radians(60), level: int = 0,
                   seed: int = 0) -> C.Course | None:
    """A course of ``n_specs`` jump specs whose spec at each position of ``fixed`` is that special spec (slime
    bounce, shaped block, neo, ...) and all others ordinary jumps of ``basic`` (``courses.generate_neos``'s
    draws). Special specs are placed by ``courses._place`` and recorded as ``courses.generate`` records them
    (slime pads, shaped platforms, their exits). None if a spec does not fit in 30 attempts."""
    assert fixed and min(fixed) >= 0 and max(fixed) < n_specs
    gx, gy, gz = grid
    s = 1 if any(fixed[j].neo for j in fixed if j == 0) else int(rng.choice([1, 2, 3], p=[0.4, 0.3, 0.3]))
    heading = int(rng.integers(4))
    hx, hz = C.HEADINGS[heading]
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
    specs, extras, shaped, pads, ladders = [], [], {}, [], []
    for j in range(n_specs):
        placed = False
        for attempt in range(30):
            if j in fixed:
                spec = fixed[j]
            else:
                lv = int(rng.integers(len(basic)))
                spec = basic[lv][int(rng.integers(len(basic[lv])))]
            h = heading
            if rng.random() < turn_prob or attempt >= 15:
                h = (heading + int(rng.choice([1, 3]))) % 4
            if j in fixed:
                if len(platforms) + 1 + (1 if spec.block or spec.bounce else 0) > C.MAX_PLATFORMS:
                    return None
                parts = C._place(platforms, extras, h, spec, grid)
                if parts is None:
                    continue
                if spec.ladder:
                    ladders.extend(parts[2])
                if spec.block:
                    shaped[len(platforms)] = spec.block
                if spec.bounce:
                    pads.append(len(platforms))
                platforms.extend(parts[0])
                extras.extend(parts[1])
                specs.append(spec)
                if spec.block:
                    specs.append(C.exit_spec(spec.block))
                if spec.bounce:
                    specs.append(C.bounce_exit(spec))
            else:
                new = C._platform_after(platforms[-1], h, spec)
                if not C._fits(new, platforms, grid, extras=extras):
                    continue
                platforms.append(new)
                specs.append(spec)
            heading = h
            placed = True
            break
        if not placed:
            return None
    c = C.Course(np.stack(platforms), np.zeros(3), 0.0, level=level, seed=seed, specs=specs,
                 extra=np.array(extras, dtype=np.int64).reshape(-1, 3), shapes=shaped, slime=pads,
                 ladders=np.array(ladders, dtype=np.int64).reshape(-1, 4))
    centres = c.centers()
    c.spawn = centres[0].copy()
    d = centres[1] - centres[0]
    c.spawn_yaw = float(math.atan2(-d[0], -d[2]) + rng.uniform(-yaw_noise, yaw_noise))
    return c


def fixed_ok(course: C.Course, fixed: dict) -> bool:
    """The special specs where ``fixed`` puts them (each followed by its exit), slime pads and shaped platforms
    exactly theirs, and no other special jump."""
    want, k = [], 0
    for j in range(max(fixed) + 1 if fixed else 0):
        if j in fixed:
            want.append((k, fixed[j]))
            k += 2 if (fixed[j].bounce or fixed[j].block) else 1
        else:
            k += 1
    pads = [k + 1 for k, sp in want if sp.bounce]  # (spec k is the jump from platform k onto platform k + 1)
    shaped = {k + 1: sp.block for k, sp in want if sp.block}
    specials = [k for k, sp in enumerate(course.specs) if sp.bounce or sp.block or sp.neo or sp.ladder or sp.runway]
    return (all(k < len(course.specs) and course.specs[k] == sp for k, sp in want) and sorted(course.slime) == pads
            and course.shapes == shaped and specials == [k for k, _ in want])


def _course_set(family: int, n: int, plan) -> list[C.Course]:
    basic = C.load_levels()
    rng = np.random.default_rng([2026, family, 6])
    out = []
    for i in range(n):
        fixed = plan(i)
        for _ in range(MAX_DRAWS):
            c = generate_fixed(rng, basic, fixed, 8, level=6, seed=i)
            if c is not None and fixed_ok(c, fixed):
                break
        else:
            raise RuntimeError(f"no course {i} of family {family} in {MAX_DRAWS} draws")
        out.append(c)
    return out


def slime_plan(i: int) -> dict:
    kinds = slime_kinds()
    return {p: kinds[(4 * i + j) % len(kinds)] for j, p in enumerate(SLIME_PATTERNS[i % len(SLIME_PATTERNS)])}


def posts_plan(i: int) -> dict:
    posts, kinds = post_kinds(), slime_kinds()
    where = POSTS_PATTERNS[i % len(POSTS_PATTERNS)]
    rest = [p for p in range(8) if p not in where]
    slime_at = (rest[i % 4], rest[(i + 2) % 4])
    out = {p: posts[(4 * i + j) % len(posts)] for j, p in enumerate(where)}
    out.update({p: kinds[(2 * i + j) % len(kinds)] for j, p in enumerate(sorted(slime_at))})
    return out


def slime_courses(family: int, n: int = 320) -> list[C.Course]:
    return _course_set(family, n, slime_plan)


def posts_courses(family: int, n: int = 320) -> list[C.Course]:
    return _course_set(family, n, posts_plan)


def held_slime_hashes() -> set:
    """``train.course_hash`` of every course of the slime and posts sets: no training course may equal one."""
    from .train import course_hash
    return ({course_hash(c) for f in (SLIME_VALIDATION_FAMILY, SLIME_TEST_FAMILY) for c in slime_courses(f)}
            | {course_hash(c) for f in (POSTS_VALIDATION_FAMILY, POSTS_TEST_FAMILY) for c in posts_courses(f)})


def slime_cps(course: C.Course) -> list[int]:
    """The checkpoints at which the slime skill plays (the course-list mode, as ``neo_assist.skill_cps`` for
    neos): for each slime pad s, s-1 and s, so it plays from the first ground contact on the platform before the
    pad until the bot first stands on the pad's exit."""
    return sorted({k for s in course.slime for k in (s - 1, s) if k >= 0})


def slime_blocks(course: C.Course) -> list[tuple[int, int]]:
    """(k, first) of each slime pad's block for a bank: the takeoff onto the pad k = s-1 (the segment rule's
    takeoff: the bot must stand on the pad, k+1, before the exit) and the platform the skill starts on (k)."""
    return [(s - 1, s - 1) for s in sorted(course.slime) if s >= 1]


def special_cps(course: C.Course) -> list[int]:
    """``slime_cps`` with the narrow posts too: for each slime pad and each shaped platform s, s-1 and s."""
    return sorted({k for s in list(course.slime) + list(course.shapes) for k in (s - 1, s) if k >= 0})


def special_blocks(course: C.Course) -> list[tuple[int, int]]:
    """``slime_blocks`` with the shaped platforms too (a post's block: onto it from k = s-1, then its exit)."""
    return [(s - 1, s - 1) for s in sorted(set(course.slime) | set(course.shapes)) if s >= 1]


POSTS_TRAIN_FAMILIES = tuple(range(90, 96))  # training courses with one post; nothing else uses them


def slime_training_courses(families=SLIME_TRAIN_FAMILIES, per_family: int = 4100,
                           posts: bool = False) -> tuple[list[C.Course], list[int]]:
    """Courses of 8 jump specs with one slime bounce (course i: kind i mod 41 at spec position (i div 41) mod 8, the
    other jumps of levels 0-5; the sets' generator), none equal to a course of the slime or posts sets; each
    course's level set to its kind (0-40). With ``posts``, one narrow post instead (``post_kinds``, types 41 on:
    a bank of both can be mixed, ``neo_skill mix``)."""
    held = held_slime_hashes()
    from .train import course_hash
    kinds = post_kinds() if posts else slime_kinds()
    offset = len(slime_kinds()) if posts else 0
    basic = C.load_levels()
    out, types = [], []
    for f in families:
        if f in (SLIME_VALIDATION_FAMILY, SLIME_TEST_FAMILY, POSTS_VALIDATION_FAMILY, POSTS_TEST_FAMILY):
            raise ValueError(f"family {f} is a held-out set")
        rng = np.random.default_rng([2026, f, 6])
        for i in range(per_family):
            t = i % len(kinds)
            fixed = {(i // len(kinds)) % 8: kinds[t]}
            for _ in range(MAX_DRAWS):
                c = generate_fixed(rng, basic, fixed, 8, level=offset + t, seed=i)
                if c is not None and fixed_ok(c, fixed):
                    break
            else:
                continue
            if course_hash(c) in held:
                continue
            out.append(c)
            types.append(offset + t)
    return out, types

"""The slime-heavy and post-heavy course sets (parkour.slime)."""
import collections

from parkour import courses as C
from parkour import slime as S


def test_slime_sets_put_the_bounces_where_the_plan_does():
    kinds = S.slime_kinds()
    assert len(kinds) == 41 and all(k.bounce for k in kinds)
    cs = S.slime_courses(S.SLIME_VALIDATION_FAMILY, 80)
    for i, c in enumerate(cs):
        fixed = S.slime_plan(i)
        assert S.fixed_ok(c, fixed) and c.n == 13 and len(c.slime) == 4
        assert sorted(fixed) == list(S.SLIME_PATTERNS[i % 8])
        for p in c.slime:  # each pad: a drop onto it, then the exit two blocks on and `bounce` rows up
            sp = c.specs[p - 1]
            assert sp.bounce and c.specs[p] == C.bounce_exit(sp) and c.platforms[p + 1][4] - c.platforms[p][4] == sp.bounce
    used = collections.Counter(k.key() for i in range(320) for k in S.slime_plan(i).values())
    assert len(used) == 41 and max(used.values()) - min(used.values()) <= 1  # 1,280 bounces over 41 kinds
    assert [c.to_json() for c in S.slime_courses(S.SLIME_VALIDATION_FAMILY, 20)] == [c.to_json() for c in cs[:20]]


def test_posts_sets_mix_posts_and_slime():
    cs = S.posts_courses(S.POSTS_VALIDATION_FAMILY, 40)
    for i, c in enumerate(cs):
        fixed = S.posts_plan(i)
        assert S.fixed_ok(c, fixed) and c.n == 15 and len(c.shapes) == 4 and len(c.slime) == 2
        assert all(C.shape_geometry(name)[2] - C.shape_geometry(name)[1] <= 0.5 for name in c.shapes.values())


def test_fixed_ok_refuses_a_course_without_its_bounce():
    c = S.slime_courses(S.SLIME_VALIDATION_FAMILY, 1)[0]
    fixed = S.slime_plan(0)
    other = dict(fixed)
    other.pop(min(other))
    assert S.fixed_ok(c, fixed) and not S.fixed_ok(c, other)


def test_sets_are_registered_and_kept_out_of_training():
    from parkour import neo_assist as NA
    from parkour.train import course_hash
    assert (NA.SLIME_TEST_FAMILY, NA.POSTS_TEST_FAMILY) == (S.SLIME_TEST_FAMILY, S.POSTS_TEST_FAMILY)
    assert {S.SLIME_TEST_FAMILY, S.POSTS_TEST_FAMILY} <= set(NA.TEST_FAMILIES)
    assert not {S.SLIME_VALIDATION_FAMILY, S.POSTS_VALIDATION_FAMILY} & set(NA.TEST_FAMILIES)
    held = S.held_slime_hashes()
    assert len(held) == 1280
    assert held == {course_hash(c) for f in (55, 56) for c in NA.course_set("slime", f)} | {
        course_hash(c) for f in (57, 58) for c in NA.course_set("posts", f)}
    cs, types = S.slime_training_courses(families=(84,), per_family=82)
    assert len(cs) == 82 and not {course_hash(c) for c in cs} & held and sorted(set(types)) == list(range(41))
    for f in (55, 58):
        try:
            S.slime_training_courses(families=(f,), per_family=1)
            raise AssertionError("a held-out family")
        except ValueError:
            pass
    for path in ("train.py", "gate.py"):  # extra levels in training: none of these sets
        from parkour.paths import ROOT
        assert "held_slime_hashes()" in (ROOT / "parkour" / path).read_text(encoding="utf-8")


def test_slime_blocks_and_checkpoints():
    c = S.slime_courses(S.SLIME_VALIDATION_FAMILY, 1)[0]
    assert S.slime_cps(c) == sorted({k for s in c.slime for k in (s - 1, s)})
    assert S.slime_blocks(c) == [(s - 1, s - 1) for s in sorted(c.slime)]


def test_generate_fixed_refuses_run_ups_and_heading_of_takes_sideways_jumps():
    import numpy as np
    import pytest
    run_up = C.JumpSpec(2, 0, 0, (1, 1), runway=3)
    with pytest.raises(ValueError):
        S.generate_fixed(np.random.default_rng(0), C.load_levels(), {1: run_up}, 8)
    prev = np.array([10, 10, 30, 30, 12])
    for h in range(4):
        for lat in (-2, -1, 0, 1, 2):
            new = C._platform_after(prev, h, C.JumpSpec(2, 0, lat, (1, 1)))
            assert C.heading_of(prev, new) == h, (h, lat)

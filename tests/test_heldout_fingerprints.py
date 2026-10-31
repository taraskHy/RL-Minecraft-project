"""The held-out course sets never change.

Every reported number was measured on these courses, so a change to course
generation, the level files or a default that moves even one test or
validation course would silently break the comparison with earlier results.
New features must be opt-in (new levels after level 8, new flags off by
default). The digests are of the full course JSON (platforms, spawn,
obstacle cells, ice) of 200 courses per level with 10 jumps, the sizes the
evaluations use (validation at 100 per level is a prefix of these, since
each level draws its courses in order from one generator). They were
computed at commit 6170803, where the final policy was trained, and checked
unchanged at af25aeb.
"""
import hashlib
import json
import unittest

from parkour import courses as C
from parkour.train import (ICE_TEST_FAMILY, ICE_VALIDATION_FAMILY, OBSTACLE_TEST_FAMILY, OBSTACLE_VALIDATION_FAMILY,
                           SKILL_ICE_PROB, TEST_SEED_FAMILY, VALIDATION_FAMILY, held_out_courses, skill_sets)

SETS = {  # (advanced levels, seed family, obstacle chance) -> sha1 prefix
    (False, TEST_SEED_FAMILY, 0.0): "4fae789a0e19dc25",
    (False, VALIDATION_FAMILY, 0.0): "ca8dee973e4fbb2e",
    (False, OBSTACLE_TEST_FAMILY, 0.5): "d870df6c9b084de7",
    (False, OBSTACLE_VALIDATION_FAMILY, 0.5): "770869517eb53280",
    (True, TEST_SEED_FAMILY, 0.0): "7a5218140b2b455e",
    (True, VALIDATION_FAMILY, 0.0): "c2610c4e0b3ab378",
    (True, OBSTACLE_TEST_FAMILY, 0.5): "c10985e50e99cdb8",
    (True, OBSTACLE_VALIDATION_FAMILY, 0.5): "4ea0b3f479b3927c",
}


ASSIST_SETS = {  # (parkour.neo_assist --set, family) -> sha1 prefix
    ("official", TEST_SEED_FAMILY): "fc844f4979844e61",
    ("official", VALIDATION_FAMILY): "3c74d2621dc577a8",
    ("random", 12): "27d7547e0119a378",
    ("random", 13): "1e5e4aef0e5b9034",
    ("mixed", 14): "3bcdee9264a4d5b5",  # these two depend on configs/jump_levels_extra.json (computed on d5943ca)
    ("mixed", 15): "607d03c215c1921c",
    # The single-neo sets (neo_assist.single_neo_courses, 320 courses each; computed on 4762c22 + the sets). With
    # every course's level set to 6 they are the reference prototype's 262f50a8c817f3f6 and 2aa66562a8eeff99.
    ("single", 40): "0f92d172cb152161",
    ("single", 41): "6f47778f3c3d0f71",
    ("single", 49): "d694d370d7c55ca1",  # the fresh single-neo test set (the neo skill's; saved 31 Oct 09:30, before any run)
    ("single", 50): "e129393d068b5b9c",  # a second fresh one, for the neo skill's later version (31 Oct 11:44, before any run)
    ("single", 51): "369070302956ceb2",  # a third fresh one, for the third DAgger round (31 Oct 13:50, before any run)
}
EXTRA_PREVIEW_SETS = {  # scripts/extra_preview.py: levels 0-5 plus one extra level, 200 courses, (level, family) -> sha1
    ("head hitter", 1): "507e74e214ef759c", ("head hitter", 2): "f3030cea3b9a0c02",
    ("hard head hitter", 1): "9babc823bff87d6b", ("hard head hitter", 2): "277046ee621634fd",
    ("ice run-up", 1): "8b08ab848f7616b2", ("ice run-up", 2): "510e18d892fe37d6",
    ("shaped blocks", 1): "1e757e579f6a838d", ("shaped blocks", 2): "0f71db0f924a2fe2",
    ("slime bounce", 1): "1e48cedfd73fd1c4", ("slime bounce", 2): "887bd5beee576ccf",
    ("ladder climb", 1): "d277b61e223a9a5c", ("ladder climb", 2): "2ed471dc5890a76d",
}
# Skills mode (docs/combine_policies.md; computed on 52be9e3 with the skills mode, 29 Oct 2026). The ice sets:
# basic levels, ice on 30% of the platforms after the start, 200 courses per level.
ICE_SETS = {ICE_TEST_FAMILY: "d024be3f3f6a351a", ICE_VALIDATION_FAMILY: "d209160a85274b57"}
# skill_sets(split): each group's courses with global level ids (test 200 per level, validation 100). Basic is
# the standard sets above; advanced and extra are the advanced and extra tables' sets at their own levels.
SKILL_SETS = {
    ("test", "basic", "standard"): "4fae789a0e19dc25", ("test", "basic", "obstacles"): "d870df6c9b084de7",
    ("test", "advanced", "standard"): "9f8ca4e046d391d9", ("test", "advanced", "obstacles"): "f46b7c54197b6714",
    ("test", "extra", "standard"): "598e6886c4e5bb81", ("test", "extra", "obstacles"): "2221abfda38a783b",
    ("test", "ice", "standard"): "1767df53b4692559",
    ("validation", "basic", "standard"): "cc637c6f1405babc", ("validation", "basic", "obstacles"): "fa59eab1e05fe468",
    ("validation", "advanced", "standard"): "250fcc06d77baac6", ("validation", "advanced", "obstacles"): "295b9db69a798ba4",
    ("validation", "extra", "standard"): "ce4fdcbb41ff1517", ("validation", "extra", "obstacles"): "dcabd242926f1ea9",
    ("validation", "ice", "standard"): "839d65fa1a56a27b",
}


def fingerprint(courses: list[C.Course]) -> str:
    return hashlib.sha1(json.dumps([c.to_json() for c in courses], sort_keys=True).encode()).hexdigest()[:16]


class HeldOutFingerprints(unittest.TestCase):
    def test_held_out_sets_are_unchanged(self):
        levels = {adv: C.load_levels(advanced=adv) for adv in (False, True)}
        for (adv, family, obstacles), digest in SETS.items():
            with self.subTest(advanced=adv, family=family):
                cs = held_out_courses(levels[adv], 200, 10, obstacle_prob=obstacles, family=family)
                self.assertEqual(fingerprint(cs), digest)

    def test_assist_course_sets_are_unchanged(self):
        # parkour.neo_assist's sets: the held-out neo courses (official) and the random neo courses; digests
        # computed on top of a22d5ea, after checking that the ladder-wall rule of neo_takeoffs left them unchanged.
        from parkour.neo_assist import course_set
        for (name, family), digest in ASSIST_SETS.items():
            with self.subTest(set=name, family=family):
                self.assertEqual(fingerprint(course_set(name, family)), digest)

    def test_extra_preview_sets_are_unchanged(self):
        from parkour.paths import CONFIGS
        basic = C.load_levels()
        levels = {L["name"]: [C._spec_from(d) for d in L["specs"]]
                  for L in json.loads((CONFIGS / "jump_levels_extra.json").read_text())["levels"]}
        for (name, family), digest in EXTRA_PREVIEW_SETS.items():
            with self.subTest(level=name, family=family):
                cs = held_out_courses(basic + [levels[name]], 200, 10, family=family)[6 * 200:]
                self.assertEqual(fingerprint(cs), digest)

    def test_ice_sets_are_unchanged(self):
        for family, digest in ICE_SETS.items():
            with self.subTest(family=family):
                cs = held_out_courses(C.load_levels(), 200, 10, family=family, ice_prob=SKILL_ICE_PROB)
                self.assertEqual(fingerprint(cs), digest)

    def test_skill_sets_are_unchanged(self):
        sets = {split: skill_sets(split, per_level, 10) for split, per_level in (("test", 200), ("validation", 100))}
        for (split, group, part), digest in SKILL_SETS.items():
            with self.subTest(split=split, group=group, part=part):
                self.assertEqual(fingerprint(sets[split][group][part]), digest)


if __name__ == "__main__":
    unittest.main()

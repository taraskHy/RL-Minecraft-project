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
from parkour.train import (OBSTACLE_TEST_FAMILY, OBSTACLE_VALIDATION_FAMILY, TEST_SEED_FAMILY, VALIDATION_FAMILY,
                           held_out_courses)

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


if __name__ == "__main__":
    unittest.main()

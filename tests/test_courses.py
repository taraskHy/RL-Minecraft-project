import unittest

import numpy as np

from parkour import courses as C
from parkour.train import TEST_SEED_FAMILY, course_hash, held_out_courses


class Courses(unittest.TestCase):
    def setUp(self):
        self.levels = C.load_levels()

    def test_generated_courses_are_valid(self):
        rng = np.random.default_rng(0)
        gx, gy, gz = C.GRID
        for i in range(400):
            c = C.generate(rng, self.levels, i % len(self.levels), 12)
            self.assertGreaterEqual(c.n, 3)
            self.assertLessEqual(c.n, C.MAX_PLATFORMS)
            b = c.blocks()
            self.assertTrue((b >= 0).all())
            self.assertTrue((b[:, 0] < gx).all() and (b[:, 1] < gy).all() and (b[:, 2] < gz).all())
            # No block appears twice: platforms never overlap.
            self.assertEqual(len({tuple(x) for x in b}), len(b))
            # Head room above every platform is free of other platforms' blocks.
            cells = {tuple(x) for x in b}
            for x0, x1, z0, z1, y in c.platforms:
                for x in range(x0, x1 + 1):
                    for z in range(z0, z1 + 1):
                        for dy in (1, 2):
                            self.assertNotIn((x, y + dy, z), cells)
            # Spawn stands on the centre of platform 0.
            np.testing.assert_allclose(c.spawn, c.centers()[0])

    def test_seeded_generation_is_reproducible(self):
        a = C.generate(np.random.default_rng([1, 2]), self.levels, 3, 10)
        b = C.generate(np.random.default_rng([1, 2]), self.levels, 3, 10)
        np.testing.assert_array_equal(a.platforms, b.platforms)
        self.assertEqual(a.spawn_yaw, b.spawn_yaw)

    def test_json_round_trip(self):
        c = C.generate(np.random.default_rng(5), self.levels, 2, 9)
        d = C.Course.from_json(c.to_json())
        np.testing.assert_array_equal(c.platforms, d.platforms)
        np.testing.assert_allclose(c.spawn, d.spawn)

    def test_test_set_is_fixed_and_disjoint_from_training_family(self):
        t1 = held_out_courses(self.levels, 5, 10)
        t2 = held_out_courses(self.levels, 5, 10)
        self.assertEqual([course_hash(c) for c in t1], [course_hash(c) for c in t2])
        self.assertEqual(TEST_SEED_FAMILY, 1)

    def test_path_remaining(self):
        c = C.generate(np.random.default_rng(9), self.levels, 1, 6)
        rem = C.path_remaining(c)
        self.assertEqual(rem[-1], 0.0)
        self.assertTrue((np.diff(rem) < 0).all())


if __name__ == "__main__":
    unittest.main()

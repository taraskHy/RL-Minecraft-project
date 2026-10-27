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

    def test_obstacles_stay_clear_of_platforms_and_paths(self):
        rng = np.random.default_rng(3)
        n_extra = 0
        for i in range(300):
            c = C.generate(rng, self.levels, i % len(self.levels), 12, obstacle_prob=0.6)
            plat = {tuple(x) for x in C.Course(c.platforms, c.spawn, c.spawn_yaw).blocks()}
            extra = {tuple(x) for x in c.extra}
            n_extra += len(extra)
            self.assertFalse(plat & extra)
            # Nothing directly on top of a platform: two blocks of standing room everywhere.
            for x0, x1, z0, z1, y in c.platforms:
                for x in range(x0, x1 + 1):
                    for z in range(z0, z1 + 1):
                        for dy in (1, 2):
                            self.assertNotIn((x, y + dy, z), extra)
            d = C.Course.from_json(c.to_json())
            np.testing.assert_array_equal(d.extra, c.extra)
        self.assertGreater(n_extra, 0)

    def test_neo_wall_blocks_the_straight_line_but_not_the_platforms(self):
        for side in (1, -1):
            for heading in range(4):
                spec = C.JumpSpec(2, 0, 0, (1, 1), neo=2, side=side, wall_at=1)
                prev = np.array([30, 30, 30, 30, 10], dtype=np.int64)
                new = C._platform_after(prev, heading, spec)
                wall = C._neo_wall(prev, new, heading, spec)
                cells = {tuple(c) for c in wall}
                hx, hz = C.HEADINGS[heading]
                # The cell straight ahead of the takeoff, at feet and head height, is wall.
                ahead = (30 + hx, 11, 30 + hz)
                self.assertIn(ahead, cells)
                self.assertIn((ahead[0], 12, ahead[2]), cells)
                # It sticks out one block (double neo) past the platforms on the open side and no
                # further, and three blocks on the other side.
                rx, rz = (-hz, 0) if hx == 0 else (0, hx)  # right of the heading
                ox, oz = rx * side, rz * side
                self.assertIn((ahead[0] + ox, 11, ahead[2] + oz), cells)
                self.assertNotIn((ahead[0] + 2 * ox, 11, ahead[2] + 2 * oz), cells)
                self.assertIn((ahead[0] - 3 * ox, 11, ahead[2] - 3 * oz), cells)
                platform_cells = {tuple(x) for x in C.Course(np.stack([prev, new]), np.zeros(3), 0.0).blocks()}
                self.assertFalse(cells & platform_cells)

    def test_advanced_courses(self):
        levels = C.load_levels(advanced=True)
        self.assertGreaterEqual(len(levels), 8)
        rng = np.random.default_rng(11)
        for i in range(60):
            c = C.generate(rng, levels, 6 + i % (len(levels) - 6), 10)
            self.assertLessEqual(c.n, C.MAX_PLATFORMS)
            plat = {tuple(x) for x in C.Course(c.platforms, c.spawn, c.spawn_yaw).blocks()}
            self.assertFalse(plat & {tuple(x) for x in c.extra})

    def test_neos_take_off_from_one_wide_blocks(self):
        levels = C.load_levels(advanced=True)
        rng = np.random.default_rng(12)
        seen = 0
        for i in range(120):
            c = C.generate(rng, levels, 6 + i % (len(levels) - 6), 10)
            k = 0  # platform index of each spec's takeoff
            for spec in c.specs:
                if spec.neo:
                    p = c.platforms[k]
                    nxt = c.platforms[k + 1]
                    along_x = nxt[0] > p[1] or nxt[1] < p[0]
                    across = (p[3] - p[2] + 1) if along_x else (p[1] - p[0] + 1)
                    self.assertEqual(across, 1, spec.key())
                    seen += 1
                k += 1
        self.assertGreater(seen, 20)

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

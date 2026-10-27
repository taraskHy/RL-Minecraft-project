"""Nothing starts inside a block: a player overlapping a block can walk out through it."""
import unittest

import numpy as np
import torch

from parkour import courses as C
from parkour import physics
from parkour.env import CoursePool


class InsideBlock(unittest.TestCase):
    def test_detects_overlap_but_not_touching(self):
        c = C.single_jump(C.JumpSpec(1, 0, 0, (1, 1), neo=2, side=1))  # wall cells at x = 5
        pool = CoursePool([c], "cpu", (24, 16, 24))
        pos = torch.tensor([[4.5, 9.0, 12.5],    # centre of the takeoff block
                            [4.7, 9.0, 12.5],    # hitbox touching the wall face
                            [4.79, 9.0, 12.5],   # hitbox 0.09 inside the wall
                            [4.5, 8.5, 12.5]],   # feet inside the takeoff block
                           dtype=torch.float64)
        got = physics.inside_solid(pool.world, torch.zeros(4, dtype=torch.long), pos).tolist()
        self.assertEqual(got, [False, False, True, True])

    def test_generated_spawns_are_clear(self):
        levels = C.load_levels(advanced=True)
        rng = np.random.default_rng(5)
        courses = [C.generate(rng, levels, i % len(levels), 10, obstacle_prob=0.4 if i % 2 else 0.0) for i in range(300)]
        pool = CoursePool(courses, "cpu", C.GRID)
        inside = physics.inside_solid(pool.world, torch.arange(len(courses)), pool.spawn)
        self.assertFalse(bool(inside.any()))



class StandingOnObstacles(unittest.TestCase):
    def test_standing_on_a_wall_counts_as_a_fall(self):
        from parkour.env import on_obstacle
        c = C.single_jump(C.JumpSpec(1, 0, 0, (1, 1), neo=1, side=1))
        pool = CoursePool([c], "cpu", (24, 16, 24))
        wx, wy, wz = (int(v) for v in max(c.extra, key=lambda e: e[1]))
        s = physics.PlayerState.at(torch.tensor([[wx + 0.5, wy + 1.0, wz + 0.5], list(c.spawn)], dtype=torch.float64),
                                   torch.zeros(2, dtype=torch.float64))
        s.on_ground = torch.tensor([True, True])
        self.assertEqual(on_obstacle(pool, s, torch.zeros(2, dtype=torch.long)).tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()

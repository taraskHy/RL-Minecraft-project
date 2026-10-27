"""A player resting against a block face cannot slip into the block through rounding.

Recorded in a training replay: the player stood against the +Z face of block
(39, 15, 31) at z = 32.3, whose hitbox edge -0.3 + 32.3 = 31.999999999999996
reads as just inside the block. Without vanilla's 1e-7 tolerance the next move
toward the block was not blocked and the hitbox ended 0.018 blocks inside it.
"""
import unittest

import torch

from parkour import physics


def world_with_block():
    solid = torch.zeros(1, 48, 24, 48, dtype=torch.bool)
    solid[0, 39, 15, 31] = True
    return physics.World(solid)


def one_tick(world):
    s = physics.PlayerState.at(torch.tensor([[38.781802, 15.42, 32.3]], dtype=torch.float64),
                               torch.tensor([0.0], dtype=torch.float64))
    s.vel = torch.tensor([[0.159966, 0.3332, -0.02]], dtype=torch.float64)
    ctrl = torch.zeros(1, 6, dtype=torch.bool)
    return physics.simulate(world, torch.zeros(1, dtype=torch.long), s, ctrl)


class CornerSlip(unittest.TestCase):
    def test_rounding_reads_as_overlap(self):
        self.assertLess(-0.3 + 32.3, 32.0)

    def test_tolerance_blocks_the_slip(self):
        world = world_with_block()
        s = one_tick(world)
        self.assertGreaterEqual(float(s.pos[0, 2]), 32.3 - 1e-9)
        self.assertFalse(bool(physics.inside_solid(world, torch.zeros(1, dtype=torch.long), s.pos, eps=1e-6)[0]))

    def test_without_tolerance_the_player_slips_in(self):
        world = world_with_block()
        old = physics.COLLISION_EPSILON
        physics.COLLISION_EPSILON = 0.0
        try:
            s = one_tick(world)
        finally:
            physics.COLLISION_EPSILON = old
        self.assertLess(float(s.pos[0, 2]), 32.29)
        self.assertTrue(bool(physics.inside_solid(world, torch.zeros(1, dtype=torch.long), s.pos, eps=1e-6)[0]))


if __name__ == "__main__":
    unittest.main()

"""The batched simulator reproduces prismarine-physics, the engine Mineflayer moves bots with."""
import unittest

import numpy as np
import torch

from parkour import parity
from parkour.paths import NODE


@unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
class PhysicsParity(unittest.TestCase):
    def check(self, device, n, ticks, seed, ice=0.0):
        rng = np.random.default_rng(seed)
        cases = [parity.random_case(rng, ticks=ticks, ice=ice) for _ in range(n)]
        r = parity.compare(cases, device)
        self.assertEqual(r["flag_mismatch_ticks"], 0, r)
        self.assertLess(r["max_pos_err"], 1e-9, r)
        self.assertLess(r["max_vel_err"], 1e-9, r)
        self.assertEqual(r["diverged_cases"], 0, r)

    def test_cpu(self):
        self.check("cpu", 60, 120, 0)

    def test_ice_cpu(self):
        # Ice under the feet changes ground friction (the optional slipperiness grid; no course uses it yet).
        self.check("cpu", 40, 120, 2, ice=0.5)

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA device")
    def test_cuda(self):
        self.check("cuda", 120, 150, 1)


if __name__ == "__main__":
    unittest.main()

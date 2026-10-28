"""The batched simulator reproduces prismarine-physics, the engine Mineflayer moves bots with."""
import unittest

import numpy as np
import torch

from parkour import parity
from parkour.paths import NODE


@unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
class PhysicsParity(unittest.TestCase):
    def check(self, device, n, ticks, seed, ice=0.0, shapes=0.0, slime=0.0, climb=0.0):
        rng = np.random.default_rng(seed)
        cases = [parity.random_case(rng, ticks=ticks, ice=ice, shapes=shapes, slime=slime, climb=climb) for _ in range(n)]
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

    def test_shapes_cpu(self):
        # Slabs, stairs, posts, carpets, ... (the optional shape grid), against the real block states.
        self.check("cpu", 40, 120, 5, shapes=0.5)

    def test_slime_cpu(self):
        # Slime blocks bounce (the optional slime grid) and have slipperiness 0.8.
        self.check("cpu", 40, 120, 7, slime=0.5)

    def test_climb_cpu(self):
        # Ladders and vines (the optional climb grid): slow fall and sideways moves, climbing.
        self.check("cpu", 40, 120, 9, climb=0.6)

    def test_shape_table_is_prismarine_blocks(self):
        import subprocess
        from parkour.paths import BRIDGE
        r = subprocess.run([str(NODE), str(BRIDGE / "export_shapes.js"), "--check"], cwd=str(BRIDGE), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA device")
    def test_cuda(self):
        self.check("cuda", 120, 150, 1)


if __name__ == "__main__":
    unittest.main()

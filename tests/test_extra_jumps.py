"""Head hitters and ice run-ups (opt-in extra levels): placement, physics of the roof, and the level files."""
import json
import unittest

import numpy as np
import torch

from parkour import courses as C
from parkour import physics
from parkour.env import MAX_EXTRA, CoursePool
from parkour.paths import CONFIGS


def extra_table():
    return C.load_levels() + C.load_levels(extra=True)[6:]


class ExtraLevels(unittest.TestCase):
    def test_existing_levels_and_keys_unchanged(self):
        self.assertEqual(len(C.load_levels()), 6)
        self.assertEqual(len(C.load_levels(advanced=True)), 9)
        self.assertEqual(len(C.load_levels(extra=True)), 9)
        self.assertEqual(len(C.load_levels(advanced=True, extra=True)), 12)
        for level in C.load_levels(advanced=True):
            for sp in level:
                self.assertEqual((sp.ceiling, sp.ice), (0, 0))
                self.assertNotIn("_hh", sp.key())
        self.assertEqual(C.MAX_EXTRA, MAX_EXTRA)

    def test_level_file_holds_only_feasible_specs(self):
        feas = {C._spec_from(r).key(): r["forgiveness"]
                for r in json.loads((CONFIGS / "jump_feasibility_extra.json").read_text())["specs"]}
        data = json.loads((CONFIGS / "jump_levels_extra.json").read_text())
        self.assertEqual([L["name"] for L in data["levels"]], ["head hitter", "hard head hitter", "ice run-up"])
        for L in data["levels"]:
            for d in L["specs"]:
                sp = C._spec_from(d)
                self.assertGreater(feas[C.JumpSpec(sp.gap, sp.dy, abs(sp.lateral), sp.size, runway=sp.runway,
                                                   ceiling=sp.ceiling, ice=sp.ice).key()], 0)
                self.assertGreater(sp.runway, 0)

    def test_roofs_and_ice_in_generated_courses(self):
        levels = extra_table()
        rng = np.random.default_rng([2026, 99])
        n_roofed = n_icy = 0
        for i in range(60):
            c = C.generate(rng, levels, 6 + i % 3, 10, seed=i)
            self.assertLessEqual(len(c.extra), MAX_EXTRA)
            cells = {tuple(int(v) for v in e) for e in c.extra}
            for x0, x1, z0, z1, y in c.platforms:  # standing room on every platform
                for x in range(x0, x1 + 1):
                    for z in range(z0, z1 + 1):
                        self.assertNotIn((x, y + 1, z), cells)
                        self.assertNotIn((x, y + 2, z), cells)
            k = 0
            for sp in c.specs:
                if sp.runway and sp.ceiling:
                    run = c.platforms[k]
                    x0, x1, z0, z1, y = (int(v) for v in run)
                    over = [(x, y + 3, z) in cells for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
                    self.assertEqual(sum(over), len(over) - 1)  # a roof over all but the first runway block
                    n_roofed += 1
                if sp.runway and sp.ice:
                    self.assertIn(k, c.ice)
                    n_icy += 1
                k += 1
            d = c.to_json()
            self.assertEqual(C.Course.from_json(d).ice, c.ice)
        self.assertGreater(n_roofed, 10)
        self.assertGreater(n_icy, 5)
        pool = CoursePool([C.generate(np.random.default_rng(i), levels, 8, 8) for i in range(4)], "cpu", C.GRID)
        self.assertIsNotNone(pool.slip)

    def test_roof_caps_the_rise_at_two_tenths(self):
        c = C.single_jump(C.JumpSpec(2, 0, 0, (1, 1), runway=3, ceiling=1))
        pool = CoursePool([c], "cpu", (24, 16, 24))
        x0, _, z0, _, y = c.platforms[0]
        s = physics.PlayerState.at(torch.tensor([[x0 + 1.5, y + 1.0, z0 + 0.5]], dtype=torch.float64),
                                   torch.tensor([0.0], dtype=torch.float64))
        s.on_ground = torch.tensor([True])
        jump = torch.zeros(1, 6, dtype=torch.bool)
        jump[0, physics.JUMP] = True
        top = 0.0
        for _ in range(6):
            s = physics.simulate(pool.world, torch.tensor([0]), s, jump)
            top = max(top, float(s.pos[0, 1]) - (y + 1.0))
        self.assertAlmostEqual(top, 0.2, places=6)


if __name__ == "__main__":
    unittest.main()

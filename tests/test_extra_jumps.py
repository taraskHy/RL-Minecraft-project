"""The opt-in extra levels (head hitters, ice run-ups, shaped blocks, slime bounces, ladder climbs): placement,
physics of the roof, and the level files."""
import dataclasses
import json
import unittest

import numpy as np
import torch

from parkour import courses as C
from parkour import physics
from parkour.env import MAX_EXTRA, CoursePool
from parkour.feasibility import EXTRA_LEVELS
from parkour.paths import CONFIGS

KIND_OF_LEVEL = {"head hitter": "ceiling", "hard head hitter": "ceiling", "ice run-up": "ice", "shaped blocks": "block",
                 "slime bounce": "bounce", "ladder climb": "ladder"}  # the JumpSpec field every spec of the level sets


def extra_table():
    return C.load_levels() + C.load_levels(extra=True)[6:]


class ExtraLevels(unittest.TestCase):
    def test_existing_levels_and_keys_unchanged(self):
        self.assertEqual(len(C.load_levels()), 6)
        self.assertEqual(len(C.load_levels(advanced=True)), 9)
        self.assertEqual(len(C.load_levels(extra=True)), 6 + len(EXTRA_LEVELS))
        self.assertEqual(len(C.load_levels(advanced=True, extra=True)), 9 + len(EXTRA_LEVELS))
        for level in C.load_levels(advanced=True):
            for sp in level:
                self.assertEqual((sp.ceiling, sp.ice), (0, 0))
                self.assertNotIn("_hh", sp.key())
        self.assertEqual(C.MAX_EXTRA, MAX_EXTRA)

    def test_level_file_holds_only_feasible_specs(self):
        feas = {C._spec_from(r).key(): r["forgiveness"]
                for r in json.loads((CONFIGS / "jump_feasibility_extra.json").read_text())["specs"]}
        data = json.loads((CONFIGS / "jump_levels_extra.json").read_text())
        self.assertEqual([L["name"] for L in data["levels"]], [name for name, _ in EXTRA_LEVELS])
        for L in data["levels"]:
            self.assertTrue(L["specs"], L["name"])
            for d in L["specs"]:
                sp = C._spec_from(d)
                self.assertGreater(feas[dataclasses.replace(sp, lateral=abs(sp.lateral)).key()], 0)
                self.assertTrue(getattr(sp, KIND_OF_LEVEL[L["name"]]), L["name"])
                if L["name"] in ("head hitter", "hard head hitter", "ice run-up"):
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

    def test_single_jump_evaluation_keeps_the_blocks(self):
        from parkour.jumpeval import jump_course, target_index
        slab = jump_course(C.JumpSpec(2, 0, 0, (1, 1), block="bottom_slab"), 0.0)
        self.assertEqual((slab.n, slab.shapes, target_index(slab.specs[0])), (4, {1: "bottom_slab"}, 1))
        pad = jump_course(C.JumpSpec(2, -3, 0, (1, 1), bounce=2), 0.0)
        self.assertEqual((pad.n, pad.slime, target_index(pad.specs[0])), (4, [1], 2))
        self.assertEqual(int(pad.platforms[2][4] - pad.platforms[1][4]), 2)  # the exit 2 above the pad
        ladder = jump_course(C.JumpSpec(1, 3, 0, (1, 1), ladder=1), 0.0)
        self.assertEqual((ladder.n, len(ladder.ladders)), (3, 4))
        plain = jump_course(C.JumpSpec(2, 0, 1, (1, 1)), 0.0)
        self.assertEqual((plain.n, plain.shapes, plain.slime, len(plain.ladders)), (3, {}, [], 0))

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

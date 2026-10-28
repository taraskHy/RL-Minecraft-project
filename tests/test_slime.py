"""Slime pads (opt-in): courses record them, the pool bounces on them, and the bot's JavaScript agrees."""
import json
import os
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from parkour import courses as C
from parkour import physics
from parkour.env import CoursePool
from parkour.feasibility import bounce_specs
from parkour.paths import BRIDGE, NODE, ROOT


def bounce_courses(n, seed):
    levels = C.load_levels() + [bounce_specs()]
    rng = np.random.default_rng(seed)
    return [C.generate(rng, levels, 6 if i % 3 else 1, 8, seed=i) for i in range(n)]


class SlimeCourses(unittest.TestCase):
    def test_pads_recorded_and_bouncy(self):
        cs = bounce_courses(30, 4)
        self.assertGreater(sum(bool(c.slime) for c in cs), 15)
        for c in cs:
            self.assertEqual(C.Course.from_json(c.to_json()).slime, c.slime)
            for k, sp in enumerate(c.specs):
                if sp.bounce:
                    self.assertIn(k + 1, c.slime)  # the jump from platform k lands on the pad k + 1
                    self.assertEqual(int(c.platforms[k + 2][4] - c.platforms[k + 1][4]), sp.bounce)
        pool = CoursePool(cs, "cpu", C.GRID)
        self.assertIsNotNone(pool.world.slime)
        i = next(i for i, c in enumerate(cs) if c.slime)
        k = cs[i].slime[0]
        x0, x1, z0, z1, y = (int(v) for v in cs[i].platforms[k])
        self.assertTrue(bool(pool.slime[i, x0, y, z0]))
        self.assertEqual(float(pool.slip[i, x0, y, z0]), physics.SLIME_SLIPPERINESS)
        plain = C.generate(np.random.default_rng(0), C.load_levels(), 2, 6)
        self.assertIsNone(CoursePool([plain], "cpu", C.GRID).world.slime)  # no slime, no slime grid

    def test_dropping_on_a_pad_bounces(self):
        cs = bounce_courses(12, 6)
        c = next(c for c in cs if c.slime)
        pool = CoursePool([c], "cpu", C.GRID)
        k = c.slime[0]
        x0, x1, z0, z1, y = (int(v) for v in c.platforms[k])
        s = physics.PlayerState.at(torch.tensor([[x0 + 0.5, y + 4.0, z0 + 0.5]], dtype=torch.float64),
                                   torch.tensor([0.0], dtype=torch.float64))
        none = torch.zeros(1, 6, dtype=torch.bool)
        ups = []
        for _ in range(40):
            s = physics.simulate(pool.world, torch.tensor([0]), s, none)
            ups.append(float(s.vel[0, 1]))
        self.assertGreater(max(ups), 0.3)  # it went back up after landing


@unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
class SlimeBotParity(unittest.TestCase):
    def test_offline_bot_plays_slime_courses_like_the_simulator(self):
        from parkour.live import SETTLE_TICKS, load_model
        from parkour.train import evaluate
        cs = bounce_courses(12, 8)
        model = load_model(ROOT / "results" / "final_policy.pt", "cpu")
        with torch.no_grad():
            py = evaluate(model, CoursePool(cs, "cpu", C.GRID), "cpu", 200, settle=SETTLE_TICKS, per_course=True)["per_course"]
        with tempfile.TemporaryDirectory() as d:
            pol, crs, out = (os.path.join(d, f) for f in ("policy.json", "courses.json", "episodes.jsonl"))
            model.export(pol)
            with open(crs, "w") as f:
                json.dump({"courses": [c.to_json() for c in cs]}, f)
            subprocess.run([str(NODE), str(BRIDGE / "offline_bot.js"), "--policy", pol, "--courses", crs, "--out", out,
                            "--settle", str(SETTLE_TICKS)], check=True, cwd=str(BRIDGE), capture_output=True)
            js = [json.loads(line) for line in open(out) if line.strip()]
        self.assertEqual([e["success"] for e in js], [bool(v) for v in py["success"]])
        self.assertEqual([e["ticks"] for e in js], [int(v) for v in py["ticks"]])
        reached_pad = sum(any(k <= e["cp"] for k in c.slime) for e, c in zip(js, cs))
        self.assertGreater(reached_pad, 3)


if __name__ == "__main__":
    unittest.main()

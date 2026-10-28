"""Ice platforms (optional): course generation, the pool's slipperiness grid, and the bot's JavaScript on ice."""
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
from parkour.paths import BRIDGE, NODE, ROOT


class IceCourses(unittest.TestCase):
    levels = C.load_levels()

    def test_layouts_do_not_change(self):
        a = C.generate(np.random.default_rng([3, 4]), self.levels, 2, 10)
        b = C.generate(np.random.default_rng([3, 4]), self.levels, 2, 10, ice_prob=1.0)
        np.testing.assert_array_equal(a.platforms, b.platforms)
        self.assertEqual(a.spawn_yaw, b.spawn_yaw)
        self.assertEqual(a.ice, [])
        self.assertEqual(b.ice, list(range(1, b.n)))  # never the start block

    def test_json_round_trip(self):
        c = C.generate(np.random.default_rng(8), self.levels, 1, 8, ice_prob=0.5)
        self.assertTrue(c.ice)
        self.assertEqual(C.Course.from_json(c.to_json()).ice, c.ice)
        plain = C.generate(np.random.default_rng(8), self.levels, 1, 8)
        self.assertNotIn("ice", plain.to_json())  # files of courses without ice stay as they were

    def test_pool_grid_follows_replace(self):
        icy = C.generate(np.random.default_rng(1), self.levels, 0, 6, ice_prob=1.0)
        plain = C.generate(np.random.default_rng(2), self.levels, 0, 6)
        pool = CoursePool([icy, plain], "cpu", C.GRID)
        cells = icy.ice_cells()
        slip = pool.slip
        self.assertTrue(bool((slip[0, cells[:, 0], cells[:, 1], cells[:, 2]] == physics.ICE_SLIPPERINESS).all()))
        self.assertEqual(int((slip[1] != physics.DEFAULT_SLIPPERINESS).sum()), 0)
        pool.replace(0, plain)
        self.assertEqual(int((pool.world.slip != physics.DEFAULT_SLIPPERINESS).sum()), 0)  # the world sees it
        self.assertIsNone(CoursePool([plain], "cpu", C.GRID).world.slip)
        with self.assertRaises(ValueError):
            CoursePool([plain], "cpu", C.GRID).replace(0, icy)

    def test_ice_changes_ground_motion(self):
        icy = C.generate(np.random.default_rng(5), self.levels, 0, 4, ice_prob=1.0)
        icy.ice = [0] + icy.ice  # the start block too, for this check
        stone = C.Course.from_json({**icy.to_json(), "ice": []})
        ends = []
        for c in (stone, icy):
            pool = CoursePool([c], "cpu", C.GRID)
            s = physics.PlayerState.at(torch.as_tensor(c.spawn[None]), torch.tensor([c.spawn_yaw]))
            s.on_ground = torch.tensor([True])
            walk = torch.tensor([[True, False, False, False, False, False]])
            for _ in range(3):
                s = physics.simulate(pool.world, torch.tensor([0]), s, walk)
            ends.append(s.vel.clone())
        self.assertGreater(float((ends[0] - ends[1]).abs().max()), 1e-3)


@unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
class IceBotParity(unittest.TestCase):
    def test_offline_bot_plays_ice_courses_like_the_simulator(self):
        from parkour.live import SETTLE_TICKS, load_model
        from parkour.train import evaluate
        levels = C.load_levels()
        rng = np.random.default_rng(21)
        cs = [C.generate(rng, levels, i % 4, 10, seed=i, ice_prob=0.6) for i in range(12)]
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
        self.assertEqual(len(js), len(cs))
        for i, ep in enumerate(js):
            self.assertEqual(ep["success"], bool(py["success"][i]), f"course {i}")
            self.assertEqual(ep["ticks"], int(py["ticks"][i]), f"course {i}")
        self.assertGreater(sum(e["success"] for e in js), len(cs) // 2)


if __name__ == "__main__":
    unittest.main()

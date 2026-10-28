"""Ladders (opt-in): course pieces, climbing in the pool's physics, and the bot's JavaScript on ladder courses."""
import json
import os
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from parkour import courses as C
from parkour import physics
from parkour.env import CoursePool, standing_on
from parkour.feasibility import ladder_specs
from parkour.paths import BRIDGE, NODE, ROOT


def ladder_courses(n, seed):
    levels = C.load_levels() + [ladder_specs()]
    rng = np.random.default_rng(seed)
    return [C.generate(rng, levels, 6 if i % 3 else 2, 8, seed=i) for i in range(n)]


class Ladders(unittest.TestCase):
    def test_pieces(self):
        cs = ladder_courses(30, 2)
        self.assertGreater(sum(len(c.ladders) > 0 for c in cs), 15)
        for c in cs:
            self.assertEqual(C.Course.from_json(c.to_json()).ladders.tolist(), c.ladders.tolist())
            extra = {tuple(e) for e in np.asarray(c.extra).reshape(-1, 3).tolist()}
            for x, y, z, f in c.ladders.tolist():
                self.assertNotIn((x, y, z), extra)  # a ladder is not an obstacle
                dx, dz = ((0, -1), (1, 0), (0, 1), (-1, 0))[f]  # facing north, east, south, west; support behind
                behind = (x - dx, y, z - dz)
                plats = {(px, py, pz) for x0, x1, z0, z1, py in c.platforms.tolist()
                         for px in range(x0, x1 + 1) for pz in range(z0, z1 + 1)}
                self.assertTrue(behind in extra or behind in plats, (x, y, z, f))
        pool = CoursePool(cs, "cpu", C.GRID)
        self.assertIsNotNone(pool.world.climb)
        self.assertIsNotNone(pool.world.shape_ids)

    def test_holding_forward_climbs_onto_the_target(self):
        for h in (2, 3, 4, 5):
            c = C.single_jump(C.JumpSpec(1, h, 0, (1, 1), ladder=1))
            pool = CoursePool([c], "cpu", (24, 16, 24))
            x0, _, z0, _, y = c.platforms[0]
            s = physics.PlayerState.at(torch.tensor([[x0 + 0.5, y + 1.0, z0 + 0.5]], dtype=torch.float64),
                                       torch.tensor([c.spawn_yaw], dtype=torch.float64))
            fwd = torch.zeros(1, 6, dtype=torch.bool)
            fwd[0, physics.FORWARD] = True
            on = False
            for _ in range(100):
                s = physics.simulate(pool.world, torch.tensor([0]), s, fwd)
                on = on or bool(standing_on(pool, s, torch.tensor([0]))[0, 1])
            self.assertTrue(on, h)


@unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
class LadderBotParity(unittest.TestCase):
    def test_offline_bot_plays_ladder_courses_like_the_simulator(self):
        from parkour.live import SETTLE_TICKS, load_model
        from parkour.train import evaluate
        cs = ladder_courses(12, 5)
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


if __name__ == "__main__":
    unittest.main()

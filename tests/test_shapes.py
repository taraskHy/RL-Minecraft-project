"""Block shapes in the simulator (physics.World(shape=...)): off by default, cubes unchanged, boxes where expected."""
import unittest

import numpy as np
import torch

from parkour import parity, physics


class Shapes(unittest.TestCase):
    def test_full_cube_shape_grid_is_the_original_path(self):
        rng = np.random.default_rng(0)
        cases = [parity.random_case(rng, ticks=80) for _ in range(20)]
        plain = parity.run_torch(cases)
        world_cls = physics.World

        class ZeroShapes(world_cls):
            def __init__(self, solid, slip=None, shape=None, slime=None):
                super().__init__(solid, slip, torch.zeros(solid.shape, dtype=torch.uint8), slime)

        parity.physics.World = ZeroShapes
        try:
            zero = parity.run_torch(cases)
        finally:
            parity.physics.World = world_cls
        for k in plain:
            np.testing.assert_array_equal(plain[k], zero[k])

    def test_table(self):
        names, boxes, counts = physics.shape_table()
        self.assertEqual(names[0], "full")
        self.assertEqual(boxes[0, 0].tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(int(counts[physics.shape_id("stairs_east")]), 2)
        self.assertEqual(float(boxes[physics.shape_id("fence_post"), 0, 4]), 1.5)
        self.assertTrue(bool(((boxes * 64) == torch.round(boxes * 64)).all()))  # exact in binary: boxes add up exactly

    def world(self, name):
        solid = torch.zeros(1, 8, 8, 8, dtype=torch.bool)
        solid[0, 3, 2, 3] = True
        shape = torch.zeros(1, 8, 8, 8, dtype=torch.uint8)
        shape[0, 3, 2, 3] = physics.shape_id(name)
        return physics.World(solid, shape=shape)

    def test_inside(self):
        c = torch.tensor([0])
        slab = self.world("bottom_slab")
        at = (lambda y: torch.tensor([[3.5, y, 3.5]], dtype=torch.float64))  # noqa: E731
        self.assertFalse(bool(physics.inside_solid(slab, c, at(2.5))[0]))
        self.assertTrue(bool(physics.inside_solid(slab, c, at(2.4))[0]))
        post = self.world("fence_post")
        self.assertTrue(bool(physics.inside_solid(post, c, at(3.2))[0]))  # the post reaches 1.5 above its cell
        self.assertFalse(bool(physics.inside_solid(post, c, at(3.5))[0]))
        self.assertFalse(bool(physics.inside_solid(post, c, torch.tensor([[4.0, 3.2, 3.5]], dtype=torch.float64))[0]))

    def test_landing_heights(self):
        c = torch.tensor([0])
        for name, top in (("bottom_slab", 2.5), ("soul_sand", 2.875), ("carpet", 2.0625), ("fence_post", 3.5),
                          ("stairs_east", 2.5)):
            w = self.world(name)
            s = physics.PlayerState.at(torch.tensor([[3.5, 4.2, 3.5]], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64))
            if name == "stairs_east":
                s.pos[0, 0] = 3.2  # over the low half of the step
            none = torch.zeros(1, 6, dtype=torch.bool)
            for _ in range(30):
                s = physics.simulate(w, c, s, none)
            self.assertTrue(bool(s.on_ground[0]), name)
            self.assertEqual(float(s.pos[0, 1]), top, name)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(__import__("parkour.paths", fromlist=["NODE"]).NODE.exists(), "portable Node runtime not installed")
class ShapedCoursesBotParity(unittest.TestCase):
    def test_offline_bot_plays_shaped_courses_like_the_simulator(self):
        import json
        import os
        import subprocess
        import tempfile

        from parkour import courses as C
        from parkour.env import CoursePool
        from parkour.feasibility import shape_specs
        from parkour.live import SETTLE_TICKS, load_model
        from parkour.paths import BRIDGE, NODE, ROOT
        from parkour.train import evaluate
        levels = C.load_levels() + [shape_specs()]
        rng = np.random.default_rng(31)
        cs = [C.generate(rng, levels, 6 if i % 3 else 2, 8, seed=i) for i in range(12)]
        self.assertTrue(sum(bool(c.shapes) for c in cs) >= 8)
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
        self.assertEqual([e["cp"] for e in js], [round(pr * (c.n - 1)) for pr, c in zip(py["progress"], cs)])
        self.assertGreater(sum(e["cp"] >= 2 for e in js), 6)  # most get past their first shaped platform

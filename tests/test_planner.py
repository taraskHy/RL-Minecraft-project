"""The route planner: perception rules, and the same plans in Python and in the bot's JavaScript."""
import dataclasses
import json
import os
import subprocess
import tempfile
import unittest

import numpy as np

from parkour import courses as C
from parkour.paths import BRIDGE, NODE, ROOT
from parkour.planner import GreedyPlanner, PlannerConfig, perceive


def solid_of(course):
    cells = {tuple(int(v) for v in b) for b in course.blocks()}
    return lambda x, y, z: (x, y, z) in cells


class Perception(unittest.TestCase):
    def test_platforms_are_found_and_columns_are_not(self):
        plats = np.array([[4, 6, 20, 22, 10], [9, 9, 21, 21, 10]], dtype=np.int64)
        pillar = [(9, y, 24) for y in range(9, 13)]  # a column beside the second platform
        c = C.Course(plats, np.array([5.5, 11.0, 21.5]), 0.0, extra=np.array(pillar, dtype=np.int64))
        found = perceive(solid_of(c), 5, 11, 21, PlannerConfig())
        self.assertEqual(sorted(p.rect() for p in found), [(4, 6, 20, 22, 10), (9, 9, 21, 21, 10)])
        found = perceive(solid_of(c), 5, 11, 21, PlannerConfig(slab_only=False))
        self.assertIn((9, 9, 24, 24, 12), [p.rect() for p in found])

    def test_plan_follows_a_straight_course(self):
        plats = np.array([[4, 6, 20, 22, 10], [9, 9, 21, 21, 10], [12, 13, 20, 21, 9], [15, 16, 20, 22, 9]], dtype=np.int64)
        c = C.Course(plats, np.array([5.5, 11.0, 21.5]), 0.0)
        goal = ((15 + 16 + 1) / 2, 10.0, (20 + 22 + 1) / 2)
        pl = GreedyPlanner(solid_of(c), goal, PlannerConfig())
        pl.start((5.5, 11.0, 21.5))
        rects, valid, is_goal = pl.slots()
        self.assertEqual(rects, [tuple(p) for p in plats.tolist()])
        self.assertEqual(valid, [True] * 4)
        self.assertEqual(is_goal, [False, False, False, True])

    def test_value_pick_takes_the_scorers_chain_and_is_off_by_default(self):
        # From the start block both the next platform (1-block gap) and the one after it (4-block gap)
        # are in reach; the greedy cost prefers the short jump.
        plats = np.array([[4, 6, 20, 22, 10], [8, 8, 21, 21, 10], [11, 11, 21, 21, 10], [14, 15, 20, 22, 10]],
                         dtype=np.int64)
        c = C.Course(plats, np.array([5.5, 11.0, 21.5]), 0.0)
        goal = ((14 + 15 + 1) / 2, 11.0, (20 + 22 + 1) / 2)
        seen = []

        def prefer_second(here, chains):
            seen.append([[p.rect() for p in ch] for ch in chains])
            return 1
        for k, first in ((0, (8, 8, 21, 21, 10)), (2, (11, 11, 21, 21, 10))):
            pl = GreedyPlanner(solid_of(c), goal, PlannerConfig(value_pick=k))
            pl.scorer = prefer_second
            pl.start((5.5, 11.0, 21.5))
            self.assertEqual(pl.chain[0].rect(), first, f"value_pick={k}")
        self.assertEqual(len(seen), 1)  # only asked when value_pick > 0
        self.assertEqual([ch[0] for ch in seen[0]], [(8, 8, 21, 21, 10), (11, 11, 21, 21, 10)])
        self.assertEqual(seen[0][1][1], (14, 15, 20, 22, 10))  # the rest of each chain is greedy


class PlannerParity(unittest.TestCase):
    @unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
    def test_python_and_javascript_plan_the_same(self):
        import torch
        from parkour.env import CoursePool
        from parkour.live import SETTLE_TICKS, load_model
        from parkour.planner import evaluate_planned
        levels = C.load_levels()
        rng = np.random.default_rng(7)
        cs = [C.generate(rng, levels, i % len(levels), 10, obstacle_prob=0.5 if i % 2 else 0.0, seed=i) for i in range(24)]
        model = load_model(ROOT / "results" / "final_policy.pt", "cpu")
        cfg = PlannerConfig()
        pool = CoursePool(cs, "cpu", C.GRID)
        with torch.no_grad():
            py = evaluate_planned(model, pool, "cpu", cfg, settle=SETTLE_TICKS, record=True)
        with tempfile.TemporaryDirectory() as d:
            pol, crs, out = (os.path.join(d, f) for f in ("policy.json", "courses.json", "episodes.jsonl"))
            model.export(pol)
            with open(crs, "w") as f:
                json.dump({"courses": [c.to_json() for c in cs]}, f)
            subprocess.run([str(NODE), str(BRIDGE / "offline_bot.js"), "--policy", pol, "--courses", crs, "--out", out,
                            "--settle", str(SETTLE_TICKS), "--planner", json.dumps(dataclasses.asdict(cfg))],
                           check=True, cwd=str(BRIDGE), capture_output=True)
            js = [json.loads(line) for line in open(out) if line.strip()]
        self.assertEqual(len(js), len(cs))
        for i, ep in enumerate(js):
            self.assertEqual(ep["plans"], py["plans"][i], f"course {i}: plans differ")
            self.assertEqual(ep["success"], bool(py["per_course"]["success"][i]), f"course {i}")
            self.assertEqual(ep["ticks"], int(py["per_course"]["ticks"][i]), f"course {i}")
        self.assertGreater(sum(e["success"] for e in js), len(cs) // 2)


if __name__ == "__main__":
    unittest.main()

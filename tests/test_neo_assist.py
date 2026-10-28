"""Neo assist (parkour.neo_assist): off it is train.evaluate; on, its plans land where the search said."""
import unittest

import numpy as np
import torch

from parkour import courses as C
from parkour import physics
from parkour.env import CoursePool, on_obstacle
from parkour.live import load_model
from parkour.neo_assist import AssistConfig, Rules, evaluate_assisted, official_neo_courses, random_neo_courses
from parkour.neo_dagger import neo_takeoffs
from parkour.paths import ROOT
from parkour.train import evaluate


class NeoAssist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = load_model(ROOT / "results" / "final_policy.pt", "cpu")
        cls.courses = official_neo_courses(2)[:6]

    def test_off_is_the_plain_evaluation(self):
        pool = CoursePool(self.courses, "cpu", C.GRID)
        with torch.no_grad():
            ref = evaluate(self.model, pool, "cpu", 200, per_course=True)["per_course"]
        res = evaluate_assisted(self.model, pool, "cpu", AssistConfig(enabled=False))["per_course"]
        self.assertEqual([c["success"] for c in res], ref["success"])
        self.assertEqual([c["fell"] for c in res], ref["fell"])
        self.assertEqual([c["ticks"] for c in res], ref["ticks"])

    def test_plans_land_as_searched(self):
        pool = CoursePool(self.courses, "cpu", C.GRID)
        res = evaluate_assisted(self.model, pool, "cpu", AssistConfig(samples=256, iterations=8))
        self.assertTrue(res["events"])
        self.assertGreater(res["plans"]["played"], 0)
        self.assertEqual(res["plans"]["landed_as_planned"], res["plans"]["played"])
        self.assertEqual(len(res["per_course"]), len(self.courses))

    def test_fast_obstacle_rule_is_the_environment_rule(self):
        pool = CoursePool(self.courses, "cpu", C.GRID)
        rules = Rules(pool)
        self.assertLess(rules.extra.shape[1], pool.extra.shape[1])
        g = torch.Generator().manual_seed(3)
        n = 4000
        course = torch.randint(0, pool.size, (n,), generator=g)
        # Feet on, near and between obstacle cells: cell corners plus small offsets, exactly on top or slightly off.
        cells = rules.extra[course, torch.randint(0, rules.extra.shape[1], (n,), generator=g)]
        off = torch.rand(n, 3, generator=g, dtype=torch.float64) * torch.tensor([1.8, 0.0, 1.8], dtype=torch.float64) - 0.4
        pos = cells + torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64) + off
        pos[::3, 1] += 1e-3
        s = physics.PlayerState.at(pos, torch.zeros(n, dtype=torch.float64))
        s.on_ground = torch.rand(n, generator=g) < 0.8
        np.testing.assert_array_equal(rules.on_obstacle(s, course).numpy(), on_obstacle(pool, s, course).numpy())

    def test_course_sets(self):
        self.assertTrue(all(neo_takeoffs(c) for c in self.courses))
        rnd = random_neo_courses(13, n=20)
        self.assertEqual(len(rnd), 20)
        self.assertGreater(np.mean([len(neo_takeoffs(c)) for c in rnd]), 2)
        self.assertEqual([c.to_json() for c in rnd[:5]], [c.to_json() for c in random_neo_courses(13, n=5)])


if __name__ == "__main__":
    unittest.main()

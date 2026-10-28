"""Neo assist (parkour.neo_assist): off it is train.evaluate; on, its plans land where the search said,
and the offline bot with the planner service plays exactly like the simulator."""
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from parkour import courses as C
from parkour import physics
from parkour.env import CoursePool, on_obstacle
from parkour.live import load_model
from parkour.neo_assist import (AssistConfig, Rules, evaluate_assisted, official_neo_courses, planned_takeoffs,
                                 random_neo_courses, spec_from_key)
from parkour.neo_dagger import neo_takeoffs
from parkour.paths import BRIDGE, NODE, ROOT
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


@unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
class AssistDefaults(unittest.TestCase):
    def test_bot_trigger_defaults_are_the_python_ones(self):
        out = subprocess.run([str(NODE), "-e", "process.stdout.write(JSON.stringify(require('./assist').DEFAULTS))"],
                             cwd=str(BRIDGE), capture_output=True, text=True, check=True)
        js = json.loads(out.stdout)
        py = dataclasses.asdict(AssistConfig())
        self.assertEqual(js, {k: py[k] for k in js})


class SpecKeys(unittest.TestCase):
    def test_every_level_spec_round_trips(self):
        for adv in (False, True):
            for level in C.load_levels(advanced=adv):
                for sp in level:
                    self.assertEqual(spec_from_key(sp.key()), sp)


@unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
class AssistBotParity(unittest.TestCase):
    def test_offline_bot_with_planner_service_plays_like_the_simulator(self):
        from parkour.live import SETTLE_TICKS
        cfg = AssistConfig(samples=256, iterations=8, approach=48, prior=0.5)
        cs = official_neo_courses(2)[:6]
        model = load_model(ROOT / "results" / "final_policy.pt", "cpu")
        res = evaluate_assisted(model, CoursePool(cs, "cpu", C.GRID), "cpu", cfg, settle=SETTLE_TICKS)
        with tempfile.TemporaryDirectory() as d:
            pol, crs, out = (os.path.join(d, f) for f in ("policy.json", "courses.json", "episodes.jsonl"))
            model.export(pol)
            with open(crs, "w") as f:
                json.dump({"courses": [{**c.to_json(), "assist_takeoffs": planned_takeoffs(c, cfg.jumps)} for c in cs]}, f)
            subprocess.run([str(NODE), str(BRIDGE / "offline_bot.js"), "--policy", pol, "--courses", crs, "--out", out,
                            "--settle", str(SETTLE_TICKS), "--assist", json.dumps(dataclasses.asdict(cfg)),
                            "--assist-python", sys.executable, "--assist-checkpoint", str(ROOT / "results" / "final_policy.pt"),
                            "--assist-root", str(ROOT), "--assist-device", "cpu"], check=True, cwd=str(BRIDGE),
                           capture_output=True)
            js = [json.loads(line) for line in open(out) if line.strip()]
        self.assertEqual([e["success"] for e in js], [c["success"] for c in res["per_course"]])
        self.assertEqual([e["ticks"] for e in js], [c["ticks"] for c in res["per_course"]])
        bot = [x for e in js for x in e["assist"]]
        self.assertEqual(len(bot), len(res["events"]))
        self.assertEqual(sum(x["found"] for x in bot), sum(e["found"] for e in res["events"]))
        self.assertTrue(all(x["verified"] == x["found"] for x in bot))  # prismarine-physics agrees with every plan
        self.assertGreater(sum(x["played"] for x in bot), 0)


if __name__ == "__main__":
    unittest.main()

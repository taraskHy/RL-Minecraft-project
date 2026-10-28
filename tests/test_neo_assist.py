"""Neo assist (parkour.neo_assist): off it is train.evaluate; on, its plans land where the search said,
and the offline bot with the planner service plays exactly like the simulator."""
import dataclasses
import json
import math
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
from parkour.neo_assist import (AssistConfig, Rules, brake_plan, courses_from_file, evaluate_assisted, jump_kind,
                                 mixed_courses, official_neo_courses, plan_target, planned_takeoffs, random_neo_courses,
                                 spec_from_key, specs_from_keys)
from parkour.neo_dagger import neo_takeoffs
from parkour.paths import BRIDGE, CONFIGS, NODE, ROOT
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
        mixed = mixed_courses(15, n=40)
        kinds = {jump_kind(c, k, neo_takeoffs(c)) for c in mixed for k in range(c.n - 1)}
        self.assertTrue({"neo", "wide", "momentum", "run-up (roof or ice)", "basic"} <= kinds, kinds)
        self.assertEqual([c.to_json() for c in mixed[:5]], [c.to_json() for c in mixed_courses(15, n=5)])

    def test_a_ladder_wall_is_not_a_neo(self):
        for h in (2, 3, 4, 5):
            self.assertEqual(neo_takeoffs(C.single_jump(C.JumpSpec(1, h, 0, (1, 1), ladder=1))), [])
        self.assertEqual(neo_takeoffs(C.single_jump(C.JumpSpec(1, 0, 0, (1, 1), neo=1))), [0])


def extra_kind_courses():
    """Two short courses each with ladder climbs, slime pads and shaped targets (feasible or not)."""
    from parkour.feasibility import bounce_specs, ladder_specs, shape_specs
    out = []
    for i, kind in enumerate((ladder_specs(), bounce_specs()[:18], shape_specs()[::12])):
        rng = np.random.default_rng([7, i])
        out += [C.generate(rng, C.load_levels() + [kind], 6, 5, seed=j) for j in range(2)]
    return out


class Brakes(unittest.TestCase):
    def test_brake_stops_the_bot_on_its_takeoff(self):
        c = official_neo_courses(2)[0]
        pool = CoursePool([c], "cpu", C.GRID)
        rules = Rules(pool)
        x0, x1, z0, z1, y = (int(v) for v in c.platforms[0])
        moving = {"course": 0, "cp": 0, "pos": [(x0 + x1 + 1) / 2, y + 1.0, (z0 + z1 + 1) / 2], "vel": [0.12, 0.0, 0.0],
                  "yaw": -math.pi / 2, "on_ground": True, "jump_ticks": 0, "collided_h": False, "collided_v": False}
        seq = brake_plan(pool, rules, moving, 14, "cpu")
        self.assertIsNotNone(seq)
        self.assertEqual(seq.shape, (14, 5))
        self.assertTrue((seq[:, 2] == 0).all() and (seq[:, 3] == 0).all())  # no jump, no sprint
        over = dict(moving, pos=[x1 + 1.25, y + 1.0, (z0 + z1 + 1) / 2], vel=[0.6, 0.0, 0.0])  # too fast at the edge
        self.assertIsNone(brake_plan(pool, rules, over, 14, "cpu"))

    def test_failed_searches_brake_and_retry(self):
        model = load_model(ROOT / "results" / "final_policy.pt", "cpu")
        cs = official_neo_courses(2)[:6]
        res = evaluate_assisted(model, CoursePool(cs, "cpu", C.GRID), "cpu",
                                AssistConfig(samples=64, iterations=3, jumps="advanced", retries=1, brake=14))
        braked = [e for e in res["events"] if e["braked"]]
        self.assertGreater(len(braked), 0)
        for e in braked:  # a retry from the same takeoff right after the brake's 14 ticks
            self.assertTrue(any(r["kind"] == "retry" and r["course"] == e["course"] and r["takeoff"] == e["takeoff"]
                                and r["tick"] == e["tick"] + 14 for r in res["events"]))
        self.assertEqual(res["plans"]["landed_as_planned"], res["plans"]["played"])


class AssistExtraKinds(unittest.TestCase):
    def test_every_jump_planned_and_slime_plans_end_past_the_pad(self):
        cs = extra_kind_courses()
        self.assertTrue(any(c.slime for c in cs) and any(c.shapes for c in cs) and any(len(c.ladders) for c in cs))
        model = load_model(ROOT / "results" / "final_policy.pt", "cpu")
        res = evaluate_assisted(model, CoursePool(cs, "cpu", C.GRID), "cpu", AssistConfig(samples=256, iterations=8, jumps="all"))
        self.assertGreater(res["plans"]["played"], 0)
        self.assertEqual(res["plans"]["landed_as_planned"], res["plans"]["played"])
        for e in res["events"]:
            c = cs[e["course"]]
            pad = e["takeoff"] < len(c.specs) and c.specs[e["takeoff"]].bounce
            self.assertEqual(e["target"], e["takeoff"] + (2 if pad else 1))
        self.assertTrue(any(e["target"] == e["takeoff"] + 2 for e in res["events"]))


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
        for adv, extra in ((False, False), (True, False), (False, True)):
            for level in C.load_levels(advanced=adv, extra=extra):
                for sp in level:
                    self.assertEqual(spec_from_key(sp.key()), sp)
        names = [s["name"] for s in json.loads((CONFIGS / "block_shapes.json").read_text())["shapes"]]
        for sp in [C.JumpSpec(2, 0, 1, (2, 2), block=b) for b in names] + [
                C.JumpSpec(2, -3, 0, (1, 1), bounce=2), C.JumpSpec(1, 4, 0, (1, 1), ladder=1),
                C.JumpSpec(3, 0, 0, (1, 1), runway=4, ceiling=2, ice=1)]:
            self.assertEqual(spec_from_key(sp.key()), sp)

    def test_hand_designed_courses_get_one_spec_per_jump(self):
        steps = [(C.JumpSpec(1, 0, 0, (1, 1)), 0), (C.JumpSpec(1, 0, 0, (1, 1), neo=1), 0),
                 (C.JumpSpec(2, 0, 0, (2, 2), block="bottom_slab"), 0),
                 (C.JumpSpec(2, -3, 0, (1, 1), bounce=1), 1), (C.JumpSpec(4, -1, 0, (2, 2), runway=3), 0),
                 (C.JumpSpec(1, 0, 0, (3, 3)), 0)]
        c = C.from_specs(steps)
        self.assertNotEqual(len(c.specs), c.n - 1)
        d = c.to_json()
        specs = specs_from_keys(d["specs"], c.n)
        self.assertEqual(len(specs), c.n - 1)
        runway = next(k for k, sp in enumerate(specs) if sp.runway)
        x0, x1, z0, z1, _ = (int(v) for v in c.platforms[runway])
        self.assertEqual(max(x1 - x0, z1 - z0) + 1, 3)  # the run-up jump starts from the 3-long runway
        pad = int(c.slime[0])
        self.assertEqual([plan_target(courses_from_file([d])[0], k) for k in range(c.n - 1)][pad - 1], pad + 1)

    def test_all_plans_every_jump(self):
        c = random_neo_courses(13, n=1)[0]
        self.assertEqual(planned_takeoffs(c, "all"), list(range(c.n - 1)))
        self.assertTrue(set(planned_takeoffs(c, "neo")) <= set(planned_takeoffs(c, "advanced")))


@unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
class AssistBotParity(unittest.TestCase):
    def test_offline_bot_with_planner_service_plays_like_the_simulator(self):
        self.check(AssistConfig(samples=256, iterations=8, approach=48, prior=0.5), official_neo_courses(2)[:6])

    def test_same_on_shaped_slime_and_ladder_courses(self):
        self.check(AssistConfig(samples=256, iterations=8, jumps="all"), extra_kind_courses())

    def test_same_with_brakes_and_retries(self):
        res, bot = self.check(AssistConfig(samples=64, iterations=3, jumps="advanced", retries=1, brake=14),
                              official_neo_courses(2)[:6])
        self.assertGreater(res["plans"]["brakes"], 0)
        self.assertEqual(sum(x.get("braked", False) for x in bot), res["plans"]["brakes"])

    def check(self, cfg, cs):
        from parkour.live import SETTLE_TICKS
        model = load_model(ROOT / "results" / "final_policy.pt", "cpu")
        res = evaluate_assisted(model, CoursePool(cs, "cpu", C.GRID), "cpu", cfg, settle=SETTLE_TICKS)
        with tempfile.TemporaryDirectory() as d:
            pol, crs, out = (os.path.join(d, f) for f in ("policy.json", "courses.json", "episodes.jsonl"))
            model.export(pol)
            with open(crs, "w") as f:
                json.dump({"courses": [{**c.to_json(), "assist_takeoffs": planned_takeoffs(c, cfg.jumps),
                                        "assist_targets": [plan_target(c, k) for k in planned_takeoffs(c, cfg.jumps)]}
                                       for c in cs]}, f)
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
        return res, bot


if __name__ == "__main__":
    unittest.main()

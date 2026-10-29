"""Neo assist (parkour.neo_assist): off it is train.evaluate; on, its plans land where the search said,
and the offline bot with the planner service plays exactly like the simulator."""
import collections
import dataclasses
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from parkour import courses as C
from parkour import neo_assist as NA
from parkour import neo_dagger as ND
from parkour import physics
from parkour import train as T
from parkour.env import CoursePool, on_obstacle
from parkour.live import load_model
from parkour.neo_assist import (SINGLE_TEST_FAMILY, SINGLE_VALIDATION_FAMILY, TEST_FAMILIES, AssistConfig, Rules,
                                 approach_platforms, assist_fields, brake_plan, courses_from_file, evaluate_assisted,
                                 jump_kind, merge_results, mixed_courses, neo_level, official_neo_courses, part_bounds,
                                 plan_target, planned_takeoffs, random_neo_courses, single_neo_courses,
                                 single_neo_hashes, single_neo_ok, single_neo_types, spec_from_key, specs_from_keys,
                                 summarize)
from parkour.neo_dagger import neo_takeoffs
from parkour.paths import BRIDGE, CONFIGS, NODE, ROOT
from parkour.train import course_hash, evaluate


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

    def test_a_run_in_parts_is_the_whole_run(self):
        cfg = AssistConfig(samples=64, iterations=3, jumps="all", retries=1, brake=14)
        whole = evaluate_assisted(self.model, CoursePool(self.courses, "cpu", C.GRID), "cpu", cfg)
        parts = [evaluate_assisted(self.model, CoursePool(self.courses[lo:hi], "cpu", C.GRID), "cpu", cfg, course_offset=lo)
                 for lo, hi in (part_bounds(6, 1, 2), part_bounds(6, 2, 2))]
        merged = merge_results(parts)
        self.assertEqual(merged["per_course"], whole["per_course"])
        key = lambda e: (e["course"], e["tick"], e["kind"])  # noqa: E731
        self.assertEqual(sorted(merged["events"], key=key), sorted(whole["events"], key=key))
        self.assertEqual(merged["plans"], whole["plans"])
        self.assertEqual([part_bounds(218, k, 4) for k in (1, 2, 3, 4)], [(0, 54), (54, 109), (109, 164), (164, 218)])

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
    def test_course_file_fields(self):
        cs = courses_from_file([c.to_json() for c in extra_kind_courses()])
        cfg = AssistConfig(jumps="all", retries=1, retry_horizon=48)
        for c in cs:
            self.assertEqual(len(c.specs), c.n - 1)
            f = assist_fields(c, cfg)
            self.assertEqual(f["assist_takeoffs"], list(range(c.n - 1)))
            for k, tgt, h in zip(f["assist_takeoffs"], f["assist_targets"], f["assist_retry_horizons"]):
                sp = c.specs[k]
                self.assertEqual(tgt, k + 2 if sp.bounce else k + 1)
                self.assertEqual(h, 80 if sp.bounce else 64 if sp.ladder else 48)
        self.assertEqual(specs_from_keys(["g1_dy0_l0_s1x1"], 5), [])  # cannot be aligned: no specs, not wrong ones

    def test_approach_before_neos_then_the_takeoff_search(self):
        cs = official_neo_courses(2)[:6]
        for c in cs:  # with every jump planned, an approach before every neo takeoff
            self.assertEqual(approach_platforms(c, planned_takeoffs(c, "all")), sorted(k - 1 for k in neo_takeoffs(c) if k >= 1))
            ks = planned_takeoffs(c, "neo")
            self.assertTrue(set(approach_platforms(c, ks)) >= {k - 1 for k in ks if k >= 1 and k - 1 not in ks})
        model = load_model(ROOT / "results" / "final_policy.pt", "cpu")
        res = evaluate_assisted(model, CoursePool(cs, "cpu", C.GRID), "cpu",
                                AssistConfig(samples=64, iterations=3, jumps="all", approach=48))
        ev = res["events"]
        approaches = [e for e in ev if e["kind"] == "approach"]
        self.assertTrue(approaches)
        for e in approaches:
            self.assertEqual(e["target"], e["takeoff"] + 1)  # from the platform before the neo's takeoff, over it
            same = [x for x in ev if x["kind"] == "takeoff" and x["course"] == e["course"] and x["takeoff"] == e["takeoff"] - 1
                    and x["tick"] == e["tick"]]
            self.assertEqual(len(same), 0 if e["found"] else 1)  # without a plan, the takeoff search at once
        self.assertEqual(res["plans"]["landed_as_planned"], res["plans"]["played"])

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
        self.check(AssistConfig(samples=256, iterations=8, jumps="all", retries=1, brake=14, retry_horizon=48),
                   extra_kind_courses())

    def test_same_with_approach_plans_before_neos(self):
        res, bot = self.check(AssistConfig(samples=128, iterations=4, jumps="all", approach=48, retries=1, brake=14),
                              official_neo_courses(2)[:6])
        self.assertGreater(sum(e["kind"] == "approach" for e in res["events"]), 0)
        self.assertEqual(sum(x["kind"] == "approach" for x in bot), sum(e["kind"] == "approach" for e in res["events"]))

    def test_same_with_brakes_and_retries(self):
        res, bot = self.check(AssistConfig(samples=64, iterations=3, jumps="advanced", retries=1, brake=14),
                              official_neo_courses(2)[:6])
        self.assertGreater(res["plans"]["brakes"], 0)
        self.assertEqual(sum(x.get("braked", False) for x in bot), res["plans"]["brakes"])

    def check(self, cfg, cs):
        from parkour.live import SETTLE_TICKS
        model = load_model(ROOT / "results" / "final_policy.pt", "cpu")
        cs = courses_from_file([c.to_json() for c in cs])  # as parkour.live reads a course file
        res = evaluate_assisted(model, CoursePool(cs, "cpu", C.GRID), "cpu", cfg, settle=SETTLE_TICKS)
        with tempfile.TemporaryDirectory() as d:
            pol, crs, out = (os.path.join(d, f) for f in ("policy.json", "courses.json", "episodes.jsonl"))
            model.export(pol)
            with open(crs, "w") as f:  # as parkour.live writes the bots' course file
                json.dump({"courses": [{**c.to_json(), **assist_fields(c, cfg)} for c in cs]}, f)
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


class SingleNeoSets(unittest.TestCase):
    """The single-neo sets (--set single, families 40 and 41) and how they stay out of training."""

    @classmethod
    def setUpClass(cls):
        cls.sets = {f: single_neo_courses(f) for f in (SINGLE_VALIDATION_FAMILY, SINGLE_TEST_FAMILY)}

    def test_one_neo_at_its_position_and_every_type_and_position_equally_often(self):
        types = single_neo_types()
        self.assertEqual([s for _, s in types], neo_level())
        self.assertEqual(collections.Counter(L for L, _ in types), {6: 22, 7: 10})
        basic = {s for L in C.load_levels() for s in L}
        for family, cs in self.sets.items():
            self.assertEqual(len(cs), 320)
            pairs = []
            for i, c in enumerate(cs):
                level, spec = types[i % 32]
                k = (i % 32 + i // 32) % 10
                self.assertEqual(neo_takeoffs(c), [k])
                self.assertEqual([j for j, sp in enumerate(c.specs) if sp.neo], [k])
                self.assertTrue(single_neo_ok(c, k))  # also: no wall cell in another jump's corridor
                self.assertEqual((c.specs[k], c.n, c.level, c.seed), (spec, 11, level, i))
                self.assertTrue(all(sp in basic for j, sp in enumerate(c.specs) if j != k))
                if k == 0:  # a 1x1 start
                    x0, x1, z0, z1, _ = c.platforms[0]
                    self.assertEqual((x0, z0), (x1, z1))
                pairs.append((spec.key(), k))
            self.assertEqual(len(set(pairs)), 320)  # every type once at every position
            self.assertEqual(set(collections.Counter(t for t, _ in pairs).values()), {10})
            self.assertEqual(collections.Counter(k for _, k in pairs), {k: 32 for k in range(10)})
        self.assertEqual([c.to_json() for c in single_neo_courses(SINGLE_VALIDATION_FAMILY, n=40)],
                         [c.to_json() for c in self.sets[SINGLE_VALIDATION_FAMILY][:40]])  # a prefix: the same courses
        self.assertIn(SINGLE_TEST_FAMILY, TEST_FAMILIES)
        self.assertNotIn(SINGLE_VALIDATION_FAMILY, TEST_FAMILIES)

    def test_kept_out_of_training(self):
        hashes = single_neo_hashes()
        self.assertEqual(hashes, {course_hash(c) for cs in self.sets.values() for c in cs})
        self.assertEqual(len(hashes), 640)
        original = T.held_out_courses
        tiny = lambda levels, per_level, n_jumps, obstacle_prob=0.0, family=1, ice_prob=0.0: original(  # noqa: E731
            levels, 1, 2, obstacle_prob=obstacle_prob, family=family, ice_prob=ice_prob)
        # neo_dagger's training courses: one equal to a course of the sets is skipped.
        with mock.patch.object(ND, "held_out_courses", tiny), mock.patch.object(NA, "single_neo_hashes", set):
            first = ND.training_neo_courses(2, 0)
        with mock.patch.object(ND, "held_out_courses", tiny), \
                mock.patch.object(NA, "single_neo_hashes", lambda: {course_hash(first[0])}):
            again = ND.training_neo_courses(2, 0)
        self.assertEqual(len(again), 2)
        self.assertNotIn(course_hash(first[0]), {course_hash(c) for c in again})

        # train.py's CourseFactory: only when the training levels contain neos (--advanced, skills with advanced).
        class Stop(Exception):
            pass

        seen = []

        def factory(cfg, levels, exclude):
            seen.append("single" in exclude)
            raise Stop

        with tempfile.TemporaryDirectory() as d, mock.patch.object(T, "RUNS", Path(d)), \
                mock.patch.object(T, "held_out_courses", tiny), mock.patch.object(T, "CourseFactory", factory), \
                mock.patch.object(NA, "single_neo_hashes", lambda: {"single"}):
            for i, flags in enumerate(({}, {"extra": True}, {"advanced": True}, {"skills": "basic,extra"},
                                       {"skills": "basic,advanced,extra"})):
                with self.assertRaises(Stop):
                    T.train(T.TrainConfig(run=f"r{i}", device="cpu", test_once=True, **flags))
        self.assertEqual(seen, [False, False, True, False, True])


class StrictNeoMetric(unittest.TestCase):
    """neo_done: the bot stood on the neo's takeoff, then on its target (not past either)."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_model(ROOT / "results" / "final_policy.pt", "cpu")

    def test_a_bypass_is_completed_but_not_neo_done(self):
        # Official validation course 1: the planner run of results/neo/assist_details/official2_all_brake_approach.json
        # completed it with an approach plan from the platform before the neo's takeoff straight onto its target.
        d = json.loads((ROOT / "results" / "neo" / "assist_details" / "official2_all_brake_approach.json").read_text())
        res = evaluate_assisted(self.model, CoursePool(official_neo_courses(2)[1:2], "cpu", C.GRID), "cpu",
                                AssistConfig(**d["row"]["config"]), course_offset=1)
        pc = res["per_course"][0]
        self.assertEqual({k: v for k, v in pc.items() if k != "neo_done"}, d["per_course"][1])  # the run, replayed
        self.assertEqual(res["events"], [e for e in d["events"] if e["course"] == 1])
        self.assertEqual((pc["success"], pc["neos"], pc["neos_reached"], pc["neo_done"]), (True, 1, 0, False))
        self.assertEqual({k: summarize(res)[k] for k in ("completed", "completed_neo_done")},
                         {"completed": 1, "completed_neo_done": 0})

    def test_policy_alone_reports_it(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "rows.jsonl"
            NA.main(["--off", "--set", "single", "--family", str(SINGLE_VALIDATION_FAMILY), "--courses", "2",
                     "--checkpoint", str(ROOT / "results" / "neo_policy.pt"), "--device", "cpu", "--out", str(out)])
            row = json.loads(out.read_text().splitlines()[-1])
        self.assertEqual((row["set"], row["split"], row["assist"], row["courses"]), ("single", "validation", False, 2))
        self.assertEqual(list(row)[list(row).index("completed") + 1], "completed_neo_done")
        self.assertEqual((row["completed"], row["completed_neo_done"]), (1, 1))  # the neo policy makes course 0's neo
        self.assertEqual(row["by_level"], {"6": {"courses": 2, "completed": 1, "completed_neo_done": 1}})
        with self.assertRaises(SystemExit):
            NA.main(["--off", "--set", "single", "--family", "2", "--courses", "1", "--device", "cpu"])

    def test_summary(self):
        rec = dict(fell=False, neos=1, neos_reached=1, neos_passed=1, stopped_at_neo=False, stopped_at_kind=None)
        pc = [dict(rec, success=True, neo_done=True, level=6), dict(rec, success=True, neo_done=False, level=7),
              dict(rec, success=False, neo_done=True, level=6), dict(rec, success=False, neo_done=False, level=7)]
        s = summarize({"per_course": pc, "events": [], "plans": {}, "search_seconds": 0.0})
        self.assertEqual(list(s)[:3], ["courses", "completed", "completed_neo_done"])
        self.assertEqual((s["completed"], s["completed_neo_done"]), (2, 1))
        self.assertEqual(s["by_level"], {"6": {"courses": 2, "completed": 1, "completed_neo_done": 1},
                                         "7": {"courses": 2, "completed": 1, "completed_neo_done": 0}})
        self.assertEqual(s["by_neo_count"], {"1": {"courses": 4, "completed": 2, "completed_neo_done": 1}})
        old = summarize({"per_course": [{k: v for k, v in c.items() if k != "neo_done"} for c in pc], "events": [],
                         "plans": {}, "search_seconds": 0.0})  # cached parts of runs from before the metric
        self.assertNotIn("completed_neo_done", old)
        self.assertEqual(old["by_level"]["6"], {"courses": 2, "completed": 1})
        bare = summarize({"per_course": [{k: v for k, v in c.items() if k not in ("neo_done", "level")} for c in pc],
                          "events": [], "plans": {}, "search_seconds": 0.0})  # records without a level: as before
        self.assertNotIn("by_level", bare)
        self.assertEqual(bare["by_neo_count"], {"1": {"courses": 4, "completed": 2}})


if __name__ == "__main__":
    unittest.main()

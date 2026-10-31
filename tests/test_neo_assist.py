"""Neo assist (parkour.neo_assist): off it is train.evaluate; on, its plans land where the search said,
and the offline bot with the planner service plays exactly like the simulator."""
import collections
import dataclasses
import io
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
from parkour.neo_assist import (NEO_LEVERS, OPTIONAL, SINGLE_FRESH_TEST_FAMILY, SINGLE_TEST_FAMILY, SINGLE_VALIDATION_FAMILY,
                                 TEST_FAMILIES, AssistConfig,
                                 Rules, approach_platforms, assist_fields, brake_plan, config_meta, courses_from_file,
                                 evaluate_assisted, jump_kind, merge_results, mixed_courses, neo_level,
                                 official_neo_courses, part_bounds, plan_once, plan_target, planned_takeoffs,
                                 random_neo_courses, request_seed, serve, single_neo_courses, single_neo_hashes,
                                 single_neo_ok, single_neo_types, spec_from_key, specs_from_keys, summarize)
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

    def test_same_with_the_neo_levers(self):
        cfg = AssistConfig(samples=64, iterations=3, jumps="neo", approach=48, retries=1, brake=14, restarts=1,
                           approach_touch=1, neo_samples=96, neo_iterations=4)  # passed to the service as strings
        res, bot = self.check(cfg, single_neo_courses(SINGLE_VALIDATION_FAMILY, n=4))
        self.assertTrue(any(e["found"] and e["restarts"] == 1 for e in res["events"]))  # landed after a restart
        by_course = sorted(res["events"], key=lambda e: e["course"])  # the bot's log: course by course
        self.assertEqual([x["kind"] for x in bot], [e["kind"] for e in by_course])

    def test_same_with_a_planned_start(self):
        # Validation courses 12 and 41 of the single-neo set: from the bots' settled start the final policy's first
        # action is a jump (on 41 the neo); with plan_start the bot asks before it, at tick 0, as the simulator searches.
        cs = single_neo_courses(SINGLE_VALIDATION_FAMILY, n=42)
        cfg = AssistConfig(samples=64, iterations=3, jumps="all", retries=1, brake=14, retry_horizon=48, plan_start=1)
        res, bot = self.check(cfg, [cs[12], cs[41]])
        by_course = sorted(res["events"], key=lambda e: e["course"])  # the bot's log: course by course
        self.assertEqual([(x["kind"], x["takeoff"], x["tick"]) for x in bot],
                         [(e["kind"], e["takeoff"], e["tick"]) for e in by_course])
        self.assertEqual([(e["kind"], e["takeoff"], e["tick"]) for e in by_course if e["tick"] == 0],
                         [("takeoff", 0, 0), ("takeoff", 0, 0)])

    def test_same_on_every_path_from_a_planned_start(self):
        # Single-neo validation courses 0 and 41 (the neo first), 1 and 11 (the neo second) and 12 (third). Weak searches
        # fail often, so the start's failures are played out: with jumps "neo" a takeoff search at the start, its brake
        # and the retry after it; an approach from the start without the takeoff search (platform 0 is not planned);
        # no request at the start; with every jump planned and no brake, retries by the timer from the start search.
        cs = single_neo_courses(SINGLE_VALIDATION_FAMILY, n=42)
        cs = [cs[i] for i in (0, 1, 11, 12, 41)]
        start, played = {}, 0
        for cfg in (AssistConfig(samples=16, iterations=1, jumps="neo", approach=48, retries=1, brake=14, plan_start=1),
                    AssistConfig(samples=16, iterations=1, jumps="all", retries=2, retry_every=3, plan_start=1)):
            res, bot = self.check(cfg, cs, played=False)  # the first setting lands no plan: only its failures
            played += sum(x["played"] for x in bot)
            by_course = sorted(res["events"], key=lambda e: e["course"])
            self.assertEqual([(x["kind"], x["takeoff"], x["tick"], x["braked"]) for x in bot],
                             [(e["kind"], e["takeoff"], e["tick"], e["braked"]) for e in by_course])
            start[cfg.jumps] = [[(e["kind"], e["takeoff"], e["tick"]) for e in res["events"] if e["course"] == i][:3]
                                for i in range(len(cs))]
        neo, every = start["neo"], start["all"]
        self.assertEqual(neo[0][:2], [("takeoff", 0, 0), ("retry", 0, 14)])  # braked from the start, retried after it
        self.assertEqual([neo[i][0] for i in (1, 2)], [("approach", 1, 0)] * 2)  # no takeoff search on platform 0
        self.assertGreater(neo[3][0][2], 0)  # nothing planned from the start
        self.assertEqual(every[1], [("takeoff", 0, 0), ("retry", 0, 3), ("retry", 0, 6)])  # by the timer
        self.assertGreater(played, 0)

    def test_same_with_the_corner_limit(self):
        # The planner service gets corner_limit with the rest of the config and drops the same landings: on
        # single-neo validation course 81 (alone in its file) the search's landing from takeoff 5, an ordinary jump,
        # crosses a corner, so the bot brakes and retries there; a service without the limit would play it instead.
        cfg = AssistConfig(samples=64, iterations=3, jumps="all", retries=1, brake=14, retry_horizon=48, plan_start=1,
                           corner_limit=0.25)
        res, bot = self.check(cfg, single_neo_courses(SINGLE_VALIDATION_FAMILY, n=82)[81:])
        self.assertTrue(res["events"] and all("corner_dropped" in e for e in res["events"]))
        self.assertEqual([(e["takeoff"], e["found"], e["braked"]) for e in res["events"] if e["corner_dropped"]],
                         [(5, False, True)])
        self.assertEqual([(x["kind"], x["takeoff"], x["tick"], x["braked"]) for x in bot],
                         [(e["kind"], e["takeoff"], e["tick"], e["braked"]) for e in res["events"]])

    def check(self, cfg, cs, played=True):
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
        if played:
            self.assertGreater(sum(x["played"] for x in bot), 0)
        return res, bot


class PlannedStart(unittest.TestCase):
    """plan_start: the searches also run on the start state. From the simulator's start (in the air) nothing changes;
    from the bots' settled start the first jump is searched before the first action."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_model(ROOT / "results" / "final_policy.pt", "cpu")
        cs = single_neo_courses(SINGLE_VALIDATION_FAMILY, n=42)
        cls.courses = [cs[12], cs[41], cs[0], cs[1]]  # 12, 41: the policy jumps at once from the settled start
        cls.cfg = AssistConfig(samples=64, iterations=3, jumps="all", retries=1, brake=14, retry_horizon=48)

    def run_with(self, plan_start, settle):
        return evaluate_assisted(self.model, CoursePool(self.courses, "cpu", C.GRID), "cpu",
                                 dataclasses.replace(self.cfg, plan_start=plan_start), settle=settle)

    def test_the_simulators_own_start_is_unchanged(self):
        before, now = self.run_with(0, 0), self.run_with(1, 0)
        for key in ("per_course", "events", "plans"):
            self.assertEqual(now[key], before[key], key)
        self.assertTrue(all(e["tick"] > 0 for e in now["events"]))  # it lands on the start block before any search
        off = dataclasses.replace(self.cfg, enabled=False, plan_start=1)
        pool = CoursePool(self.courses, "cpu", C.GRID)
        self.assertEqual(evaluate_assisted(self.model, pool, "cpu", off, settle=10)["per_course"],
                         evaluate_assisted(self.model, pool, "cpu", dataclasses.replace(off, plan_start=0),
                                           settle=10)["per_course"])

    def test_a_settled_start_searches_its_first_jump(self):
        from parkour.live import SETTLE_TICKS
        before, now = self.run_with(0, SETTLE_TICKS), self.run_with(1, SETTLE_TICKS)
        first = lambda res, i: [e for e in res["events"] if e["course"] == i][0]  # noqa: E731
        for i in range(len(self.courses)):  # every course: a search from the start state, before any action
            self.assertEqual({k: first(now, i)[k] for k in ("kind", "takeoff", "tick", "after_plan")},
                             {"kind": "takeoff", "takeoff": 0, "tick": 0, "after_plan": False})
        for i in (0, 1):  # without it the first action was the policy's jump, never searched
            self.assertFalse(any(e["takeoff"] == 0 for e in before["events"] if e["course"] == i))
        for i in (2, 3):  # a first action without a jump: the same search one tick later
            self.assertEqual((first(before, i)["takeoff"], first(before, i)["tick"]), (0, 1))
        self.assertEqual(now["plans"]["landed_as_planned"], now["plans"]["played"])

    def test_without_plans_only_the_start_counts_differently(self):
        # Course 41: its neo is the first jump and the policy jumps at once. With searches that never land (no
        # rounds), both runs play the policy alone; with plan_start the start on the neo's takeoff counts as reached,
        # as the state after any tick does.
        from parkour.live import SETTLE_TICKS
        self.assertEqual(neo_takeoffs(self.courses[1]), [0])
        pool = CoursePool(self.courses[1:2], "cpu", C.GRID)
        cfg = AssistConfig(iterations=0, jumps="all")
        res = [evaluate_assisted(self.model, pool, "cpu", dataclasses.replace(cfg, plan_start=p), settle=SETTLE_TICKS)
               for p in (0, 1)]
        self.assertEqual([[(e["kind"], e["takeoff"], e["tick"], e["found"]) for e in r["events"] if e["takeoff"] == 0]
                          for r in res], [[], [("takeoff", 0, 0, False)]])
        drop = lambda pc: {k: v for k, v in pc.items() if k not in ("neos_reached", "neos_passed")}  # noqa: E731
        self.assertEqual(drop(res[1]["per_course"][0]), drop(res[0]["per_course"][0]))
        self.assertEqual([r["per_course"][0]["neos_reached"] for r in res], [0, 1])


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
        fresh = [single_neo_courses(f) for f in (SINGLE_FRESH_TEST_FAMILY, NA.SINGLE_FRESH_TEST_FAMILY_2)]  # the neo skill's
        self.assertEqual(hashes, {course_hash(c) for cs in list(self.sets.values()) + fresh for c in cs})
        self.assertEqual(len(hashes), 1280)
        self.assertIn(SINGLE_FRESH_TEST_FAMILY, TEST_FAMILIES)
        self.assertIn(NA.SINGLE_FRESH_TEST_FAMILY_2, TEST_FAMILIES)
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
        cls.neo_model = load_model(ROOT / "results" / "neo_policy.pt", "cpu")

    def test_the_rule(self):
        # The values the checkpoint took (it only grows): the neo's takeoff, then its target, each before a later one.
        self.assertTrue(NA.neo_done([3], {0, 1, 2, 3, 4, 10}))
        self.assertFalse(NA.neo_done([3], {0, 1, 2, 4, 10}))  # jumped past the takeoff
        self.assertFalse(NA.neo_done([3], {0, 1, 2, 3, 5, 10}))  # past the target (on it later, if at all, from 5 on)
        self.assertFalse(NA.neo_done([3, 6], {0, 1, 2, 3, 4, 5, 6, 8, 9, 10}))  # every neo
        self.assertTrue(NA.neo_done([9], set(range(11))))
        self.assertTrue(NA.neo_done([], {0}))

    def test_policy_alone_past_the_target_onto_the_goal_and_a_fall_after(self):
        cs = single_neo_courses(SINGLE_VALIDATION_FAMILY, n=162)
        self.assertEqual(neo_takeoffs(cs[60]), [cs[60].n - 2])
        for i, want in ((161, (True, False)),  # stood on takeoff 6, then landed on 8, past its target 7
                        (60, (True, True)),  # the target is the goal: the checkpoint takes it only in the last tick
                        (34, (False, True))):  # did the neo from 3, fell after 7
            pc = evaluate_assisted(self.neo_model, CoursePool(cs[i:i + 1], "cpu", C.GRID), "cpu",
                                   AssistConfig(enabled=False))["per_course"][0]
            self.assertEqual((pc["success"], pc["neo_done"]), want, i)

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
        self.assertFalse((set(NEO_LEVERS) | {"plan_start"}) & set(row["config"]))  # off: the config as before them
        self.assertNotIn("settle", row)  # the simulator's own start: the row as before --settle existed
        with tempfile.TemporaryDirectory() as d:  # the bots' start: the row says so, and the run started that way
            out, details = Path(d) / "rows.jsonl", Path(d) / "details.json"
            NA.main(["--off", "--set", "single", "--family", str(SINGLE_VALIDATION_FAMILY), "--courses", "2", "--settle",
                     "10", "--checkpoint", str(ROOT / "results" / "neo_policy.pt"), "--device", "cpu", "--out", str(out),
                     "--details", str(details)])
            self.assertEqual(json.loads(out.read_text().splitlines()[-1])["settle"], 10)
            pool = CoursePool(single_neo_courses(SINGLE_VALIDATION_FAMILY, n=2), "cpu", C.GRID)
            settled = evaluate_assisted(self.neo_model, pool, "cpu", AssistConfig(enabled=False), settle=10)["per_course"]
            plain = evaluate_assisted(self.neo_model, pool, "cpu", AssistConfig(enabled=False))["per_course"]
            self.assertEqual(json.loads(details.read_text())["per_course"], settled)
            self.assertNotEqual([c["ticks"] for c in settled], [c["ticks"] for c in plain])
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
        self.assertNotIn("by_level", old)  # their summary as before, so a rerun of the split run republishes its row
        self.assertEqual(old["by_neo_count"], {"1": {"courses": 4, "completed": 2}})
        bare = summarize({"per_course": [{k: v for k, v in c.items() if k not in ("neo_done", "level")} for c in pc],
                          "events": [], "plans": {}, "search_seconds": 0.0})  # records without a level: as before
        self.assertNotIn("by_level", bare)
        self.assertEqual(bare["by_neo_count"], {"1": {"courses": 4, "completed": 2}})


def plan_once_4762c22(model, pool, rules, arrival, kind, attempt, cfg, horizon, device):
    """``neo_assist.plan_once`` of 4762c22, the commit before the neo levers (its request_seed inline), searching with
    ``search_4762c22``; and 0 restarts, as plan_once returns them now."""
    pr = NA.policy_prior(model, pool, [arrival], device, horizon) if cfg.prior > 0 else None
    seed = (cfg.seed * 1_000_003 + arrival.get("seed_course", arrival["course"]) * 4099 + arrival["cp"] * 67
            + {"takeoff": 0, "retry": 1, "approach": 2}[kind] * 7 + attempt)
    f, r = search_4762c22(pool, rules, [arrival], device, cfg, seed, pr, horizon)
    return (f[0][0] if f[0] else None), r[0], 0


def search_4762c22(pool, rules, arrivals, device, cfg, seed, prior=None, horizon=ND.HORIZON):
    """``neo_assist.search`` of 4762c22 (git show 4762c22:parkour/neo_assist.py), verbatim but for the module its
    helpers are named by."""
    gen = torch.Generator(device=device).manual_seed(seed)
    A, S, T = len(arrivals), cfg.samples, horizon
    steps = torch.tensor(NA.YAW_STEPS_DEG, dtype=torch.float64, device=device) * (math.pi / 180)
    init = NA._state_rows(arrivals, device)
    course = torch.tensor([a["course"] for a in arrivals], device=device)
    target = torch.tensor([a.get("target", a["cp"] + 1) for a in arrivals], device=device)
    tops = torch.stack([pool.top[a["course"], a["cp"]:a.get("target", a["cp"] + 1) + 1].min() for a in arrivals])
    goal = pool.centre[course, target]
    p = {"move": torch.tensor([0.05, 0.15, 0.8], device=device).repeat(A, T, 1),
         "strafe": torch.full((A, T, 3), 1 / 3, device=device),
         "jump": torch.tensor([0.7, 0.3], device=device).repeat(A, T, 1),
         "sprint": torch.tensor([0.2, 0.8], device=device).repeat(A, T, 1),
         "yaw": torch.tensor([0.05, 0.1, 0.15, 0.4, 0.15, 0.1, 0.05], device=device).repeat(A, T, 1)}
    if prior is not None and cfg.prior > 0:
        for key in p:
            p[key] = (cfg.prior * prior[key] + (1 - cfg.prior) * p[key]).clamp(min=0.01)
            p[key] = p[key] / p[key].sum(-1, keepdim=True)
    found = [[] for _ in range(A)]
    seen = [set() for _ in range(A)]
    rounds = [0] * A
    k_elite = max(1, int(S * NA.ELITE))
    active = list(range(A))
    for it in range(cfg.iterations):
        act_idx = torch.tensor(active, device=device)
        a_n = len(active)
        n = a_n * S
        rep = lambda x: x[act_idx].repeat_interleave(S, dim=0)  # noqa: E731
        acts = {}
        for key, v in p.items():
            flat = v[act_idx].reshape(a_n * T, -1)
            draw = torch.multinomial(flat, S, replacement=True, generator=gen)
            acts[key] = draw.reshape(a_n, T, S).permute(0, 2, 1).reshape(n, T)
        s = physics.PlayerState(rep(init.pos), rep(init.vel), rep(init.yaw), rep(init.on_ground), rep(init.jump_ticks),
                                rep(init.collided_h), rep(init.collided_v))
        cidx, tgt, top_r, goal_r = rep(course), rep(target), rep(tops), rep(goal)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        landed = torch.zeros(n, dtype=torch.bool, device=device)
        land_tick = torch.full((n,), T, dtype=torch.int64, device=device)
        closest = torch.full((n,), 1e9, dtype=torch.float64, device=device)
        for t in range(T):
            y = s.yaw + steps[acts["yaw"][:, t]]
            s.yaw = torch.remainder(y + math.pi, 2 * math.pi) - math.pi
            a_t = torch.stack([acts[k][:, t] for k in NA.HEADS], -1)
            s = physics.simulate(pool.world, cidx, s, NA.action_to_controls(a_t))
            alive &= ~rules.on_obstacle(s, cidx) & (s.pos[:, 1] > top_r - 1.0)
            now = NA.standing_on(pool, s, cidx).gather(1, tgt[:, None]).squeeze(1) & alive & ~landed
            land_tick = torch.where(now, torch.full_like(land_tick, t + 1), land_tick)
            landed |= now
            closest = torch.where(alive & ~landed, torch.minimum(closest, torch.linalg.norm(s.pos - goal_r, dim=-1)),
                                  closest)
        L = landed.cpu().numpy()
        if L.any():
            lt = land_tick.cpu().numpy()
            stack = torch.stack([acts[k] for k in NA.HEADS], -1).cpu().numpy()
            for j in np.flatnonzero(L):
                a = active[j // S]
                if len(found[a]) >= cfg.keep:
                    continue
                seq = stack[j, :lt[j]]
                b = seq.tobytes()
                if b not in seen[a]:
                    seen[a].add(b)
                    found[a].append(seq)
        for a in active:
            rounds[a] = it + 1
        score = (landed.double() * 100.0 - torch.where(landed, torch.zeros_like(closest), closest)).reshape(a_n, S)
        top = torch.topk(score, k_elite, dim=1).indices
        for key, v in p.items():
            sel = acts[key].reshape(a_n, S, T).gather(1, top[:, :, None].expand(-1, -1, T))
            freq = torch.nn.functional.one_hot(sel, v.shape[-1]).float().mean(1)
            upd = (0.7 * freq + 0.3 * v[act_idx]).clamp(min=0.01)
            v[act_idx] = upd / upd.sum(-1, keepdim=True)
        active = [a for a in active if len(found[a]) < cfg.keep]
        if not active:
            break
    return found, rounds


class NeoLevers(unittest.TestCase):
    """restarts, neo_samples, neo_iterations, approach_touch: only at neos, off at 0, the same in the bots' service."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_model(ROOT / "results" / "final_policy.pt", "cpu")

    def test_off_the_planner_plays_as_before(self):
        cs = single_neo_courses(SINGLE_VALIDATION_FAMILY, n=4)  # (also compared with a run of 4762c22's code itself)
        cfg = AssistConfig(samples=32, iterations=2, jumps="neo", approach=48, retries=1, brake=14, retry_horizon=48)
        self.assertFalse(any(getattr(cfg, k) for k in NEO_LEVERS))
        batched = dataclasses.replace(cfg, batch=2)  # searches batched per tick: course 2 twice, searched together
        for c, courses in ((cfg, cs), (batched, [cs[2], cs[2]])):
            now = evaluate_assisted(self.model, CoursePool(courses, "cpu", C.GRID), "cpu", c)
            with mock.patch.object(NA, "plan_once", plan_once_4762c22), \
                    mock.patch.object(NA, "search", wraps=search_4762c22) as old:
                ref = evaluate_assisted(self.model, CoursePool(courses, "cpu", C.GRID), "cpu", c)
            self.assertEqual(max((len(x.args[2]) for x in old.call_args_list), default=1), c.batch)
            for key in ("per_course", "events", "plans"):
                self.assertEqual(now[key], ref[key], (key, c.batch))
            self.assertEqual({e["kind"] for e in now["events"]}, {"approach", "takeoff", "retry"}, c.batch)
            self.assertGreater(now["plans"]["brakes"], 0, c.batch)
            self.assertNotIn("restarts", now["events"][0])
            for pc in now["per_course"]:  # standing on the takeoff counts it as reached
                self.assertTrue(not pc["neo_done"] or pc["neos_reached"] == pc["neos"])
        self.assertEqual(config_meta(cfg), {k: v for k, v in dataclasses.asdict(cfg).items() if k not in OPTIONAL})
        self.assertNotIn("plan_start", config_meta(cfg))
        self.assertEqual(config_meta(dataclasses.replace(cfg, plan_start=1))["plan_start"], 1)

    def test_off_v1s_setting_replays_its_stored_run(self):
        # The planner track's V1 setting (jumps all, retries 1, brake 14, retry_horizon 48), run on CPU by earlier code.
        d = json.loads((ROOT / "results" / "neo" / "assist_details" / "official2_all_brake.json").read_text())
        res = evaluate_assisted(self.model, CoursePool(official_neo_courses(2)[:1], "cpu", C.GRID), "cpu",
                                AssistConfig(**d["row"]["config"]))
        self.assertEqual([{k: v for k, v in pc.items() if k != "neo_done"} for pc in res["per_course"]],
                         d["per_course"][:1])
        self.assertEqual(res["events"], [e for e in d["events"] if e["course"] == 0])

    def test_touch_gates_the_landing(self):
        # From the bot's state on platform 2 of course 0, one search round lands on platform 3; the same draws land
        # nothing when they must first stand on the last platform (a touch platform they cannot reach).
        c = single_neo_courses(SINGLE_VALIDATION_FAMILY, n=1)[0]
        pool = CoursePool([c], "cpu", C.GRID)
        real, arrivals = NA.plan_once, []

        class Stop(Exception):
            pass

        def grab(model, pool_, rules, arrival, kind, attempt, cfg, horizon, device):
            if arrival["cp"] == 2:
                arrivals.append(arrival)
                raise Stop
            return real(model, pool_, rules, arrival, kind, attempt, cfg, horizon, device)

        with mock.patch.object(NA, "plan_once", grab), self.assertRaises(Stop):
            evaluate_assisted(self.model, pool, "cpu", AssistConfig(jumps="all", retries=1, brake=14, retry_horizon=48))
        self.assertEqual((arrivals[0]["target"], neo_takeoffs(c)), (3, [0]))  # an ordinary jump
        one, seed, rules = AssistConfig(iterations=1), request_seed(0, 0, 2, "takeoff", 0), Rules(pool)
        self.assertTrue(NA.search(pool, rules, arrivals, "cpu", one, seed)[0][0])
        self.assertFalse(NA.search(pool, rules, [dict(arrivals[0], touch=c.n - 1)], "cpu", one, seed)[0][0])

    def test_touch_turns_the_bypass_into_the_neo(self):
        # Official validation course 1, which the official2_all_brake_approach run completed with an approach plan
        # straight past the neo's takeoff (StrictNeoMetric): with approach_touch the plan from that state stands on it.
        d = json.loads((ROOT / "results" / "neo" / "assist_details" / "official2_all_brake_approach.json").read_text())
        cfg = dataclasses.replace(AssistConfig(**d["row"]["config"]), approach_touch=1)
        res = evaluate_assisted(self.model, CoursePool(official_neo_courses(2)[1:2], "cpu", C.GRID), "cpu", cfg,
                                course_offset=1)
        stored = [e for e in d["events"] if e["course"] == 1]
        self.assertEqual(res["events"][:-1], stored[:-1])  # the same play up to the approach before the neo
        last = res["events"][-1]
        self.assertEqual((last["kind"], last["takeoff"], last["tick"], last["found"]),
                         ("approach", 9, stored[-1]["tick"], True))
        # Steered over the takeoff (distance + 1000 until it stands there), the search lands sooner than the search
        # without the touch rule did on its bypass (7 rounds against 9; the gate without the steering also takes 9).
        self.assertLess(last["rounds"], stored[-1]["rounds"])
        pc = res["per_course"][0]
        self.assertEqual((pc["success"], pc["neos_reached"], pc["neo_done"]), (True, 1, True))

    def test_levers_act_at_neos_only(self):
        c = single_neo_courses(SINGLE_VALIDATION_FAMILY, n=3)[2]
        k = neo_takeoffs(c)[0]
        self.assertEqual(k, 2)
        pool = CoursePool([c], "cpu", C.GRID)
        calls = []

        def search(pool_, rules, arrivals, device, cfg, seed, prior=None, horizon=32):  # lands on the calls in hit
            calls.append((seed, cfg.samples, cfg.iterations, arrivals[0].get("touch")))
            return ([[np.zeros((3, 5), dtype=np.int64)]] if len(calls) in hit else [[]]), [cfg.iterations]

        def plan(cp, kind, attempt, cfg):
            calls.clear()
            arrival = {"course": 0, "cp": cp, "seed_course": 7}
            seq, rounds, restarts = plan_once(None, pool, Rules(pool), arrival, kind, attempt, cfg, 32, "cpu")
            return seq is not None, rounds, restarts

        cfg = AssistConfig(samples=64, iterations=3, restarts=2, neo_samples=256, neo_iterations=5, approach_touch=1)
        with mock.patch.object(NA, "search", search):
            hit = {2}  # the neo's takeoff: found by the first restart, with the next seed
            self.assertEqual(plan(k, "takeoff", 0, cfg), (True, 10, 1))
            s = request_seed(0, 7, k, "takeoff", 0)
            self.assertEqual(calls, [(s, 256, 5, None), (s + 1_000_000_007, 256, 5, None)])
            hit = set()  # the approach before it must stand on the takeoff first; 1 + 2 searches without a landing
            self.assertEqual(plan(k - 1, "approach", 0, cfg), (False, 15, 2))
            s = request_seed(0, 7, k - 1, "approach", 0)
            self.assertEqual(calls, [(s + r * 1_000_000_007, 256, 5, k) for r in range(3)])
            self.assertEqual(plan(k, "retry", 1, cfg), (False, 15, 2))  # its retries
            self.assertEqual(calls[0], (request_seed(0, 7, k, "retry", 1), 256, 5, None))
            for cp, kind in ((k + 1, "takeoff"), (k + 1, "approach"), (k - 1, "takeoff")):  # not a neo's: as before
                self.assertEqual(plan(cp, kind, 0, cfg), (False, 3, 0))
                self.assertEqual(calls, [(request_seed(0, 7, cp, kind, 0), 64, 3, None)])
            off = AssistConfig(samples=64, iterations=3)
            self.assertEqual(plan(k, "takeoff", 0, off), (False, 3, 0))
            self.assertEqual(calls, [(request_seed(0, 7, k, "takeoff", 0), 64, 3, None)])
        with self.assertRaises(ValueError):  # batched searches cannot apply them per request
            evaluate_assisted(self.model, pool, "cpu", dataclasses.replace(cfg, batch=4))

    def test_service_plans_as_the_simulator_with_restarts_and_touch(self):
        cs = single_neo_courses(SINGLE_VALIDATION_FAMILY, n=5)  # course 4: the approach plan lands after a restart
        cfg = AssistConfig(samples=64, iterations=3, jumps="neo", approach=48, restarts=1, approach_touch=1)
        calls = []
        real = NA.plan_once

        def spy(model, pool, rules, arrival, kind, attempt, cfg_, horizon, device):
            out = real(model, pool, rules, arrival, kind, attempt, cfg_, horizon, device)
            calls.append((arrival, kind, attempt, out))
            return out

        with mock.patch.object(NA, "plan_once", spy):
            res = evaluate_assisted(self.model, CoursePool(cs[4:], "cpu", C.GRID), "cpu", cfg, course_offset=4)
        approaches = [x for x in calls if x[1] == "approach"]
        self.assertTrue(any(out[0] is not None and out[2] == 1 for _, _, _, out in approaches))
        self.assertEqual([e.get("restarts") for e in res["events"]], [out[2] for _, _, _, out in calls])
        lines = [json.dumps({"index": 4, "cp": a["cp"], "target": a["target"], "kind": kind, "attempt": attempt,
                             "state": {"pos": a["pos"], "vel": a["vel"], "yaw": a["yaw"], "onGround": a["on_ground"],
                                       "jumpTicks": a["jump_ticks"], "collidedH": a["collided_h"]},
                             "lastAction": a["last_action"]}) for a, kind, attempt, _ in calls]
        out = io.StringIO()
        serve(self.model, cs, cfg, "cpu", inp=io.StringIO("\n".join(lines) + "\n"), out=out)
        answers = [json.loads(x) for x in out.getvalue().splitlines()]
        self.assertEqual(len(answers), len(calls))
        for (_, _, _, (seq, rounds, restarts)), ans in zip(calls, answers):
            self.assertEqual(ans["plan"], None if seq is None else seq.tolist())
            self.assertEqual((ans["rounds"], ans["restarts"]), (rounds, restarts))
        # The approach plan stood on the neo's takeoff before it landed on the target: the neo is done.
        self.assertEqual(res["plans"]["landed_as_planned"], res["plans"]["played"])
        self.assertTrue(res["per_course"][0]["neo_done"])


if __name__ == "__main__":
    unittest.main()

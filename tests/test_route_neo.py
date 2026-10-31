"""Neo routing (parkour.route_neo): the router's blocked-path choices planned by the neo planner. With the setting
off (or "keep") it is route_env.evaluate_routed; on, its plans land where the search said, and the offline bot
(bridge/router.js, bridge/assist.js RouteAssist) with the planner service plays exactly like the simulator."""
import dataclasses
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from parkour import courses as C
from parkour.env import CoursePool
from parkour.live import load_model
from parkour.neo_assist import SINGLE_VALIDATION_FAMILY, AssistConfig, courses_from_file, serve, single_neo_courses
from parkour.neo_dagger import neo_takeoffs
from parkour.paths import BRIDGE, NODE, ROOT
from parkour.planner import PlannerConfig
from parkour.route_env import RouteConfig, RouteTables, Scene, evaluate_routed
from parkour.route_neo import check_config, evaluate_routed_neo, merge_parts, part_bounds
from parkour.route_policy import load_router
from parkour.route_train import router_choosers

ROUTER = ROOT / "results" / "routing" / "router_policy.pt"
FILL = torch.load(ROUTER, map_location="cpu", weights_only=False)["config"]["fill"]


def neo_courses(idx):
    cs = single_neo_courses(SINGLE_VALIDATION_FAMILY, n=max(idx) + 1)
    return [cs[i] for i in idx]


def standard_courses(idx):
    """Courses of the router's standard validation set (family 2, 300 per level, route_eval's order)."""
    from parkour.route_neo import course_set
    cs = course_set("standard", 2)
    return [cs[i] for i in idx]


class Shared(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model(ROOT / "results" / "final_policy.pt", "cpu")
        cls.net = load_router(ROUTER, "cpu")
        cls.choosers = router_choosers(cls.net, True)  # (a tuple: functions on the class would become methods)

    def routed(self, cs, blocked, cfg=None, settle=0, **kw):
        pool = CoursePool(cs, "cpu", C.GRID)
        choose, fill = self.choosers
        with torch.no_grad():
            if cfg is None:
                return evaluate_routed(self.model, pool, RouteTables(pool), choose, fill,
                                       rcfg=RouteConfig(fill=FILL, blocked=blocked), settle=settle, record=True, **kw)
            return evaluate_routed_neo(self.model, pool, RouteTables(pool), choose, fill,
                                       rcfg=RouteConfig(fill=FILL, blocked=blocked), acfg=cfg, settle=settle, record=True, **kw)


class RouteNeoSimulator(Shared):
    def test_the_default_is_the_planners_rule(self):
        self.assertEqual(RouteConfig().blocked, "drop")
        with self.assertRaises(ValueError):
            RouteConfig(blocked="planned")
        with self.assertRaises(ValueError):  # the planner is route_neo's
            self.routed(neo_courses([0]), "plan")
        self.assertNotIn("blocked", dataclasses.asdict(PlannerConfig()))  # the greedy planner and planner.js: unchanged

    def test_drop_and_keep_are_evaluate_routed(self):
        from parkour.train import OBSTACLE_VALIDATION_FAMILY, held_out_courses
        cs = neo_courses([0, 5, 17]) + held_out_courses(C.load_levels(), 1, 10, obstacle_prob=0.5,
                                                        family=OBSTACLE_VALIDATION_FAMILY)
        outcomes = {}
        for blocked in ("drop", "keep"):
            ref = self.routed(cs, blocked)
            mine = self.routed(cs, blocked, AssistConfig())
            self.assertEqual([c["success"] for c in mine["per_course"]], ref["per_course"]["success"], blocked)
            self.assertEqual([c["ticks"] for c in mine["per_course"]], ref["per_course"]["ticks"], blocked)
            self.assertEqual(mine["decisions"], ref["events"], blocked)
            self.assertEqual(mine["events"], [])  # no planner
            outcomes[blocked] = ref["events"]
        self.assertNotEqual(outcomes["drop"][:3], outcomes["keep"][:3])  # the neos' landings are chosen now

    def test_keep_makes_the_neos_landing_selectable_and_blocked(self):
        cs = neo_courses(range(12))
        pool = CoursePool(cs, "cpu", C.GRID)
        tables = RouteTables(pool)
        ks = [neo_takeoffs(c)[0] for c in cs]
        cells = torch.tensor([[int(c.platforms[k][0]), int(c.platforms[k][4]), int(c.platforms[k][2])] for c, k in zip(cs, ks)])
        pos = torch.stack((cells[:, 0] + 0.5, cells[:, 1] + 1.0, cells[:, 2] + 0.5), -1).to(torch.float64)
        course = torch.arange(len(cs))
        for keep in (False, True):
            sc = Scene(pool, tables, course, pos, cells, PlannerConfig(), keep_blocked=keep)
            here = sc.token_of(torch.tensor(ks))
            target = sc.token_of(torch.tensor(ks) + 1)
            excluded = torch.zeros(len(cs), sc.kt, dtype=torch.bool)
            excluded[course, here] = True
            ok, *_ = sc.options(here, excluded)
            self.assertEqual(ok[course, target].tolist(), [keep] * len(cs))  # dropped by the straight-path rule
            self.assertEqual(sc.blocked[course, target].tolist(), [keep] * len(cs))
            self.assertEqual(sc.walled[course, target].tolist(), [keep] * len(cs))  # by the neo's wall
            self.assertTrue(bool((sc.blocked <= ok).all()) and bool((sc.walled <= sc.blocked).all()))
        with self.assertRaises(ValueError):
            Scene(pool, tables, course, pos, cells, PlannerConfig(clear_path=False), keep_blocked=True)

    def test_a_path_blocked_only_by_platforms_is_no_neo(self):
        # Standard validation courses (family 2) 311, 818 and 1346: the line to the next platform passes through another
        # platform (a jump past it), not a wall. It is selectable with "keep" and "plan", but only a wall makes a neo: on
        # these courses "plan" searches nothing and plays as "keep" (the planner's plans touched the platform in the way).
        cs = standard_courses([311, 818, 1346])
        ks = [8, 4, 6]
        pool = CoursePool(cs, "cpu", C.GRID)
        cells = torch.tensor([[int(c.platforms[k][0]), int(c.platforms[k][4]), int(c.platforms[k][2])] for c, k in zip(cs, ks)])
        pos = torch.stack((cells[:, 0] + 0.5, cells[:, 1] + 1.0, cells[:, 2] + 0.5), -1).to(torch.float64)
        course = torch.arange(len(cs))
        sc = Scene(pool, RouteTables(pool), course, pos, cells, PlannerConfig(), keep_blocked=True)
        here, target = sc.token_of(torch.tensor(ks)), sc.token_of(torch.tensor(ks) + 1)
        excluded = torch.zeros(len(cs), sc.kt, dtype=torch.bool)
        excluded[course, here] = True
        ok, *_ = sc.options(here, excluded)
        self.assertEqual((ok[course, target].tolist(), sc.blocked[course, target].tolist(), sc.walled[course, target].tolist()),
                         ([True] * 3, [True] * 3, [False] * 3))
        keep = self.routed(cs, "keep")
        plan = self.routed(cs, "plan", AssistConfig(samples=64, iterations=3))
        self.assertEqual(plan["events"], [])
        self.assertEqual([c["success"] for c in plan["per_course"]], keep["per_course"]["success"])
        self.assertEqual(plan["decisions"], keep["events"])

    def test_plans_land_as_searched_and_only_the_neos_are_planned(self):
        cs = neo_courses([0, 1, 2, 3, 41])
        res = self.routed(cs, "plan", AssistConfig(samples=256, iterations=8, restarts=1))
        self.assertEqual(sorted((e["course"], e["kind"], e["takeoff"], e["target"]) for e in res["events"]),
                         [(i, "takeoff", neo_takeoffs(c)[0], neo_takeoffs(c)[0] + 1) for i, c in enumerate(cs)])
        self.assertGreater(res["plans"]["played"], 0)
        self.assertEqual(res["plans"]["landed_as_planned"], res["plans"]["played"])
        self.assertTrue(all(c["neo_done"] for c in res["per_course"] if c["success"]))
        self.assertGreaterEqual(sum(c["success"] for c in res["per_course"]), 4)

    def test_plan_on_plans_the_jumps_after_a_planned_landing(self):
        cs = neo_courses([1, 2, 3, 11])
        res = self.routed(cs, "plan", AssistConfig(samples=256, iterations=8, restarts=1, plan_on=1))
        for i, c in enumerate(cs):
            k = neo_takeoffs(c)[0]
            ev = [e for e in res["events"] if e["course"] == i]
            self.assertEqual((ev[0]["takeoff"], ev[0]["target"]), (k, k + 1))  # the neo first, as without plan_on
            for a, b in zip(ev, ev[1:]):  # then each landing's next choice: a chain of planned jumps
                if a["found"] and b["kind"] == "takeoff":
                    self.assertEqual(b["takeoff"], a["target"])
            self.assertGreater(len(ev), 1)
        self.assertEqual(res["plans"]["landed_as_planned"], res["plans"]["played"])
        without = self.routed(cs, "plan", AssistConfig(samples=256, iterations=8, restarts=1))
        self.assertEqual(sorted((e["course"], e["takeoff"]) for e in without["events"]),
                         [(i, neo_takeoffs(c)[0]) for i, c in enumerate(cs)])
        from parkour.neo_assist import evaluate_assisted
        with self.assertRaises(ValueError):  # the course-list planner plans every jump with jumps "all" instead
            evaluate_assisted(self.model, CoursePool(cs[:1], "cpu", C.GRID), "cpu", AssistConfig(plan_on=1))

    def test_failed_searches_brake_and_retry(self):
        cs = neo_courses([0, 1, 2, 3, 4, 5])
        res = self.routed(cs, "plan", AssistConfig(samples=16, iterations=1, retries=1, brake=14, retry_horizon=48))
        braked = [e for e in res["events"] if e["braked"]]
        self.assertGreater(len(braked), 0)
        for e in braked:  # a retry from the same takeoff right after the brake's 14 ticks
            self.assertTrue(any(r["kind"] == "retry" and r["course"] == e["course"] and r["takeoff"] == e["takeoff"]
                                and r["tick"] == e["tick"] + 14 for r in res["events"]))
        self.assertTrue(all(sum(1 for e in res["events"] if e["course"] == i) <= 2 for i in range(len(cs))))
        self.assertEqual(res["plans"]["landed_as_planned"], res["plans"]["played"])

    def test_a_run_in_parts_is_the_whole_run(self):
        cs = neo_courses([0, 1, 2, 3, 4, 5])
        cfg = AssistConfig(samples=64, iterations=3, retries=1, brake=14, retry_horizon=48, restarts=1)
        whole = self.routed(cs, "plan", cfg)
        parts = [self.routed(cs[lo:hi], "plan", cfg, course_offset=lo) for lo, hi in (part_bounds(6, 1, 2), part_bounds(6, 2, 2))]
        merged = merge_parts(parts)
        self.assertEqual(merged["per_course"], whole["per_course"])
        self.assertEqual(merged["decisions"], whole["decisions"])
        key = lambda e: (e["course"], e["tick"], e["kind"])  # noqa: E731
        self.assertEqual(sorted(merged["events"], key=key), sorted(whole["events"], key=key))
        self.assertEqual(merged["plans"], whole["plans"])
        self.assertTrue(any(e["braked"] for e in whole["events"]) and any(e["found"] for e in whole["events"]))

    def test_the_command_line_in_parts_writes_the_rows_of_one_run(self):
        from parkour.route_neo import main
        cfg = json.dumps({"samples": 64, "iterations": 3, "retries": 1, "brake": 14, "retry_horizon": 48, "restarts": 1})
        with tempfile.TemporaryDirectory() as d:
            common = ["--set", "single", "--family", str(SINGLE_VALIDATION_FAMILY), "--courses", "4", "--device", "cpu",
                      "--assist", cfg, "--out", os.path.join(d, "rows.jsonl"), "--parts-dir", d, "--tag", "t"]
            whole = main(common + ["--name", "whole", "--details", os.path.join(d, "whole.json.gz")])
            split = main(common + ["--name", "split", "--details", os.path.join(d, "split.json.gz"), "--parts", "2",
                                   "--jobs", "2"])
            rows = [json.loads(line) for line in open(os.path.join(d, "rows.jsonl"))]
            self.assertTrue(os.path.exists(os.path.join(d, "split", "1of2.json")))
            with self.assertRaises(SystemExit):  # a name is used once
                main(common + ["--name", "whole", "--details", os.path.join(d, "whole.json.gz")])
        self.assertEqual(len(rows), 2)
        same = ("completed", "completed_neo_done", "fell", "searches", "plans_played", "plans_landed_as_planned", "brakes")
        self.assertEqual({k: whole.get(k) for k in same}, {k: split.get(k) for k in same})
        self.assertEqual((whole["split"], whole["assist"], whole["blocked"], whole["config"]["restarts"]),
                         ("validation", True, "plan", 1))
        self.assertEqual((split["parts"], whole.get("parts")), (2, None))

    def test_fields_that_need_the_course_list_are_refused(self):
        for bad in (dict(prior=0.5), dict(approach=48), dict(batch=2), dict(jumps="all")):
            with self.assertRaises(ValueError, msg=str(bad)):
                check_config(RouteConfig(blocked="plan"), AssistConfig(**bad))
        check_config(RouteConfig(blocked="plan"), AssistConfig(restarts=3, retries=1, brake=14, plan_start=1, corner_limit=0.25))

    def test_the_service_plans_a_router_request_as_the_simulator(self):
        # Every request of a run (with failures: brakes and retries), sent to the bots' planner service as the bots
        # send it: the same plans, and the same brakes after none.
        import parkour.route_neo as RN
        cs = courses_from_file([c.to_json() for c in neo_courses([1, 2, 3, 4, 5])])
        cfg = AssistConfig(samples=64, iterations=3, restarts=1, retries=1, brake=14, retry_horizon=48, seed=5)
        seen, real = [], RN.plan_once

        def spy(model, pool, rules, arr, kind, attempt, c, horizon, device, stats=None):
            out = real(model, pool, rules, arr, kind, attempt, c, horizon, device, stats)
            seen.append((dict(arr), kind, attempt, out[0]))
            return out
        with mock.patch.object(RN, "plan_once", spy):
            res = self.routed(cs, "plan", cfg)
        self.assertEqual(len(seen), len(res["events"]))
        self.assertTrue(any(s[3] is None for s in seen) and any(s[3] is not None for s in seen))
        pool = CoursePool(cs, "cpu", C.GRID)
        for arr, *_ in seen:
            self.assertEqual((arr["neo"], arr["target"]), (True, arr["cp"] + 1))
            self.assertEqual(arr["floor"], float(min(pool.top[arr["course"], arr["cp"]], pool.top[arr["course"], arr["target"]])))
        reqs = "".join(json.dumps({"index": arr["course"], "cp": arr["cp"], "target": arr["target"], "kind": kind,
                                   "attempt": att, "lastAction": arr["last_action"], "route": True, "floor": arr["floor"],
                                   "state": {"pos": arr["pos"], "vel": arr["vel"], "yaw": arr["yaw"],
                                             "onGround": arr["on_ground"], "jumpTicks": arr["jump_ticks"],
                                             "collidedH": arr["collided_h"]}}) + "\n" for arr, kind, att, _ in seen)
        out = io.StringIO()
        serve(self.model, cs, cfg, "cpu", io.StringIO(reqs), out)
        got = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual([g["plan"] for g in got], [None if s[3] is None else s[3].tolist() for s in seen])
        braked = [e["braked"] for e in res["events"]]
        self.assertEqual([g["brake"] is not None for g in got], braked)
        self.assertTrue(any(braked))


@unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
class RouteNeoBotParity(Shared):
    def test_offline_bot_with_planner_service_plays_like_the_simulator(self):
        # Five single-neo courses (their neos planned) and three standard ones whose blocked choices are no neos.
        res, bot = self.check(AssistConfig(samples=256, iterations=8, restarts=1),
                              neo_courses([1, 2, 3, 12, 33]) + standard_courses([311, 818, 1346]))
        self.assertGreater(sum(x["played"] for x in bot), 0)
        self.assertEqual({e["course"] for e in res["events"]}, {0, 1, 2, 3, 4})

    def test_same_with_brakes_and_retries(self):
        res, bot = self.check(AssistConfig(samples=16, iterations=1, retries=1, brake=14, retry_horizon=48),
                              neo_courses([1, 2, 3, 4]), played=False)
        self.assertGreater(res["plans"]["brakes"], 0)
        self.assertEqual(sum(x["braked"] for x in bot), res["plans"]["brakes"])

    def test_same_with_a_planned_start_and_the_corner_limit(self):
        # Courses 0 and 41: the neo is the first jump, from the settled start; plan_start asks before the first action.
        cfg = AssistConfig(samples=128, iterations=6, retries=1, brake=14, retry_horizon=48, restarts=1, plan_start=1,
                           corner_limit=0.25)
        res, bot = self.check(cfg, neo_courses([0, 41, 5]))
        self.assertEqual([(e["course"], e["kind"], e["tick"]) for e in res["events"] if e["tick"] == 0],
                         [(0, "takeoff", 0), (1, "takeoff", 0)])
        self.assertTrue(all("corner_dropped" in e for e in res["events"]))

    def test_same_with_plan_on(self):
        # The jumps after each planned landing are planned too; a weak search fails some, which brake and retry.
        cfg = AssistConfig(samples=64, iterations=3, restarts=1, retries=1, brake=14, retry_horizon=48, plan_on=1)
        res, bot = self.check(cfg, neo_courses([1, 2, 3, 11]))
        neo = [neo_takeoffs(c)[0] for c in neo_courses([1, 2, 3, 11])]
        self.assertTrue(any(e["takeoff"] > neo[e["course"]] for e in res["events"]))  # planned past the neo
        self.assertGreater(res["plans"]["brakes"], 0)

    def check(self, cfg, cs, played=True):
        from parkour.live import SETTLE_TICKS
        cs = courses_from_file([c.to_json() for c in cs])  # as parkour.live reads a course file
        res = self.routed(cs, "plan", cfg, settle=SETTLE_TICKS)
        with tempfile.TemporaryDirectory() as d:
            pol, crs, out, rj = (os.path.join(d, f) for f in ("policy.json", "courses.json", "episodes.jsonl", "router.json"))
            self.model.export(pol)
            self.net.export(rj)
            with open(crs, "w") as f:  # as parkour.live writes the bots' course file in router mode (no assist fields)
                json.dump({"courses": [c.to_json() for c in cs]}, f)
            planner = {**dataclasses.asdict(PlannerConfig()), "router": rj, "fill": FILL, "blocked": "plan"}
            subprocess.run([str(NODE), str(BRIDGE / "offline_bot.js"), "--policy", pol, "--courses", crs, "--out", out,
                            "--settle", str(SETTLE_TICKS), "--planner", json.dumps(planner),
                            "--assist", json.dumps(dataclasses.asdict(cfg)), "--assist-python", sys.executable,
                            "--assist-checkpoint", str(ROOT / "results" / "final_policy.pt"), "--assist-root", str(ROOT),
                            "--assist-device", "cpu"], check=True, cwd=str(BRIDGE), capture_output=True)
            js = [json.loads(line) for line in open(out) if line.strip()]
        self.assertEqual([e["success"] for e in js], [c["success"] for c in res["per_course"]])
        self.assertEqual([e["ticks"] for e in js], [c["ticks"] for c in res["per_course"]])
        self.assertEqual([e["plans"] for e in js], res["decisions"])  # the router's decisions, tick by tick
        bot = [x for e in js for x in e["assist"]]
        by_course = sorted(res["events"], key=lambda e: e["course"])  # the bot's log: course by course
        self.assertEqual([(x["kind"], x["cp"], x["tick"], x["found"], x["braked"]) for x in bot],
                         [(e["kind"], e["takeoff"], e["tick"], e["found"], e["braked"]) for e in by_course])
        self.assertTrue(all(x["verified"] == x["found"] for x in bot))  # prismarine-physics agrees with every plan
        if played:
            self.assertGreater(sum(x["played"] for x in bot), 0)
        return res, bot


if __name__ == "__main__":
    unittest.main()

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


SCRATCH_SINGLE, SCRATCH_STANDARD, SCRATCH_OBSTACLES = 90, 91, 92  # scratch families, used nowhere else: smoke checks of new levers run here


def scratch_neo_courses(idx):
    """Single-neo courses of a scratch family: course i has type i mod 32 (22 to 31: level 7), as in families 40, 41."""
    cs = single_neo_courses(SCRATCH_SINGLE, n=max(idx) + 1)
    return [cs[i] for i in idx]


def scratch_standard_courses(per_level=1):
    """Standard courses (no obstacle blocks, so no walls) of a scratch family, levels 0 to 5."""
    from parkour.train import held_out_courses
    return held_out_courses(C.load_levels(), per_level, 10, family=SCRATCH_STANDARD)


def scratch_obstacle_courses(idx, per_level=150):
    """Obstacle courses (a pillar or beam at half the jumps, as families 5 and 4) of a scratch family, levels 0 to 5."""
    from parkour.train import held_out_courses
    cs = held_out_courses(C.load_levels(), per_level, 10, obstacle_prob=0.5, family=SCRATCH_OBSTACLES)
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
        cs = neo_courses(range(32))  # every single-neo type once (22 to 31: level 7)
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
        for i, (c, k) in enumerate(zip(cs, ks)):  # from k the router picks k + 1, past the platform in the way
            plats = [tuple(int(v) for v in r) for r in c.platforms]
            chain = next(ch for _, here, ch in keep["events"][i] if tuple(here) == plats[k])
            self.assertEqual(tuple(chain[0]), plats[k + 1])
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
        import parkour.route_neo as RN
        cs = neo_courses([1, 2, 3, 11])
        seen, real = [], RN.plan_once

        def spy(model, pool, rules, arr, *rest):
            seen.append(dict(arr))
            return real(model, pool, rules, arr, *rest)
        with mock.patch.object(RN, "plan_once", spy):
            res = self.routed(cs, "plan", AssistConfig(samples=256, iterations=8, restarts=1, plan_on=1))
        longest = 0
        for i, c in enumerate(cs):
            k = neo_takeoffs(c)[0]
            ev = [e for e in res["events"] if e["course"] == i]
            self.assertEqual((ev[0]["takeoff"], ev[0]["target"]), (k, k + 1))  # the neo first, as without plan_on
            run = 1
            for a, b in zip(ev, ev[1:]):  # then each landing's next choice: a chain of planned jumps
                if a["found"] and b["kind"] == "takeoff":
                    self.assertEqual(b["takeoff"], a["target"])
                run = run + 1 if a["found"] and b["kind"] == "takeoff" else 1
                longest = max(longest, run)
            self.assertGreater(len(ev), 1)
        self.assertGreaterEqual(longest, 3)  # "and so on while plans land": three planned jumps in a row
        self.assertEqual(res["plans"]["landed_as_planned"], res["plans"]["played"])
        # Each planned jump is searched as a neo (the neo levers, here restarts, apply), the neo's and those after it.
        self.assertTrue(all(arr["neo"] for arr in seen))
        self.assertTrue(any(arr["cp"] != neo_takeoffs(cs[arr["course"]])[0] for arr in seen))
        without = self.routed(cs, "plan", AssistConfig(samples=256, iterations=8, restarts=1))
        self.assertEqual(sorted((e["course"], e["takeoff"]) for e in without["events"]),
                         [(i, neo_takeoffs(c)[0]) for i, c in enumerate(cs)])
        from parkour.neo_assist import evaluate_assisted
        with self.assertRaises(ValueError):  # the course-list planner plans every jump with jumps "all" instead
            evaluate_assisted(self.model, CoursePool(cs[:1], "cpu", C.GRID), "cpu", AssistConfig(plan_on=1))

    def test_plan_ahead_plans_the_jumps_that_lead_to_a_wall(self):
        cs = scratch_neo_courses([2, 3, 13, 24])  # the neo's takeoff is platform 2, 3, 3, 4 (24: a level-7 type)
        cfg = dict(samples=256, iterations=8, restarts=1)
        res = self.routed(cs, "plan", AssistConfig(**cfg, plan_ahead=1))
        onto = 0
        for i, c in enumerate(cs):
            k = neo_takeoffs(c)[0]
            ev = [e for e in res["events"] if e["course"] == i]
            before = [e for e in ev if e["takeoff"] < k]
            self.assertTrue(all(k - 2 <= e["takeoff"] and e["target"] <= k for e in before))  # only the approach
            onto += (k - 1, k) in [(e["takeoff"], e["target"]) for e in before]  # the jump onto the neo's takeoff
            if res["per_course"][i]["neos_reached"]:
                self.assertIn((k, k + 1), [(e["takeoff"], e["target"]) for e in ev])  # the neo, as without plan_ahead
        self.assertGreaterEqual(onto, 3)  # the router's route crossed the wall one or two slots ahead
        self.assertEqual(res["plans"]["landed_as_planned"], res["plans"]["played"])
        without = self.routed(cs, "plan", AssistConfig(**cfg))
        self.assertTrue(all(e["takeoff"] == neo_takeoffs(cs[e["course"]])[0] for e in without["events"]))
        # Courses with no wall plan nothing.
        flat = scratch_standard_courses()
        keep = self.routed(flat, "keep")
        plan = self.routed(flat, "plan", AssistConfig(samples=64, iterations=3, plan_on=1, plan_ahead=1))
        self.assertEqual(plan["events"], [])
        self.assertEqual(plan["decisions"], keep["events"])
        from parkour.neo_assist import evaluate_assisted
        with self.assertRaises(ValueError):  # the course-list planner plans every jump with jumps "all" instead
            evaluate_assisted(self.model, CoursePool(cs[:1], "cpu", C.GRID), "cpu", AssistConfig(plan_ahead=1))

    def test_plan_mode_on_obstacle_courses(self):
        # Pillars and beams are the walls of the router's obstacle sets. Of 900 scratch obstacle courses, 392, 545, 743
        # and 888 have a first choice walled, and 546, 695 and 815 a wall only in a later slot of a route. Only those are
        # planned, from and onto course platforms (here also skips and a step back), and the rest play as "keep".
        cs = scratch_obstacle_courses([392, 545, 546, 695, 743, 815, 888])
        keep = self.routed(cs, "keep")
        cfg = dict(samples=64, iterations=3)
        for extra, searched in (({}, {0, 1, 4, 6}), (dict(plan_on=1, plan_ahead=1), set(range(7)))):
            res = self.routed(cs, "plan", AssistConfig(**cfg, **extra))
            self.assertEqual({e["course"] for e in res["events"]}, searched, extra)
            n = [len(c.platforms) for c in cs]
            self.assertTrue(all(0 <= e["takeoff"] < n[e["course"]] and 0 <= e["target"] < n[e["course"]] for e in res["events"]))
            self.assertEqual(res["plans"]["landed_as_planned"], res["plans"]["played"])
            for i in set(range(7)) - searched:
                self.assertEqual((res["per_course"][i]["success"], res["per_course"][i]["ticks"], res["decisions"][i]),
                                 (keep["per_course"]["success"][i], keep["per_course"]["ticks"][i], keep["events"][i]))
        self.assertTrue(any(e["target"] != e["takeoff"] + 1 for e in res["events"]))

    def test_the_bots_refuse_blocked_choices_without_the_path_check(self):
        from parkour import live
        with tempfile.TemporaryDirectory() as tmp:  # refused before any course is played or anything is written
            for extra in (["--router-blocked", "keep"], ["--assist", "{}"]):
                out = os.path.join(tmp, "run")
                with self.assertRaises(SystemExit):
                    live.main(["--offline", "--checkpoint", str(ROOT / "results" / "final_policy.pt"), "--router",
                               str(ROUTER), "--planner-config", json.dumps({"clear_path": False}), "--out", out] + extra)
                self.assertFalse(os.path.exists(out))

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
        # The same through each entry point, before anything runs or is written.
        from parkour import live
        from parkour.route_neo import main
        with self.assertRaises(ValueError):
            self.routed(scratch_neo_courses([0]), "plan", AssistConfig(prior=0.5))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                main(["--set", "single", "--family", str(SINGLE_VALIDATION_FAMILY), "--courses", "1", "--device", "cpu",
                      "--assist", json.dumps({"approach": 48}), "--out", os.path.join(d, "rows.jsonl"), "--parts-dir", d,
                      "--name", "refused", "--details", os.path.join(d, "refused.json.gz"), "--tag", "t"])
            self.assertEqual(os.listdir(d), [])
            for bad in ({"jumps": "all"}, {"batch": 2}):
                out = os.path.join(d, "run")
                with self.assertRaises((ValueError, SystemExit), msg=str(bad)):
                    live.main(["--offline", "--checkpoint", str(ROOT / "results" / "final_policy.pt"), "--router",
                               str(ROUTER), "--assist", json.dumps(bad), "--out", out])
                self.assertFalse(os.path.exists(out))

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

    def test_the_service_plans_skipping_and_backward_targets_as_the_simulator(self):
        # The router's choice can be any course platform: one further on (a skip) or one behind, each with its own
        # fall line, the lower top of the two. The service plans such requests as plan_once does.
        import parkour.route_neo as RN
        cs = courses_from_file([c.to_json() for c in scratch_neo_courses([4, 5, 6])])
        cfg = AssistConfig(samples=64, iterations=3, restarts=1, seed=7)
        seen, real = [], RN.plan_once

        def spy(model, pool, rules, arr, kind, attempt, c, horizon, device, stats=None):
            seen.append((pool, rules, dict(arr), kind, attempt, horizon))
            return real(model, pool, rules, arr, kind, attempt, c, horizon, device, stats)
        with mock.patch.object(RN, "plan_once", spy):
            self.routed(cs, "plan", cfg)
        top = CoursePool(cs, "cpu", C.GRID).top
        reqs, plans, floors = [], [], []
        for pool, rules, arr, kind, attempt, horizon in seen:
            i, cp = arr["course"], arr["cp"]
            for target in (cp + 2, cp - 1):
                if not 0 <= target < len(cs[i].platforms):
                    continue
                a = dict(arr, target=target, floor=float(min(top[i, cp], top[i, target])))
                seq = real(self.model, pool, rules, a, kind, attempt, cfg, horizon, "cpu")[0]
                plans.append(None if seq is None else seq.tolist())
                floors.append((a["floor"], float(top[i, cp])))
                reqs.append(json.dumps({"index": i, "cp": cp, "target": target, "kind": kind, "attempt": attempt,
                                        "lastAction": a["last_action"], "route": True, "floor": a["floor"],
                                        "state": {"pos": a["pos"], "vel": a["vel"], "yaw": a["yaw"], "onGround": a["on_ground"],
                                                  "jumpTicks": a["jump_ticks"], "collidedH": a["collided_h"]}}) + "\n")
        out = io.StringIO()
        serve(self.model, cs, cfg, "cpu", io.StringIO("".join(reqs)), out)
        self.assertEqual([json.loads(line)["plan"] for line in out.getvalue().splitlines()], plans)
        self.assertGreaterEqual(len(plans), 4)
        self.assertTrue(any(p is not None for p in plans))
        self.assertTrue(any(f < t for f, t in floors))  # a target lower than the platform stood on sets the fall line


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

    def test_same_when_the_corner_limit_drops_a_landing(self):
        # Scratch course 38 from the bots' start: the corner limit drops the landing its takeoff search finds without
        # the limit, so the bot brakes and retries (here 217 ticks instead of 181). The bot plays like the simulator
        # only if it and the planner service apply the limit as plan_once does.
        from parkour.live import SETTLE_TICKS
        cfg = AssistConfig(samples=128, iterations=6, retries=1, brake=14, retry_horizon=48, restarts=1, plan_start=1,
                           corner_limit=0.25)
        res, bot = self.check(cfg, scratch_neo_courses([38]))
        first = res["events"][0]
        self.assertTrue(first["corner_dropped"] > 0 and not first["found"] and first["braked"])
        free = self.routed(courses_from_file([c.to_json() for c in scratch_neo_courses([38])]), "plan",
                           dataclasses.replace(cfg, corner_limit=0.0), settle=SETTLE_TICKS)
        self.assertEqual([(e["kind"], e["takeoff"], e["tick"]) for e in free["events"][:1]],
                         [(first["kind"], first["takeoff"], first["tick"])])
        self.assertTrue(free["events"][0]["found"])  # the same search, without the limit, lands

    def test_same_with_plan_on(self):
        # The jumps after each planned landing are planned too; a weak search fails some, which brake and retry.
        cfg = AssistConfig(samples=64, iterations=3, restarts=1, retries=1, brake=14, retry_horizon=48, plan_on=1)
        res, bot = self.check(cfg, neo_courses([1, 2, 3, 11]))
        neo = [neo_takeoffs(c)[0] for c in neo_courses([1, 2, 3, 11])]
        self.assertTrue(any(e["takeoff"] > neo[e["course"]] for e in res["events"]))  # planned past the neo
        self.assertGreater(res["plans"]["brakes"], 0)

    def test_same_with_plan_ahead(self):
        # The jumps that lead to each neo are planned too (the router's route crosses the wall one or two slots
        # ahead), then the jumps after planned landings; a level-7 type among them, and courses with no wall.
        cfg = AssistConfig(samples=64, iterations=3, restarts=1, retries=1, brake=14, retry_horizon=48, plan_on=1,
                           plan_ahead=1)
        cs = scratch_neo_courses([2, 3, 24]) + scratch_standard_courses()[:2]
        res, bot = self.check(cfg, cs)
        neo = [neo_takeoffs(c)[0] for c in cs[:3]]
        self.assertTrue(any(e["course"] == i and e["takeoff"] < k for i, k in enumerate(neo) for e in res["events"]))
        self.assertFalse(any(e["course"] >= 3 for e in res["events"]))

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
        top = CoursePool(cs, "cpu", C.GRID).top
        self.assertEqual([(x["kind"], x["cp"], x["target"], x["floor"], x["tick"], x["found"], x["braked"]) for x in bot],
                         [(e["kind"], e["takeoff"], e["target"], float(min(top[e["course"], e["takeoff"]], top[e["course"], e["target"]])),
                           e["tick"], e["found"], e["braked"]) for e in by_course])
        self.assertTrue(all(x["verified"] == x["found"] for x in bot))  # prismarine-physics agrees with every plan
        if played:
            self.assertGreater(sum(x["played"] for x in bot), 0)
        return res, bot


if __name__ == "__main__":
    unittest.main()

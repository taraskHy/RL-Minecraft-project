"""The neo-skill mode in the bots (parkour.live --skill, parkour.neo_skill): on single-neo validation courses the
offline bot, its second network acting while the checkpoint is one of skill_cps, plays move for move as
neo_assist.evaluate_assisted(..., skill=...) does on the CPU; with the learned router (--router, --router-blocked
keep) it plays move for move as route_neo.evaluate_routed_neo(..., skill=...) does, the router's decisions handing
the skill each neo (their walled flags as the simulator's, and the count they set checked with a stub router too);
without --skill nothing changes; parkour.live refuses the skill outside these two modes before anything is
written."""
import dataclasses
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from parkour import courses as C
from parkour import physics
from parkour.env import OBS_DIM_EXT, CoursePool, EnvConfig, ParkourEnv
from parkour.live import SETTLE_TICKS, load_model
from parkour.model import ActorCritic
from parkour.neo_assist import (SINGLE_VALIDATION_FAMILY, AssistConfig, courses_from_file, evaluate_assisted,
                                single_neo_courses, skill_cps)
from parkour.neo_dagger import neo_takeoffs
from parkour.paths import BRIDGE, NODE, ROOT
from parkour.planner import PlannerConfig
from parkour.route_env import RouteConfig, RouteRunner, RouteTables
from parkour.route_neo import evaluate_routed_neo
from parkour.route_policy import load_router
from parkour.route_train import router_choosers

FINAL = ROOT / "results" / "final_policy.pt"
SKILL = ROOT / "results" / "neo_policy.pt"  # a stand-in for the neo skill: a network that sees the same 255 numbers
ROUTER = ROOT / "results" / "routing" / "router_policy.pt"
ROUTER_FILL = torch.load(ROUTER, map_location="cpu", weights_only=False)["config"].get("fill", "greedy")
MARK = torch.tensor([100, 0, 0, 0, 0])  # added to the skill's actions, so a spy on ParkourEnv.step sees who chose them
# Single-neo validation course 142 (its neo is jump 8): at the router's decision on platform 4 its route's second jump,
# platform 5 to 6, is blocked by a block of a platform in view, not by a wall, so nothing is handed to the skill (the
# jumper falls at jump 5); a router that took "blocked" for "walled" would hand it platforms 4 and 5.
BLOCKED_NOT_WALLED = 142


def validation_raw(n: int) -> list[dict]:
    """The first n single-neo validation courses (family 40 only: 41 and 49 are test sets) as a course file lists
    them; course i < 10 has its neo as jump i."""
    return [c.to_json() for c in single_neo_courses(SINGLE_VALIDATION_FAMILY, n)]


def off(model, cs, skill=None) -> list[tuple]:
    """(success, ticks) per course of the neo-skill mode (or the policy alone) from the bots' start, CPU."""
    pc = evaluate_assisted(model, CoursePool(cs, "cpu", C.GRID), "cpu", AssistConfig(enabled=False), settle=SETTLE_TICKS,
                           skill=skill)["per_course"]
    return [(c["success"], c["ticks"]) for c in pc]


def play(model, skill, cs, settle=SETTLE_TICKS, max_ticks=4000) -> list[dict]:
    """Per course, the first episode's success, ticks, actions and the ticks the skill chose: evaluate_assisted with
    the assist off, written out (the skill's action while the checkpoint before the step is one of skill_cps)."""
    n = len(cs)
    pool = CoursePool(cs, "cpu", C.GRID)
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=200, max_ticks=max_ticks), sampler=None)
    obs = env.reset_all(torch.arange(n))
    if settle:  # the bots' start: ticks without keys on the start block
        idle = torch.zeros(n, 6, dtype=torch.bool)
        for _ in range(settle):
            env.state = physics.simulate(pool.world, env.course, env.state, idle)
        env.dist = env._distance(env.state.pos, env.course, env.cp)
        env.reset_memory(torch.ones(n, dtype=torch.bool))
        obs = env.observe()
    window = torch.zeros(n, C.MAX_PLATFORMS, dtype=torch.bool)
    for i, c in enumerate(cs):
        window[i, skill_cps(c)] = True
    rows = torch.arange(n)
    out = [{"success": False, "ticks": 0, "actions": [], "skill_ticks": 0} for _ in cs]
    finished = torch.zeros(n, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(max_ticks):
            use = window[rows, env.cp]
            a = torch.where(use[:, None], skill.act(obs, True)[0], model.act(obs, True)[0])
            for i in torch.nonzero(~finished).flatten().tolist():
                out[i]["actions"].append(a[i].tolist())
                out[i]["skill_ticks"] += int(use[i])
            obs, _, _, _, _, info = env.step(a)
            for i in torch.nonzero(info["done"] & ~finished).flatten().tolist():
                out[i]["success"], out[i]["ticks"] = bool(info["success"][i]), int(info["ep_len"][i])
            finished |= info["done"]
            if bool(finished.all()):
                break
    return out


def offline_bot(cs, d, skill=None, with_cps=True) -> list[dict]:
    """The offline bot's records on courses ``cs`` (files in folder ``d``), with --skill when ``skill`` is given; the
    course file carries skill_cps as parkour.live --skill writes it (unless ``with_cps`` is false)."""
    pol, crs, out = (os.path.join(d, f) for f in ("policy.json", "courses.json", "episodes.jsonl"))
    load_model(FINAL, "cpu").export(pol)
    with open(crs, "w") as f:
        json.dump({"courses": [{**c.to_json(), **({"skill_cps": skill_cps(c)} if with_cps else {})} for c in cs]}, f)
    cmd = [str(NODE), str(BRIDGE / "offline_bot.js"), "--policy", pol, "--courses", crs, "--out", out,
           "--settle", str(SETTLE_TICKS)]
    if skill is not None:
        sk = os.path.join(d, "skill.json")
        load_model(skill, "cpu").export(sk)
        cmd += ["--skill", sk]
    subprocess.run(cmd, check=True, cwd=str(BRIDGE), capture_output=True)
    with open(out) as f:
        return [json.loads(line) for line in f if line.strip()]


class Marked:
    """The skill with its actions marked (MARK added): a spy on ParkourEnv.step removes the mark and so learns, tick by
    tick, which rows evaluate_routed_neo gave to the skill."""

    def __init__(self, net):
        self.net, self.obs_dim = net, net.obs_dim

    def act(self, obs, deterministic=False):
        a, *rest = self.net.act(obs, deterministic)
        return (a + MARK, *rest)


def routed(model, cs, choosers, skill=None) -> tuple[dict, list[dict]]:
    """route_neo.evaluate_routed_neo with the learned router, its blocked choices kept and the assist off (with
    ``skill``: the neo-skill mode), from the bots' start on the CPU, with the router's decisions; and per course the
    actions it played until the first episode ended, how many of them the skill chose (a spy on ParkourEnv.step) and
    at each router decision [its tick, whether its route's next jump and its second cross a wall] (RouteRunner.decide's
    walled_chain[:, 0] and [:, 1], from which the skill's count is set; a spy on RouteRunner.decide)."""
    pool = CoursePool(cs, "cpu", C.GRID)
    choose, fill = choosers
    steps, real = [], ParkourEnv.step
    walls, decide, depth = [[] for _ in cs], RouteRunner.decide, [0]

    def step(env, a):
        used = a[:, 0] >= MARK[0]
        a = torch.where(used[:, None], a - MARK, a)
        out = real(env, a)
        steps.append((a, used, out[5]["done"].clone()))
        return out

    def spy(run, rows, cell, choose=None, fill_choose=None):
        depth[0] += 1  # decide calls itself for the rows on a platform: the outermost call returns the decisions
        try:
            res = decide(run, rows, cell, choose, fill_choose)
        finally:
            depth[0] -= 1
        if res is not None and depth[0] == 0:
            for i, w in zip(res["rows"].tolist(), res["walled_chain"][:, :2].tolist()):
                walls[i].append(w)
        return res
    with torch.no_grad(), mock.patch.object(ParkourEnv, "step", step), mock.patch.object(RouteRunner, "decide", spy):
        res = evaluate_routed_neo(model, pool, RouteTables(pool), choose, fill, PlannerConfig(),
                                  RouteConfig(fill=ROUTER_FILL, blocked="keep"), AssistConfig(enabled=False),
                                  settle=SETTLE_TICKS, record=True, skill=None if skill is None else Marked(skill))
    played = [{"actions": [], "skill_ticks": 0,
               "walls": [[d[0], *w] for d, w in zip(res["decisions"][i], walls[i], strict=True)]}
              for i in range(len(cs))]
    done = torch.zeros(len(cs), dtype=torch.bool)
    for a, used, d in steps:
        for i in torch.nonzero(~done).flatten().tolist():
            played[i]["actions"].append(a[i].tolist())
            played[i]["skill_ticks"] += int(used[i])
        done |= d
    return res, played


def router_bot(cs, d, net, skill=None, walls=False) -> list[dict]:
    """The offline bot's records with the learned router ``net`` (files in folder ``d``), its course file and planner
    JSON as parkour.live --router R --router-blocked keep writes them (no skill_cps), with --skill when ``skill`` is
    given; with ``walls`` run through WALLS (the records with the router's walled flags at its decisions)."""
    pol, crs, out, rj = (os.path.join(d, f) for f in ("policy.json", "courses.json", "episodes.jsonl", "router.json"))
    load_model(FINAL, "cpu").export(pol)
    net.export(rj)
    with open(crs, "w") as f:
        json.dump({"courses": [c.to_json() for c in cs]}, f)
    planner = {**dataclasses.asdict(PlannerConfig()), "router": rj, "fill": ROUTER_FILL, "blocked": "keep"}
    bot = ["-e", WALLS, "--"] if walls else [str(BRIDGE / "offline_bot.js")]
    cmd = [str(NODE), *bot, "--policy", pol, "--courses", crs, "--out", out,
           "--settle", str(SETTLE_TICKS), "--planner", json.dumps(planner)]
    if skill is not None:
        sk = os.path.join(d, "skill.json")
        load_model(skill, "cpu").export(sk)
        cmd += ["--skill", sk]
    subprocess.run(cmd, check=True, cwd=str(BRIDGE), capture_output=True)
    with open(out) as f:
        return [json.loads(line) for line in f if line.strip()]


# A policy.js Episode built with the neo skill (the policy itself as the skill) and the given planner, assist and
# skill_cps: its error, or "ok".
EPISODE = """
const P = require('./policy')
const pol = P.Policy.load(process.argv[1])
const cfg = JSON.parse(process.argv[2])
try {
  new P.Episode(pol, [[0, 0, 0, 0, 0], [3, 3, 0, 0, 0]], () => false,
    { planner: cfg.planner, assist: cfg.assist, skill: { policy: pol, cps: cfg.cps } })
  console.log('ok')
} catch (e) { console.log(e.message) }
"""

# The offline bot (bridge/offline_bot.js, its arguments after "--"), policy.js Episode wrapped so that each record also
# has the walled flags the router-skill mode's count read at each router decision ("walls": [tick, walledChoice,
# walledNext]); the bot itself is unchanged.
WALLS = """
const P = require('./policy')
const count = P.Episode.prototype._skillDecision
P.Episode.prototype._skillDecision = function () {
  (this._walls = this._walls || []).push([this.ticks, this.planner.walledChoice, this.planner.walledNext])
  count.call(this)
}
const record = P.Episode.prototype.record
P.Episode.prototype.record = function (extra) { return { ...record.call(this, extra), walls: this._walls || [] } }
process.argv = [process.argv[0], 'offline_bot.js', ...process.argv.slice(1)]
require('./offline_bot')
"""

# A policy.js Episode in the router-skill mode whose router is a stub (no network, no course): it decides at the ticks
# of the plan ({tick: [walledNext, walledChoice]}; the start's at tick 0) while the bot stands on the first of two
# platforms; per tick, who chose the action (1: the skill, 0: the jumper) and the count after that tick's decision.
COUNT = """
const fs = require('fs')
const P = require('./policy')
const [file, plan, n] = [process.argv[1], JSON.parse(process.argv[2]), Number(process.argv[3])]
fs.writeFileSync(file, JSON.stringify({ format: 'parkour-router-v1' })) // the Episode loads it; the stub decides
const net = (move) => ({ obsDim: P.OBS_DIM, act: () => [move, 1, 0, 0, 3] })
const ep = new P.Episode(net(0), [[0, 0, 0, 0, 0], [3, 3, 0, 0, 0]], () => false,
  { planner: { router: file, blocked: 'keep' }, skill: { policy: net(2) } })
const r = { progress: 0, events: [], slots: () => [] }
const decide = (t) => { if (!(t in plan)) return false; [r.walledNext, r.walledChoice] = plan[t]; return true }
r.start = () => decide(0)
r.update = (pos, onGround, t) => decide(t)
ep.planner = r
const s = { pos: [0.5, 1, 0.5], vel: [0, 0, 0], yaw: 0, onGround: true, jumpTicks: 0, collidedH: false }
const ticks = []
for (let t = 0; t < n; t++) ticks.push([ep.step(s)[0] === 2 ? 1 : 0, ep.skill.left])
const rec = ep.record({})
console.log(JSON.stringify({ ticks, skill_ticks: rec.skill_ticks, skill_at: rec.skill_at }))
"""


@unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
class SkillBotParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.skill = load_model(FINAL, "cpu"), load_model(SKILL, "cpu")
        cls.courses = courses_from_file(validation_raw(10))  # as parkour.live reads a course file
        cls.sim = off(cls.model, cls.courses, cls.skill)
        cls.alone = off(cls.model, cls.courses)

    def test_offline_bot_plays_the_neo_skill_mode_like_the_simulator(self):
        cs = self.courses
        self.assertEqual([neo_takeoffs(c) for c in cs], [[k] for k in range(10)])  # the neo as jump 0 to 9
        ref = play(self.model, self.skill, cs)
        self.assertEqual([(r["success"], r["ticks"]) for r in ref], self.sim)  # the loop is the mode
        with tempfile.TemporaryDirectory() as d:
            js = offline_bot(cs, d, skill=SKILL)
        self.assertEqual([(e["success"], e["ticks"]) for e in js], self.sim)
        self.assertEqual([e["actions"] for e in js], [r["actions"] for r in ref])  # move for move
        self.assertEqual([e["skill_ticks"] for e in js], [r["skill_ticks"] for r in ref])
        self.assertTrue(ref[0]["skill_ticks"] > 0 and ref[1]["skill_ticks"] > 0)  # neo as jump 0 or 1: from the start
        self.assertGreater(sum(r["skill_ticks"] > 0 for r in ref), 5)
        self.assertNotEqual(self.sim, self.alone)  # the skill changed the play

    def test_without_the_skill_the_bot_plays_the_policy_alone(self):
        # The same course file, skill_cps included, without --skill: the policy alone, and the records as before.
        with tempfile.TemporaryDirectory() as d:
            js = offline_bot(self.courses, d)
        self.assertEqual([(e["success"], e["ticks"]) for e in js], self.alone)
        self.assertFalse(any("skill_ticks" in e for e in js))

    def test_the_bot_refuses_a_skill_without_skill_cps(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(subprocess.CalledProcessError) as err:
                offline_bot(self.courses[:1], d, skill=SKILL, with_cps=False)
        self.assertIn("skill_cps", err.exception.stderr.decode())

    def test_live_writes_the_skill_and_checks_the_bot_against_the_skill_mode(self):
        from parkour import live
        raw = validation_raw(3)
        cs = courses_from_file(raw)
        ref = play(self.model, self.skill, cs)
        with tempfile.TemporaryDirectory() as d:
            crs = os.path.join(d, "courses.json")
            with open(crs, "w") as f:
                json.dump({"courses": raw}, f)
            runs = {"skill": Path(d) / "skill", "plain": Path(d) / "plain"}
            with mock.patch("torch.cuda.is_available", return_value=False):  # the simulator re-run on the CPU
                for name, extra in (("skill", ["--skill", str(SKILL)]), ("plain", [])):
                    live.main(["--offline", "--checkpoint", str(FINAL), "--courses-file", crs, "--out", str(runs[name])]
                              + extra)
            load_model(SKILL, "cpu").export(Path(d) / "skill_ref.json", {"checkpoint": str(SKILL)})
            skill_ref = (Path(d) / "skill_ref.json").read_text()
            files = {name: {f: (out / f).read_text() if (out / f).exists() else None
                            for f in ("courses.json", "skill.json", "summary.json", "summary.md")}
                     for name, out in runs.items()}
            with open(runs["skill"] / "episodes.jsonl") as f:
                eps = [json.loads(line) for line in f if line.strip()]
        sk, plain = files["skill"], files["plain"]
        # With --skill: each course's skill_cps, the bots' copy of the skill, the skill mode's re-run and the summary.
        with_cps = [{**c.to_json(), "skill_cps": skill_cps(c)} for c in cs]
        self.assertEqual(sk["courses.json"], json.dumps({"courses": with_cps}))
        self.assertEqual(sk["skill.json"], skill_ref)
        s = json.loads(sk["summary.json"])
        self.assertEqual(s["skill"], str(SKILL))
        rows = s["episodes"]
        sim = off(self.model, cs, self.skill)
        self.assertEqual([(r["sim_success"], r["sim_ticks"]) for r in rows], sim)  # evaluate_assisted with the skill
        self.assertEqual([(r["result"] == "success", r["ticks"]) for r in rows], sim)
        self.assertEqual([(r["success"], r["ticks"]) for r in ref], sim)
        self.assertEqual([r["skill_ticks"] for r in rows], [r["skill_ticks"] for r in ref])
        self.assertEqual([e["actions"] for e in eps], [r["actions"] for r in ref])
        self.assertEqual((s["outcome_agreement"], s["replay_exact_episodes"]), (1.0, 3))
        self.assertTrue(all(r["skill_ticks"] > 0 for r in rows))  # the neo as jump 0, 1, 2: the skill plays
        self.assertIn("Neo skill", sk["summary.md"])
        # Without it: today's course file byte for byte, no skill file, nothing about a skill in the summary.
        self.assertEqual(plain["courses.json"], json.dumps({"courses": [c.to_json() for c in cs]}))
        self.assertIsNone(plain["skill.json"])
        p = json.loads(plain["summary.json"])
        self.assertNotIn("skill", p)
        self.assertFalse(any("skill_ticks" in r for r in p["episodes"]))
        self.assertNotIn("Neo skill", plain["summary.md"])
        self.assertEqual(p["outcome_agreement"], 1.0)


@unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
class RouterSkillBotParity(unittest.TestCase):
    """The neo skill with the learned router (parkour.live --router R --router-blocked keep --skill S): each router
    decision counts down the decisions the skill still plays, or sets them to 3 if its route's second jump crosses a
    wall and to 2 if its next one does, whichever is more (route_neo.evaluate_routed_neo with ``skill``); the skill
    acts while the count is above 0."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.skill = load_model(FINAL, "cpu"), load_model(SKILL, "cpu")
        cls.net = load_router(ROUTER, "cpu")
        cls.choosers = router_choosers(cls.net, True)  # (a tuple: functions on the class would become methods)
        raw = validation_raw(BLOCKED_NOT_WALLED + 1)
        cls.courses = courses_from_file(raw[:10] + raw[BLOCKED_NOT_WALLED:])  # as parkour.live reads a course file
        cls.sim, cls.ref = routed(cls.model, cls.courses, cls.choosers, cls.skill)

    def test_offline_bot_plays_the_router_skill_mode_like_the_simulator(self):
        cs, pc, ref = self.courses, self.sim["per_course"], self.ref
        self.assertEqual([neo_takeoffs(c) for c in cs], [[k] for k in range(10)] + [[8]])  # the neo as jump 0 to 9, 8
        with tempfile.TemporaryDirectory() as d:
            js = router_bot(cs, d, self.net, skill=SKILL)
        self.assertEqual([(e["success"], e["ticks"]) for e in js], [(c["success"], c["ticks"]) for c in pc])
        self.assertEqual([e["plans"] for e in js], self.sim["decisions"])  # the router's decisions, tick by tick
        self.assertEqual([e["actions"] for e in js], [r["actions"] for r in ref])  # move for move
        self.assertEqual([e["skill_ticks"] for e in js], [r["skill_ticks"] for r in ref])  # the ticks the skill chose
        self.assertEqual([e["skill_at"] for e in js], [c["skill_at"] for c in pc])  # and the checkpoints it chose at
        # On these courses the router hands the skill each neo's block as skill_cps does with the course list: from the
        # platform before the takeoff (for a neo as the first jump, from the start) until the bot stands on the one
        # after the target, where the episode got that far.
        for c, p in zip(cs, pc):
            end = c.n - 1 if p["success"] else p["stopped_at"]
            self.assertEqual(p["skill_at"], [k for k in skill_cps(c) if k <= end and k < c.n - 1])
        self.assertTrue(all(r["skill_ticks"] > 0 for r in ref[:10]))
        self.assertTrue(all(0 < r["skill_ticks"] < len(r["actions"]) for r, p in zip(ref, pc) if p["success"]))
        # Course 142: the jumper falls at jump 5, before the neo's block, and the skill never plays.
        self.assertEqual((pc[10]["stopped_at"], pc[10]["skill_at"], ref[10]["skill_ticks"]), (5, [], 0))

    def test_the_bot_router_flags_the_walled_jumps_where_the_simulator_does(self):
        # At each router decision, whether its route's next jump and its second cross a wall (walledChoice and
        # walledNext: the count's 2 and 3) are RouteRunner.decide's walled_chain[:, 0] and [:, 1] in the simulator; on
        # course 142 the second jump of one decision is blocked, but not by a wall.
        with tempfile.TemporaryDirectory() as d:
            js = router_bot(self.courses, d, self.net, skill=SKILL, walls=True)
        self.assertEqual([e["actions"] for e in js], [r["actions"] for r in self.ref])  # the wrapper changes nothing
        self.assertEqual([e["walls"] for e in js], [r["walls"] for r in self.ref])
        flags = [w for r in self.ref for w in r["walls"]]
        self.assertTrue(any(w[1] for w in flags) and any(w[2] for w in flags))  # both kinds of walled decision
        self.assertEqual([w[1:] for w in self.ref[10]["walls"]], [[False, False]] * len(self.ref[10]["walls"]))

    def test_without_the_skill_the_router_bot_plays_the_jumper_alone(self):
        sim, ref = routed(self.model, self.courses, self.choosers)
        with tempfile.TemporaryDirectory() as d:
            js = router_bot(self.courses, d, self.net)
        pc = sim["per_course"]
        self.assertEqual([(e["success"], e["ticks"]) for e in js], [(c["success"], c["ticks"]) for c in pc])
        self.assertEqual([e["plans"] for e in js], sim["decisions"])
        self.assertEqual([e["actions"] for e in js], [r["actions"] for r in ref])
        self.assertFalse(any("skill_ticks" in e or "skill_at" in e for e in js))  # the records as before
        self.assertEqual([r["skill_ticks"] for r in ref], [0] * len(ref))
        self.assertFalse(any("skill_at" in c for c in pc))
        # The skill changed the play: it completes more of these courses than the jumper alone.
        self.assertGreater(sum(c["success"] for c in self.sim["per_course"]), sum(c["success"] for c in pc))

    def test_the_episode_takes_the_skill_with_the_course_list_or_the_learned_router_alone(self):
        with tempfile.TemporaryDirectory() as d:
            pol, rj = os.path.join(d, "policy.json"), os.path.join(d, "router.json")
            load_model(FINAL, "cpu").export(pol)
            self.net.export(rj)
            greedy = dataclasses.asdict(PlannerConfig())
            router = {**greedy, "router": rj, "fill": ROUTER_FILL}
            assist = {"config": {}, "takeoffs": []}

            def says(**cfg):
                r = subprocess.run([str(NODE), "-e", EPISODE, pol, json.dumps(cfg)], cwd=str(BRIDGE),
                                   capture_output=True, text=True, check=True)
                return r.stdout.strip()
            self.assertEqual(says(cps=[0]), "ok")  # the course list
            self.assertIn("skill_cps", says())
            self.assertEqual(says(planner={**router, "blocked": "keep"}), "ok")  # the router decides: no skill_cps
            self.assertEqual(says(planner={**router, "blocked": "plan"}), "ok")  # without an assist "plan" is "keep"
            for bad in (dict(planner=greedy, cps=[0]), dict(assist=assist, cps=[0]),
                        dict(planner={**router, "blocked": "plan"}, assist=assist, cps=[0]),
                        dict(planner={**router, "blocked": "keep"}, assist=assist, cps=[0]),
                        dict(planner=greedy), dict(planner={**router, "blocked": "keep"}, assist=assist)):  # no cps either
                self.assertIn("no greedy planner, no assist", says(**bad), str(bad))
            self.assertIn("blocked choices kept", says(planner=router))  # "drop": no neo is ever chosen
            self.assertIn("blocked choices kept", says(planner={**router, "blocked": "drop"}))

    def test_live_runs_the_router_skill_mode_and_checks_the_bot_against_it(self):
        from parkour import live
        raw = validation_raw(3)
        cs = courses_from_file(raw)
        sim, ref = routed(self.model, cs, self.choosers, self.skill)
        with tempfile.TemporaryDirectory() as d:
            crs = os.path.join(d, "courses.json")
            with open(crs, "w") as f:
                json.dump({"courses": raw}, f)
            out = Path(d) / "run"
            with mock.patch("torch.cuda.is_available", return_value=False):  # the simulator re-run on the CPU
                live.main(["--offline", "--checkpoint", str(FINAL), "--courses-file", crs, "--out", str(out),
                           "--router", str(ROUTER), "--router-blocked", "keep", "--skill", str(SKILL)])
            load_model(SKILL, "cpu").export(Path(d) / "skill_ref.json", {"checkpoint": str(SKILL)})
            skill_ref = (Path(d) / "skill_ref.json").read_text()
            files = {f: (out / f).read_text() for f in ("courses.json", "skill.json", "summary.json", "summary.md")}
            with open(out / "episodes.jsonl") as f:
                eps = [json.loads(line) for line in f if line.strip()]
        # No skill_cps (the router decides), the bots' copy of the skill, the router-skill mode's re-run, the summary.
        self.assertEqual(files["courses.json"], json.dumps({"courses": [c.to_json() for c in cs]}))
        self.assertEqual(files["skill.json"], skill_ref)
        s = json.loads(files["summary.json"])
        self.assertEqual((s["skill"], s["router"], s["router_blocked"]), (str(SKILL), str(ROUTER), "keep"))
        rows, pc = s["episodes"], sim["per_course"]
        same = [(c["success"], c["ticks"]) for c in pc]
        self.assertEqual([(r["sim_success"], r["sim_ticks"]) for r in rows], same)  # evaluate_routed_neo with the skill
        self.assertEqual([(r["result"] == "success", r["ticks"]) for r in rows], same)
        self.assertEqual([e["actions"] for e in eps], [r["actions"] for r in ref])
        self.assertEqual([r["skill_ticks"] for r in rows], [r["skill_ticks"] for r in ref])
        self.assertEqual([(r["skill_at"], r["sim_skill_at"]) for r in rows], [(c["skill_at"],) * 2 for c in pc])
        self.assertTrue(all(r["skill_ticks"] > 0 for r in rows))  # the neo as jump 0, 1, 2: the skill plays
        self.assertEqual((s["outcome_agreement"], s["replay_exact_episodes"]), (1.0, 3))
        self.assertIn("Neo skill", files["summary.md"])
        self.assertIn("at the simulator's checkpoints on 3 of 3 courses", files["summary.md"])


@unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
class RouterSkillCount(unittest.TestCase):
    """policy.js Episode's count in the router-skill mode with a stub router, every case written out: at each router
    decision max(count - 1, 3 if its route's second jump crosses a wall (walledNext), else 2 if its next one does
    (walledChoice), else 0), as route_neo.evaluate_routed_neo's ``left``; the start's decision before the first action
    and each later one before that tick's action; no change between decisions; the skill acts while it is above 0."""

    def test_the_count_follows_the_router_decisions(self):
        # (tick, walledNext, walledChoice) of each decision, and the count it leaves.
        decisions = [(0, False, True, 2),  # a neo as the first jump: the start's decision and the next one
                     (2, False, False, 1), (4, False, False, 0), (5, False, False, 0),  # never below 0
                     (6, True, False, 3),  # the route's second jump crosses a wall: three decisions
                     (7, False, False, 2), (9, False, False, 1), (11, False, False, 0),
                     (13, True, True, 3),  # both: 3
                     (14, False, False, 2), (15, False, False, 1), (16, False, False, 0),
                     (18, False, True, 2), (19, True, False, 3),  # a later walled jump extends the count
                     (20, False, False, 2), (21, False, False, 1), (22, False, False, 0)]
        n = 24
        at = {t: c for t, _, _, c in decisions}
        count = [0] * n
        for t in range(n):
            count[t] = at.get(t, count[t - 1] if t else 0)
        with tempfile.TemporaryDirectory() as d:
            plan = json.dumps({t: [nxt, choice] for t, nxt, choice, _ in decisions})
            r = subprocess.run([str(NODE), "-e", COUNT, os.path.join(d, "router.json"), plan, str(n)], cwd=str(BRIDGE),
                               capture_output=True, text=True, check=True)
        out = json.loads(r.stdout)
        self.assertEqual([c for _, c in out["ticks"]], count)
        self.assertEqual([s for s, _ in out["ticks"]], [int(c > 0) for c in count])  # the skill acts while above 0
        self.assertEqual((out["skill_ticks"], out["skill_at"]), (sum(c > 0 for c in count), [0]))


class SkillRefusals(unittest.TestCase):
    def test_live_refuses_the_skill_outside_the_course_list_mode(self):
        from parkour import live
        with tempfile.TemporaryDirectory() as d:
            crs = os.path.join(d, "courses.json")  # validation courses (should a refusal ever fail), offline
            with open(crs, "w") as f:
                json.dump({"courses": validation_raw(1)}, f)
            other = os.path.join(d, "extended.pt")  # a network that sees another observation
            torch.save({"model": ActorCritic(obs_dim=OBS_DIM_EXT, hidden=(8,)).state_dict(), "config": {"hidden": (8,)}},
                       other)
            cases = [(SKILL, ["--planner"]), (SKILL, ["--router", str(ROUTER)]), (SKILL, ["--assist", "{}"]),
                     (SKILL, ["--assist", json.dumps({"enabled": False})]),
                     (SKILL, ["--router", str(ROUTER), "--assist", "{}"]), (other, [])]
            with mock.patch("torch.cuda.is_available", return_value=False):
                for skill, extra in cases:
                    out = os.path.join(d, "run")
                    with self.assertRaises(SystemExit, msg=str(extra)):
                        live.main(["--offline", "--checkpoint", str(FINAL), "--skill", str(skill), "--courses-file", crs,
                                   "--out", out] + extra)
                    self.assertFalse(os.path.exists(out), str(extra))

    def test_live_refuses_the_skill_with_the_router_unless_its_blocked_choices_are_kept_without_an_assist(self):
        from parkour import live
        with tempfile.TemporaryDirectory() as d:
            crs = os.path.join(d, "courses.json")  # validation courses (should a refusal ever fail), offline
            with open(crs, "w") as f:
                json.dump({"courses": validation_raw(1)}, f)
            other = os.path.join(d, "extended.pt")  # a network that sees another observation
            torch.save({"model": ActorCritic(obs_dim=OBS_DIM_EXT, hidden=(8,)).state_dict(), "config": {"hidden": (8,)}},
                       other)
            r = ["--router", str(ROUTER)]
            cases = [(SKILL, r), (SKILL, r + ["--router-blocked", "drop"]),
                     (SKILL, r + ["--router-blocked", "plan", "--assist", "{}"]), (SKILL, r + ["--assist", "{}"]),
                     (SKILL, r + ["--router-blocked", "keep", "--assist", "{}"]),
                     (SKILL, r + ["--router-blocked", "keep", "--assist", json.dumps({"enabled": False})]),
                     (SKILL, r + ["--router-blocked", "keep", "--planner-config", json.dumps({"clear_path": False})]),
                     (SKILL, ["--planner", "--router-blocked", "keep"]), (SKILL, ["--router-blocked", "keep"]),
                     (other, r + ["--router-blocked", "keep"])]
            with mock.patch("torch.cuda.is_available", return_value=False):
                for skill, extra in cases:
                    out = os.path.join(d, "run")
                    with self.assertRaises(SystemExit, msg=str(extra)):
                        live.main(["--offline", "--checkpoint", str(FINAL), "--skill", str(skill), "--courses-file", crs,
                                   "--out", out] + extra)
                    self.assertFalse(os.path.exists(out), str(extra))


if __name__ == "__main__":
    unittest.main()

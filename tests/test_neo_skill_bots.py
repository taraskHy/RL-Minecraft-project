"""The neo-skill mode in the bots (parkour.live --skill, parkour.neo_skill): on single-neo validation courses the
offline bot, its second network acting while the checkpoint is one of skill_cps, plays move for move as
neo_assist.evaluate_assisted(..., skill=...) does on the CPU; without --skill nothing changes; parkour.live refuses
the skill outside the course-list mode before anything is written."""
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

FINAL = ROOT / "results" / "final_policy.pt"
SKILL = ROOT / "results" / "neo_policy.pt"  # a stand-in for the neo skill: a network that sees the same 255 numbers
ROUTER = ROOT / "results" / "routing" / "router_policy.pt"


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


if __name__ == "__main__":
    unittest.main()

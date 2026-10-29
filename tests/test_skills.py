"""One network for every skill (docs/combine_policies.md): --skills, --teachers, --distill-coef, --distill-decay."""
import csv
import dataclasses
import hashlib
import json
import math

import pytest
import torch

from parkour import courses as C
from parkour import train as T
from parkour.env import CoursePool, EnvConfig, LevelSampler, ParkourEnv

GROUPS = ["basic", "advanced", "extra"]
SPECIAL = ("neo", "runway", "ceiling", "ice", "block", "bounce", "ladder")
# CourseFactory(TrainConfig(seed=0, obstacle_prob=0.3, **flags), table).make(24) + make(8), computed with
# parkour/train.py at 52be9e3, before the skills mode (the training loop itself was compared bit for bit
# there too: runs/skills-defaults-check-20261029).
DEFAULT_FACTORY = {
    "basic": ({}, {"ice_prob": 0.3}, "d22b5b60f4ba2664"),
    "advanced": ({"advanced": True}, {}, "be34f97c6a437066"),
    "extra": ({"extra": True}, {}, "0117b01e0b71ecf4"),
}


def fingerprint(courses) -> str:
    return hashlib.sha1(json.dumps([c.to_json() for c in courses], sort_keys=True).encode()).hexdigest()[:16]


def test_defaults_make_the_same_courses():
    assert (T.TrainConfig().skills, T.TrainConfig().teachers, T.TrainConfig().distill_coef) == ("", "", 0.0)
    cli = T.parse_args([])
    assert (cli.skills, cli.teachers, cli.distill_coef, cli.distill_decay) == ("", "", 0.0, False)
    for name, (table, flags, digest) in DEFAULT_FACTORY.items():
        factory = T.CourseFactory(T.TrainConfig(seed=0, obstacle_prob=0.3, **flags), C.load_levels(**table), set())
        assert fingerprint(factory.make(24) + factory.make(8)) == digest, name


def test_skills_courses_come_from_their_groups_tables():
    cfg = T.TrainConfig(seed=0, obstacle_prob=0.3, ice_prob=0.3, skills=",".join(GROUPS))
    courses = T.CourseFactory(cfg, C.load_levels(), set()).make(90)
    plan = {gid: (g, L) for gid, g, L in T.skill_plan(GROUPS)}
    tables = T.skill_tables(GROUPS)
    assert [c.level for c in courses] == [k % 15 for k in range(90)]  # the training levels in turn, global ids
    for c in courses:
        g, L = plan[c.level]
        own = {s for lv in tables[g][:L + 1] for s in lv}
        special = [s for s in c.specs if any(getattr(s, a) for a in SPECIAL)]
        # Every special jump is one of the group's own, at its level or below (plain jumps, run-ups and exits
        # are the generator's helpers).
        assert all(s in own for s in special), (g, L, [s.key() for s in special if s not in own])
        if g == "basic":
            assert not special
        if g == "advanced":
            assert not any(s.ceiling or s.ice or s.block or s.bounce or s.ladder for s in c.specs)
            assert not len(c.ice)  # --ice-prob only on basic courses
        if g == "extra":
            assert not any(s.neo or (s.runway and not (s.ceiling or s.ice)) for s in c.specs)
            assert not len(c.ice) or any(s.ice for s in c.specs)
    assert any(len(c.ice) for c in courses if c.level < 6)


def test_held_out_groups_are_the_specialists_courses():
    sets = T.skill_sets("validation", 2, 3)
    assert list(sets) == ["basic", "advanced", "extra", "ice"]
    for g, table in (("basic", {}), ("advanced", {"advanced": True}), ("extra", {"extra": True})):
        _, lo, hi, first = T.SKILL_GROUPS[g]
        own = T.held_out_courses(C.load_levels(**table), 2, 3, family=T.VALIDATION_FAMILY)
        own = [c for c in own if lo <= c.level <= hi]
        assert [c.platforms.tobytes() for c in own] == [c.platforms.tobytes() for c in sets[g]["standard"]]
        assert [c.level for c in sets[g]["standard"]] == [first + c.level - lo for c in own]
        assert len(sets[g]["obstacles"]) == len(own)
    ice = sets["ice"]["standard"]
    assert sorted({c.level for c in ice}) == list(range(15, 21)) and not sets["ice"]["obstacles"]
    assert any(len(c.ice) for c in ice)
    assert T.macro_score(T.group_scores({0: {"success": 1.0, "n": 2}, 6: {"success": 0.5, "n": 4},
                                         15: {"success": 0.0, "n": 1}})) == pytest.approx(0.5)


def test_kl_is_zero_against_itself_and_distillation_lowers_it():
    torch.manual_seed(0)
    teacher = T.ActorCritic(hidden=(32,))
    with torch.no_grad():
        teacher.actor[-1].weight.mul_(300)  # peaked distributions, as a trained policy's
    obs = torch.randn(256, teacher.obs_dim)
    with torch.no_grad():  # as in the rollout
        target = T.teacher_logp(teacher, obs)
    assert target.shape == (256, sum(T.ACTION_DIMS))
    assert T.distill_kl(teacher.logits(teacher.norm(obs)), target).abs().max() < 1e-5
    student = T.ActorCritic(hidden=(32,))
    opt = torch.optim.Adam(student.parameters(), lr=1e-2)
    kl = [T.distill_kl(student.logits(student.norm(obs)), target).mean()]
    for _ in range(30):
        opt.zero_grad()
        kl[-1].backward()
        opt.step()
        kl.append(T.distill_kl(student.logits(student.norm(obs)), target).mean())
    assert kl[0] > 0.1 and kl[-1] < 0.5 * kl[0]


def test_teachers_follow_the_course_level_also_after_a_pool_refresh():
    factory = T.CourseFactory(T.TrainConfig(seed=0, skills=",".join(GROUPS)), C.load_levels(), set())
    pool = CoursePool(factory.make(15), "cpu", C.GRID, ice=True, shapes=True, slime=True)
    env = ParkourEnv(pool, EnvConfig(num_envs=40), sampler=LevelSampler(15, fixed=[1.0] * 15), seed=0)
    table = T.teacher_table({"basic": "a.pt", "advanced": "b.pt"}, GROUPS, ["a.pt", "b.pt"])
    assert table.tolist() == [0] * 6 + [1] * 3 + [-1] * 6  # extra has no teacher here
    expect = lambda: [table[int(pool.level[c])].item() for c in env.course]
    assert T.teacher_rows(env, table).tolist() == expect()
    # The refresh of train(): course 0 (basic, level 0) becomes an advanced course; its players restart.
    new = next(c for c in factory.make(15) if c.level == 6)
    pool.replace(0, new)
    env.rebuild_level_table()
    env.set_level_probs(env.sampler.probs())
    env.reset_rows(env.course == 0)
    on_new = torch.zeros(40, dtype=torch.bool)
    on_new[:3] = True
    env._apply_reset(on_new, torch.zeros(40, dtype=torch.int64))
    rows = T.teacher_rows(env, table)
    assert rows.tolist() == expect() and rows[:3].tolist() == [1, 1, 1]
    with pytest.raises(ValueError, match="not in --skills"):
        T.parse_teachers("ice=a.pt", GROUPS)


def one_course_per_level(monkeypatch):
    original = T.held_out_courses
    monkeypatch.setattr(T, "held_out_courses", lambda levels, per_level, n_jumps, obstacle_prob=0.0, family=1, ice_prob=0.0:
                        original(levels, 1, 2, obstacle_prob=obstacle_prob, family=family, ice_prob=ice_prob))


def test_matrix_and_the_fixed_choice_rules(monkeypatch, tmp_path):
    from parkour import skills as S
    one_course_per_level(monkeypatch)
    torch.manual_seed(0)
    cands = []
    for name in ("a", "b"):
        torch.save({"model": T.ActorCritic(hidden=(8,)).state_dict(), "config": {"hidden": [8]}, "iteration": 7},
                   tmp_path / f"{name}.pt")
        cands += ["--candidate", f"{name}={tmp_path / f'{name}.pt'}"]
    for k in (1, 2):
        S.main(["matrix", "--device", "cpu", "--part", f"{k}/2", "--out", str(tmp_path / f"part{k}.json"), *cands])
    S.main(["choose", str(tmp_path / "part2.json"), str(tmp_path / "part1.json"), "--out", str(tmp_path / "m.json"),
            *[c.replace("--candidate", "--order") for c in cands]])
    m = json.loads((tmp_path / "m.json").read_text())
    assert list(m["candidates"]) == ["a", "b"] and m["families"] == {"standard": 2, "obstacles": 5, "ice": 17}
    c = m["candidates"]["a"]
    assert list(c["groups"]) == ["basic", "ice", "advanced", "extra"] and c["groups"]["ice"]["obstacles"] is None
    assert c["groups"]["basic"]["n"] == 12 and c["iteration"] == 7
    assert c["macro"] == pytest.approx(sum(v["all"] for v in c["groups"].values()) / 4)
    assert set(m["choice"]["teachers"]) == {"basic", "advanced", "extra"} and m["choice"]["init"] in ("a", "b")
    tie = {n: {"groups": {g: {"all": 0.5} for g in T.GROUP_ORDER}, "macro": 0.5} for n in ("x", "y")}
    tie["y"]["groups"]["ice"]["all"] = 0.6  # basic's teacher counts ice too
    assert S.choose(tie)["teachers"] == {"basic": "y", "advanced": "x", "extra": "x"} and S.choose(tie)["init"] == "x"
    S.main(["matrix", "--device", "cpu", "--split", "test", "--groups", "ice", "--out", str(tmp_path / "t.json"), *cands[:2]])
    t = json.loads((tmp_path / "t.json").read_text())
    assert list(t["candidates"]["a"]["groups"]) == ["ice"] and t["families"]["ice"] == 16
    with pytest.raises(ValueError, match="validation only"):
        S.main(["choose", str(tmp_path / "t.json"), "--out", str(tmp_path / "bad.json")])


def tiny(monkeypatch, tmp_path, **flags):
    monkeypatch.setattr(T, "RUNS", tmp_path)
    one_course_per_level(monkeypatch)
    return T.TrainConfig(run="tiny-skills", device="cpu", threads=1, num_envs=4, rollout=2, pool_size=15,
                         pool_refresh_every=1, hidden=(8,), epochs=1, minibatches=1, eval_every=1, iterations=3,
                         obstacle_prob=0.3, ice_prob=0.3, test_once=True, curriculum="uniform",
                         skills=",".join(GROUPS), **flags)


def test_flag_combinations_are_checked(monkeypatch, tmp_path):
    cfg = tiny(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="test-once"):
        T.train(dataclasses.replace(cfg, test_once=False))
    with pytest.raises(ValueError, match="distillation needs"):
        T.train(dataclasses.replace(cfg, distill_coef=1.0))
    with pytest.raises(ValueError, match="not in --skills"):
        T.train(dataclasses.replace(cfg, skills="basic", teachers="extra=x.pt", distill_coef=1.0))
    with pytest.raises(ValueError, match="each once"):
        T.train(dataclasses.replace(cfg, skills="basic,ice"))
    assert not (tmp_path / cfg.run).exists()


def test_tiny_distillation_run_pauses_resumes_and_tests_once(monkeypatch, tmp_path):
    torch.manual_seed(0)
    paths = {}
    for name in ("a", "b"):
        paths[name] = tmp_path / f"teacher_{name}.pt"
        torch.save({"model": T.ActorCritic(hidden=(8,)).state_dict(), "config": {"hidden": [8], "hide": ""}}, paths[name])
    init = tmp_path / "initial.pt"
    torch.save({"model": T.ActorCritic(hidden=(8,)).state_dict()}, init)
    teachers = f"basic={paths['a']},advanced={paths['b']},extra={paths['a']}"
    cfg = tiny(monkeypatch, tmp_path, teachers=teachers, distill_coef=1.0, distill_decay=True, init_from=str(init))
    out = tmp_path / cfg.run
    out.mkdir()
    (out / "pause_requested").touch()
    T.train(cfg)
    assert json.loads((out / "pause_ready.json").read_text())["iteration"] == 0
    (out / "pause_requested").unlink()
    monkeypatch.chdir(tmp_path)
    T.train(dataclasses.replace(cfg, init_from="tiny-skills/checkpoints/latest.pt"))

    assert json.loads((out / "config.json").read_text())["teachers"] == teachers
    log = list(csv.DictReader((out / "train_log.csv").open()))
    assert [int(r["iteration"]) for r in log] == [0, 1, 2]
    assert all(math.isfinite(float(r["distill_kl"])) and float(r["distill_kl"]) >= 0 for r in log)
    assert [float(r["distill_w"]) for r in log] == pytest.approx([1.0, 2 / 3, 1 / 3])  # the schedule survived the pause
    assert "train_success_L14" in log[0] and "train_success_L15" not in log[0]
    with (out / "eval_val_log.csv").open() as f:
        header, *rows = list(csv.reader(f))
    assert header == ["iteration", "env_steps", "success", "success_basic", "success_ice", "success_advanced", "success_extra"]
    assert len(rows) == 3
    for r in rows:  # the selection score is the mean over the four groups
        assert float(r[2]) == pytest.approx(sum(map(float, r[3:])) / 4, abs=2e-4)
    summary = json.loads((out / "summary.json").read_text())
    best = summary["best"]
    assert best["validation_success"] == pytest.approx(max(float(r[2]) for r in rows), abs=1e-4)
    assert set(best["validation_groups"]) == {"basic", "ice", "advanced", "extra"}
    test = summary["test_eval"]
    assert summary["completed"] and test["iteration"] == best["iteration"]
    assert list(test["groups"]) == ["basic", "ice", "advanced", "extra"] and test["groups"]["ice"]["obstacles"] is None
    assert test["macro"] == pytest.approx(sum(v["all"] for v in test["groups"].values()) / 4)
    with (out / "eval_log.csv").open() as f:
        header, *rows = list(csv.reader(f))
    assert len(header) == 3 + 3 * 21 and len(rows) == 1  # tested once, levels 0-20
    with (out / "eval_obstacles_log.csv").open() as f:
        header, row = list(csv.reader(f))
    assert len(header) == 3 + 21 and row[3 + 15:] == [""] * 6  # no ice group among the obstacle courses

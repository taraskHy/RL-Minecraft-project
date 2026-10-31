"""The neo skill (parkour.neo_skill): its block, its training episodes and the switch in neo_assist."""
import dataclasses
import json

import torch

from parkour import courses as C
from parkour import neo_skill as NS
from parkour import train as T
from parkour.env import CoursePool, EnvConfig, ParkourEnv
from parkour.live import load_model
from parkour.neo_assist import (SINGLE_VALIDATION_FAMILY, AssistConfig, evaluate_assisted, single_neo_courses,
                                single_neo_hashes, skill_cps)
from parkour.neo_dagger import neo_takeoffs
from parkour.paths import ROOT

FINAL = ROOT / "results" / "final_policy.pt"
NEO = ROOT / "results" / "neo_policy.pt"


def courses(n=12):
    return single_neo_courses(SINGLE_VALIDATION_FAMILY, n)


def test_block_and_switch_cover_k_minus_1_to_k_plus_1():
    for c in courses(32):
        (k,) = neo_takeoffs(c)
        assert NS.block(c) == (k, max(k - 1, 0), min(k + 2, c.n - 1))
        assert skill_cps(c) == list(range(max(k - 1, 0), k + 2))


def test_block_rule():
    k, end = torch.tensor([4] * 7), torch.tensor([6] * 7)
    cp = torch.tensor([3, 3, 4, 4, 5, 5, 3])
    new = torch.tensor([4, 5, 5, 6, 6, 7, 3])
    success, skipped = NS.block_rule(cp, new, k, end)
    assert skipped.tolist() == [False, True, False, True, False, False, False]  # past the takeoff, past the target
    assert success.tolist() == [False, False, False, False, True, True, False]


def test_switch_plays_the_skill_inside_the_block_and_the_model_outside():
    """evaluate_assisted with a skill equals a loop that picks the skill's action while cp is in skill_cps."""
    cs = courses(10)
    model, skill = load_model(FINAL, "cpu"), load_model(NEO, "cpu")
    res = evaluate_assisted(model, CoursePool(cs, "cpu", C.GRID), "cpu", AssistConfig(enabled=False), skill=skill)
    pool = CoursePool(cs, "cpu", C.GRID)
    n = len(cs)
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=200, max_ticks=4000), sampler=None)
    obs = env.reset_all(torch.arange(n))
    window = torch.zeros(n, C.MAX_PLATFORMS, dtype=torch.bool)
    for i, c in enumerate(cs):
        window[i, skill_cps(c)] = True
    finished = torch.zeros(n, dtype=torch.bool)
    success = torch.zeros(n, dtype=torch.bool)
    ticks = torch.zeros(n, dtype=torch.int64)
    used = torch.zeros(n, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(4000):
            use = window[torch.arange(n), env.cp] & ~finished
            used |= use
            a = torch.where(use[:, None], skill.act(obs, True)[0], model.act(obs, True)[0])
            obs, _, _, _, _, info = env.step(a)
            first = info["done"] & ~finished
            success |= first & info["success"]
            ticks = torch.where(first, info["ep_len"], ticks)
            finished |= info["done"]
            if bool(finished.all()):
                break
    assert bool(used.all())
    assert [c["success"] for c in res["per_course"]] == success.tolist()
    assert [c["ticks"] for c in res["per_course"]] == ticks.tolist()
    alone = evaluate_assisted(model, CoursePool(cs, "cpu", C.GRID), "cpu", AssistConfig(enabled=False))
    same = evaluate_assisted(model, CoursePool(cs, "cpu", C.GRID), "cpu", AssistConfig(enabled=False), skill=model)
    assert same["per_course"] == alone["per_course"]  # the model as its own skill changes nothing


def test_a_segment_starts_as_the_switch_hands_over():
    """The first observation of a training episode equals the observation the skill gets from the switch."""
    cs = [c for c in courses(40) if NS.block(c)[1] >= 1][:6]
    model = load_model(FINAL, "cpu")
    starts = NS.collect_starts(model, cs, "cpu")
    keep = [i for i, s in enumerate(starts) if s is not None]
    assert keep
    cs = [cs[i] for i in keep]
    starts = [starts[i] for i in keep]
    n = len(cs)
    # The handover in a full episode: the first tick the furthest platform reaches the block's first platform.
    pool = CoursePool(cs, "cpu", C.GRID)
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=200, max_ticks=2000), sampler=None)
    obs = env.reset_all(torch.arange(n))
    first = torch.tensor([NS.block(c)[1] for c in cs])
    handover = [None] * n
    finished = torch.zeros(n, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(2000):
            a, _, _ = model.act(obs, True)
            obs, _, _, _, _, info = env.step(a)
            finished |= info["done"]
            for i in torch.nonzero(~finished & (env.cp >= first)).flatten().tolist():
                if handover[i] is None:
                    handover[i] = obs[i].clone()
            if all(h is not None for h in handover):
                break
    kinds = {"pos": torch.float64, "vel": torch.float64, "yaw": torch.float64, "on_ground": torch.bool,
             "jump_ticks": torch.int64, "collided_h": torch.bool, "collided_v": torch.bool, "cp": torch.int64,
             "last_action": torch.int64}
    slots = {key: torch.tensor([s[key] for s in starts], dtype=dt)[:, None] for key, dt in kinds.items()}
    slots["valid"] = torch.ones(n, 1, dtype=torch.bool)
    slots["k"] = torch.tensor([NS.block(c)[0] for c in cs])
    slots["end"] = torch.tensor([NS.block(c)[2] for c in cs])
    seg = NS.SegmentEnv(CoursePool(cs, "cpu", C.GRID), EnvConfig(num_envs=n), slots=slots)
    o = seg.reset_all(torch.arange(n))
    assert torch.equal(o, torch.stack(handover))
    assert seg.cp.tolist() == [s["cp"] for s in starts]


def test_the_base_plays_the_jump_after_the_block():
    """With a base policy the episode goes on to k+3, and from k+2 on the base's own action is played."""
    cs = [c for c in courses(40) if 1 <= NS.block(c)[1] and NS.block(c)[0] + 3 < c.n][:4]
    base = load_model(FINAL, "cpu")
    starts = NS.collect_starts(base, cs, "cpu")
    keep = [i for i, s in enumerate(starts) if s is not None]
    cs, starts = [cs[i] for i in keep], [starts[i] for i in keep]
    n = len(cs)
    kinds = {"pos": torch.float64, "vel": torch.float64, "yaw": torch.float64, "on_ground": torch.bool,
             "jump_ticks": torch.int64, "collided_h": torch.bool, "collided_v": torch.bool, "cp": torch.int64,
             "last_action": torch.int64}
    slots = {key: torch.tensor([s[key] for s in starts], dtype=dt)[:, None] for key, dt in kinds.items()}
    slots["valid"] = torch.ones(n, 1, dtype=torch.bool)
    slots["k"] = torch.tensor([NS.block(c)[0] for c in cs])
    slots["end"] = torch.tensor([NS.block(c)[2] for c in cs])
    env = NS.SegmentEnv(CoursePool(cs, "cpu", C.GRID), EnvConfig(num_envs=n), slots=slots, base=base)
    env.reset_all(torch.arange(n))
    given = torch.tensor([[0, 0, 0, 0, 0]] * n)  # back, left, no jump, no sprint, -20 degrees: never the base's
    env.cp = torch.where(torch.arange(n) % 2 == 0, slots["k"] + 2, env.cp)  # half the rows past the block
    expect = base.act(env.observe(), deterministic=True)[0]
    _, _, _, _, _, info = env.step(given)
    handed = torch.arange(n) % 2 == 0
    assert info["ctrl"].tolist() == (~handed).tolist()
    played = env.last_action  # (no row has finished in one tick)
    assert torch.equal(played[handed], expect[handed]) and torch.equal(played[~handed], given[~handed])
    k, end = slots["k"], slots["end"]
    success, _ = NS.block_rule(k + 2, k + 2, k, torch.minimum(end + 1, torch.tensor([c.n - 1 for c in cs])))
    assert not bool(success.any())  # standing on k+2 is not the end any more: k+3 is


def test_router_hands_the_block_to_the_skill_where_the_course_list_does():
    """With the learned router (no course list) the skill plays at the same checkpoints as skill_cps."""
    from parkour.planner import PlannerConfig
    from parkour.route_env import RouteConfig, RouteTables
    from parkour.route_neo import evaluate_routed_neo
    from parkour.route_policy import load_router
    from parkour.route_train import router_choosers
    cs = courses(16)
    jumper, skill = load_model(FINAL, "cpu"), load_model(NEO, "cpu")
    path = ROOT / "results" / "routing" / "router_policy.pt"
    choose, fill = router_choosers(load_router(path, "cpu"), True)
    fill_mode = torch.load(path, map_location="cpu", weights_only=False)["config"].get("fill", "greedy")
    pool = CoursePool(cs, "cpu", C.GRID)
    res = evaluate_routed_neo(jumper, pool, RouteTables(pool), choose, fill, PlannerConfig(),
                              RouteConfig(fill=fill_mode, blocked="plan"), AssistConfig(enabled=False), skill=skill)
    for c, pc in zip(cs, res["per_course"]):
        end = c.n - 1 if pc["success"] else pc["stopped_at"]
        assert pc["skill_at"] == [k for k in skill_cps(c) if k <= end and k < c.n - 1]
    for bad in (RouteConfig(fill=fill_mode, blocked="drop"),):
        try:
            evaluate_routed_neo(jumper, pool, RouteTables(pool), choose, fill, PlannerConfig(), bad,
                                AssistConfig(enabled=False), skill=skill)
            raise AssertionError("the skill needs the blocked choices kept")
        except ValueError:
            pass


def test_bank_and_training_on_segments(tmp_path, monkeypatch):
    bank = tmp_path / "bank.pt"
    meta = NS.build_bank(FINAL, bank, families=(60,), per_family=8, device="cpu", batch=8)
    assert meta["generated"] == 8 and 1 <= meta["kept"] <= 8
    b = NS.SegmentBank(bank, "cpu")
    assert not {T.course_hash(c) for c in b.courses} & single_neo_hashes()
    assert sorted(c.level for c in b.courses) == sorted(json.loads(json.dumps(torch.load(bank, weights_only=False)["types"])))
    first = b.take(3)
    assert len(b.take(len(b.courses))) == len(b.courses) and b.next == 3 % len(b.courses)
    rows = b.rows(first)
    assert rows["pos"].shape == (3, len(NS.SETTLES), 3) and bool(rows["valid"].any(1).all())
    assert rows["pos"].dtype == torch.float64 and rows["last_action"].dtype == torch.int64
    starts = NS.collect_starts(load_model(FINAL, "cpu"), [b.courses[0]], "cpu")  # the bank keeps the exact state
    if starts[0] is not None and bool(rows["valid"][0, 0]):
        assert rows["pos"][0, 0].tolist() == starts[0]["pos"] and float(rows["yaw"][0, 0]) == starts[0]["yaw"]
    # A few PPO updates on the block alone (tiny network, CPU).
    monkeypatch.setattr(T, "RUNS", tmp_path)
    original = T.held_out_courses
    monkeypatch.setattr(T, "held_out_courses", lambda levels, per_level, n_jumps, obstacle_prob=0, family=1, ice_prob=0.0:
                        original(levels, 1, 2, obstacle_prob=obstacle_prob, family=family))
    cfg = T.TrainConfig(run="tiny-skill", device="cpu", threads=1, num_envs=8, rollout=4, pool_size=4, pool_refresh_every=2,
                        pool_refresh_frac=0.5, hidden=(8,), epochs=1, minibatches=1, eval_every=2, iterations=3, test_once=True,
                        segments=str(bank), save_at="1,3")
    out = T.train(cfg)
    log = (out / "train_log.csv").read_text().splitlines()
    assert len(log) == 4 and "train_success_L31" in log[0]
    assert (out / "checkpoints" / "updates_00003.pt").exists()
    assert json.loads((out / "config.json").read_text())["segments"] == str(bank)
    based = T.train(dataclasses.replace(cfg, run="tiny-skill-base", segment_base=str(FINAL), hidden=(512, 256, 128),
                                        init_from=str(FINAL)))
    assert len((based / "train_log.csv").read_text().splitlines()) == 4
    try:
        T.train(dataclasses.replace(cfg, run="base-alone", segments="", segment_base=str(FINAL)))
        raise AssertionError("--segment-base needs --segments")
    except ValueError:
        pass
    try:
        T.train(dataclasses.replace(cfg, run="no-test", test_once=False))
        raise AssertionError("--segments needs --test-once")
    except ValueError:
        pass

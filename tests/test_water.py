"""Water pieces in courses and environments (parkour.water; the physics itself: test_physics_parity.test_water_cpu)."""
import json

import numpy as np
import torch

from parkour import courses as C
from parkour import physics
from parkour import water as W
from parkour.env import CoursePool, EnvConfig, ParkourEnv, in_water
from parkour.paths import ROOT


def water_courses(n_per_spec=1, seed=7):
    rng = np.random.default_rng([2026, 96, seed])
    basic = C.load_levels()
    out = []
    for i, sp in enumerate(W.water_specs() * n_per_spec):
        for _ in range(500):
            c = W.water_course(rng, basic, {2: sp}, 5, seed=i)
            if c is not None:
                out.append((sp, c))
                break
    return out


def test_water_in_json_and_only_there():
    sp, c = water_courses()[0]
    d = c.to_json()
    assert d["water"] and all(level == 0 for *_, level in d["water"])
    back = C.Course.from_json(json.loads(json.dumps(d)))
    nospecs = lambda j: {k: v for k, v in j.items() if k != "specs"}  # noqa: E731 (from_json leaves the specs out)
    assert np.array_equal(back.water, c.water) and nospecs(back.to_json()) == nospecs(d)
    plain = C.Course.from_json({k: v for k, v in d.items() if k != "water"})
    assert "water" not in plain.to_json() and len(plain.water) == 0


def test_pools_and_basins():
    cs = water_courses()
    assert len(cs) == len(W.water_specs())  # every kind fits
    for sp, c in cs:
        wet = {tuple(w[:3]) for w in c.water.tolist()}
        solid = {tuple(b) for b in c.blocks().tolist()}
        assert not wet & solid
        assert len(wet) == sp.length * sp.width * sp.depth
        for x, y, z in wet:  # still water: no air beside it or below it (in the real game it would flow there)
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, -1, 0)):
                q = (x + dx, y + dy, z + dz)
                assert q in wet or q in solid, (sp.key(), q)
        exit_ = c.platforms[3]  # the piece is jump 2: platform 2 the takeoff, 3 the exit
        assert exit_[4] == max(y for _, y, _ in wet)  # the exit is level with the surface's row
    pool = CoursePool([c for _, c in cs], "cpu", C.GRID)
    assert pool.water is not None and pool.world.water is not None
    dry = CoursePool([C.Course.from_json({k: v for k, v in cs[0][1].to_json().items() if k != "water"})], "cpu", C.GRID)
    assert dry.water is None and dry.world.water is None


def test_in_water_is_no_fall():
    """A player put into a pool sinks without the episode ending; on the same course without water, it falls."""
    sp, c = next((sp, c) for sp, c in water_courses() if sp.kind == "drop")
    x, y, z = c.water[0][:3]
    for with_water in (True, False):
        course = c if with_water else C.Course.from_json({k: v for k, v in c.to_json().items() if k != "water"})
        pool = CoursePool([course], "cpu", C.GRID)
        env = ParkourEnv(pool, EnvConfig(num_envs=1, stall_ticks=200, max_ticks=400), sampler=None)
        env.reset_all(torch.arange(1))
        env.state.pos = torch.tensor([[x + 0.5, y + 0.2, z + 0.5]], dtype=torch.float64)
        env.state.vel = torch.zeros(1, 3, dtype=torch.float64)
        env.state.on_ground = torch.zeros(1, dtype=torch.bool)
        env.cp = torch.tensor([2])
        fell = False
        for _ in range(60):
            _, _, _, _, _, info = env.step(torch.tensor([[1, 1, 0, 0, 3]]))  # no keys
            fell |= bool(info["fell"][0])
            if bool(info["done"][0]):
                break
        assert fell != with_water
        if with_water:
            assert bool(in_water(pool, env.state, env.course)[0])


def test_no_piece_is_passed_dry():
    """The final policy (which never saw water) completes no water piece without going into the water."""
    from parkour.live import load_model
    cs = water_courses()
    pool = CoursePool([c for _, c in cs], "cpu", C.GRID)
    model = load_model(ROOT / "results" / "final_policy.pt", "cpu")
    n = len(cs)
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=200, max_ticks=1200), sampler=None)
    obs = env.reset_all(torch.arange(n))
    done = torch.zeros(n, dtype=torch.bool)
    wet = torch.zeros(n, dtype=torch.bool)
    past = torch.zeros(n, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(1200):
            a, _, _ = model.act(obs, deterministic=True)
            obs, _, _, _, _, info = env.step(a)
            wet |= in_water(pool, env.state, env.course) & ~done
            past |= (env.cp >= 3) & ~done
            done |= info["done"]
            if bool(done.all()):
                break
    assert not bool((past & ~wet).any())


def test_water_observation_extends_the_standard_one():
    """EnvConfig.water_obs appends OBS_LAYOUT_WATER's features after the standard 255 numbers, which stay as they
    were; in a pool the player reads in water with the surface above the feet, on a dry course nothing."""
    from parkour.env import OBS_DIM, OBS_DIM_WATER
    sp, c = next((sp, c) for sp, c in water_courses() if sp.kind == "drop")
    x, y, z = c.water[0][:3]
    pool = CoursePool([c, c], "cpu", C.GRID)
    envs = [ParkourEnv(pool, EnvConfig(num_envs=2, water_obs=w), sampler=None) for w in (False, True)]
    obs = []
    for env in envs:
        env.reset_all(torch.arange(2))
        env.state.pos[1] = torch.tensor([x + 0.5, y + 0.2, z + 0.5], dtype=torch.float64)
        obs.append(env.observe())
    assert obs[0].shape == (2, OBS_DIM) and obs[1].shape == (2, OBS_DIM_WATER)
    assert torch.equal(obs[1][:, :OBS_DIM], obs[0])
    wet = obs[1][:, OBS_DIM:]
    assert wet[0, 0] == 0 and wet[1, 0] == 1  # on the spawn platform, in the pool
    assert wet[1, 1] > 0  # the surface is above the feet
    dry = CoursePool([C.Course.from_json({k: v for k, v in c.to_json().items() if k != "water"})], "cpu", C.GRID)
    env = ParkourEnv(dry, EnvConfig(num_envs=1, water_obs=True), sampler=None)
    env.reset_all(torch.arange(1))
    o = env.observe()[0, OBS_DIM:]
    assert o[0] == 0 and o[1] == 0 and bool((o[2:] < 0).all())  # no water anywhere: NO_FLOOR / 3


def test_expand_to_water_acts_as_before():
    """A standard network widened to the water observation (train.expand_to_water) acts and values as before."""
    from parkour.env import OBS_DIM, OBS_DIM_WATER
    from parkour.model import ActorCritic
    from parkour.train import expand_to_water
    torch.manual_seed(0)
    small = ActorCritic(OBS_DIM)
    small.norm.mean.uniform_(-1, 1)
    small.norm.var.uniform_(0.5, 2.0)
    big = ActorCritic(OBS_DIM_WATER)
    big.load_state_dict(expand_to_water(small.state_dict()))
    obs = torch.randn(64, OBS_DIM_WATER) * 3
    with torch.no_grad():
        for a, b in zip(small.logits(small.norm(obs[:, :OBS_DIM])), big.logits(big.norm(obs))):
            assert torch.allclose(a, b, atol=1e-5)
        assert torch.allclose(small.value(small.norm(obs[:, :OBS_DIM])), big.value(big.norm(obs)), atol=1e-5)


def test_water_sets_hold_their_pieces():
    """The water sets' pieces are where water_plan puts them, every 12 courses hold every kind once, and the skill's
    checkpoints are each piece's t-1 and t."""
    cs = W.water_set(W.WATER_VALIDATION_FAMILY, 18)
    assert len(W.water_specs()) == 24
    for i, c in enumerate(cs):
        plan = W.water_plan(i)
        assert W.water_ok(c, plan) and W.water_takeoffs(c) == sorted(plan)
        assert W.water_cps(c) == sorted({k for t in plan for k in (t - 1, t)})
        assert W.water_blocks(c) == [(t - 1, t - 1) for t in sorted(plan)]
        assert len(c.extra) <= C.MAX_EXTRA
    assert sorted(sp.key() for i in range(12) for sp in W.water_plan(i).values()) == sorted(
        sp.key() for sp in W.water_specs())
    assert W.water_set(W.WATER_VALIDATION_FAMILY, 3)[2].to_json() == cs[2].to_json()  # (a longer set: the same first)


def test_a_widened_final_policy_as_the_water_skill_changes_nothing(tmp_path, monkeypatch):
    """The water bank (neo_skill bank --water) and its segments with the water observation, and the course-list
    mode with a water skill beside the final policy: with the final policy widened as the skill, every course ends
    as with the final policy alone (the skill reads the same prefix with zero weights on the rest)."""
    from parkour import neo_skill as NS
    from parkour.live import load_model
    from parkour.neo_assist import AssistConfig, evaluate_assisted
    from parkour.model import ActorCritic
    from parkour.env import OBS_DIM_WATER
    from parkour.train import expand_to_water, load_frozen
    monkeypatch.setattr(W, "held_water_hashes", lambda: frozenset())  # (the test needs no 648 held-out courses)
    ck = ROOT / "results" / "final_policy.pt"
    meta = NS.build_bank(ck, tmp_path / "bank.pt", families=(102,), per_family=6, device="cpu", water=True)
    assert meta["kind"] == "water" and meta["n_types"] == len(W.water_specs()) and meta["kept"] >= 1
    bank = NS.SegmentBank(tmp_path / "bank.pt", "cpu")
    assert all(int(k) == W.water_blocks(c)[0][0] for k, c in zip(bank.k.tolist(), bank.courses))
    n = len(bank.courses)
    pool = CoursePool(bank.courses, "cpu", C.GRID)
    base = load_frozen(str(ck), "cpu")
    env = NS.SegmentEnv(pool, EnvConfig(num_envs=n, water_obs=True), sampler=None, slots=bank.rows(list(range(n))),
                        base=base)
    obs = env.reset_all(torch.arange(n))
    assert obs.shape == (n, OBS_DIM_WATER)
    final = load_model(ck, "cpu")
    skill = ActorCritic(OBS_DIM_WATER, hidden=final.hidden)
    skill.load_state_dict(expand_to_water(final.state_dict()))
    skill.eval()
    with torch.no_grad():
        for _ in range(300):
            a, _, _ = skill.act(obs, deterministic=True)
            obs, *_ = env.step(a)
    cs = W.water_set(W.WATER_VALIDATION_FAMILY, 4)
    runs = []
    for sk in (None, skill):
        res = evaluate_assisted(final, CoursePool(cs, "cpu", C.GRID), "cpu", AssistConfig(enabled=False), skill=sk,
                                skill_blocks="water")
        runs.append([(r["success"], r["ticks"], r["progress"]) for r in res["per_course"]])
    assert runs[0] == runs[1]


def test_bridge_water_observation_matches():
    """bridge/policy.js's water observation (OBS_DIM_WATER) and in-water test match the Python ones on the final
    policy's own states over water pieces (it ends up in the pools) and on states put into the water."""
    import os
    import subprocess
    import tempfile

    import pytest
    from parkour.env import OBS_DIM_WATER
    from parkour.live import load_model
    from parkour.paths import BRIDGE, NODE
    if not NODE.exists():
        pytest.skip("portable Node runtime not installed (run setup.cmd)")
    cs = [c for _, c in water_courses()]
    pool = CoursePool(cs, "cpu", C.GRID)
    n = len(cs)
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=10 ** 6, max_ticks=10 ** 6, water_obs=True), sampler=None)
    obs = env.reset_all(torch.arange(n))
    model = load_model(ROOT / "results" / "final_policy.pt", "cpu")
    g = torch.Generator().manual_seed(5)
    cases, py_obs, py_wet = [], [], []

    def record(o):
        s = env.state
        wet = in_water(pool, s, env.course)
        for i in range(n):
            cases.append({"course": int(env.course[i]), "cp": int(env.cp[i]), "lastAction": env.last_action[i].tolist(),
                          "state": {"pos": s.pos[i].tolist(), "vel": s.vel[i].tolist(), "yaw": float(s.yaw[i]),
                                    "onGround": bool(s.on_ground[i]), "jumpTicks": int(s.jump_ticks[i]),
                                    "collidedH": bool(s.collided_h[i])}})
            py_obs.append(o[i].numpy())
            py_wet.append(bool(wet[i]))

    with torch.no_grad():
        for t in range(400):
            a, _, _ = model.act(obs[:, :model.obs_dim], deterministic=True)
            a[:, 4] = torch.where(torch.rand(n, generator=g) < 0.3, torch.randint(0, 7, (n,), generator=g), a[:, 4])
            obs, *_ = env.step(a)
            if t % 5 == 0:
                record(obs)
        for k in range(6):  # into every pool: a random cell, at random heights in and over it
            w = torch.stack([torch.as_tensor(c.water[int(torch.randint(len(c.water), (1,), generator=g))][:3])
                             for c in cs]).to(torch.float64)
            env.state.pos = w + torch.stack((torch.rand(n, generator=g), torch.rand(n, generator=g) * 2.4 - 0.4,
                                             torch.rand(n, generator=g)), -1).to(torch.float64)
            env.state.yaw = (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1) * np.pi
            record(env.observe())
    assert sum(py_wet) > 200 and len(py_wet) - sum(py_wet) > 200
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, "in.json")
        with open(inp, "w") as f:
            json.dump({"policy": None, "courses": [{"platforms": c.platforms.tolist(), "extra": c.extra.tolist(),
                                                    "water": c.water.tolist()} for c in cs], "cases": cases}, f)
        out = subprocess.run([str(NODE), str(BRIDGE / "obs_parity.js"), inp], capture_output=True, text=True,
                             check=True, cwd=str(BRIDGE))
    js = json.loads(out.stdout)
    js_obs = np.array(js["obs"])
    assert js_obs.shape == (len(cases), OBS_DIM_WATER)
    assert np.abs(js_obs - np.array(py_obs)).max() < 1e-5  # (Python's observation is float32)
    assert js["wet"] == py_wet


def test_offline_bot_plays_water_like_the_simulator(tmp_path):
    """The offline bot (prismarine-physics without a server) with a water skill beside the final policy plays each
    water course as neo_assist.evaluate_assisted(..., skill_blocks="water") does on the CPU: the same outcome in the
    same tick. The skill is the final policy widened with random weights on the water inputs, so it acts on them."""
    import os
    import subprocess

    import pytest
    from parkour.env import OBS_DIM
    from parkour.live import SETTLE_TICKS, load_model
    from parkour.model import ActorCritic
    from parkour.neo_assist import AssistConfig, evaluate_assisted
    from parkour.paths import BRIDGE, NODE
    from parkour.train import expand_to_water
    if not NODE.exists():
        pytest.skip("portable Node runtime not installed (run setup.cmd)")
    cs = [c for _, c in water_courses()]
    final = load_model(ROOT / "results" / "final_policy.pt", "cpu")
    state = expand_to_water(final.state_dict())
    g = torch.Generator().manual_seed(1)
    for key in ("actor.0.weight", "critic.0.weight"):
        state[key][:, OBS_DIM:] = torch.randn(state[key][:, OBS_DIM:].shape, generator=g) * 0.3
    from parkour.env import OBS_DIM_WATER
    skill = ActorCritic(OBS_DIM_WATER, hidden=final.hidden)
    skill.load_state_dict(state)
    skill.eval()
    torch.save({"model": skill.state_dict(), "config": {"hidden": list(final.hidden)}}, tmp_path / "skill.pt")
    res = evaluate_assisted(final, CoursePool(cs, "cpu", C.GRID), "cpu", AssistConfig(enabled=False), settle=SETTLE_TICKS,
                            skill=load_model(tmp_path / "skill.pt", "cpu"), skill_blocks="water")["per_course"]
    pol, crs, out, sk = (str(tmp_path / f) for f in ("policy.json", "courses.json", "episodes.jsonl", "skill.json"))
    final.export(pol)
    skill.export(sk)
    with open(crs, "w") as f:
        json.dump({"courses": [{**c.to_json(), "skill_cps": W.water_cps(c)} for c in cs]}, f)
    subprocess.run([str(NODE), str(BRIDGE / "offline_bot.js"), "--policy", pol, "--courses", crs, "--out", out,
                    "--settle", str(SETTLE_TICKS), "--skill", sk], check=True, cwd=str(BRIDGE), capture_output=True)
    with open(out) as f:
        js = [json.loads(line) for line in f if line.strip()]
    assert [(r["success"], r["ticks"]) for r in res] == [(r["success"], r["ticks"]) for r in js]
    assert sum(r["skill_ticks"] for r in js) > 200  # (the skill played, in and over the water)
    assert os.path.getsize(out) > 0


def test_water_steps_without_the_host_syncs_are_the_same(monkeypatch):
    """physics.SYNC_SKIPS: on a device not in it (the GPU) the water code computes every cell's current and the jump
    out of the water for all rows instead of skipping what no row needs; the states are the same to the last bit."""
    cs = [c for _, c in water_courses()]
    n = len(cs)
    runs = []
    for skips in (("cpu",), ()):
        monkeypatch.setattr(physics, "SYNC_SKIPS", skips)
        pool = CoursePool(cs, "cpu", C.GRID)
        env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=10 ** 6, max_ticks=10 ** 6), sampler=None)
        env.reset_all(torch.arange(n))
        g = torch.Generator().manual_seed(2)
        x, y, z = (torch.stack([torch.as_tensor(c.water[0][i]) for c in cs]).to(torch.float64) for i in range(3))
        env.state.pos = torch.stack((x + 0.5, y + 0.9, z + 0.5), -1)  # in every pool, then random keys
        out = []
        for _ in range(150):
            a = torch.stack([torch.randint(0, d, (n,), generator=g) for d in (3, 3, 2, 2, 7)], -1)
            a[:, 0] = 2
            env.step(a)
            s = env.state
            out.append(torch.cat((s.pos, s.vel, s.on_ground[:, None].double(), s.collided_h[:, None].double()), -1))
        runs.append(torch.stack(out))
    assert torch.equal(runs[0], runs[1])


def test_in_water_matches_the_current_s():
    rng = np.random.default_rng(3)
    from parkour import parity
    cases = [parity.random_case(rng, ticks=2, water=0.7) for _ in range(4)]
    n = len(cases)
    ext = np.max([c.blocks.max(axis=0) for c in cases], axis=0) + 8
    solid = torch.zeros(n, *ext.tolist(), dtype=torch.bool)
    water = torch.full(solid.shape, -1, dtype=torch.int16)
    for i, c in enumerate(cases):
        b = torch.as_tensor(c.blocks)
        solid[i, b[:, 0], b[:, 1], b[:, 2]] = True
        w = torch.as_tensor(c.water)
        water[i, w[:, 0], w[:, 1], w[:, 2]] = w[:, 3].to(torch.int16)
    world = physics.World(solid, water=water)
    pos = torch.tensor([[float(rng.uniform(30, 40)), float(rng.uniform(1.5, 5)), float(rng.uniform(30, 40))]
                        for _ in range(n)], dtype=torch.float64)
    course = torch.arange(n)
    _, wet = physics._water_current(world, course, pos, torch.zeros(n, 3, dtype=torch.float64))
    assert torch.equal(physics.in_water(world, course, pos), wet)

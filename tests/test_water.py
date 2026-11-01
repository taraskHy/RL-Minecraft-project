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

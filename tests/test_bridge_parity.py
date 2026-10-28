"""The live bot's JavaScript observation, rules and network match the Python ones."""
import json
import os
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from parkour import courses as C
from parkour import physics
from parkour.env import CoursePool, EnvConfig, LevelSampler, ParkourEnv, standing_on
from parkour.model import ActorCritic
from parkour.paths import BRIDGE, NODE


def record_states(n_envs=64, steps=120, seed=0, shaped=False):
    levels = C.load_levels()
    rng = np.random.default_rng(seed)
    if shaped:  # courses whose level-6 jumps land on slabs, soul sand, posts, ...
        from parkour.feasibility import shape_specs
        levels = levels + [shape_specs()]
        cs = [C.generate(rng, levels, 6, 8) for i in range(32)]
        assert all(c.shapes for c in cs)
    else:
        # Half the courses carry obstacles (beams and pillars), so the scan sees non-platform blocks.
        cs = [C.generate(rng, levels, i % len(levels), 8, obstacle_prob=0.6 if i % 2 else 0.0) for i in range(32)]
    pool = CoursePool(cs, "cpu", C.GRID)
    env = ParkourEnv(pool, EnvConfig(num_envs=n_envs), LevelSampler(len(levels), fixed=[1] * len(levels)), seed=seed)
    model = ActorCritic()
    torch.manual_seed(seed)
    obs = env.observe()
    cases, py_obs, py_logits, py_stand = [], [], [], []
    for t in range(steps):
        with torch.no_grad():
            a, _, _ = model.act(obs)
            # Bias towards forward + sprint + jump so the states cover flights and landings.
            a[:, 0] = torch.where(torch.rand(n_envs) < 0.7, 2, a[:, 0])
            a[:, 3] = torch.where(torch.rand(n_envs) < 0.6, 1, a[:, 3])
        obs, _, _, _, _, _ = env.step(a)
        if t % 6 == 0:
            s = env.state
            stand = standing_on(pool, s, env.course)
            k = torch.arange(stand.shape[1])
            top = torch.where(stand, k, torch.full_like(k, -1)).amax(1)
            with torch.no_grad():
                lg = torch.cat(model.logits(model.norm(obs)), -1)
            for i in range(n_envs):
                cases.append({"course": int(env.course[i]), "cp": int(env.cp[i]),
                              "lastAction": env.last_action[i].tolist(),
                              "state": {"pos": s.pos[i].tolist(), "vel": s.vel[i].tolist(), "yaw": float(s.yaw[i]),
                                        "onGround": bool(s.on_ground[i]), "jumpTicks": int(s.jump_ticks[i]),
                                        "collidedH": bool(s.collided_h[i])}})
                py_obs.append(obs[i].numpy())
                py_logits.append(lg[i].numpy())
                py_stand.append(int(top[i]))
    if shaped:  # and states standing on every shaped platform: centre and the four edges (overhanging by 0.29)
        g = torch.Generator().manual_seed(seed)
        rows = []
        for ci, c in enumerate(pool.courses):
            for k in c.shapes:
                f, top = pool.foot[ci, k].tolist(), float(pool.top[ci, k])
                mx, mz = (f[0] + f[1]) / 2, (f[2] + f[3]) / 2
                for x, z in ((mx, mz), (f[0] - 0.29, mz), (f[1] + 0.29, mz), (mx, f[2] - 0.29), (mx, f[3] + 0.29)):
                    rows.append((ci, k, [x, top, z]))
        n = len(rows)
        st = physics.PlayerState.at(torch.tensor([r[2] for r in rows], dtype=torch.float64),
                                    (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1) * np.pi)
        st.on_ground = torch.ones(n, dtype=torch.bool)
        course = torch.tensor([r[0] for r in rows])
        cp = torch.tensor([r[1] - int(torch.randint(0, 2, (1,), generator=g)) for r in rows])
        last = torch.stack([torch.randint(0, d, (n,), generator=g) for d in (3, 3, 2, 2, 7)], -1)
        o = env.obs_fn(pool, course, st, cp, last)
        stand = standing_on(pool, st, course)
        k = torch.arange(stand.shape[1])
        top_k = torch.where(stand, k, torch.full_like(k, -1)).amax(1)
        with torch.no_grad():
            lg = torch.cat(model.logits(model.norm(o)), -1)
        for i in range(n):
            cases.append({"course": int(course[i]), "cp": int(cp[i]), "lastAction": last[i].tolist(),
                          "state": {"pos": st.pos[i].tolist(), "vel": [0.0, 0.0, 0.0], "yaw": float(st.yaw[i]),
                                    "onGround": True, "jumpTicks": 0, "collidedH": False}})
            py_obs.append(o[i].numpy())
            py_logits.append(lg[i].numpy())
            py_stand.append(int(top_k[i]))
    return pool, model, cases, np.array(py_obs), np.array(py_logits), np.array(py_stand)


class BridgeParity(unittest.TestCase):
    @unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
    def test_observation_logits_and_rules_match(self):
        self.check(record_states())

    @unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
    def test_shaped_platforms_match(self):
        rec = record_states(seed=3, shaped=True)
        pool, cases, stand = rec[0], rec[2], rec[5]
        on_shaped = sum(1 for c, k in zip(cases, stand) if k in pool.courses[c["course"]].shapes)
        self.assertGreater(on_shaped, 100)
        self.check(rec)

    def check(self, rec):
        pool, model, cases, py_obs, py_logits, py_stand = rec
        with tempfile.TemporaryDirectory() as d:
            pol = os.path.join(d, "policy.json")
            model.export(pol)
            inp = os.path.join(d, "in.json")
            with open(inp, "w") as f:
                json.dump({"policy": pol, "courses": [{"platforms": c.platforms.tolist(), "extra": c.extra.tolist(),
                                                       "shapes": {str(k): v for k, v in c.shapes.items()}}
                                                      for c in pool.courses],
                           "cases": cases}, f)
            out = subprocess.run([str(NODE), str(BRIDGE / "obs_parity.js"), inp], capture_output=True, text=True,
                                 check=True, cwd=str(BRIDGE))
        js = json.loads(out.stdout)
        js_obs = np.array(js["obs"])
        js_logits = np.array(js["logits"])
        # Python casts observations to float32; JavaScript keeps doubles.
        self.assertLess(np.abs(js_obs - py_obs).max(), 1e-5)
        self.assertLess(np.abs(js_logits - py_logits).max(), 1e-3)
        self.assertTrue((np.array(js["standing"]) == py_stand).all())


if __name__ == "__main__":
    unittest.main()

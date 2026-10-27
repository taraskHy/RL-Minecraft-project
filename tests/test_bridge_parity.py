"""The live bot's JavaScript observation, rules and network match the Python ones."""
import json
import os
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from parkour import courses as C
from parkour.env import CoursePool, EnvConfig, LevelSampler, ParkourEnv, standing_on
from parkour.model import ActorCritic
from parkour.paths import BRIDGE, NODE


def record_states(n_envs=64, steps=120, seed=0):
    levels = C.load_levels()
    rng = np.random.default_rng(seed)
    cs = [C.generate(rng, levels, i % len(levels), 8) for i in range(32)]
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
    return pool, model, cases, np.array(py_obs), np.array(py_logits), np.array(py_stand)


class BridgeParity(unittest.TestCase):
    @unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
    def test_observation_logits_and_rules_match(self):
        pool, model, cases, py_obs, py_logits, py_stand = record_states()
        with tempfile.TemporaryDirectory() as d:
            pol = os.path.join(d, "policy.json")
            model.export(pol)
            inp = os.path.join(d, "in.json")
            with open(inp, "w") as f:
                json.dump({"policy": pol, "courses": [{"platforms": c.platforms.tolist()} for c in pool.courses],
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

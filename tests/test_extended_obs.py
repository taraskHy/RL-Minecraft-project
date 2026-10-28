"""The extended observation (far scan and landing memory) is the same in Python and in the bot."""
import json
import os
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from parkour import courses as C
from parkour import physics
from parkour.env import OBS_DIM, OBS_DIM_EXT, CoursePool, EnvConfig, ParkourEnv
from parkour.paths import BRIDGE, NODE, ROOT

SETTLE = 10


class ExtendedObservationParity(unittest.TestCase):
    @unittest.skipUnless(NODE.exists(), "portable Node runtime not installed (run setup.cmd)")
    def test_far_scan_and_memory_match(self):
        from parkour.live import load_model
        levels = C.load_levels()
        rng = np.random.default_rng(3)
        cs = [C.generate(rng, levels, i % len(levels), 8, obstacle_prob=0.5 if i % 2 else 0.0, seed=i) for i in range(8)]
        n, ticks = len(cs), 200
        # The final policy plays (on the standard part of the observation), so the player lands
        # on many platforms (memory) and sees each course from many places (scan).
        model = load_model(ROOT / "results" / "final_policy.pt", "cpu")
        pool = CoursePool(cs, "cpu", C.GRID)
        env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=10 ** 6, max_ticks=10 ** 6, extended_obs=True), sampler=None)
        env.reset_all(torch.arange(n))
        idle = torch.zeros(n, 6, dtype=torch.bool)
        for _ in range(SETTLE):
            env.state = physics.simulate(pool.world, env.course, env.state, idle)
        env.dist = env._distance(env.state.pos, env.course, env.cp)
        env.reset_memory(torch.ones(n, dtype=torch.bool))
        py = [[] for _ in range(n)]
        acts = [[] for _ in range(n)]
        alive = np.ones(n, dtype=bool)
        most_memory = 0
        obs = env.observe()
        for t in range(ticks):
            with torch.no_grad():
                a, _, _ = model.act(obs[:, :OBS_DIM], deterministic=True)
            for i in np.flatnonzero(alive):
                py[i].append(obs[i].double().numpy())
                acts[i].append(a[i].tolist())
            obs, _, _, _, _, info = env.step(a)
            alive &= ~info["done"].numpy()
            most_memory = max(most_memory, int(env.mem_valid[torch.as_tensor(alive)].sum(1).max()) if alive.any() else 0)
        self.assertEqual(most_memory, 3)

        with tempfile.TemporaryDirectory() as d:
            cases, out = os.path.join(d, "cases.json"), os.path.join(d, "obs.json")
            with open(cases, "w") as f:
                json.dump({"courses": [c.to_json() for c in cs], "actions": acts, "settle": SETTLE}, f)
            subprocess.run([str(NODE), str(BRIDGE / "obs_ext_parity.js"), "--cases", cases, "--out", out],
                           check=True, cwd=str(BRIDGE), capture_output=True)
            js = json.load(open(out))
        checked = 0
        for i in range(n):
            k = min(len(py[i]), len(js[i]))
            self.assertGreater(k, 20)
            a, b = np.array(py[i][:k]), np.array(js[i][:k])
            self.assertEqual(a.shape[1], OBS_DIM_EXT)
            np.testing.assert_allclose(a, b, atol=2e-6, err_msg=f"course {i}")
            checked += k
        self.assertGreater(checked, 500)


if __name__ == "__main__":
    unittest.main()

import math
import unittest

import numpy as np
import torch

from parkour import courses as C
from parkour.env import OBS_DIM, CoursePool, EnvConfig, ParkourEnv, RewardConfig


def straight_course(gaps=(1, 1, 1), size=3):
    """Platforms in a straight line along +X, each ``size`` wide."""
    plats = [[4, 4 + size - 1, 20, 20 + size - 1, 10]]
    for g in gaps:
        x0 = plats[-1][1] + g + 1
        plats.append([x0, x0 + size - 1, 20, 20 + size - 1, 10])
    c = C.Course(np.array(plats, dtype=np.int64), np.zeros(3), 0.0)
    c.spawn = c.centers()[0]
    c.spawn_yaw = -math.pi / 2  # facing +X
    return c


def make_env(course, n=1, **reward):
    pool = CoursePool([course], "cpu", C.GRID)
    return ParkourEnv(pool, EnvConfig(num_envs=n, reward=RewardConfig(**reward)))


def act(move=1, strafe=1, jump=0, sprint=0, yaw=3, n=1):
    return torch.tensor([[move, strafe, jump, sprint, yaw]] * n)


class Env(unittest.TestCase):
    def test_observation_shape(self):
        env = make_env(straight_course(), n=3)
        self.assertEqual(env.observe().shape, (3, OBS_DIM))

    def test_standing_still_truncates_without_reward_beyond_time(self):
        env = make_env(straight_course())
        env.cfg.stall_ticks = 30
        total = 0.0
        for t in range(30):
            _, r, term, trunc, _, info = env.step(act())
            total += float(r)
        self.assertTrue(bool(trunc[0]))
        self.assertFalse(bool(term[0]))
        self.assertAlmostEqual(total, -0.01 * 30, places=6)

    def test_stall_penalty_ends_the_episode_as_a_failure(self):
        env = make_env(straight_course(), stall=5.0)
        env.cfg.stall_ticks = 30
        total = 0.0
        for t in range(30):
            _, r, term, trunc, _, info = env.step(act())
            total += float(r)
        self.assertTrue(bool(term[0]))
        self.assertFalse(bool(trunc[0]))
        self.assertFalse(bool(info["fell"][0]))
        self.assertAlmostEqual(total, -0.01 * 30 - 5.0, places=5)

    def test_walking_off_the_side_is_a_fall(self):
        env = make_env(straight_course())
        fell = False
        for t in range(80):
            _, r, term, trunc, _, info = env.step(act(strafe=0))  # strafe left off the platform
            if bool(info["done"][0]):
                fell = bool(info["fell"][0])
                break
        self.assertTrue(fell)

    def test_sprint_jumping_forward_lands_and_counts_platforms(self):
        env = make_env(straight_course(gaps=(1, 1, 1)), landing=1.0, goal=10.0)
        landed, rewards = 0, []
        for t in range(200):
            _, r, term, trunc, _, info = env.step(act(move=2, jump=1, sprint=1))
            rewards.append(float(r))
            if bool(info["done"][0]):
                self.assertTrue(bool(info["success"][0]), "a straight 1-gap course should be solvable by sprint-jumping")
                break
        self.assertGreater(sum(rewards), 10.0)

    def test_progress_reward_telescopes(self):
        # Without bonuses and time cost, the return equals the path length covered.
        course = straight_course(gaps=(1, 1, 1))
        env = make_env(course, landing=0.0, goal=0.0, fall=0.0, time=0.0)
        start = float(env.dist[0])
        total = 0.0
        for t in range(200):
            _, r, term, trunc, _, info = env.step(act(move=2, jump=1, sprint=1))
            total += float(r)
            if bool(info["done"][0]):
                break
        self.assertTrue(bool(info["success"][0]))
        # The return is the start distance minus what is left when standing on the
        # goal platform (at most the half-diagonal of its 3x3 top plus a little).
        self.assertGreater(total, start - 1.5)
        self.assertLess(total, start + 0.01)

    def test_yaw_wraps(self):
        env = make_env(straight_course())
        for _ in range(40):
            env.step(act(yaw=6))
        self.assertTrue(bool((env.state.yaw.abs() <= math.pi + 1e-12).all()))


if __name__ == "__main__":
    unittest.main()

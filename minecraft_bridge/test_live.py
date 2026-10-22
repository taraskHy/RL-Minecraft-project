"""Adapter contract tests with recorded observation fixtures, not a game test."""
import unittest

from minecraft_rl.environment import EnvConfig, potential
from minecraft_rl.live import LiveBridgeError, MinecraftEnv


class LiveObservationTests(unittest.TestCase):
    def test_rejects_invalid_observed_states(self):
        for state in ([5, 0, 3, 0, 0, 0], [1, 1, 3, 0, 0, 0],
                      [0, 0, 3, -1, 0, 0], [0, 0, 3, True, 0, 0]):
            with self.subTest(state=state), self.assertRaises(LiveBridgeError):
                MinecraftEnv._observed_state({"state": state, "success": False, "death": False})

    def make_env(self, observation, *, max_steps=100):
        env = object.__new__(MinecraftEnv)
        env.config = EnvConfig(max_steps=max_steps)
        env._state = (4, 4, 0, 0, 6, 4)
        env._steps = 0
        env._needs_reset = False
        env._request = lambda *args, **kwargs: observation
        return env

    def test_success_uses_real_observation_and_zero_terminal_potential(self):
        env = self.make_env({"state": [4, 4, 0, 0, 3, 2], "success": True, "death": False})
        old = env.state
        state, reward, terminated, truncated, info = env.step(7)
        self.assertEqual(state, (4, 4, 0, 0, 3, 2))
        self.assertAlmostEqual(reward, 9.99 - potential(old))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["simulator_slip_applied"])
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_truncation_preserves_observed_state_potential(self):
        observation = {"state": [4, 4, 0, 0, 6, 4], "success": False, "death": False}
        env = self.make_env(observation, max_steps=1)
        _, reward, terminated, truncated, _ = env.step(2)
        self.assertAlmostEqual(reward, -0.01 + (env.config.gamma - 1) * potential(env.state))
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_bridge_failure_invalidates_episode_without_inventing_a_transition(self):
        env = self.make_env({})
        with self.assertRaises(LiveBridgeError):
            env.step(4)
        self.assertEqual(env._steps, 0)
        self.assertTrue(env._needs_reset)


if __name__ == "__main__":
    unittest.main()

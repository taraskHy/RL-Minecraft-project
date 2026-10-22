import json
from pathlib import Path
import tempfile
import unittest

from minecraft_rl.agents import TabularAgent
from minecraft_rl.environment import EnvConfig, N_ACTIONS
from minecraft_rl.training import TrainingConfig, evaluate, train

S = (0, 0, 3, 0, 0, 0)
T = (1, 0, 3, 0, 0, 0)


class AgentTests(unittest.TestCase):
    def test_q_learning_bootstraps_from_maximum(self):
        agent = TabularAgent(alpha=0.5, gamma=0.9)
        agent.q[T] = [2.0, 4.0] + [0.0] * (N_ACTIONS - 2)
        agent.update(S, 0, 1, T, False)
        self.assertAlmostEqual(agent.q[S][0], 2.3)

    def test_sarsa_uses_sampled_behavior_action(self):
        agent = TabularAgent("sarsa", alpha=0.5, gamma=0.9)
        agent.q[T] = [2.0, 4.0] + [0.0] * (N_ACTIONS - 2)
        agent.update(S, 0, 1, T, False, next_action=0)
        self.assertAlmostEqual(agent.q[S][0], 1.4)

    def test_terminal_does_not_bootstrap(self):
        for algorithm in ("q_learning", "sarsa"):
            agent = TabularAgent(algorithm, alpha=0.5)
            agent.q[T] = [100.0] * N_ACTIONS
            agent.update(S, 0, 1, T, True)
            self.assertEqual(agent.q[S][0], 0.5)

    def test_evaluation_does_not_change_q_or_rng(self):
        agent = TabularAgent()
        rng = agent.rng.getstate()
        evaluate(agent, EnvConfig(max_steps=5), [2_000_000, 2_000_001])
        self.assertEqual(agent.q, {})
        self.assertEqual(agent.rng.getstate(), rng)

    def test_reproducible_training(self):
        for algorithm in ("q_learning", "sarsa"):
            tc = TrainingConfig(algorithm=algorithm, episodes=20, seed=7)
            ec = EnvConfig(max_steps=20)
            agent1, rows1 = train(tc, ec)
            agent2, rows2 = train(tc, ec)
            self.assertEqual(agent1.q, agent2.q)
            self.assertEqual(rows1, rows2)

    def test_safe_json_roundtrip_and_action_version(self):
        agent = TabularAgent()
        agent.q[S] = [1.0] * N_ACTIONS
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            agent.save(path)
            self.assertEqual(TabularAgent.load(path).q, agent.q)
            payload = json.loads(path.read_text())
            payload["actions"].reverse()
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                TabularAgent.load(path)

    def test_reject_mismatched_shaping_discount(self):
        with self.assertRaises(ValueError):
            train(TrainingConfig(episodes=1, gamma=0.9), EnvConfig(gamma=0.97))

    def test_epsilon_schedule(self):
        config = TrainingConfig(episodes=100)
        self.assertAlmostEqual(config.epsilon(0), 1)
        self.assertAlmostEqual(config.epsilon(99), 0.05)


if __name__ == "__main__":
    unittest.main()

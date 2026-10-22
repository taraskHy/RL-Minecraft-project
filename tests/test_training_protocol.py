"""Regression checks for time-limit learning and honest experiment selection."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, call, patch

from minecraft_rl.__main__ import main, parser
from minecraft_rl.agents import TabularAgent
from minecraft_rl.environment import CraftingEnv, EnvConfig, N_ACTIONS
from minecraft_rl.experiments import run_experiment
from minecraft_rl.training import TrainingConfig, train


S = (0, 0, 3, 0, 0, 0)
T = (1, 0, 3, 0, 0, 0)
ROOT = Path(__file__).resolve().parents[1]


def evaluation_rows(seeds, success):
    return [{"episode_seed": seed, "return": 9.99 if success else -0.01,
             "base_return": 9.99 if success else -0.01, "steps": 1,
             "success": int(success), "death": 0,
             "truncated": int(not success), "unseen_state_steps": 0}
            for seed in seeds]


def episode_seeds(spec, split):
    start = spec[f"{split}_seed_start"]
    count = spec[f"{split}_episodes"] * len(spec["training_seeds"])
    return set(range(start, start + count))


class TrainingProtocolTests(unittest.TestCase):
    def test_training_loop_bootstraps_at_truncation_for_both_algorithms(self):
        for algorithm, expected_value in (("q_learning", 2.3), ("sarsa", 1.4)):
            with self.subTest(algorithm=algorithm):
                config = TrainingConfig(algorithm=algorithm, episodes=1,
                                        alpha=0.5, gamma=0.9,
                                        epsilon_start=0.0, epsilon_end=0.0)
                agent = TabularAgent(algorithm, alpha=0.5, gamma=0.9)
                agent.q[T] = [2.0, 4.0] + [0.0] * (N_ACTIONS - 2)
                env = Mock(spec=CraftingEnv)
                env.reset.return_value = S
                env.step.return_value = (T, 1.0, False, True,
                                         {"base_reward": 1.0, "steps": 1,
                                          "success": False, "death": False})
                with (patch("minecraft_rl.training.CraftingEnv", return_value=env),
                      patch("minecraft_rl.training.TabularAgent", return_value=agent),
                      patch.object(agent, "act", side_effect=[0, 0]) as act):
                    trained, rows = train(config, EnvConfig(gamma=0.9, max_steps=1))
                # The next behavior action is 0, while the maximum Q is action 1.
                # A mistaken terminal update would produce 0.5 for BOTH methods.
                self.assertAlmostEqual(trained.q[S][0], expected_value)
                self.assertEqual(act.call_args_list, [call(S, 0.0), call(T, 0.0)])
                env.step.assert_called_once_with(0)
                self.assertEqual(rows[0]["truncated"], 1)
                self.assertEqual(rows[0]["total_steps"], 1)

    def test_training_loop_never_bootstraps_or_samples_after_termination(self):
        for algorithm in ("q_learning", "sarsa"):
            with self.subTest(algorithm=algorithm):
                config = TrainingConfig(algorithm=algorithm, episodes=1,
                                        alpha=0.5, gamma=0.9,
                                        epsilon_start=0.0, epsilon_end=0.0)
                agent = TabularAgent(algorithm, alpha=0.5, gamma=0.9)
                agent.q[T] = [100.0] * N_ACTIONS
                env = Mock(spec=CraftingEnv)
                env.reset.return_value = S
                env.step.return_value = (T, 1.0, True, False,
                                         {"base_reward": 1.0, "steps": 1,
                                          "success": True, "death": False})
                with (patch("minecraft_rl.training.CraftingEnv", return_value=env),
                      patch("minecraft_rl.training.TabularAgent", return_value=agent),
                      patch.object(agent, "act", return_value=0) as act):
                    trained, rows = train(config, EnvConfig(gamma=0.9))
                self.assertAlmostEqual(trained.q[S][0], 0.5)
                act.assert_called_once_with(S, 0.0)
                self.assertEqual(rows[0]["truncated"], 0)

    def test_selection_is_frozen_before_tests_and_retains_every_training_seed(self):
        spec = {"episodes": 1, "training_seeds": [11, 23],
                "algorithms": ["q_learning"], "reward_modes": ["sparse"],
                "alphas": [0.1, 0.3], "gammas": [0.9], "decay_fractions": [0.75],
                "validation_episodes": 2, "test_episodes": 3,
                "validation_seed_start": 1_000_000, "test_seed_start": 2_000_000,
                "max_steps": 5, "slip_probability": 0.0}
        calls = []

        def fake_train(config, env_config):
            agent = TabularAgent(config.algorithm, config.alpha, config.gamma, config.seed)
            # Keep the training-seed marker through real policy save/load.
            agent.q[S] = [float(config.seed)] + [0.0] * (N_ACTIONS - 1)
            return agent, [{"episode": 1, "success": 0, "total_steps": 1}]

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "experiment"

            def fake_evaluate(agent, env_config, seeds):
                selection_path = out / "selected_configs.json"
                if seeds[0] < spec["test_seed_start"]:
                    self.assertFalse(selection_path.exists())
                    training_seed = int(agent.q[S][0])
                    calls.append(("validation", agent.alpha, training_seed, tuple(seeds)))
                    # Candidate 0 has one perfect seed but a worse aggregate score.
                    return evaluation_rows(seeds, agent.alpha == 0.3 or training_seed == 11)

                self.assertTrue(selection_path.exists(), "Freeze selection before reading test outcomes")
                selected = json.loads(selection_path.read_text(encoding="utf-8"))
                self.assertEqual(len(selected), 1)
                self.assertEqual(selected[0]["training"]["alpha"], 0.3)
                if agent.algorithm == "random":
                    calls.append(("random_test", None, None, tuple(seeds)))
                    return evaluation_rows(seeds, False)
                self.assertEqual(agent.alpha, 0.3, "A rejected candidate reached the test split")
                training_seed = int(agent.q[S][0])
                calls.append(("test", agent.alpha, training_seed, tuple(seeds)))
                # Keep a failed final run: all selected training seeds must count.
                return evaluation_rows(seeds, training_seed == 23)

            with (patch("minecraft_rl.experiments.train", side_effect=fake_train),
                  patch("minecraft_rl.experiments.evaluate", side_effect=fake_evaluate),
                  patch("minecraft_rl.experiments.provenance", return_value={"test_fixture": True}),
                  redirect_stdout(io.StringIO())):
                result = run_experiment(spec, out)

            self.assertEqual([entry[0] for entry in calls],
                             ["validation"] * 4 + ["test"] * 2 + ["random_test"])
            test_calls = [entry for entry in calls if entry[0] == "test"]
            self.assertEqual([entry[2] for entry in test_calls], [11, 23])
            self.assertEqual([entry[3] for entry in test_calls],
                             [(2_000_000, 2_000_001, 2_000_002),
                              (2_000_003, 2_000_004, 2_000_005)])
            self.assertFalse(any((out / "candidate_000").rglob("test.csv")))
            self.assertEqual(len(list((out / "candidate_001").rglob("test.csv"))), 2)
            method = result["methods"][0]
            self.assertEqual(method["test"]["training_seeds"], 2)
            self.assertEqual(method["test"]["mean_success_rate"], 0.5)
            self.assertEqual(len(method["policy_paths"]), 2)

    def test_shipped_development_and_final_episode_seed_ranges_are_disjoint(self):
        specs = {name: json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))
                 for name in ("smoke", "course_experiment")}
        ranges = {f"{name}/{split}": episode_seeds(spec, split)
                  for name, spec in specs.items() for split in ("validation", "test")}
        entries = list(ranges.items())
        for index, (name, seeds) in enumerate(entries):
            self.assertGreaterEqual(min(seeds), 1_000_000, name)
            for other_name, other_seeds in entries[index + 1:]:
                self.assertTrue(seeds.isdisjoint(other_seeds), f"{name} overlaps {other_name}")

        final_seeds = ranges["course_experiment/test"]
        evaluate_args = parser().parse_args(["evaluate", "--policy", "unused.json"])
        diagnostic_seeds = set(range(evaluate_args.seed_start,
                                     evaluate_args.seed_start + evaluate_args.episodes))
        demo_args = parser().parse_args(["demo", "--policy", "unused.json"])
        self.assertTrue(diagnostic_seeds.isdisjoint(final_seeds))
        self.assertNotIn(demo_args.seed, final_seeds)

    def test_quick_training_evaluates_diagnostic_seeds_without_touching_final_test(self):
        final_spec = json.loads((ROOT / "configs" / "course_experiment.json").read_text(encoding="utf-8"))
        final_seeds = episode_seeds(final_spec, "test")
        agent = TabularAgent()

        def diagnostic_evaluate(policy, env_config, seeds):
            self.assertTrue(set(seeds).isdisjoint(final_seeds))
            self.assertEqual(seeds[0], 1_000_000)
            return evaluation_rows(seeds, False)

        with tempfile.TemporaryDirectory() as directory:
            with (patch("minecraft_rl.__main__.train", return_value=(agent, [{"episode": 1, "success": 0, "total_steps": 1}])),
                  patch("minecraft_rl.__main__.evaluate", side_effect=diagnostic_evaluate) as evaluate_mock,
                  patch("minecraft_rl.__main__.provenance", return_value={"test_fixture": True}),
                  redirect_stdout(io.StringIO())):
                self.assertEqual(main(["train", "--episodes", "1", "--out", directory]), 0)
            evaluate_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()

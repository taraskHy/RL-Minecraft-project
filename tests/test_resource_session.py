"""Check learning/evaluation evidence with real agents and a simulated transport."""

from contextlib import contextmanager, ExitStack, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from infrastructure import resource_session
from minecraft_rl.resources import ResourceAgent, state_from_observation


class ResourceSessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(resource_session, "ROOT", self.root))
        source_names = (
            "minecraft_bridge/resource-bridge.js", "minecraft_bridge/pnpm-lock.yaml",
            "minecraft_rl/resources.py", "infrastructure/resource_session.py",
            "minecraft_rl/resource_live.py", "infrastructure/resource_server.py",
        )
        for name in source_names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test source fixture\n", encoding="utf-8")
        self.stack.enter_context(patch.object(resource_session, "__file__", str(
            self.root / "infrastructure" / "resource_session.py")))
        self.stack.enter_context(patch.object(resource_session, "runtime_paths", return_value={"node": sys.executable}))
        self.stack.enter_context(patch.object(resource_session.time, "sleep"))
        self.stack.enter_context(patch("builtins.input", return_value=""))
        self.camera = MagicMock()
        self.camera.following.side_effect = lambda: nullcontext()
        self.stack.enter_context(patch.object(resource_session, "Viewer", return_value=self.camera))
        self.bridge = MagicMock()
        self.bridge.__enter__.return_value = self.bridge
        self.bridge.connection = {"backend": "test-only", "game_version": "1.21.4"}
        self.stack.enter_context(patch.object(resource_session, "ResourceBridge", return_value=self.bridge))
        self.server_exits = []

        @contextmanager
        def fake_server(**settings):
            try:
                yield {"port": 25567, **settings}
            finally:
                self.server_exits.append(settings["episode"])

        self.stack.enter_context(patch.object(resource_session, "fresh_resource_server", side_effect=fake_server))
        self.start = {
            "inventory_counts": {"logs": 0, "planks": 0, "sticks": 0, "tables": 0, "pickaxes": 0},
            "nearby_log": False, "nearby_table": False, "table_placed": False,
            "health": 20, "food": 20, "position": {"x": 0., "y": 64., "z": 0.},
            "event": "spawned", "success": False, "death": False,
        }
        self.bridge.observe.return_value = self.start

    def _policy(self, suffix, following, algorithm="q_learning"):
        agent = ResourceAgent(algorithm=algorithm, alpha=.2, gamma=.97)
        state = state_from_observation(self.start)
        next_state = state_from_observation(following)
        agent.q[state] = [1., 0., 0., 0., 0., 0., 0., 0.]
        agent.q[next_state] = [4.] * 8
        path = self.root / f"policy-{suffix}.json"
        agent.save(path)
        return path, state

    def _run(self, name, policy, **settings):
        output = self.root / "runs" / name
        resource_session.run_resources(
            output=output, episodes=1, max_steps=1, policy=policy,
            epsilon=0, epsilon_end=0, pause=0, **settings)
        return output

    def test_online_learning_records_real_updates_and_bootstraps_only_time_limits(self):
        for algorithm in ("q_learning", "sarsa"):
            for dead in (False, True):
                with self.subTest(algorithm=algorithm, dead=dead):
                    following = deepcopy(self.start)
                    following.update(nearby_log=True, health=0 if dead else 20,
                                     death=dead, event="died" if dead else "found_log")
                    self.bridge.step.return_value = following
                    name = f"{algorithm}-{'death' if dead else 'time-limit'}"
                    policy, state = self._policy(name, following, algorithm)
                    output = self._run(name, policy)
                    record = json.loads((output / "episode-0001" / "transitions.jsonl").read_text())
                    self.assertEqual(record["terminated"], dead)
                    self.assertEqual(record["truncated"], not dead)
                    self.assertEqual(record["q_before"][0], 1.)
                    bootstrap = 0. if dead else .97 * 4.
                    expected_td = record["reward"] + bootstrap - 1.
                    expected_q = 1. + .2 * expected_td
                    self.assertAlmostEqual(record["td_error"], expected_td)
                    self.assertAlmostEqual(record["q_after"][0], expected_q)
                    self.assertAlmostEqual(ResourceAgent.load(output / "policy.json").values(state)[0], expected_q)
                    self.assertEqual(ResourceAgent.load(output / "initial_policy.json").values(state)[0], 1.)
                    outcome = json.loads((output / "episode-0001" / "outcome.json").read_text())
                    self.assertEqual(outcome["status"], "completed")
                    self.assertEqual(outcome["steps"], 1)
                    self.assertEqual(outcome["death"], dead)
        self.assertEqual(self.server_exits, [1, 1, 1, 1])

    def test_frozen_evaluation_never_updates_and_records_frozen_values(self):
        following = deepcopy(self.start)
        following.update(nearby_log=True, event="found_log")
        self.bridge.step.return_value = following
        policy, _ = self._policy("frozen", following)
        original_q = json.loads(policy.read_text())["q"]
        with patch.object(ResourceAgent, "update", side_effect=AssertionError("Evaluation must never learn")) as update:
            output = self._run("evaluation", policy, evaluate=True)
            update.assert_not_called()
        record = json.loads((output / "episode-0001" / "transitions.jsonl").read_text())
        self.assertIsNone(record["td_error"])
        self.assertEqual(record["q_before"], record["q_after"])
        self.assertEqual(record["mode"], "frozen_evaluation")
        self.assertEqual(record["epsilon"], 0)
        self.assertEqual(json.loads((output / "policy.json").read_text())["q"], original_q)
        self.assertEqual(json.loads(policy.read_text())["q"], original_q)
        self.assertEqual(self.server_exits, [1])

    def test_missing_or_failed_observation_is_invalid_without_invented_policy_result(self):
        following = deepcopy(self.start)
        following.update(nearby_log=True, event="found_log")
        policy, _ = self._policy("invalid", following)
        original_q = json.loads(policy.read_text())["q"]
        for failure in ("missing-food", "transport-error"):
            with self.subTest(failure=failure):
                if failure == "missing-food":
                    broken = deepcopy(following)
                    del broken["food"]
                    self.bridge.step.return_value = broken
                    self.bridge.step.side_effect = None
                else:
                    self.bridge.step.side_effect = RuntimeError("Game transport disconnected")
                with patch.object(ResourceAgent, "update", side_effect=AssertionError("Invalid data must not update")) as update:
                    with self.assertRaises((ValueError, RuntimeError)):
                        self._run(failure, policy)
                    update.assert_not_called()
                output = self.root / "runs" / failure
                outcome = json.loads((output / "episode-0001" / "outcome.json").read_text())
                self.assertEqual(outcome["status"], "invalid")
                self.assertIsNone(outcome["success"])
                self.assertIsNone(outcome["death"])
                self.assertEqual(outcome["recorded_steps"], 0)
                self.assertEqual((output / "episode-0001" / "transitions.jsonl").read_text(), "")
                self.assertFalse((output / "summary.json").exists())
                self.assertFalse((output / "episodes.csv").exists())
                self.assertEqual(json.loads((output / "policy.json").read_text())["q"], original_q)
        self.assertEqual(self.server_exits, [1, 1])


if __name__ == "__main__":
    unittest.main()

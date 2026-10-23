"""Check learning/evaluation evidence with real agents and a simulated transport."""

from contextlib import contextmanager, ExitStack, nullcontext
from copy import deepcopy
import csv
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

    def test_masked_random_baseline_is_frozen_labeled_and_never_trains(self):
        following = deepcopy(self.start)
        following.update(nearby_log=True, event="found_log")
        self.bridge.step.return_value = following
        with patch.object(ResourceAgent, "update", side_effect=AssertionError("The baseline must never learn")) as update:
            output = self._run("baseline", None, evaluate=True, random_baseline=True)
            update.assert_not_called()
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual((metadata["mode"], metadata["algorithm"], metadata["baseline"]),
                         ("frozen_evaluation", "random", "masked_random"))
        self.assertIn("canopy descent", metadata["scripted_support"])
        self.assertIn("foliage escape", metadata["scripted_support"])
        self.assertIn("PLACE_TABLE", metadata["scripted_support"])
        self.assertIn("leaf blocks", metadata["recovery_helpers"])
        self.assertIn("leaf_escape", metadata["recovery_helpers"])
        self.assertNotIn("No other block is dug outside CHOP;", metadata["recovery_helpers"])
        record = json.loads((output / "episode-0001" / "transitions.jsonl").read_text())
        self.assertEqual(record["mode"], "frozen_evaluation")
        self.assertIsNone(record["td_error"])
        self.assertIn(record["action"], ("SEARCH",))
        self.assertEqual(json.loads((output / "policy.json").read_text())["algorithm"], "random")
        self.assertEqual(json.loads((output / "summary.json").read_text())["baseline"], "masked_random")
        for settings in ({"evaluate": False, "random_baseline": True},
                         {"evaluate": True, "random_baseline": True, "pretrain": 5},
                         {"evaluate": True, "random_baseline": False}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self._run("baseline-invalid", None, **settings)
        self.assertEqual(self.server_exits, [1])

    def test_recovery_records_are_kept_in_full_and_summarized_per_step(self):
        following = deepcopy(self.start)
        descent = {"type": "canopy_descent", "layers_cleared": 2, "blocks_dug": ["oak_leaves", "oak_leaves"],
                   "falls": [{"planned_fall": 1, "actual_fall": 1.0, "damage": 0}, {"planned_fall": 4, "actual_fall": 4.0, "damage": 1}],
                   "moves": 0, "damage": 1, "start_y": 76.0, "end_y": 71.0, "descended": True, "stopped": None, "refused": None}
        following.update(nearby_log=True, event="cleared_2_leaf_layers;descended_canopy;log_reached", recovery=descent)
        self.bridge.step.return_value = following
        policy, _ = self._policy("recovery", following)
        output = self._run("recovery", policy)
        record = json.loads((output / "episode-0001" / "transitions.jsonl").read_text())
        self.assertEqual(record["after"]["recovery"], descent)
        with (output / "episode-0001" / "steps.csv").open(encoding="utf-8", newline="") as handle:
            steps = list(csv.DictReader(handle))
        self.assertEqual(steps[0]["recovery"], "canopy_descent:layers=2,descent=5.0,damage=1,stopped=-")
        self.assertEqual(steps[0]["event"], "cleared_2_leaf_layers;descended_canopy;log_reached")
        self.assertEqual(resource_session.recovery_summary(None), "")
        self.assertEqual(resource_session.recovery_summary({"type": "canopy_descent", "layers_cleared": 0, "stopped": "fall_too_far"}),
                         "canopy_descent:layers=0,descent=None,damage=0,stopped=fall_too_far")
        # Natural leaf drops collected during a descent are attributed in the note.
        self.assertEqual(resource_session.recovery_summary({"type": "canopy_descent", "layers_cleared": 3, "damage": 0,
                                                            "start_y": 76.0, "end_y": 72.0, "stopped": None,
                                                            "items_picked_up": {"stick": 2, "oak_sapling": 1}}),
                         "canopy_descent:layers=3,descent=4.0,damage=0,stopped=-,picked_up=oak_sapling:1+stick:2")
        # A foliage escape is summarized on its own or as the tail of the descent that led to it.
        self.assertEqual(resource_session.recovery_summary({"type": "leaf_escape", "blocks_dug": ["oak_leaves"],
                                                            "direction": "-x", "escaped": True, "stopped": None,
                                                            "items_picked_up": {"stick": 1}}),
                         "leaf_escape:blocks=1,direction=-x,escaped=True,stopped=-,picked_up=stick:1")
        self.assertEqual(resource_session.recovery_summary({"type": "canopy_descent", "layers_cleared": 2, "damage": 0,
                                                            "start_y": 78.0, "end_y": 76.0, "stopped": None,
                                                            "escape": {"type": "leaf_escape", "blocks_dug": ["oak_leaves", "oak_leaves"],
                                                                       "direction": "+x", "escaped": True, "stopped": None}}),
                         "canopy_descent:layers=2,descent=2.0,damage=0,stopped=-,escape_blocks=2,escaped=True")

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
                # The seed stays on record even when nothing was observed.
                self.assertEqual((outcome["episode"], outcome["world_seed"]), (1, 0))
                self.assertIsNone(outcome["success"])
                self.assertIsNone(outcome["death"])
                self.assertEqual(outcome["recorded_steps"], 0)
                self.assertEqual((output / "episode-0001" / "transitions.jsonl").read_text(), "")
                self.assertFalse((output / "summary.json").exists())
                self.assertFalse((output / "episodes.csv").exists())
                self.assertEqual(json.loads((output / "policy.json").read_text())["q"], original_q)
        self.assertEqual(self.server_exits, [1, 1])

    def _train(self, name, condition, steps, **settings):
        output = self.root / "runs" / name
        resource_session.run_resources(output=output, episodes=1, max_steps=steps, epsilon=0, epsilon_end=0,
                                       pause=0, condition=condition, **settings)
        return output, [json.loads(line) for line in (output / "episode-0001" / "transitions.jsonl").read_text().splitlines()]

    def test_unmasked_training_keeps_infeasible_actions_as_observed_transitions(self):
        # The bridge reports an unchanged world: the chosen action was infeasible.
        unchanged = deepcopy(self.start)
        unchanged.update(event="missing_materials")
        self.bridge.step.return_value = unchanged
        output, records = self._train("unmasked", "unmasked", 3, algorithm="q_learning")
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual((metadata["condition"], metadata["action_mask"], metadata["state_version"], metadata["baseline"]),
                         ("unmasked", "none", "resource_v1", None))
        self.assertIn("No action mask", metadata["scripted_support"])
        self.assertIn("infeasible action fails as an observed transition", metadata["condition_note"])
        for record in records:
            self.assertEqual(record["condition"], "unmasked")
            self.assertEqual(record["allowed_actions"], list(range(8)))
            self.assertEqual(record["masked_actions"], [0])  # what the mask would have allowed at the start
            self.assertEqual(record["base_reward"], -0.01)
            self.assertEqual(record["state"], record["next_state"])
            self.assertIsNotNone(record["td_error"])
        self.assertEqual(json.loads((output / "policy.json").read_text())["condition"], "unmasked")
        self.assertEqual(json.loads((output / "summary.json").read_text())["condition"], "unmasked")

    def test_history_bit_follows_whether_the_previous_decision_changed_anything(self):
        found = deepcopy(self.start)
        found.update(nearby_log=True, event="log_reached")
        unchanged = deepcopy(found)
        unchanged.update(event="no_reachable_log")
        self.bridge.step.side_effect = [found, unchanged, unchanged]
        output, records = self._train("history", "unmasked_history", 3, algorithm="sarsa")
        self.assertEqual([r["state"][-1] for r in records], [0, 0, 1])
        self.assertEqual([r["next_state"][-1] for r in records], [0, 1, 1])
        self.assertTrue(all(len(r["state"]) == 10 for r in records))
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual((metadata["condition"], metadata["state_version"]), ("unmasked_history", "resource_v2"))
        self.bridge.step.side_effect = None

    def test_random_baselines_follow_the_condition_and_policies_keep_theirs(self):
        following = deepcopy(self.start)
        following.update(nearby_log=True, event="log_reached")
        self.bridge.step.return_value = following
        output, records = self._train("uniform", "unmasked", 1, evaluate=True, random_baseline=True)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual((metadata["baseline"], metadata["condition"]), ("uniform_random", "unmasked"))
        self.assertEqual(records[0]["allowed_actions"], list(range(8)))
        masked_output, masked_records = self._train("masked-random", None, 1, evaluate=True, random_baseline=True)
        self.assertEqual(json.loads((masked_output / "metadata.json").read_text())["condition"], "masked")
        self.assertEqual(masked_records[0]["allowed_actions"], [0])
        with self.assertRaises(ValueError):
            self._train("history-random", "unmasked_history", 1, evaluate=True, random_baseline=True)
        with self.assertRaises(ValueError):
            self._train("bad-condition", "unknown", 1, algorithm="q_learning")
        # A masked checkpoint cannot be evaluated as unmasked, and an unmasked one keeps its condition.
        masked_policy, _ = self._policy("masked-ckpt", following)
        with self.assertRaises(ValueError):
            self._train("mismatch", "unmasked", 1, evaluate=True, policy=masked_policy)
        unmasked_policy = output.parent / "unmasked-ckpt.json"
        ResourceAgent(action_mask="none").save(unmasked_policy)
        kept, kept_records = self._train("kept", None, 1, evaluate=True, policy=unmasked_policy)
        self.assertEqual(json.loads((kept / "metadata.json").read_text())["condition"], "unmasked")
        self.assertEqual(kept_records[0]["allowed_actions"], list(range(8)))

    def test_simulator_pretraining_behind_a_checkpoint_stays_in_the_metadata(self):
        following = deepcopy(self.start)
        following.update(nearby_log=True, event="log_reached")
        self.bridge.step.return_value = following
        pretrained = self.root / "pretrained.json"
        ResourceAgent().save(pretrained, {"pretrain_simulator_episodes": 300})
        evaluated, _ = self._train("pretrained-eval", None, 1, evaluate=True, policy=pretrained)
        metadata = json.loads((evaluated / "metadata.json").read_text())
        self.assertEqual((metadata["pretrain_simulator_episodes"], metadata["checkpoint_pretrain_simulator_episodes"]), (0, 300))
        # A resumed run carries it forward again.
        resumed, _ = self._train("pretrained-resume", None, 1, policy=evaluated / "policy.json")
        self.assertEqual(json.loads((resumed / "metadata.json").read_text())["checkpoint_pretrain_simulator_episodes"], 300)
        fresh, _ = self._train("fresh", "masked", 1, algorithm="q_learning")
        self.assertEqual(json.loads((fresh / "metadata.json").read_text())["checkpoint_pretrain_simulator_episodes"], 0)

    def test_command_line_passes_the_condition(self):
        from infrastructure.__main__ import resources
        with patch.object(resource_session, "run_resources", return_value=0) as run:
            resources(["--condition", "unmasked_history", "--episodes", "2", "--out", str(self.root / "runs" / "cli")])
            self.assertEqual(run.call_args.kwargs["condition"], "unmasked_history")
            resources(["--random-baseline", "--condition", "unmasked", "--out", str(self.root / "runs" / "cli2")], evaluate=True)
            self.assertEqual((run.call_args.kwargs["condition"], run.call_args.kwargs["random_baseline"]), ("unmasked", True))
            resources(["--out", str(self.root / "runs" / "cli3")])
            self.assertIsNone(run.call_args.kwargs["condition"])


    def test_port_and_parallel_slots_are_recorded_and_checked(self):
        following = deepcopy(self.start)
        following.update(nearby_log=True, event="log_reached")
        self.bridge.step.return_value = following
        output, _ = self._train("parallel", "masked", 1, algorithm="q_learning", port=25570, parallel_slots=4)
        self.assertEqual(resource_session.fresh_resource_server.call_args.kwargs["port"], 25570)
        execution = json.loads((output / "metadata.json").read_text())["execution"]
        self.assertEqual((execution["server_port"], execution["parallel_slots"]), (25570, 4))
        outcome = json.loads((output / "episode-0001" / "outcome.json").read_text())
        self.assertLessEqual(outcome["started_at"], outcome["finished_at"])
        alone, _ = self._train("alone", "masked", 1, algorithm="q_learning")
        execution = json.loads((alone / "metadata.json").read_text())["execution"]
        self.assertEqual((execution["server_port"], execution["parallel_slots"]), (25567, 1))
        refused = ({"port": 25566}, {"port": 25600}, {"port": True}, {"parallel_slots": 0}, {"parallel_slots": 9},
                   {"parallel_slots": True}, {"parallel_slots": 2, "viewer": "Watcher"})
        for index, bad in enumerate(refused):
            output = self.root / "runs" / f"refused-{index}"
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resource_session.run_resources(output=output, episodes=1, max_steps=1, condition="masked",
                                               pause=0, **bad)
            self.assertFalse(output.exists())

    def test_command_line_passes_port_and_parallel_slots(self):
        from infrastructure.__main__ import resources
        with patch.object(resource_session, "run_resources", return_value=0) as run:
            resources(["--port", "25569", "--parallel-slots", "3", "--out", str(self.root / "runs" / "cli")])
            self.assertEqual((run.call_args.kwargs["port"], run.call_args.kwargs["parallel_slots"]), (25569, 3))
            resources(["--random-baseline", "--condition", "masked", "--out", str(self.root / "runs" / "cli2")],
                      evaluate=True)
            self.assertEqual((run.call_args.kwargs["port"], run.call_args.kwargs["parallel_slots"]), (25567, 1))


if __name__ == "__main__":
    unittest.main()

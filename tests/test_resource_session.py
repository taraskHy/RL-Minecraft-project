"""Check learning/evaluation evidence with real agents and a simulated transport."""

from contextlib import contextmanager, ExitStack, nullcontext, redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from infrastructure import resource_session
from minecraft_rl import resource_live
from minecraft_rl.resource_live import GOALS, ResourceBridge
from minecraft_rl.resources import ACTION_NAMES, ResourceAgent, potential, reward_transition, state_from_observation

# The fields the bridge adds under the stone goal or the survey (design section 3.6), and the survey record.
STONE_FIELDS = ("nearby_stone", "stone_visible", "stone_targets", "pickaxe_uses_left", "pickaxe_digs", "hostile_nearby",
                "time_of_day", "depth_below_stage_start", "diagnostic_only")
SURVEY_RECORD = {"lit_only": True,
                 "depth_to_stone": {"depth": 5, "block": "stone", "layers": [["grass_block", 1], ["dirt", 3]], "stopped": None},
                 "staircases": {"+x": {"stopped": "stone_exposed", "dig_counts": [1, 2, 3, 3, 2], "end_y": 60, "steps": []}},
                 "fan": {"ms": 12.5, "rays": 416, "reads": 12000, "hits": 100, "targets": 0, "targets_any_light": 0, "nearest": []},
                 "reach": {"ms": 0.9, "candidates": 0, "candidates_any_light": 0}, "survey_ms": 45.0}


class ResourceSessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(resource_session, "ROOT", self.root))
        source_names = (
            "minecraft_bridge/resource-bridge.js", "minecraft_bridge/stone-helpers.js", "minecraft_bridge/pnpm-lock.yaml",
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
        self.live_lock = self.stack.enter_context(
            patch.object(resource_session, "live_run_lock", side_effect=lambda owner: nullcontext()))
        self.owner = self.stack.enter_context(
            patch.object(resource_session, "current_owner", return_value="schedule px:abc:t0"))
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

    def test_a_policy_held_open_for_a_moment_is_still_replaced(self):
        agent = MagicMock()
        agent.save.side_effect = lambda path, metadata: Path(path).write_text("new", encoding="utf-8")
        policy = self.root / "policy.json"
        policy.write_text("old", encoding="utf-8")
        real_replace, refusals = Path.replace, []

        def replace(source, target):
            if len(refusals) < 2:
                refusals.append(1)
                raise PermissionError(5, "Access is denied")
            return real_replace(source, target)

        with patch.object(Path, "replace", replace):
            resource_session._save_policy(agent, policy, {})
        self.assertEqual((policy.read_text(encoding="utf-8"), len(refusals)), ("new", 2))
        self.assertFalse(policy.with_suffix(".tmp").exists())
        # A file that stays locked still fails loudly instead of being skipped.
        with patch.object(Path, "replace", side_effect=PermissionError(5, "Access is denied")):
            with self.assertRaises(PermissionError):
                resource_session._save_policy(agent, policy, {})

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
        output, _ = self._train("parallel", "masked", 1, algorithm="q_learning", port=25570, parallel_slots=4,
                                schedule_id="px:abc:t0")
        self.assertEqual(resource_session.fresh_resource_server.call_args.kwargs["port"], 25570)
        execution = json.loads((output / "metadata.json").read_text())["execution"]
        self.assertEqual((execution["server_port"], execution["parallel_slots"], execution["schedule_id"]),
                         (25570, 4, "px:abc:t0"))
        outcome = json.loads((output / "episode-0001" / "outcome.json").read_text())
        self.assertLessEqual(outcome["started_at"], outcome["finished_at"])
        alone, _ = self._train("alone", "masked", 1, algorithm="q_learning")
        execution = json.loads((alone / "metadata.json").read_text())["execution"]
        self.assertEqual((execution["server_port"], execution["parallel_slots"], execution["schedule_id"]),
                         (25567, 1, None))
        refused = ({"port": 25566}, {"port": 25600}, {"port": True}, {"parallel_slots": 0}, {"parallel_slots": 9},
                   {"parallel_slots": True}, {"parallel_slots": 2},
                   {"parallel_slots": 2, "schedule_id": "px:abc:t0", "viewer": "Watcher"},
                   {"schedule_id": ""}, {"schedule_id": "a\nb"}, {"schedule_id": 5})
        for index, bad in enumerate(refused):
            output = self.root / "runs" / f"refused-{index}"
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resource_session.run_resources(output=output, episodes=1, max_steps=1, condition="masked",
                                               pause=0, **bad)
            self.assertFalse(output.exists())

    def test_initial_q_is_set_recorded_and_carried_by_the_policy(self):
        following = deepcopy(self.start)
        following.update(nearby_log=True, event="log_reached")
        self.bridge.step.return_value = following
        output, _ = self._train("pessimistic", "unmasked", 1, algorithm="q_learning", initial_q=-1.0)
        self.assertEqual(json.loads((output / "metadata.json").read_text())["initial_q"], -1.0)
        self.assertEqual(ResourceAgent.load(output / "policy.json").initial_q, -1.0)
        # A continuation inherits it from the policy; repeating the same value is allowed.
        resumed, _ = self._train("pessimistic-resume", None, 1, policy=output / "policy.json")
        self.assertEqual(json.loads((resumed / "metadata.json").read_text())["initial_q"], -1.0)
        self._train("pessimistic-same", None, 1, policy=output / "policy.json", initial_q=-1.0)
        default, _ = self._train("default-q", "unmasked", 1, algorithm="q_learning")
        self.assertEqual(json.loads((default / "metadata.json").read_text())["initial_q"], 0.0)
        # A frozen evaluation records the evaluated policy's value, which the declared analysis checks;
        # a random baseline has none.
        frozen, _ = self._train("pessimistic-test", None, 1, evaluate=True, policy=output / "policy.json")
        self.assertEqual(json.loads((frozen / "metadata.json").read_text())["initial_q"], -1.0)
        baseline, _ = self._train("random-test", "unmasked", 1, evaluate=True, random_baseline=True)
        self.assertIsNone(json.loads((baseline / "metadata.json").read_text())["initial_q"])
        # Simulator pretraining keeps the value in the agent it hands to live training.
        with patch.object(resource_session, "train_simulator", wraps=resource_session.train_simulator) as pretrain:
            warmed, _ = self._train("pessimistic-pretrained", "unmasked", 1, algorithm="q_learning", initial_q=-1.0,
                                    pretrain=3)
        self.assertEqual(pretrain.call_args.kwargs["initial_q"], -1.0)
        self.assertEqual((ResourceAgent.load(warmed / "initial_policy.json").initial_q,
                          json.loads((warmed / "metadata.json").read_text())["initial_q"]), (-1.0, -1.0))
        policy = output / "policy.json"
        refused = ({"policy": policy, "initial_q": 0.0}, {"policy": policy, "initial_q": -1.0, "evaluate": True},
                   {"condition": "unmasked", "random_baseline": True, "evaluate": True, "initial_q": -1.0},
                   {"condition": "unmasked", "initial_q": float("nan")}, {"condition": "unmasked", "initial_q": True},
                   {"condition": "unmasked", "initial_q": -101})
        for index, bad in enumerate(refused):
            target = self.root / "runs" / f"refused-q-{index}"
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resource_session.run_resources(output=target, episodes=1, max_steps=1, pause=0, **bad)
            self.assertFalse(target.exists())

    def test_command_line_passes_initial_q(self):
        from infrastructure.__main__ import resources
        with patch.object(resource_session, "run_resources", return_value=0) as run:
            resources(["--initial-q", "-1.5", "--out", str(self.root / "runs" / "cli")])
            self.assertEqual(run.call_args.kwargs["initial_q"], -1.5)
            resources(["--out", str(self.root / "runs" / "cli2")])
            self.assertIsNone(run.call_args.kwargs["initial_q"])
            # Frozen evaluation takes the policy's own value and has no such option.
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                resources(["--initial-q", "-1", "--random-baseline", "--condition", "unmasked",
                           "--out", str(self.root / "runs" / "cli3")], evaluate=True)

    def test_command_line_passes_port_and_parallel_slots(self):
        from infrastructure.__main__ import resources
        with patch.object(resource_session, "run_resources", return_value=0) as run:
            resources(["--port", "25569", "--parallel-slots", "3", "--schedule-id", "px:abc:t0",
                       "--out", str(self.root / "runs" / "cli")])
            self.assertEqual((run.call_args.kwargs["port"], run.call_args.kwargs["parallel_slots"],
                              run.call_args.kwargs["schedule_id"]), (25569, 3, "px:abc:t0"))
            resources(["--random-baseline", "--condition", "masked", "--out", str(self.root / "runs" / "cli2")],
                      evaluate=True)
            self.assertEqual((run.call_args.kwargs["port"], run.call_args.kwargs["parallel_slots"],
                              run.call_args.kwargs["schedule_id"]), (25567, 1, None))


    def test_a_run_started_by_hand_holds_the_live_lock_and_a_schedule_id_must_be_its_schedules(self):
        following = deepcopy(self.start)
        following.update(nearby_log=True, event="log_reached")
        self.bridge.step.return_value = following
        alone, _ = self._train("by-hand", "masked", 1, algorithm="q_learning")
        self.assertTrue(self.live_lock.call_args.args[0].startswith("run by-hand (pid "))
        self.live_lock.side_effect = RuntimeError("A live resource run is already going on this machine (schedule x)")
        output = self.root / "runs" / "refused"
        with self.assertRaisesRegex(RuntimeError, "already going"):
            resource_session.run_resources(output=output, episodes=1, max_steps=1, condition="masked", pause=0)
        self.assertFalse(output.exists())
        # A schedule's own runs carry its id and run under the schedule's lock; a copied or stale id is refused.
        self.live_lock.reset_mock()
        scheduled, _ = self._train("scheduled", "masked", 1, algorithm="q_learning", schedule_id="px:abc:t0")
        self.assertTrue((scheduled / "episode-0001" / "outcome.json").is_file())
        self.live_lock.assert_not_called()
        for owner in ("schedule px:abc:other", None):
            self.owner.return_value = owner
            output = self.root / "runs" / f"stale-{owner is None}"
            with self.subTest(owner=owner), self.assertRaisesRegex(RuntimeError, "set only by resource-parallel"):
                resource_session.run_resources(output=output, episodes=1, max_steps=1, condition="masked", pause=0,
                                               schedule_id="px:abc:t0")
            self.assertFalse(output.exists())

    def test_an_episode_never_runs_on_sources_that_changed_after_the_run_started(self):
        following = deepcopy(self.start)
        following.update(nearby_log=True, event="log_reached")
        source = self.root / "minecraft_rl" / "resources.py"

        def step(_action):
            source.write_text("changed during the run\n", encoding="utf-8")
            return following

        self.bridge.step.side_effect = step
        output = self.root / "runs" / "changed"
        with self.assertRaisesRegex(RuntimeError, "sources changed"):
            resource_session.run_resources(output=output, episodes=2, max_steps=1, condition="masked", pause=0)
        self.assertTrue((output / "episode-0001" / "outcome.json").is_file())
        self.assertFalse((output / "episode-0002").exists())

    def test_the_goal_and_the_survey_reach_the_bridge_and_the_metadata(self):
        following = deepcopy(self.start)
        following.update(nearby_log=True, event="log_reached")
        self.bridge.step.return_value = following
        wood, _ = self._train("wood-goal", "masked", 1, algorithm="q_learning")
        settings = resource_session.ResourceBridge.call_args.kwargs
        self.assertEqual((settings["goal"], settings["stone_survey"]), ("wooden_pickaxe", False))
        metadata = json.loads((wood / "metadata.json").read_text())
        self.assertEqual((metadata["goal"], metadata["stone_survey"], metadata["stone_survey_note"]), ("wooden_pickaxe", False, None))
        self.assertEqual(metadata["actions"], list(ACTION_NAMES))
        self.assertEqual(len(metadata["actions"]), 8)
        # The S2 survey: the frozen wood checkpoint on development worlds, with the survey on.
        policy, _ = self._policy("survey", following)
        surveyed, _ = self._train("survey", None, 1, evaluate=True, policy=policy, world_seed=61000, stone_survey=True)
        self.assertIs(resource_session.ResourceBridge.call_args.kwargs["stone_survey"], True)
        self.assertEqual(resource_session.ResourceBridge.call_args.kwargs["goal"], "wooden_pickaxe")
        metadata = json.loads((surveyed / "metadata.json").read_text())
        self.assertEqual((metadata["goal"], metadata["stone_survey"], metadata["first_world_seed"]), ("wooden_pickaxe", True, 61000))
        for phrase in ("diagnostic_only", "nothing is dug", "No state encoder reads them", "cannot see"):
            self.assertIn(phrase, metadata["stone_survey_note"])
        # The stone goal waits for the stage-2 coordinator (step S3); nothing else is a goal.
        for index, bad in enumerate(({"goal": "stone_pickaxe"}, {"goal": "iron_pickaxe"}, {"goal": None},
                                     {"stone_survey": 1}, {"stone_survey": "on"}, {"stone_survey": None})):
            output = self.root / "runs" / f"refused-goal-{index}"
            with self.subTest(bad=bad), self.assertRaises(ValueError) as refusal:
                resource_session.run_resources(output=output, episodes=1, max_steps=1, condition="masked", pause=0, **bad)
            if bad.get("goal") == "stone_pickaxe":
                self.assertIn("step S3", str(refusal.exception))
            self.assertFalse(output.exists())

    def test_the_stone_helpers_are_digested_and_a_change_to_them_stops_the_next_episode(self):
        following = deepcopy(self.start)
        following.update(nearby_log=True, event="log_reached")
        self.bridge.step.return_value = following
        output, _ = self._train("digested", "masked", 1, algorithm="q_learning")
        digests = json.loads((output / "metadata.json").read_text())["source_sha256"]
        self.assertIn(str(Path("minecraft_bridge/stone-helpers.js")), digests)
        self.assertIn(str(Path("minecraft_bridge/resource-bridge.js")), digests)
        source = self.root / "minecraft_bridge" / "stone-helpers.js"

        def step(_action):
            source.write_text("changed during the run\n", encoding="utf-8")
            return following

        self.bridge.step.side_effect = step
        changed = self.root / "runs" / "helpers-changed"
        with self.assertRaisesRegex(RuntimeError, "sources changed"):
            resource_session.run_resources(output=changed, episodes=2, max_steps=1, condition="masked", pause=0)
        self.assertTrue((changed / "episode-0001" / "outcome.json").is_file())
        self.assertFalse((changed / "episode-0002").exists())
        self.bridge.step.side_effect = None

    def test_survey_observations_are_recorded_in_full_and_never_change_the_state_or_reward(self):
        start = deepcopy(self.start)
        start["inventory_counts"].update(stone_materials=0, stone_pickaxes=0)
        start.update(dict.fromkeys(STONE_FIELDS))
        crafted = deepcopy(start)
        crafted["inventory_counts"].update(planks=1, sticks=2, pickaxes=1)
        crafted.update(table_placed=True, nearby_table=True, held_item="wooden_pickaxe", success=True,
                       event="crafted_wooden_pickaxe", nearby_stone=True, stone_visible=True,
                       stone_targets=[{"name": "stone", "x": 2, "y": 63, "z": 0, "distance": 2.3, "drop": "cobblestone", "rest_y": 63}],
                       pickaxe_uses_left=59, pickaxe_digs=0, hostile_nearby=[], time_of_day=2400,
                       depth_below_stage_start=0, diagnostic_only=SURVEY_RECORD)
        plain = {key: value for key, value in crafted.items() if key not in STONE_FIELDS}
        plain["inventory_counts"] = {key: crafted["inventory_counts"][key] for key in ("logs", "planks", "sticks", "tables", "pickaxes")}
        self.bridge.observe.return_value = start
        self.bridge.step.return_value = crafted
        policy, _ = self._policy("survey-record", crafted)
        output = self._run("survey-record", policy, evaluate=True, stone_survey=True)
        record = json.loads((output / "episode-0001" / "transitions.jsonl").read_text())
        self.assertEqual(record["after"], crafted)
        self.assertEqual(record["before"]["diagnostic_only"], None)
        self.assertEqual((record["state"], record["next_state"]),
                         (list(state_from_observation(self.start)), list(state_from_observation(plain))))
        expected, _ = reward_transition(self.start, plain, gamma=.97, success=True, death=False)
        self.assertAlmostEqual(record["reward"], expected)
        self.assertTrue(json.loads((output / "episode-0001" / "outcome.json").read_text())["success"])
        # A stone material at the start is not an empty inventory.
        held = deepcopy(start)
        held["inventory_counts"]["stone_materials"] = 1
        self.bridge.observe.return_value = held
        with self.assertRaisesRegex(RuntimeError, "Fresh-start inventory was not empty"):
            self._run("stone-at-start", policy, evaluate=True)

    def test_the_frozen_wood_checkpoint_runs_the_same_with_and_without_the_survey(self):
        checkpoint = Path(__file__).resolve().parents[1] / "results" / "pc_resource_runs" / "87673b8" / "mx-unmasked-q-train" / "policy.json"
        if not checkpoint.is_file():
            self.skipTest("the retained 87673b8 checkpoint is not in this checkout")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.assertEqual(digest, "7d0fcce8653df8d4afc0867b321fd9670a8c64ab6ef03a876c64b9cd1c9e0349")
        following = deepcopy(self.start)
        following.update(nearby_log=True, event="log_reached")
        self.bridge.step.return_value = following
        chosen = []
        for survey in (False, True):
            self.bridge.step.reset_mock()
            output = self._run(f"checkpoint-survey-{survey}", checkpoint, evaluate=True, stone_survey=survey)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual((metadata["initial_policy_sha256"], metadata["condition"], metadata["state_version"]),
                             (digest, "unmasked", "resource_v1"))
            action = self.bridge.step.call_args.args[0]
            self.assertIn(action, range(8))
            record = json.loads((output / "episode-0001" / "transitions.jsonl").read_text())
            self.assertEqual((record["action"], len(record["q_before"])), (ACTION_NAMES[action], 8))
            chosen.append((action, record["state"], record["q_before"]))
        self.assertEqual(chosen[0], chosen[1])
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), digest)

    def test_command_line_turns_on_the_survey_and_never_sets_the_goal(self):
        from infrastructure.__main__ import resources
        with patch.object(resource_session, "run_resources", return_value=0) as run:
            resources(["--policy", "results/pc_resource_runs/87673b8/mx-unmasked-q-train/policy.json", "--stone-survey",
                       "--world-seed", "61000", "--episodes", "20", "--out", str(self.root / "runs" / "cli")], evaluate=True)
            self.assertEqual((run.call_args.kwargs["stone_survey"], run.call_args.kwargs["world_seed"],
                              run.call_args.kwargs["episodes"]), (True, 61000, 20))
            self.assertNotIn("goal", run.call_args.kwargs)
            resources(["--stone-survey", "--out", str(self.root / "runs" / "cli2")])
            self.assertIs(run.call_args.kwargs["stone_survey"], True)
            resources(["--out", str(self.root / "runs" / "cli3")])
            self.assertIs(run.call_args.kwargs["stone_survey"], False)
            # The stone goal has no command-line option until its coordinator exists (step S3).
            for evaluate in (False, True):
                with self.subTest(evaluate=evaluate), self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                    resources(["--goal", "stone_pickaxe", "--random-baseline", "--out", str(self.root / "runs" / "cli4")],
                              evaluate=evaluate)


class LauncherCheckTests(unittest.TestCase):
    def test_the_checks_run_the_stone_helper_tests_and_their_syntax_check(self):
        from infrastructure import __main__ as launcher
        commands = []
        with patch.object(launcher, "run", side_effect=lambda command: commands.append([str(value) for value in command]) or 0), \
                patch("infrastructure.runtime.runtime_paths", return_value={"node": sys.executable}), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.checks(), 0)
        tests = next(command[2:] for command in commands if command[1:2] == ["--test"])
        self.assertEqual(tests, ["minecraft_bridge/bridge.test.js", "minecraft_bridge/survival-probe.test.js",
                                 "minecraft_bridge/resource-bridge.test.js", "minecraft_bridge/stone-helpers.test.js",
                                 "minecraft_bridge/stone-mechanics.test.js"])
        root = Path(launcher.__file__).resolve().parents[1]
        self.assertTrue(all((root / name).is_file() for name in tests))
        checked = [command[2] for command in commands if command[1:2] == ["--check"]]
        self.assertIn("minecraft_bridge/stone-helpers.js", checked)
        self.assertIn("minecraft_bridge/resource-bridge.js", checked)
        # package.json runs the same test files and syntax-checks the stone helpers too.
        scripts = json.loads((root / "minecraft_bridge" / "package.json").read_text(encoding="utf-8"))["scripts"]
        self.assertEqual(scripts["test"].split()[2:], [Path(name).name for name in tests])
        self.assertIn("node --check stone-helpers.js", scripts["check"])


class StoneObservationTests(unittest.TestCase):
    """The stone fields and the survey record ride along in observations; the wood stage never reads them."""

    def test_the_wood_state_and_reward_never_read_the_stone_fields_or_the_survey(self):
        stone_counts = ("stone_materials", "stone_pickaxes")

        class Guarded(dict):
            def __init__(self, values, forbidden):
                super().__init__(values)
                self.forbidden = forbidden

            def __getitem__(self, key):
                if key in self.forbidden:
                    raise AssertionError(f"the wood stage read {key}")
                return super().__getitem__(key)

            def get(self, key, default=None):
                if key in self.forbidden:
                    raise AssertionError(f"the wood stage read {key}")
                return super().get(key, default)

        plain = {"inventory_counts": {"logs": 1, "planks": 3, "sticks": 2, "tables": 0, "pickaxes": 1},
                 "nearby_log": True, "nearby_table": True, "table_placed": True, "health": 20, "food": 18,
                 "position": {"x": 0.5, "y": 64.0, "z": 0.5}, "success": True, "death": False}
        stone = {**plain, "inventory_counts": {**plain["inventory_counts"], "stone_materials": 2, "stone_pickaxes": 0},
                 "nearby_stone": True, "stone_visible": True, "stone_targets": [], "pickaxe_uses_left": 58, "pickaxe_digs": 1,
                 "hostile_nearby": [{"name": "zombie", "x": 5.5, "y": 64.0, "z": 3.5, "distance": 5.83}],
                 "time_of_day": 12500, "depth_below_stage_start": 4, "diagnostic_only": SURVEY_RECORD}
        guarded = Guarded({**stone, "inventory_counts": Guarded(stone["inventory_counts"], stone_counts)}, STONE_FIELDS)
        self.assertEqual(state_from_observation(guarded), state_from_observation(plain))
        self.assertEqual(potential(guarded), potential(plain))
        self.assertEqual(reward_transition(guarded, guarded, success=True), reward_transition(plain, plain, success=True))
        # Null stone fields, as before the wooden pickaxe, are ignored the same way.
        nulls = {**plain, **dict.fromkeys(STONE_FIELDS)}
        self.assertEqual(state_from_observation(nulls), state_from_observation(plain))


class ResourceBridgeCommandTests(unittest.TestCase):
    """The transport passes the goal and the survey option to the bridge process, and nothing else changes."""

    def _open(self, **settings):
        launched = []

        class Process:
            def __init__(self, command, **_options):
                launched.append(command)
                self.stdin = io.StringIO()
                self.stdout = io.StringIO(json.dumps({"id": 1, "ok": True, "result": {"connected": True}}) + "\n")

            def wait(self, timeout=None):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        with patch.object(resource_live.subprocess, "Popen", Process):
            with ResourceBridge(node="node.exe", port=25570, **settings) as bridge:
                self.assertEqual(bridge.connection, {"connected": True})
        return launched, bridge

    def test_the_bridge_process_gets_the_goal_and_the_survey_option(self):
        launched, bridge = self._open()
        self.assertTrue(launched[0][1].endswith("resource-bridge.js"))
        self.assertEqual(launched[0][2:], ["--port", "25570", "--goal", "wooden_pickaxe", "--stone-survey", "off"])
        self.assertEqual((bridge.goal, bridge.stone_survey), ("wooden_pickaxe", False))
        launched, _ = self._open(stone_survey=True)
        self.assertEqual(launched[0][-4:], ["--goal", "wooden_pickaxe", "--stone-survey", "on"])
        launched, _ = self._open(goal="stone_pickaxe")
        self.assertEqual(launched[0][-4:], ["--goal", "stone_pickaxe", "--stone-survey", "off"])
        for bad in ({"goal": "stone"}, {"goal": None}, {"stone_survey": "on"}, {"stone_survey": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self._open(**bad)

    def test_the_python_goals_match_the_bridge(self):
        source = (Path(resource_live.__file__).resolve().parents[1] / "minecraft_bridge" / "resource-bridge.js").read_text(encoding="utf-8")
        declared = re.search(r"^const GOALS = \[(.*)\]$", source, re.MULTILINE)
        self.assertIsNotNone(declared)
        self.assertEqual(tuple(re.findall(r"'([a-z_]+)'", declared.group(1))), GOALS)


if __name__ == "__main__":
    unittest.main()

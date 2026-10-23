"""Summaries of retained resource runs must count only what the saved files say."""

import contextlib
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from infrastructure.__main__ import resource_report as resource_report_command
from minecraft_rl.resource_report import (
    EVIDENCE, classify_failure, compare_to_baselines, fisher_exact, load_run, render_markdown, summarize_runs,
    wilson_interval, write_report,
)


def observation(logs=0, planks=0, sticks=0, tables=0, pickaxes=0, table_placed=False, event="observe", recovery=None):
    return {"inventory_counts": {"logs": logs, "planks": planks, "sticks": sticks, "tables": tables, "pickaxes": pickaxes},
            "table_placed": table_placed, "nearby_log": False, "nearby_table": table_placed,
            "health": 20, "food": 20, "position": {"x": 0.5, "y": 70.0, "z": 0.5},
            "event": event, "recovery": recovery, "success": pickaxes > 0, "death": False}


def transition(step, action, after, world_seed=1):
    return {"step": step, "world_seed": world_seed, "action": action, "reward": -0.01,
            "before": observation(), "after": after}


def write_run(root: Path, name: str, metadata: dict, episodes: list[dict]) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for index, episode in enumerate(episodes, start=1):
        folder = run / f"episode-{index:04d}"
        folder.mkdir()
        (folder / "transitions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in episode.get("transitions", [])), encoding="utf-8")
        if "outcome" in episode:
            (folder / "outcome.json").write_text(json.dumps(episode["outcome"]), encoding="utf-8")
    return run


class ResourceReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # A descent that ended walled in, followed by a one-leaf escape in the same decision.
        descent = {"type": "canopy_descent", "layers_cleared": 2, "blocks_dug": ["oak_leaves", "oak_leaves"], "damage": 1,
                   "start_y": 76.0, "end_y": 71.0, "stopped": None, "descended": True,
                   "escape": {"type": "leaf_escape", "blocks_dug": ["oak_leaves"], "direction": "-x", "escaped": True, "stopped": None}}
        success_steps = [
            transition(1, "SEARCH", observation(event="cleared_2_leaf_layers;descended_canopy;cleared_1_leaf_wall_block;log_reached",
                                                recovery=descent)),
            transition(2, "CHOP", observation(logs=1, event="log_collected")),
            transition(3, "CRAFT_PICKAXE", observation(planks=3, sticks=2, pickaxes=1, table_placed=True, event="crafted_wooden_pickaxe")),
        ]
        self.training = write_run(self.root, "train-a", {"evidence": EVIDENCE, "mode": "online_training",
                                                         "algorithm": "q_learning", "baseline": None,
                                                         "episodes_requested": 3, "first_world_seed": 1,
                                                         "max_steps": 40}, [
            {"transitions": success_steps,
             "outcome": {"status": "completed", "world_seed": 1, "success": True, "death": False, "truncated": False,
                         "steps": 3, "seconds": 42.5, "return": 9.8, "final_observation": success_steps[-1]["after"]}},
            {"transitions": [transition(s, "SEARCH", observation(event="exploration_blocked"), 2) for s in range(1, 41)],
             "outcome": {"status": "completed", "world_seed": 2, "success": False, "death": False, "truncated": True,
                         "steps": 40, "seconds": 120.0, "return": -0.4, "final_observation": observation(event="exploration_blocked")}},
            {"transitions": [transition(1, "SEARCH", observation(event="explored"), 3)],
             "outcome": {"status": "invalid", "success": None, "death": None, "recorded_steps": 1,
                         "error": "RuntimeError: transport timeout"}},
        ])
        self.baseline = write_run(self.root, "demo-random", {"evidence": EVIDENCE, "mode": "frozen_evaluation",
                                                             "algorithm": "random", "baseline": "masked_random",
                                                             "episodes_requested": 2, "first_world_seed": 10000,
                                                             "max_steps": 40}, [
            {"transitions": [transition(1, "SEARCH", observation(logs=1, event="log_collected"), 10000)],
             "outcome": {"status": "completed", "world_seed": 10000, "success": False, "death": True, "truncated": False,
                         "steps": 1, "seconds": 9.0, "return": -10.01, "final_observation": observation(logs=1)}},
            {"transitions": [transition(1, "SEARCH", observation(event="cleared_2_leaf_wall_blocks;explored", recovery={
                                 "type": "leaf_escape", "blocks_dug": ["oak_leaves", "oak_leaves"], "direction": "+x",
                                 "escaped": True, "stopped": None}), 10001),
                             transition(2, "PLACE_TABLE", observation(planks=1, table_placed=True, event="table_placed"), 10001)],
             "outcome": {"status": "completed", "world_seed": 10001, "success": False, "death": False, "truncated": True,
                         "steps": 40, "seconds": 130.0, "return": -0.2,
                         "final_observation": observation(planks=1, table_placed=True)}},
        ])

    def test_wilson_interval_matches_closed_form_and_rejects_bad_counts(self):
        self.assertIsNone(wilson_interval(0, 0))
        low, high = wilson_interval(1, 2)
        z = 1.96
        center = (0.5 + z * z / 4) / (1 + z * z / 2)
        half = z * math.sqrt(0.25 / 2 + z * z / 16) / (1 + z * z / 2)
        self.assertAlmostEqual(low, center - half)
        self.assertAlmostEqual(high, center + half)
        self.assertEqual(wilson_interval(3, 3)[1], 1.0)
        for bad in ((3, 2), (-1, 2), (True, 2), (1.5, 2)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                wilson_interval(*bad)

    def test_failure_classification_uses_only_saved_progress(self):
        completed = {"status": "completed", "success": False, "death": False}
        self.assertEqual(classify_failure({**completed, "final_observation": observation()}, []), "no_wood_collected")
        # Wood that was collected earlier and then consumed still counts as collected.
        self.assertEqual(classify_failure({**completed, "final_observation": observation()},
                                          [transition(1, "CHOP", observation(logs=1))]), "table_not_placed")
        self.assertEqual(classify_failure({**completed, "final_observation": observation(planks=2)}, []), "table_not_placed")
        # Sticks alone can come from leaf blocks cleared during a canopy descent, so they are not wood evidence.
        self.assertEqual(classify_failure({**completed, "final_observation": observation(sticks=2)}, []), "no_wood_collected")
        self.assertEqual(classify_failure({**completed, "final_observation": observation(planks=3, table_placed=True)}, []), "pickaxe_not_crafted")
        self.assertEqual(classify_failure({**completed, "death": True, "final_observation": observation()}, []), "death")
        self.assertEqual(classify_failure({**completed, "success": True}, []), "success")
        self.assertEqual(classify_failure({"status": "invalid"}, []), "invalid_run")
        self.assertEqual(classify_failure({"status": "interrupted"}, []), "interrupted_run")

    def test_groups_keep_training_evaluation_and_baseline_apart(self):
        summary = summarize_runs([self.training, self.baseline])
        self.assertEqual([(g["mode"], g["algorithm"], g["baseline"]) for g in summary["groups"]],
                         [("frozen_evaluation", "random", "masked_random"), ("online_training", "q_learning", None)])
        baseline, training = summary["groups"]
        self.assertEqual((training["episodes_attempted"], training["episodes_completed"], training["episodes_invalid"]), (3, 2, 1))
        self.assertEqual(training["successes"], 1)
        self.assertEqual(training["success_rate"], 0.5)
        self.assertEqual(training["success_wilson_95"], wilson_interval(1, 2))
        # Policy failures count completed episodes only; the invalid one is listed apart.
        self.assertEqual(training["failure_causes"], {"no_wood_collected": 1})
        self.assertEqual(training["not_measured"], {"invalid": 1})
        self.assertEqual((training["max_steps"], training["checkpoint_sha256"]), (40, None))
        self.assertEqual(training["mean_decisions_success"], 3)
        self.assertEqual(training["mean_decisions_all"], 21.5)
        self.assertEqual(training["mean_task_seconds_success"], 42.5)
        self.assertEqual(training["canopy_layers_cleared"], 2)
        self.assertEqual(training["recovery_damage"], 1)
        # Nested escapes count their own blocks only, never the descent's leaf layers.
        self.assertEqual((training["leaf_wall_blocks_cleared"], baseline["leaf_wall_blocks_cleared"]), (1, 2))
        self.assertEqual(training["event_counts"]["cleared_1_leaf_wall_block"], 1)
        self.assertEqual(baseline["event_counts"]["cleared_2_leaf_wall_blocks"], 1)
        self.assertEqual(training["event_counts"]["exploration_blocked"], 40)
        self.assertEqual(training["event_counts"]["descended_canopy"], 1)
        self.assertEqual(training["world_seeds"], [1, 2, 3])
        self.assertEqual(training["return_label"], "exploratory shaped training return")
        self.assertEqual((baseline["deaths"], baseline["truncations"], baseline["successes"]), (1, 1, 0))
        self.assertEqual(baseline["success_rate"], 0.0)
        self.assertEqual(baseline["failure_causes"], {"death": 1, "pickaxe_not_crafted": 1})
        self.assertEqual(baseline["not_measured"], {})
        self.assertIsNone(baseline["mean_decisions_success"])
        invalid = [e for e in summary["episodes"] if e["status"] == "invalid"][0]
        self.assertEqual((invalid["steps"], invalid["success"], invalid["failure_cause"]), (1, None, "invalid_run"))
        self.assertIn("transport timeout", invalid["error"])

    def test_report_files_are_written_once_and_never_overwrite(self):
        summary = summarize_runs([self.training])
        out = write_report(summary, self.root / "report")
        saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["groups"][0]["successes"], 1)
        self.assertEqual(saved["evidence"], summary["evidence"])
        with (out / "episodes.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["failure_cause"] for row in rows], ["success", "no_wood_collected", "invalid_run"])
        self.assertEqual(json.loads(rows[1]["events"]), {"exploration_blocked": 40})
        markdown = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("NOT FULL SURVIVAL COMPLETION", markdown)
        self.assertIn("online_training / q_learning", markdown)
        self.assertIn("no_wood_collected ×1", markdown)
        self.assertIn("Not measured (invalid, interrupted, or unfinished; excluded from every rate): invalid ×1.", markdown)
        self.assertEqual(markdown, render_markdown(summary))
        with self.assertRaises(ValueError):
            write_report(summary, out)

    def test_episode_without_outcome_is_unfinished_not_a_result(self):
        run = write_run(self.root, "cut-off", {"evidence": EVIDENCE, "mode": "online_training", "algorithm": "sarsa",
                                               "baseline": None},
                        [{"transitions": [transition(1, "SEARCH", observation(event="explored"))]}])
        loaded = load_run(run)
        episode = loaded["episodes"][0]
        self.assertEqual((episode["status"], episode["success"], episode["steps"], episode["failure_cause"]),
                         ("unfinished", None, 1, "unfinished_run"))
        group = summarize_runs([run])["groups"][0]
        self.assertEqual((group["episodes_completed"], group["episodes_invalid"], group["success_rate"]), (0, 1, None))
        self.assertIsNone(group["success_wilson_95"])
        self.assertEqual((group["failure_causes"], group["not_measured"]), ({}, {"unfinished": 1}))
        with self.assertRaises(ValueError):
            summarize_runs([])

    def test_only_natural_resource_run_folders_are_summarized(self):
        # Arena, simulator, and watch runs write the same layout; their successes
        # must never appear under the natural-resource banner.
        arena = write_run(self.root, "watch-arena", {"evidence": "REAL GAME DEVELOPMENT ARENA; NOT FRESH SURVIVAL",
                                                     "algorithm": "q_learning"},
                          [{"transitions": [], "outcome": {"status": "completed", "success": True, "death": False,
                                                           "truncated": False, "steps": 12}}])
        with self.assertRaises(ValueError) as context:
            summarize_runs([self.training, arena])
        self.assertIn("watch-arena", str(context.exception))
        no_mode = write_run(self.root, "no-mode", {"evidence": EVIDENCE, "algorithm": "q_learning"}, [])
        with self.assertRaises(ValueError):
            load_run(no_mode)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            resource_report_command([str(arena), "--out", str(self.root / "arena-report")])
        self.assertFalse((self.root / "arena-report").exists())

    def test_duplicate_same_named_and_empty_run_folders_are_refused(self):
        with self.assertRaises(ValueError):
            summarize_runs([self.training, self.training])
        with self.assertRaises(ValueError):
            summarize_runs([self.training, Path(str(self.training).replace("train-a", "./train-a"))])
        elsewhere = Path(tempfile.mkdtemp(dir=self.root))
        twin = write_run(elsewhere, "train-a", {"evidence": EVIDENCE, "mode": "online_training",
                                                "algorithm": "q_learning", "baseline": None, "max_steps": 40}, [])
        with self.assertRaises(ValueError) as context:
            summarize_runs([self.training, twin])
        self.assertIn("train-a", str(context.exception))
        with self.assertRaises(ValueError) as context:
            summarize_runs([twin])
        self.assertIn(str(twin), str(context.exception))

    def test_frozen_evaluations_of_different_checkpoints_and_step_limits_stay_apart(self):
        def frozen(name, checkpoint, max_steps, seed):
            return write_run(self.root, name, {"evidence": EVIDENCE, "mode": "frozen_evaluation",
                                               "algorithm": "q_learning", "baseline": None,
                                               "initial_policy_sha256": checkpoint, "max_steps": max_steps},
                             [{"transitions": [transition(1, "SEARCH", observation(event="explored"), seed)],
                               "outcome": {"status": "completed", "world_seed": seed, "success": False, "death": False,
                                           "truncated": True, "steps": max_steps, "seconds": 50.0, "return": -0.3,
                                           "final_observation": observation()}}])
        first = frozen("frozen-a", "a" * 64, 60, 1)
        second = frozen("frozen-b", "b" * 64, 60, 2)
        longer = frozen("frozen-a-long", "a" * 64, 80, 3)
        again = frozen("frozen-a-again", "a" * 64, 60, 4)
        summary = summarize_runs([first, second, longer, again])
        groups = {(g["checkpoint_sha256"], g["max_steps"]): g for g in summary["groups"]}
        self.assertEqual(set(groups), {("a" * 64, 60), ("b" * 64, 60), ("a" * 64, 80)})
        self.assertEqual(groups[("a" * 64, 60)]["runs"], ["frozen-a", "frozen-a-again"])
        self.assertEqual(groups[("a" * 64, 60)]["episodes_completed"], 2)
        markdown = render_markdown(summary)
        for label in ("checkpoint aaaaaaaaaaaa / max_steps 60", "checkpoint bbbbbbbbbbbb / max_steps 60",
                      "checkpoint aaaaaaaaaaaa / max_steps 80"):
            self.assertIn(label, markdown)
        self.assertEqual([r["initial_policy_sha256"] for r in summary["runs"]], ["a" * 64, "b" * 64, "a" * 64, "a" * 64])
        # Training runs start empty, so they stay pooled per algorithm and step limit.
        self.assertIsNone(summarize_runs([self.training])["groups"][0]["checkpoint_sha256"])

    def test_console_copy_of_the_report_survives_a_limited_code_page(self):
        out = self.root / "cli-report"
        console = io.TextIOWrapper(io.BytesIO(), encoding="cp437")
        with contextlib.redirect_stdout(console):
            code = resource_report_command([str(self.training), str(self.baseline), "--out", str(out)])
        console.flush()
        printed = console.buffer.getvalue().decode("cp437")
        self.assertEqual(code, 0)
        self.assertIn("Natural-resource run summary", printed)
        self.assertIn("no_wood_collected ?1", printed)
        self.assertIn("no_wood_collected ×1", (out / "report.md").read_text(encoding="utf-8"))
        self.assertTrue((out / "episodes.csv").is_file())


    def _condition_run(self, name, condition, mode, algorithm, outcomes, *, baseline=None, checkpoint=None,
                       first_seed=20000, decisions=None):
        metadata = {"evidence": EVIDENCE, "mode": mode, "algorithm": algorithm, "baseline": baseline,
                    "initial_policy_sha256": checkpoint, "max_steps": 60}
        if condition is not None:
            metadata["condition"] = condition
        episodes = []
        for index, success in enumerate(outcomes):
            rows = decisions(index) if decisions else [transition(1, "SEARCH", observation(event="explored"), first_seed + index)]
            episodes.append({"transitions": rows,
                             "outcome": {"status": "completed", "world_seed": first_seed + index, "success": success,
                                         "death": False, "truncated": not success, "steps": len(rows), "seconds": 10.0,
                                         "return": 1.0, "final_observation": observation(pickaxes=int(success))}})
        return write_run(self.root, name, metadata, episodes)

    def test_conditions_are_never_pooled_and_legacy_runs_read_as_masked(self):
        legacy = self._condition_run("legacy", None, "frozen_evaluation", "random", [True, True], baseline="masked_random")
        unmasked = self._condition_run("unmasked", "unmasked", "frozen_evaluation", "random", [False, False],
                                       baseline="uniform_random")
        summary = summarize_runs([legacy, unmasked])
        self.assertEqual(sorted((g["condition"], g["action_mask"]) for g in summary["groups"]),
                         [("masked", "recipe_and_budget_v1"), ("unmasked", "none")])
        self.assertEqual({r["run"]: r["condition"] for r in summary["runs"]}, {"legacy": "masked", "unmasked": "unmasked"})
        self.assertIn("unmasked / frozen_evaluation / random / uniform_random", render_markdown(summary))
        bad = self._condition_run("bad", "sideways", "frozen_evaluation", "random", [True])
        with self.assertRaises(ValueError):
            load_run(bad)

    def test_decision_fractions_count_single_choices_masked_out_choices_and_no_progress(self):
        def rows(_):
            start, found = observation(), observation(logs=1)
            return [
                # One allowed action, progress made.
                {**transition(1, "CHOP", found), "state": [0] * 9, "next_state": [1] + [0] * 8,
                 "allowed_actions": [1], "masked_actions": [1]},
                # Unmasked: the chosen action is outside the mask and changes nothing.
                {**transition(2, "CRAFT_PICKAXE", found), "state": [1] + [0] * 8, "next_state": [1] + [0] * 8,
                 "allowed_actions": list(range(8)), "masked_actions": [2]},
                # Legacy record without masked_actions: the allowed set was the mask.
                {**transition(3, "CRAFT_PLANKS", start), "state": [1] + [0] * 8 + [1],
                 "next_state": [0, 4] + [0] * 7 + [0], "allowed_actions": [1, 2]},
            ]
        run = self._condition_run("fractions", "unmasked", "frozen_evaluation", "random", [False],
                                  baseline="uniform_random", decisions=rows)
        group = summarize_runs([run])["groups"][0]
        self.assertEqual(group["decisions"], 3)
        self.assertAlmostEqual(group["single_allowed_fraction"], 1 / 3)
        self.assertAlmostEqual(group["masked_out_fraction"], 1 / 3)
        self.assertAlmostEqual(group["no_progress_fraction"], 1 / 3)
        self.assertIn("| 3 | 33% | 33% | 33% |", render_markdown(summarize_runs([run])))

    def test_learned_policies_are_compared_only_with_a_matching_random_baseline(self):
        self.assertAlmostEqual(fisher_exact(10, 10, 0, 10), 2 / math.comb(20, 10))
        self.assertEqual(fisher_exact(9, 10, 10, 10), 1.0)
        self.assertAlmostEqual(fisher_exact(3, 5, 1, 5), 0.5238095238095238)
        for bad in ((1, 0, 0, 1), (2, 1, 0, 1), (-1, 2, 0, 2), (True, 2, 0, 2)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                fisher_exact(*bad)
        uniform = self._condition_run("uniform", "unmasked", "frozen_evaluation", "random", [False] * 9 + [True],
                                      baseline="uniform_random")
        strong = self._condition_run("q-unmasked", "unmasked", "frozen_evaluation", "q_learning", [True] * 10,
                                     checkpoint="a" * 64)
        history = self._condition_run("q-history", "unmasked_history", "frozen_evaluation", "q_learning",
                                      [True] * 4 + [False] * 6, checkpoint="b" * 64)
        masked_random = self._condition_run("masked-random", "masked", "frozen_evaluation", "random", [True] * 10,
                                            baseline="masked_random")
        masked_q = self._condition_run("q-masked", "masked", "frozen_evaluation", "q_learning", [True] * 9 + [False],
                                       checkpoint="c" * 64)
        weak = self._condition_run("sarsa-masked", "masked", "frozen_evaluation", "sarsa", [False] * 8 + [True] * 2,
                                   checkpoint="d" * 64)
        other_seeds = self._condition_run("q-other-seeds", "unmasked", "frozen_evaluation", "sarsa", [True] * 10,
                                          checkpoint="e" * 64, first_seed=30000)
        training = self._condition_run("train-unmasked", "unmasked", "online_training", "q_learning", [True] * 10)
        summary = summarize_runs([uniform, strong, history, masked_random, masked_q, weak, other_seeds, training])
        verdicts = {(c["condition"], c["algorithm"], c["checkpoint_sha256"][:1]): c for c in summary["comparisons"]}
        self.assertEqual(len(summary["comparisons"]), 5)  # training runs are never compared
        self.assertTrue(verdicts[("unmasked", "q_learning", "a")]["verdict"].startswith("benefit shown"))
        self.assertEqual(verdicts[("unmasked", "q_learning", "a")]["baseline"], "uniform_random")
        # unmasked_history is compared with the state-independent uniform random baseline.
        self.assertEqual(verdicts[("unmasked_history", "q_learning", "b")]["baseline_condition"], "unmasked")
        self.assertTrue(verdicts[("unmasked_history", "q_learning", "b")]["verdict"].startswith("no benefit shown"))
        self.assertTrue(verdicts[("masked", "q_learning", "c")]["verdict"].startswith("no benefit shown"))
        self.assertEqual(verdicts[("masked", "q_learning", "c")]["baseline"], "masked_random")
        self.assertTrue(verdicts[("masked", "sarsa", "d")]["verdict"].startswith("worse than random"))
        self.assertTrue(verdicts[("unmasked", "sarsa", "e")]["verdict"].startswith("not compared"))
        self.assertIsNone(verdicts[("unmasked", "sarsa", "e")]["fisher_p"])
        self.assertAlmostEqual(verdicts[("unmasked", "q_learning", "a")]["fisher_p"], fisher_exact(10, 10, 1, 10))
        out = write_report(summary, self.root / "comparison-report")
        with (out / "comparisons.csv").open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 5)
        markdown = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("## Comparison with the random baseline", markdown)
        self.assertIn("no benefit shown: the intervals overlap", markdown)
        self.assertEqual(compare_to_baselines([]), [])
        # An episode that failed before any transition still counts its seed, so seed sets still match.
        cut = self._condition_run("q-cut", "unmasked", "frozen_evaluation", "q_learning", [True] * 10, checkpoint="f" * 64)
        (cut / "episode-0004" / "transitions.jsonl").write_text("", encoding="utf-8")
        (cut / "episode-0004" / "outcome.json").write_text(json.dumps({"status": "invalid", "success": None, "death": None,
                                                                       "recorded_steps": 0}), encoding="utf-8")
        meta = json.loads((cut / "metadata.json").read_text(encoding="utf-8"))
        meta["first_world_seed"] = 20000
        (cut / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        paired = summarize_runs([uniform, cut])["comparisons"][0]
        self.assertEqual((paired["learned_completed"], paired["baseline"]), (9, "uniform_random"))
        self.assertTrue(paired["verdict"].startswith("benefit shown"))
        # Counted on the seeds both completed; the lost seed is listed with a conservative check.
        self.assertEqual((paired["seeds_compared"], paired["learned_lost_seeds"], paired["baseline_lost_seeds"]),
                         (9, [20003], []))
        self.assertEqual(paired["baseline_successes"], 1)
        self.assertTrue(paired["conservative_verdict"].startswith("benefit shown"))
        self.assertIn("seeds not measured for the learned policy [20003]", render_markdown(summarize_runs([uniform, cut])))

    def test_pairing_requires_the_same_pause_and_helper_revision_and_flags_pretraining(self):
        def run(name, algorithm, outcomes, **meta):
            folder = self._condition_run(name, "unmasked", "frozen_evaluation", algorithm, outcomes,
                                         baseline="uniform_random" if algorithm == "random" else None,
                                         checkpoint=None if algorithm == "random" else name[0] * 64)
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            metadata.update({"pause_between_decisions": 0, "source_sha256": {"bridge": "x"}, **meta})
            (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return folder
        baseline = run("random", "random", [False] * 10)
        learned = run("aq", "q_learning", [True] * 10)
        paused = run("bq", "q_learning", [True] * 10, pause_between_decisions=0.5)
        other_helpers = run("cq", "q_learning", [True] * 10, source_sha256={"bridge": "y"})
        pretrained = run("dq", "q_learning", [True] * 10, checkpoint_pretrain_simulator_episodes=2000)
        summary = summarize_runs([baseline, learned, paused, other_helpers, pretrained])
        verdicts = {c["checkpoint_sha256"][0]: c for c in summary["comparisons"]}
        self.assertTrue(verdicts["a"]["verdict"].startswith("benefit shown"))
        self.assertTrue(verdicts["b"]["verdict"].startswith("not compared"))
        self.assertTrue(verdicts["c"]["verdict"].startswith("not compared"))
        self.assertTrue(verdicts["d"]["verdict"].startswith("simulator-pretrained checkpoint, not a live-learning result"))
        self.assertTrue(verdicts["d"]["simulator_pretrained"])
        self.assertIn("/ simulator-pretrained", render_markdown(summary))

    def test_history_bit_is_judged_only_against_the_unmasked_policy_of_the_same_algorithm(self):
        plain_q = self._condition_run("pq", "unmasked", "frozen_evaluation", "q_learning", [False] * 10, checkpoint="p" * 64)
        plain_s = self._condition_run("ps", "unmasked", "frozen_evaluation", "sarsa", [True] * 9 + [False], checkpoint="s" * 64)
        history_q = self._condition_run("hq", "unmasked_history", "frozen_evaluation", "q_learning", [True] * 10,
                                        checkpoint="h" * 64)
        history_s = self._condition_run("hs", "unmasked_history", "frozen_evaluation", "sarsa", [True] * 10,
                                        checkpoint="g" * 64)
        summary = summarize_runs([plain_q, plain_s, history_q, history_s])
        rows = {row["algorithm"]: row for row in summary["history_comparisons"]}
        self.assertEqual(set(rows), {"q_learning", "sarsa"})
        self.assertTrue(rows["q_learning"]["verdict"].startswith("effect of the bit shown"))
        self.assertTrue(rows["sarsa"]["verdict"].startswith("no effect of the bit shown"))
        self.assertEqual(rows["q_learning"]["unmasked_checkpoint"], "p" * 64)
        markdown = render_markdown(summary)
        self.assertIn("## Effect of the history bit", markdown)
        self.assertIn("no effect of the bit shown: the intervals overlap", markdown)
        out = write_report(summary, self.root / "history-report")
        self.assertTrue((out / "history_comparisons.csv").is_file())


    def test_parallel_slots_are_separate_groups_and_never_paired_with_runs_alone(self):
        def run(name, algorithm, outcomes, slots=None):
            folder = self._condition_run(name, "unmasked", "frozen_evaluation", algorithm, outcomes,
                                         baseline="uniform_random" if algorithm == "random" else None,
                                         checkpoint=None if algorithm == "random" else name[0] * 64)
            if slots is not None:
                metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
                metadata["execution"] = {"server_port": 25568, "parallel_slots": slots}
                (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return folder
        alone = run("random-alone", "random", [False] * 10)  # recorded before resource-parallel: one slot
        four = run("random-four", "random", [False] * 10, slots=4)
        learned_alone = run("aq", "q_learning", [True] * 10, slots=1)
        learned_four = run("bq", "q_learning", [True] * 10, slots=4)
        learned_two = run("cq", "q_learning", [True] * 10, slots=2)
        summary = summarize_runs([alone, four, learned_alone, learned_four, learned_two])
        baselines = {g["parallel_slots"]: g for g in summary["groups"] if g["algorithm"] == "random"}
        self.assertEqual(set(baselines), {1, 4})
        self.assertEqual((baselines[1]["runs"], baselines[4]["runs"]), (["random-alone"], ["random-four"]))
        rows = {c["checkpoint_sha256"][0]: c for c in summary["comparisons"]}
        self.assertEqual(len(summary["comparisons"]), 3)
        self.assertEqual((rows["a"]["baseline"], rows["a"]["parallel_slots"]), ("uniform_random", 1))
        self.assertTrue(rows["a"]["verdict"].startswith("benefit shown"))
        self.assertTrue(rows["b"]["verdict"].startswith("benefit shown"))
        self.assertTrue(rows["c"]["verdict"].startswith("not compared"))
        self.assertIn("/ 4 parallel slots", render_markdown(summary))
        broken = run("broken", "random", [False], slots=0)
        with self.assertRaisesRegex(ValueError, "parallel_slots"):
            summarize_runs([broken])


if __name__ == "__main__":
    unittest.main()

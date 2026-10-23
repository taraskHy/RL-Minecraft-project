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
    EVIDENCE, classify_failure, load_run, render_markdown, summarize_runs, wilson_interval, write_report,
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


if __name__ == "__main__":
    unittest.main()

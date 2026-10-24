"""The declared initial-Q analysis must count the right arms and the loop and untried-choice measures."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "initial_q_analysis.py"
spec = importlib.util.spec_from_file_location("initial_q_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

STUCK = [0, 2, 2, 0, 0, 0, 1, 1, 2]
START = [0, 0, 0, 0, 0, 0, 0, 0, 2]


def decision(action, state, next_state, q_before):
    return {"action": action, "state": state, "next_state": next_state, "q_before": q_before,
            "masked_actions": [0], "allowed_actions": list(range(8))}


class InitialQAnalysisTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.schedule = Path(temporary.name) / "iq-experiment"
        self.units = []

    def run_folder(self, name, algorithm, initial_q, episodes):
        folder = self.schedule / name
        folder.mkdir(parents=True)
        (folder / "metadata.json").write_text(json.dumps({"algorithm": algorithm, "initial_q": initial_q}))
        for number, (outcome, transitions) in enumerate(episodes, 1):
            episode = folder / f"episode-{number:04d}"
            episode.mkdir()
            (episode / "outcome.json").write_text(json.dumps(outcome))
            (episode / "transitions.jsonl").write_text("".join(json.dumps(t) + "\n" for t in transitions))
        self.units.append({"name": name, "parts": [{"folder": name}]})

    def write_schedule(self):
        (self.schedule / "schedule.json").write_text(json.dumps({"units": self.units}))

    def output(self):
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            analysis.main([str(self.schedule)])
        return text.getvalue()

    def test_pooled_arms_loops_untried_choices_and_training(self):
        loop = [decision("CRAFT_TABLE", STUCK, STUCK, [-0.014, -0.014] + [0.0] * 6)] * 20
        win = [decision("SEARCH", START, STUCK, [0.5] * 8)]
        self.run_folder("iq-q-zero-s0-test", "q_learning", 0.0, [
            ({"status": "completed", "success": False, "world_seed": 40000}, loop),
            ({"status": "completed", "success": True, "world_seed": 40001}, win),
            ({"status": "invalid", "success": None, "world_seed": 40002}, [])])
        self.run_folder("iq-q-pess-s0-test", "q_learning", -1.0, [
            ({"status": "completed", "success": True, "world_seed": 40000}, win),
            ({"status": "completed", "success": True, "world_seed": 40001}, win)])
        self.run_folder("iq-random-test", "random", None, [
            ({"status": "completed", "success": False, "world_seed": 40000}, win)])
        self.run_folder("iq-q-pess-s0-train", "q_learning", -1.0, [
            ({"status": "completed", "success": True, "world_seed": 100}, win),
            ({"status": "completed", "success": False, "world_seed": 112}, win)])
        self.write_schedule()
        text = self.output()
        # One loop failure (20 unchanged decisions, untried CRAFT_TABLE at 0) and one invalid episode.
        self.assertIn("| q_learning | 0 | 0 | 1/2 | ", text)
        self.assertIn("| 1 | 1 | 95% |", text)
        self.assertIn("| q_learning | 1/2 (", text)
        self.assertIn("2/2 (", text)
        self.assertIn("no effect shown: the intervals overlap", text)
        self.assertIn("Uniform random baseline on the same worlds: 0/1", text)
        self.assertIn("| q_learning | -1 | 0 | 2 | 1 | 1, 0, 0, 0, 0 |", text)

    def test_a_run_whose_record_contradicts_its_name_stops_the_analysis(self):
        self.run_folder("iq-q-pess-s0-test", "q_learning", 0.0, [])
        self.write_schedule()
        with self.assertRaisesRegex(ValueError, "not q_learning with -1.0"):
            self.output()


if __name__ == "__main__":
    unittest.main()

"""The declared initial-Q analysis must pair the arms, count every attempted episode, and name each failure kind."""

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

START = [0, 0, 0, 0, 0, 0, 0, 0, 2]
STUCK = [0, 2, 2, 0, 0, 0, 1, 1, 2]
WOOD = {"inventory_counts": {"logs": 0, "planks": 2, "sticks": 4, "tables": 0, "pickaxes": 0}, "table_placed": True}
NONE = {"inventory_counts": {"logs": 0, "planks": 0, "sticks": 0, "tables": 0, "pickaxes": 0}, "table_placed": False}


def decision(action, state, next_state, q_value, *, event="missing_materials", after=WOOD, chosen_only=True):
    q_before = [0.5] * 8
    q_before[analysis.ACTION_NAMES.index(action)] = q_value
    if not chosen_only:
        q_before = [q_value] * 8
        q_before[analysis.ACTION_NAMES.index(action)] = 0.5
    return {"action": action, "state": state, "next_state": next_state, "q_before": q_before,
            "masked_actions": [0], "after": {**after, "event": event}}


def win(q_value=0.5):
    # SEARCH (index 0) is inside the mask and makes progress.
    return [decision("SEARCH", START, STUCK, q_value, event="explored")]


def loop(initial_q):
    # CRAFT_TABLE is outside the mask, changes nothing, and still has the arm's initial value.
    return [decision("CRAFT_TABLE", STUCK, STUCK, initial_q)] * 20


def completed(world, success, transitions):
    return {"status": "completed", "success": success, "world_seed": world, "death": False,
            "final_observation": transitions[-1]["after"] if transitions else NONE}, transitions


class InitialQAnalysisTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.schedule = Path(temporary.name) / "iq-experiment"
        self.units = []

    def run_folder(self, name, algorithm, initial_q, episodes, *, first_world=40000, status="done", no_outcome=()):
        folder = self.schedule / name
        folder.mkdir(parents=True)
        (folder / "metadata.json").write_text(json.dumps({"algorithm": algorithm, "initial_q": initial_q,
                                                          "first_world_seed": first_world}))
        for number, (outcome, transitions) in enumerate(episodes, 1):
            episode = folder / f"episode-{number:04d}"
            episode.mkdir()
            if number not in no_outcome:
                (episode / "outcome.json").write_text(json.dumps(outcome))
            (episode / "transitions.jsonl").write_text("".join(json.dumps(t) + "\n" for t in transitions))
        self.units.append({"name": name, "status": status, "attempted": len(episodes), "episodes": 20,
                           "parts": [{"folder": name}]})

    def output(self):
        (self.schedule / "schedule.json").write_text(json.dumps({"status": "completed", "units": self.units}))
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            analysis.main([str(self.schedule)])
        return text.getvalue()

    def arms(self, algorithm, short, zero_successes, pess_successes, worlds=20):
        for arm, value, successes in (("zero", 0.0, zero_successes), ("pess", -1.0, pess_successes)):
            for seed in (0, 1, 2):
                self.run_folder(f"iq-{short}-{arm}-s{seed}-test", algorithm, value,
                                [completed(40000 + i, i < successes, win() if i < successes else loop(value))
                                 for i in range(worlds)])

    def row(self, text, prefix):
        return next(line for line in text.splitlines() if line.startswith(prefix))

    def test_the_verdict_points_the_right_way(self):
        self.arms("q_learning", "q", zero_successes=5, pess_successes=20)
        self.arms("sarsa", "sarsa", zero_successes=20, pess_successes=5)
        text = self.output()
        self.assertIn("effect shown: initial Q -1 better", self.row(text, "| q_learning | 60 |"))
        self.assertIn("effect shown: initial Q -1 worse", self.row(text, "| sarsa | 60 |"))
        self.assertIn("| q_learning | 60 | 15/60 (", text)
        self.assertIn("| 5/20, 5/20, 5/20 | 20/20, 20/20, 20/20 | 0 / 0 |", text)

    def test_intervals_that_only_touch_show_no_effect(self):
        # 57 of 60 and 46 of 60 separate; 57 and 47 do not (pinned against wilson_interval).
        low_57 = analysis.wilson_interval(57, 60)[0]
        self.assertLess(analysis.wilson_interval(46, 60)[1], low_57)
        self.assertGreater(analysis.wilson_interval(47, 60)[1], low_57)
        self.assertEqual(analysis._verdict((57, 60), (46, 60)), "effect shown: initial Q -1 better")
        self.assertEqual(analysis._verdict((57, 60), (47, 60)), "no effect shown: the intervals overlap")
        self.assertEqual(analysis._verdict((0, 0), (0, 0)), "not compared: no seed completed by both arms")

    def test_failure_kinds_untried_choices_shares_and_training(self):
        canopy = [decision("SEARCH", STUCK, STUCK, 0.5, event="canopy_descent_refused:fall_too_far")] * 20
        no_wood = [decision("SEARCH", START, START, 0.5, event="explored_locally", after=NONE)] * 20
        self.run_folder("iq-q-pess-s0-test", "q_learning", -1.0, [
            completed(40000, False, loop(-1.0)), completed(40001, False, canopy), completed(40002, False, no_wood),
            completed(40003, True, win(-1.0))])
        self.run_folder("iq-q-zero-s0-test", "q_learning", 0.0, [completed(40000 + i, True, win()) for i in range(4)])
        self.run_folder("iq-random-test", "random", None, [completed(40000, False, win())])
        self.run_folder("iq-filler-random", "random", None, [completed(40100, True, win())], first_world=40100)
        self.run_folder("iq-q-pess-s0-train", "q_learning", -1.0, [
            completed(100, True, win()), completed(112, True, win()), completed(113, False, win())],
            first_world=100)
        text = self.output()
        # Per agent: one of each failure kind. 21 of 61 decisions chose an action still at -1 (the loop and the win);
        # the loop's 20 decisions are outside the mask; the loop, canopy and no-wood decisions changed nothing.
        self.assertIn("| q_learning | -1 | 0 | 1/4 | 0.05–0.70 | 0 | 1 | 1 | 1 | 0 | 34% | 33% | 98% |", text)
        self.assertIn("| q_learning | -1 | 1 | 1 | 1 | 0 | 34% | 33% | 98% |", text)
        self.assertIn("Uniform random baseline on the same worlds: 0/1 (", text)
        self.assertNotIn("iq-filler", text.split("## Test")[1])
        self.assertIn("| q_learning | -1 | 0 | 3 | 0 | 2 | 1, 1, 0, 0, 0 |", text)

    def test_lost_episodes_are_paired_out_and_checked_conservatively(self):
        # Zero: 20/20 on seed 0 only. Pess: 20 on seed 0, two of them invalid and one without an outcome file.
        self.run_folder("iq-q-zero-s0-test", "q_learning", 0.0, [completed(40000 + i, False, loop(0.0)) for i in range(20)])
        episodes = [completed(40000 + i, True, win()) for i in range(20)]
        episodes[3] = ({"status": "invalid", "success": None, "world_seed": 40003}, [])
        episodes[7] = ({"status": "invalid", "success": None, "world_seed": 40007}, [])
        self.run_folder("iq-q-pess-s0-test", "q_learning", -1.0, episodes, no_outcome=(12,), status="interrupted")
        text = self.output()
        row = self.row(text, "| q_learning | 17 |")
        # Compared on the 17 pairs both completed; the other 43 (agent seeds 1 and 2) and 3 lost count as not measured.
        self.assertIn("| 0/17 (", row)
        self.assertIn("17/17 (", row)
        self.assertIn("| 40 / 43 | effect shown: initial Q -1 better |", row)
        # Conservatively, pess 17/60 against zero 0 + 40 lost = 40/60: the claim does not hold.
        self.assertTrue(row.endswith("| the claim does not hold (initial Q -1 worse) |"), row)
        self.assertIn("**Incomplete:** these units did not finish: iq-q-pess-s0-test", text)
        self.assertIn("| iq-q-pess-s0-test | interrupted | 20 | 20 |", text)
        self.assertIn("| q_learning | -1 | 0 | 17/17 | 0.82–1.00 | 3 |", text)

    def test_a_run_whose_record_contradicts_its_name_stops_the_analysis(self):
        self.run_folder("iq-q-pess-s0-test", "q_learning", 0.0, [])
        with self.assertRaisesRegex(ValueError, "not q_learning with -1.0"):
            self.output()


if __name__ == "__main__":
    unittest.main()

"""The parallel scheduler must keep runs apart, continue only as declared, and never reuse a folder."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from infrastructure import parallel_runs, server
from infrastructure.parallel_runs import epsilon_at, parse_plan, run_schedule


def make_plan(**overrides):
    plan = {"schema": 1, "name": "px", "slots": 2,
            "defaults": {"agent_seed": 0, "max_steps": 60, "pause": 0},
            "jobs": [
                {"name": "px-q-train", "kind": "train", "condition": "unmasked", "algorithm": "q_learning",
                 "episodes": 4, "first_world_seed": 100, "epsilon": 0.6, "epsilon_end": 0.1},
                {"name": "px-q-test", "kind": "evaluate", "policy_from": "px-q-train",
                 "episodes": 6, "first_world_seed": 20000, "chunks": 3},
                {"name": "px-random-test", "kind": "evaluate", "random_baseline": True, "condition": "unmasked",
                 "episodes": 6, "first_world_seed": 20000, "chunks": 3},
            ]}
    plan.update(overrides)
    return plan


def option(command, name):
    return command[command.index(name) + 1]


class FakeRun:
    """Stands in for one resource run: writes its episode folders at launch, then exits after a few polls."""

    def __init__(self, owner, command, statuses, code, polls):
        self.owner, self.command, self.code, self.polls = owner, command, code, polls
        self.folder = Path(option(command, "--out"))
        self.closed = False
        owner.alive.append(self)
        owner.max_alive = max(owner.max_alive, len(owner.alive))
        ports = [option(run.command, "--port") for run in owner.alive]
        owner.port_clash |= len(ports) != len(set(ports))
        owner.events.append(("start", self.folder.name))
        self.folder.mkdir(parents=True)
        (self.folder / "metadata.json").write_text("{}", encoding="utf-8")
        if "resource-train" in command:
            (self.folder / "policy.json").write_text(json.dumps({"saved_by": self.folder.name}), encoding="utf-8")
        for number, status in enumerate(statuses, 1):
            episode = self.folder / f"episode-{number:04d}"
            episode.mkdir()
            (episode / "outcome.json").write_text(json.dumps({"status": status}), encoding="utf-8")
            if status != "completed":
                break

    def poll(self):
        if self.polls > 0:
            self.polls -= 1
            return None
        if self in self.owner.alive:
            self.owner.alive.remove(self)
            self.owner.events.append(("exit", self.folder.name))
        return self.code

    def close(self):
        self.closed = True


class ParallelRunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(parallel_runs, "ROOT", self.root))
        self.code = self.stack.enter_context(patch.object(
            parallel_runs, "code_state", return_value={"head": "abc123", "uncommitted_code": [], "plan_committed": True}))
        self.port_check = self.stack.enter_context(patch.object(server, "_check_port"))
        self.alive, self.events, self.commands, self.runs = [], [], [], []
        self.max_alive, self.port_clash = 0, False
        # Per run folder: the episode statuses it writes, and its exit code.
        self.outcomes, self.codes = {}, {}

    def launcher(self, command, log_path):
        self.assertTrue(log_path.parent.is_dir())
        self.commands.append(command)
        folder = Path(option(command, "--out"))
        statuses = self.outcomes.get(folder.name, ["completed"] * int(option(command, "--episodes")))
        code = self.codes.get(folder.name, 2 if any(s != "completed" for s in statuses) else 0)
        run = FakeRun(self, command, statuses, code, polls=2)
        self.runs.append(run)
        return run

    def write_plan(self, plan):
        path = self.root / "configs" / "plan.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(plan), encoding="utf-8")
        return path

    def run_plan(self, plan):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_schedule(self.write_plan(plan), launcher=self.launcher, poll_seconds=0)

    def schedule(self, name="px"):
        return json.loads((self.root / "runs" / name / "schedule.json").read_text(encoding="utf-8"))

    def command_for(self, folder):
        return next(c for c in self.commands if Path(option(c, "--out")).name == folder)

    def test_plan_expands_chunks_and_can_interleave_jobs(self):
        _, units = parse_plan(json.dumps(make_plan()))
        self.assertEqual([u.name for u in units], ["px-q-train", "px-q-test-c1", "px-q-test-c2", "px-q-test-c3",
                                                   "px-random-test-c1", "px-random-test-c2", "px-random-test-c3"])
        self.assertEqual([(u.first_world_seed, u.episodes) for u in units[1:4]], [(20000, 2), (20002, 2), (20004, 2)])
        self.assertEqual({u.depends_on for u in units[1:4]}, {"px-q-train"})
        self.assertIsNone(units[4].depends_on)
        _, units = parse_plan(json.dumps(make_plan(interleave=True)))
        self.assertEqual([u.name for u in units], ["px-q-train", "px-q-test-c1", "px-random-test-c1", "px-q-test-c2",
                                                   "px-random-test-c2", "px-q-test-c3", "px-random-test-c3"])
        uneven = make_plan()
        uneven["jobs"][2].update(episodes=10, chunks=3)
        _, units = parse_plan(json.dumps(uneven))
        self.assertEqual([(u.first_world_seed, u.episodes) for u in units[4:]], [(20000, 4), (20004, 3), (20007, 3)])

    def test_invalid_plans_are_refused(self):
        policy = self.root / "policies" / "sarsa.json"
        policy.parent.mkdir()
        policy.write_text(json.dumps({"algorithm": "sarsa", "condition": "masked"}), encoding="utf-8")

        def job(index, **changes):
            def change(plan):
                plan["jobs"][index].update(changes)
            return change

        def drop(index, key):
            def change(plan):
                plan["jobs"][index].pop(key)
            return change

        cases = {
            "unknown plan key": lambda p: p.update(extra=1), "schema": lambda p: p.update(schema=2),
            "trailing dot": lambda p: p.update(name="px."), "slash": lambda p: p.update(name="a/b"),
            "no slots": lambda p: p.update(slots=0), "too many slots": lambda p: p.update(slots=9),
            "arena port": lambda p: p.update(first_port=25566), "ports past range": lambda p: p.update(first_port=25599),
            "interleave not bool": lambda p: p.update(interleave="yes"),
            "unknown default": lambda p: p["defaults"].update(epsilon=0.5),
            "unknown job key": job(1, epsilon=0.5), "kind": job(0, kind="test"),
            "duplicate": job(2, name="px-q-test"), "case duplicate": job(2, name="PX-Q-TEST"),
            "part suffix": job(0, name="px-q-train-part1"),
            "chunk collides": lambda p: p["jobs"].append({**p["jobs"][2], "name": "px-q-test-c1", "chunks": 1}),
            "policy_from unknown": job(1, policy_from="nothing"),
            "policy_from evaluation": lambda p: p["jobs"].append({**p["jobs"][1], "name": "late", "chunks": 1,
                                                                   "policy_from": "px-random-test"}),
            "two sources": job(2, policy_from="px-q-train"), "no source": job(2, random_baseline=False),
            "baseline without condition": drop(2, "condition"),
            "history baseline": job(2, condition="unmasked_history"), "random_baseline not bool": job(2, random_baseline=1),
            "chunks above episodes": job(1, chunks=7), "zero episodes": job(1, episodes=0),
            "missing max_steps": lambda p: p["defaults"].pop("max_steps"), "pause": lambda p: p["defaults"].update(pause=11),
            "training without condition": drop(0, "condition"), "epsilon order": job(0, epsilon_end=0.7),
            "algorithm": job(0, algorithm="dqn"), "seed": job(0, first_world_seed=True),
            "missing policy file": job(1, policy_from=None, policy="policies/missing.json"),
            "policy condition": job(1, policy_from=None, policy=str(policy), condition="unmasked"),
            "policy algorithm": job(0, policy=str(policy), condition=None),
            "pretrain with policy": job(0, policy=str(policy), condition=None, algorithm="sarsa", pretrain=10),
        }
        for label, change in cases.items():
            plan = make_plan()
            change(plan)
            plan["jobs"] = [{k: v for k, v in j.items() if v is not None} for j in plan["jobs"]]
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    parse_plan(json.dumps(plan))
        # The matching policy is accepted, and its path is stored resolved.
        plan = make_plan()
        plan["jobs"][0].update(policy="policies/sarsa.json", algorithm="sarsa", condition="masked")
        _, units = parse_plan(json.dumps(plan))
        self.assertEqual(units[0].job["policy"], str(policy))

    def test_continuation_epsilon_resumes_the_declared_linear_schedule(self):
        for k in range(1, 51):
            self.assertAlmostEqual(epsilon_at(0.6, 0.1, k, 50), 0.6 - 0.5 * (k - 1) / 49)
        self.assertEqual((epsilon_at(0.6, 0.1, 1, 50), epsilon_at(0.6, 0.1, 50, 50), epsilon_at(0.6, 0.1, 1, 1)),
                         (0.6, 0.1, 0.6))
        self.assertTrue(all(0.1 <= epsilon_at(0.6, 0.1, k, 7) <= 0.6 for k in range(1, 8)))

    def test_runs_respect_slots_distinct_ports_and_training_dependencies(self):
        plan_path = self.write_plan(make_plan())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_schedule(plan_path, launcher=self.launcher, poll_seconds=0), 0)
        self.assertEqual(self.max_alive, 2)
        self.assertFalse(self.port_clash)
        self.assertEqual(self.port_check.call_args_list, [unittest.mock.call(25567), unittest.mock.call(25568)])
        trained = self.events.index(("exit", "px-q-train"))
        for chunk in ("px-q-test-c1", "px-q-test-c2", "px-q-test-c3"):
            self.assertGreater(self.events.index(("start", chunk)), trained)
            command = self.command_for(chunk)
            self.assertEqual(Path(option(command, "--policy")), self.root / "runs" / "px" / "px-q-train" / "policy.json")
            self.assertEqual(option(command, "--max-steps"), "60")
        # A baseline needs no training, so it starts while training is still running.
        self.assertLess(self.events.index(("start", "px-random-test-c1")), trained)
        for command in self.commands:
            self.assertEqual(option(command, "--parallel-slots"), "2")
            self.assertIn(option(command, "--port"), ("25567", "25568"))
            self.assertEqual((option(command, "--seed"), option(command, "--pause")), ("0", "0.0"))
        self.assertEqual([option(self.command_for(f"px-random-test-c{i}"), "--world-seed") for i in (1, 2, 3)],
                         ["20000", "20002", "20004"])
        train = self.command_for("px-q-train")
        self.assertEqual((option(train, "--condition"), option(train, "--epsilon"), option(train, "--epsilon-end")),
                         ("unmasked", "0.6", "0.1"))
        record = self.schedule()
        self.assertEqual(record["status"], "completed")
        self.assertEqual({u["status"] for u in record["units"]}, {"done"})
        self.assertEqual(record["max_running_observed"], 2)
        self.assertEqual((self.root / "runs" / "px" / "plan.json").read_bytes(), plan_path.read_bytes())
        self.assertFalse(record["code_changed_during_schedule"])
        self.assertTrue(self.runs and all(run.closed for run in self.runs))

    def test_evaluations_can_wait_until_all_training_has_ended(self):
        self.assertEqual(self.run_plan(make_plan(evaluate_after_all_training=True)), 0)
        trained = self.events.index(("exit", "px-q-train"))
        self.assertGreater(self.events.index(("start", "px-random-test-c1")), trained)

    def test_invalid_episode_is_continued_from_the_next_seed_with_the_schedule(self):
        self.outcomes = {"px-q-train": ["completed", "invalid"], "px-random-test-c2": ["invalid"]}
        self.assertEqual(self.run_plan(make_plan()), 0)
        part = self.command_for("px-q-train-part1")
        self.assertEqual((option(part, "--episodes"), option(part, "--world-seed")), ("2", "102"))
        self.assertEqual(Path(option(part, "--policy")), self.root / "runs" / "px" / "px-q-train" / "policy.json")
        self.assertEqual(option(part, "--epsilon"), repr(epsilon_at(0.6, 0.1, 3, 4)))
        self.assertNotIn("--pretrain", part)
        # Evaluation uses the last training part's policy.
        test = self.command_for("px-q-test-c1")
        self.assertEqual(Path(option(test, "--policy")), self.root / "runs" / "px" / "px-q-train-part1" / "policy.json")
        baseline = self.command_for("px-random-test-c2-part1")
        self.assertEqual((option(baseline, "--episodes"), option(baseline, "--world-seed")), ("1", "20003"))
        units = {u["name"]: u for u in self.schedule()["units"]}
        self.assertEqual([p["folder"] for p in units["px-q-train"]["parts"]], ["px-q-train", "px-q-train-part1"])
        self.assertEqual(units["px-q-train"]["attempted"], 4)
        self.assertEqual(units["px-q-train"]["parts"][1]["attempted_before"], 2)
        # The invalid episode stays on disk and is never replaced.
        invalid = self.root / "runs" / "px" / "px-q-train" / "episode-0002" / "outcome.json"
        self.assertEqual(json.loads(invalid.read_text())["status"], "invalid")

    def test_a_run_without_progress_fails_and_blocks_its_evaluations(self):
        self.outcomes, self.codes = {"px-q-train": []}, {"px-q-train": 2}
        self.assertEqual(self.run_plan(make_plan()), 1)
        units = {u["name"]: u["status"] for u in self.schedule()["units"]}
        self.assertEqual(units["px-q-train"], "failed")
        self.assertEqual({units[f"px-q-test-c{i}"] for i in (1, 2, 3)}, {"blocked"})
        self.assertEqual({units[f"px-random-test-c{i}"] for i in (1, 2, 3)}, {"done"})
        self.assertEqual(self.schedule()["status"], "finished_with_failures")
        self.assertFalse(any("px-q-test" in option(c, "--out") for c in self.commands))

    def test_continuations_are_capped(self):
        self.outcomes = {"px-q-train": ["invalid"], "px-q-train-part1": ["invalid"]}
        self.assertEqual(self.run_plan(make_plan(max_continuations=1)), 1)
        units = {u["name"]: u for u in self.schedule()["units"]}
        self.assertEqual(units["px-q-train"]["status"], "gave_up")
        self.assertEqual(len(units["px-q-train"]["parts"]), 2)
        self.assertEqual(units["px-q-test-c1"]["status"], "blocked")

    def test_refuses_existing_folder_uncommitted_code_busy_port_and_a_second_schedule(self):
        plan_path = self.write_plan(make_plan())
        (self.root / "runs" / "px").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            run_schedule(plan_path, launcher=self.launcher, poll_seconds=0)
        (self.root / "runs" / "px").rmdir()
        self.code.return_value = {"head": "abc123", "uncommitted_code": [" M infrastructure/x.py"], "plan_committed": False}
        with self.assertRaisesRegex(RuntimeError, "Commit the code"):
            run_schedule(plan_path, launcher=self.launcher, poll_seconds=0)
        self.code.return_value = {"head": "abc123", "uncommitted_code": [], "plan_committed": True}
        self.port_check.side_effect = server.ServerError("Local port 25568 is already in use")
        with self.assertRaisesRegex(server.ServerError, "25568"):
            run_schedule(plan_path, launcher=self.launcher, poll_seconds=0)
        self.port_check.side_effect = None
        with parallel_runs._machine_lock():
            with self.assertRaisesRegex(RuntimeError, "Another resource-parallel schedule"):
                run_schedule(plan_path, launcher=self.launcher, poll_seconds=0)
        self.assertEqual(self.commands, [])
        self.assertFalse((self.root / "runs" / "px").exists())

    def test_interrupt_starts_nothing_new_and_waits_for_running_runs(self):
        self.outcomes = {"px-q-train": ["interrupted"], "px-random-test-c1": ["interrupted"]}
        self.codes = {"px-q-train": 130, "px-random-test-c1": 130}
        calls = []

        def sleep(_seconds):
            calls.append(1)
            if len(calls) == 1:
                raise KeyboardInterrupt

        with patch.object(parallel_runs.time, "sleep", side_effect=sleep):
            self.assertEqual(self.run_plan(make_plan()), 130)
        self.assertEqual(len(self.commands), 2)
        units = {u["name"]: u["status"] for u in self.schedule()["units"]}
        self.assertEqual((units["px-q-train"], units["px-random-test-c1"]), ("interrupted", "interrupted"))
        self.assertEqual(units["px-random-test-c2"], "not_started")
        self.assertIn(units["px-q-test-c1"], ("blocked", "not_started"))
        self.assertEqual(self.schedule()["status"], "interrupted")
        self.assertEqual(self.alive, [])

    def test_dry_run_prints_the_runs_and_creates_nothing(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(run_schedule(self.write_plan(make_plan()), dry_run=True, launcher=self.launcher), 0)
        text = output.getvalue()
        for expected in ("px-q-test-c3", "seeds 20004-20005", "--port <slot port>", "<final policy.json of px-q-train>",
                         "--random-baseline --condition unmasked"):
            self.assertIn(expected, text)
        self.assertFalse((self.root / "runs").exists())
        self.code.assert_not_called()
        self.port_check.assert_not_called()

    def test_command_line_dispatches_the_plan(self):
        from infrastructure.__main__ import main
        with patch.object(parallel_runs, "run_schedule", return_value=0) as run:
            self.assertEqual(main(["resource-parallel", "configs/plan.json", "--dry-run"]), 0)
        self.assertEqual(run.call_args.kwargs, {"dry_run": True})
        self.assertEqual(run.call_args.args[0], Path("configs/plan.json"))


if __name__ == "__main__":
    unittest.main()

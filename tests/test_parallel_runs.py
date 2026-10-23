"""The parallel scheduler must keep runs apart, continue only as declared, and never reuse a folder."""

import contextlib
import io
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import call, patch

from infrastructure import parallel_runs, resource_server, server
from infrastructure.parallel_runs import START_PORTS, epsilon_at, parse_plan, run_schedule, schedule_run_folders

CLEAN = {"head": "abc123", "code_tree": "040000 tree t1\tinfrastructure", "uncommitted_code": [], "plan_committed": True}


def make_plan(**overrides):
    plan = {"schema": 1, "name": "px", "slots": 2, "interleave": True, "evaluate_after_all_training": True,
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
        self.pid = 4000 + len(owner.runs)
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
        self.stack.enter_context(patch.object(resource_server, "LOCK_PATH", self.root / "lock" / "parallel.lock"))
        self.code = self.stack.enter_context(patch.object(parallel_runs, "code_state", return_value=dict(CLEAN)))
        self.port_check = self.stack.enter_context(patch.object(server, "_check_port"))
        self.alive, self.events, self.commands, self.runs = [], [], [], []
        self.max_alive, self.port_clash = 0, False
        # Per run folder: the episode statuses it writes, and its exit code.
        self.outcomes, self.codes = {}, {}
        self.on_launch = None

    def launcher(self, command, log_path):
        self.assertTrue(log_path.parent.is_dir())
        self.commands.append(command)
        folder = Path(option(command, "--out"))
        statuses = self.outcomes.get(folder.name, ["completed"] * int(option(command, "--episodes")))
        code = self.codes.get(folder.name, 2 if any(s != "completed" for s in statuses) else 0)
        run = FakeRun(self, command, statuses, code, polls=2)
        self.runs.append(run)
        if self.on_launch:
            self.on_launch(len(self.commands))
        return run

    def write_plan(self, plan):
        path = self.root / "configs" / "plan.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(plan), encoding="utf-8")
        return path

    def run_plan(self, plan, output=None):
        with contextlib.redirect_stdout(output or io.StringIO()):
            return run_schedule(self.write_plan(plan), launcher=self.launcher, poll_seconds=0)

    def schedule(self, name="px"):
        return json.loads((self.root / "runs" / name / "schedule.json").read_text(encoding="utf-8"))

    def units(self):
        return {u["name"]: u for u in self.schedule()["units"]}

    def command_for(self, folder):
        return next(c for c in self.commands if Path(option(c, "--out")).name == folder)

    def test_plan_expands_chunks_with_their_own_agent_seeds_and_can_interleave(self):
        alone = {"slots": 1, "interleave": False, "evaluate_after_all_training": False}
        _, units = parse_plan(json.dumps(make_plan(**alone)))
        self.assertEqual([u.name for u in units], ["px-q-train", "px-q-test-c1", "px-q-test-c2", "px-q-test-c3",
                                                   "px-random-test-c1", "px-random-test-c2", "px-random-test-c3"])
        self.assertEqual([(u.first_world_seed, u.episodes) for u in units[1:4]], [(20000, 2), (20002, 2), (20004, 2)])
        # No two chunks start from one random stream.
        self.assertEqual([u.agent_seed for u in units], [0, 0, 1, 2, 0, 1, 2])
        self.assertEqual({u.depends_on for u in units[1:4]}, {"px-q-train"})
        self.assertIsNone(units[4].depends_on)
        _, units = parse_plan(json.dumps(make_plan()))
        self.assertEqual([u.name for u in units], ["px-q-train", "px-q-test-c1", "px-random-test-c1", "px-q-test-c2",
                                                   "px-random-test-c2", "px-q-test-c3", "px-random-test-c3"])
        uneven = make_plan(**alone)
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

        def extra(**fields):
            def change(plan):
                plan["jobs"].append({**plan["jobs"][2], "chunks": 1, **fields})
            return change

        cases = {
            "unknown plan key": lambda p: p.update(extra=1), "schema": lambda p: p.update(schema=2),
            "trailing dot": lambda p: p.update(name="px."), "slash": lambda p: p.update(name="a/b"),
            "plan part suffix": lambda p: p.update(name="px-part2"),
            "no slots": lambda p: p.update(slots=0), "too many slots": lambda p: p.update(slots=9),
            "arena port": lambda p: p.update(first_port=25566), "ports past range": lambda p: p.update(first_port=25599),
            "interleave not bool": lambda p: p.update(interleave="yes"),
            "unknown default": lambda p: p["defaults"].update(epsilon=0.5),
            "unknown job key": job(1, epsilon=0.5), "kind": job(0, kind="test"),
            "duplicate": job(2, name="px-q-test"), "case duplicate": job(2, name="PX-Q-TEST"),
            "part suffix": job(0, name="px-q-train-part1"),
            "chunk collides": extra(name="px-q-test-c1"),
            "named like the plan": extra(name="PX"), "named logs": extra(name="logs"),
            "named like the record": extra(name="Schedule.json"), "named like the plan file": extra(name="plan.json"),
            "named like the fallback record": extra(name="schedule.final.json"),
            "chunk names leave no room for continuations": job(1, name="x" * 56),
            "training name leaves no room for continuations": lambda p: (p["jobs"][0].update(name="t" * 59),
                                                                         p["jobs"][1].update(policy_from="t" * 59)),
            "parallel evaluations without interleaving": lambda p: p.update(interleave=False),
            "parallel evaluations beside training": lambda p: p.update(evaluate_after_all_training=False),
            "policy_from unknown": job(1, policy_from="nothing"),
            "chunk names past 64 characters": job(1, name="y" * 63),
            "policy_from evaluation": extra(name="late", random_baseline=None, condition=None,
                                            policy_from="px-random-test"),
            "two sources": job(2, policy_from="px-q-train"), "no source": job(2, random_baseline=False),
            "baseline without condition": drop(2, "condition"),
            "history baseline": job(2, condition="unmasked_history"), "random_baseline not bool": job(2, random_baseline=1),
            "chunks above episodes": job(1, chunks=7), "zero episodes": job(1, episodes=0),
            "missing max_steps": lambda p: p["defaults"].pop("max_steps"), "pause": lambda p: p["defaults"].update(pause=11),
            "missing agent_seed": lambda p: p["defaults"].pop("agent_seed"),
            "chunk seeds overflow": job(1, agent_seed=2**63 - 2),
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
        # The machine must be free at the start; then each slot's port is checked again before every launch.
        self.assertEqual(self.port_check.call_args_list[:len(START_PORTS)], [call(p) for p in START_PORTS])
        self.assertEqual(self.port_check.call_args_list[len(START_PORTS):], [call(int(option(c, "--port")))
                                                                             for c in self.commands])
        # Every evaluation, the random baseline included, waits until all training has ended.
        trained = self.events.index(("exit", "px-q-train"))
        for chunk in ("px-q-test-c1", "px-q-test-c2", "px-q-test-c3", "px-random-test-c1"):
            self.assertGreater(self.events.index(("start", chunk)), trained)
        for chunk in ("px-q-test-c1", "px-q-test-c2", "px-q-test-c3"):
            command = self.command_for(chunk)
            self.assertEqual(Path(option(command, "--policy")), self.root / "runs" / "px" / "px-q-train" / "policy.json")
            self.assertEqual(option(command, "--max-steps"), "60")
        # Interleaved: the arms alternate.
        starts = [name for kind, name in self.events if kind == "start"]
        self.assertEqual(starts, ["px-q-train", "px-q-test-c1", "px-random-test-c1", "px-q-test-c2",
                                  "px-random-test-c2", "px-q-test-c3", "px-random-test-c3"])
        record = self.schedule()
        for command in self.commands:
            self.assertEqual(option(command, "--parallel-slots"), "2")
            self.assertIn(option(command, "--port"), ("25568", "25569"))
            self.assertEqual(option(command, "--pause"), "0.0")
            self.assertEqual(option(command, "--schedule-id"), record["schedule_id"])
        self.assertTrue(record["schedule_id"].startswith("px:"))
        self.assertEqual([option(self.command_for(f"px-random-test-c{i}"), "--world-seed") for i in (1, 2, 3)],
                         ["20000", "20002", "20004"])
        self.assertEqual([option(self.command_for(f"px-random-test-c{i}"), "--seed") for i in (1, 2, 3)], ["0", "1", "2"])
        train = self.command_for("px-q-train")
        self.assertEqual((option(train, "--condition"), option(train, "--epsilon"), option(train, "--epsilon-end"),
                          option(train, "--seed")), ("unmasked", "0.6", "0.1", "0"))
        self.assertEqual(record["status"], "completed")
        self.assertEqual({u["status"] for u in record["units"]}, {"done"})
        self.assertEqual(record["max_running_observed"], 2)
        parts = {p["folder"]: p for u in record["units"] for p in u["parts"]}
        self.assertEqual((parts["px-q-train"]["max_running"], parts["px-q-train"]["pid"]), (1, 4000))
        self.assertEqual((parts["px-q-test-c1"]["running_at_launch"], parts["px-q-test-c1"]["max_running"]), (1, 2))
        self.assertEqual((self.root / "runs" / "px" / "plan.json").read_bytes(), plan_path.read_bytes())
        self.assertFalse(record["code_changed_during_schedule"])
        self.assertEqual(record["code_checks"], len(self.commands))
        self.assertTrue(self.runs and all(run.closed for run in self.runs))
        found, missing = schedule_run_folders(self.root / "runs" / "px")
        self.assertEqual((len(found), missing), (7, []))

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
        self.assertEqual((option(baseline, "--episodes"), option(baseline, "--world-seed"), option(baseline, "--seed")),
                         ("1", "20003", "1"))
        units = self.units()
        self.assertEqual([p["folder"] for p in units["px-q-train"]["parts"]], ["px-q-train", "px-q-train-part1"])
        self.assertEqual(units["px-q-train"]["attempted"], 4)
        self.assertEqual(units["px-q-train"]["parts"][1]["attempted_before"], 2)
        # The invalid episode stays on disk and is never replaced.
        invalid = self.root / "runs" / "px" / "px-q-train" / "episode-0002" / "outcome.json"
        self.assertEqual(json.loads(invalid.read_text())["status"], "invalid")
        found, _ = schedule_run_folders(self.root / "runs" / "px")
        self.assertIn(self.root / "runs" / "px" / "px-random-test-c2-part1", found)

    def test_an_invalid_final_episode_finishes_the_run_without_an_empty_continuation(self):
        self.outcomes = {"px-q-train": ["completed", "completed", "completed", "invalid"],
                         "px-random-test-c3": ["completed", "invalid"]}
        self.assertEqual(self.run_plan(make_plan()), 0)
        folders = [Path(option(c, "--out")).name for c in self.commands]
        self.assertFalse(any("part" in name for name in folders), folders)
        units = self.units()
        self.assertEqual((units["px-q-train"]["status"], units["px-random-test-c3"]["status"]), ("done", "done"))
        self.assertEqual(Path(option(self.command_for("px-q-test-c1"), "--policy")),
                         self.root / "runs" / "px" / "px-q-train" / "policy.json")
        self.assertTrue(all(int(option(c, "--episodes")) >= 1 for c in self.commands))

    def test_a_run_without_progress_fails_and_blocks_its_evaluations(self):
        self.outcomes, self.codes = {"px-q-train": []}, {"px-q-train": 2}
        self.assertEqual(self.run_plan(make_plan()), 1)
        units = {name: u["status"] for name, u in self.units().items()}
        self.assertEqual(units["px-q-train"], "failed")
        self.assertEqual({units[f"px-q-test-c{i}"] for i in (1, 2, 3)}, {"blocked"})
        self.assertEqual({units[f"px-random-test-c{i}"] for i in (1, 2, 3)}, {"done"})
        self.assertEqual(self.schedule()["status"], "finished_with_failures")
        self.assertFalse(any("px-q-test" in option(c, "--out") for c in self.commands))

    def test_continuations_are_capped(self):
        self.outcomes = {"px-q-train": ["invalid"], "px-q-train-part1": ["invalid"]}
        self.assertEqual(self.run_plan(make_plan(max_continuations=1)), 1)
        units = self.units()
        self.assertEqual(units["px-q-train"]["status"], "gave_up")
        self.assertEqual(len(units["px-q-train"]["parts"]), 2)
        self.assertEqual(units["px-q-test-c1"]["status"], "blocked")

    def test_a_port_taken_during_the_schedule_retires_its_slot_and_stops_nothing(self):
        def check(port):
            if port == 25568 and len(self.port_check.call_args_list) > len(START_PORTS):
                raise server.ServerError("Local port 25568 is already in use")
        self.port_check.side_effect = check
        self.assertEqual(self.run_plan(make_plan()), 0)
        self.assertEqual({option(c, "--port") for c in self.commands}, {"25569"})
        self.assertEqual(self.max_alive, 1)
        record = self.schedule()
        self.assertEqual([r["port"] for r in record["retired_ports"]], [25568])
        self.assertEqual(record["status"], "completed")

    def test_a_code_change_during_the_schedule_stops_new_launches(self):
        changed = {**CLEAN, "code_tree": "040000 tree t2\tinfrastructure"}
        # The start check and the first two launches see the committed code; later checks see a change.
        self.code.side_effect = lambda _plan: dict(CLEAN) if self.code.call_count <= 3 else dict(changed)
        self.assertEqual(self.run_plan(make_plan()), 1)
        self.assertEqual(len(self.commands), 2)
        record = self.schedule()
        self.assertEqual(record["status"], "code_changed")
        self.assertTrue(record["code_changed_during_schedule"])
        self.assertEqual(record["code_change"]["state"]["code_tree"], changed["code_tree"])
        units = {name: u["status"] for name, u in self.units().items()}
        self.assertEqual((units["px-q-train"], units["px-q-test-c1"]), ("done", "done"))
        self.assertEqual(units["px-random-test-c1"], "not_started")

    def test_the_code_is_also_checked_while_runs_are_going(self):
        changed = {**CLEAN, "uncommitted_code": [" M minecraft_bridge/resource-bridge.js"]}
        self.code.side_effect = lambda _plan: dict(CLEAN) if self.code.call_count <= 2 else dict(changed)
        with patch.object(parallel_runs, "CODE_CHECK_SECONDS", 0):
            self.assertEqual(self.run_plan(make_plan()), 1)
        self.assertEqual(len(self.commands), 1)
        record = self.schedule()
        self.assertEqual(record["status"], "code_changed")
        units = {name: u["status"] for name, u in self.units().items()}
        self.assertEqual((units["px-q-train"], units["px-q-test-c1"]), ("done", "not_started"))

    def test_refuses_existing_folder_uncommitted_code_busy_ports_and_a_live_run(self):
        plan_path = self.write_plan(make_plan())
        (self.root / "runs" / "px").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            run_schedule(plan_path, launcher=self.launcher, poll_seconds=0)
        (self.root / "runs" / "px").rmdir()
        for state in ({**CLEAN, "uncommitted_code": [" M infrastructure/x.py"]}, {**CLEAN, "plan_committed": False}):
            self.code.return_value = state
            with self.subTest(state=state), self.assertRaisesRegex(RuntimeError, "Commit the run code and the plan"):
                run_schedule(plan_path, launcher=self.launcher, poll_seconds=0)
        self.code.return_value = dict(CLEAN)
        # Any Minecraft port in use, not only the plan's own, means the PC is not free.
        for busy_port in (25566, 25567, 25569, 25599):
            def busy(port, busy_port=busy_port):
                if port == busy_port:
                    raise server.ServerError(f"Local port {port} is already in use")
            self.port_check.reset_mock()
            self.port_check.side_effect = busy
            with self.subTest(port=busy_port), self.assertRaisesRegex(server.ServerError, f"to itself.*{busy_port}"):
                run_schedule(plan_path, launcher=self.launcher, poll_seconds=0)
            self.assertEqual(self.port_check.call_args_list[-1], call(busy_port))
        self.port_check.side_effect = None
        with resource_server.live_run_lock("run by hand (pid 1)"):
            self.assertEqual(resource_server.current_owner(), "run by hand (pid 1)")
            with self.assertRaisesRegex(RuntimeError, r"already going on this machine \(run by hand"):
                run_schedule(plan_path, launcher=self.launcher, poll_seconds=0)
        self.assertIsNone(resource_server.current_owner())
        self.assertEqual(self.commands, [])
        self.assertFalse((self.root / "runs" / "px").exists())

    def test_interrupt_starts_nothing_new_and_waits_for_running_runs(self):
        self.outcomes, self.codes = {"px-q-train": ["interrupted"]}, {"px-q-train": 130}
        calls = []

        def sleep(_seconds):
            calls.append(1)
            if len(calls) in (1, 2):
                raise KeyboardInterrupt

        output = io.StringIO()
        with patch.object(parallel_runs.time, "sleep", side_effect=sleep):
            self.assertEqual(self.run_plan(make_plan(), output), 130)
        # The second Ctrl+C only warns; the scheduler keeps waiting until the run has exited.
        self.assertIn("Do not close this window", output.getvalue())
        self.assertIn("possibly not reached by that Ctrl+C: px-q-train", output.getvalue())
        self.assertEqual(len(self.commands), 1)
        units = {name: u["status"] for name, u in self.units().items()}
        self.assertEqual(units["px-q-train"], "interrupted")
        self.assertEqual({units[f"px-random-test-c{i}"] for i in (1, 2, 3)}, {"not_started"})
        self.assertEqual(self.schedule()["status"], "interrupted")
        self.assertEqual(self.alive, [])
        self.assertTrue(all(p["finished_at"] for u in self.schedule()["units"] for p in u["parts"]))

    def test_ctrl_c_just_before_a_launch_launches_nothing(self):
        real, calls = parallel_runs._deferred_interrupt, []

        @contextlib.contextmanager
        def held():
            calls.append(1)
            if len(calls) == 1:
                yield [True]
                return
            with real() as received:
                yield received

        with patch.object(parallel_runs, "_deferred_interrupt", held):
            self.assertEqual(self.run_plan(make_plan()), 130)
        self.assertEqual(self.commands, [])
        record = self.schedule()
        self.assertEqual({u["status"] for u in record["units"]}, {"not_started"})
        self.assertEqual([u["parts"] for u in record["units"]], [[]] * 7)

    def test_a_continuation_that_cannot_start_is_not_called_not_started(self):
        self.outcomes = {"px-q-train": ["completed", "invalid"]}
        changed = {**CLEAN, "code_tree": "040000 tree t2\tinfrastructure"}
        self.code.side_effect = lambda _plan: dict(CLEAN) if self.code.call_count <= 2 else dict(changed)
        output = io.StringIO()
        self.assertEqual(self.run_plan(make_plan(), output), 1)
        self.assertEqual(self.units()["px-q-train"]["status"], "continuation_not_started")
        self.assertEqual(self.units()["px-q-test-c1"]["status"], "not_started")
        self.assertIn("continuation_not_started", output.getvalue())

    def test_ctrl_c_during_a_launch_still_tracks_and_waits_for_that_run(self):
        self.outcomes, self.codes = {"px-q-test-c1": ["interrupted"]}, {"px-q-test-c1": 130}
        self.on_launch = lambda count: signal.raise_signal(signal.SIGINT) if count == 2 else None
        self.assertEqual(self.run_plan(make_plan()), 130)
        self.assertEqual(len(self.commands), 2)
        units = {name: u["status"] for name, u in self.units().items()}
        self.assertEqual((units["px-q-train"], units["px-q-test-c1"]), ("done", "interrupted"))
        self.assertEqual(units["px-random-test-c1"], "not_started")
        self.assertEqual(self.alive, [])
        self.assertIs(signal.getsignal(signal.SIGINT), signal.default_int_handler)

    def test_ctrl_c_during_the_closing_code_check_still_writes_the_final_record(self):
        def state(_plan):
            # Only the closing check runs once every run has been launched and has exited.
            if len(self.commands) == 7 and not self.alive:
                raise KeyboardInterrupt
            return dict(CLEAN)

        self.code.side_effect = state
        self.assertEqual(self.run_plan(make_plan()), 130)
        record = self.schedule()
        self.assertEqual(record["status"], "interrupted")
        self.assertTrue(record["finished_at"])
        self.assertIn("Ctrl+C", record["code_at_end"]["error"])
        self.assertEqual({u["status"] for u in record["units"]}, {"done"})

    def test_a_ctrl_c_held_while_closing_marks_the_schedule_interrupted(self):
        real_close = parallel_runs._Schedule._close

        def close(schedule, interrupted, error, pressed):
            pressed.append(True)  # a press that arrives while the closing record is being written
            return real_close(schedule, interrupted, error, pressed)

        with patch.object(parallel_runs._Schedule, "_close", close), self.assertRaises(KeyboardInterrupt):
            self.run_plan(make_plan())
        record = self.schedule()
        self.assertEqual((record["status"], bool(record["finished_at"])), ("interrupted", True))
        self.assertIs(signal.getsignal(signal.SIGINT), signal.default_int_handler)

    def test_git_runs_in_its_own_process_group(self):
        done = subprocess.CompletedProcess(["git"], 0, stdout="abc123\n", stderr="")
        with patch.object(parallel_runs.subprocess, "run", return_value=done) as run:
            self.assertEqual(parallel_runs._git("rev-parse", "HEAD"), "abc123")
        self.assertEqual(run.call_args.kwargs["creationflags"], getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        stopped = subprocess.CompletedProcess(["git"], 3221225786, stdout="", stderr="")
        with patch.object(parallel_runs.subprocess, "run", return_value=stopped), self.assertRaises(KeyboardInterrupt):
            parallel_runs._git("rev-parse", "HEAD")

    def test_the_fallback_final_record_is_the_one_read(self):
        folder = self.root / "runs" / "px"
        for name in ("px-a", "px-b"):
            (folder / name).mkdir(parents=True)
            (folder / name / "metadata.json").write_text("{}", encoding="utf-8")

        def record(*names, status):
            return json.dumps({"status": status, "units": [{"name": "px-a", "parts": [{"folder": n} for n in names]}]})

        (folder / "schedule.json").write_text(record("px-a", status="running"), encoding="utf-8")
        self.assertEqual(parallel_runs.schedule_record_path(folder), folder / "schedule.json")
        self.assertEqual(schedule_run_folders(folder), ([folder / "px-a"], []))
        # Written only by the final save when schedule.json was held, so it is the newest record.
        (folder / "schedule.final.json").write_text(record("px-a", "px-b", status="completed"), encoding="utf-8")
        self.assertEqual(parallel_runs.schedule_record_path(folder), folder / "schedule.final.json")
        self.assertEqual(schedule_run_folders(folder), ([folder / "px-a", folder / "px-b"], []))

    def test_an_unexpected_error_waits_for_the_running_runs_before_it_is_raised(self):
        def fail(count):
            if count == 3:
                raise RuntimeError("launcher broke")
        self.on_launch = fail
        with self.assertRaisesRegex(RuntimeError, "launcher broke"):
            self.run_plan(make_plan())
        record = self.schedule()
        self.assertEqual(record["status"], "error")
        self.assertIn("launcher broke", record["error"])
        units = self.units()
        # The run already going was waited for and recorded.
        self.assertEqual((units["px-q-train"]["status"], units["px-q-test-c1"]["status"]), ("done", "done"))
        # Launched but never seen to exit: recorded as interrupted with an unknown state, never as not started.
        self.assertEqual(units["px-random-test-c1"]["status"], "interrupted")
        self.assertIn("state unknown", units["px-random-test-c1"]["parts"][0]["last_status"])
        self.assertEqual(self.alive, [self.runs[2]])

    def test_dry_run_prints_the_runs_and_creates_nothing(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(run_schedule(self.write_plan(make_plan()), dry_run=True, launcher=self.launcher), 0)
        text = output.getvalue()
        for expected in ("px-q-test-c3", "seeds 20004-20005, agent seed 2", "--port <slot port>",
                         "<final policy.json of px-q-train>", "--random-baseline --condition unmasked",
                         "ports 25568-25569"):
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

    def test_committed_plans_parse(self):
        configs = Path(__file__).resolve().parents[1] / "configs"
        names = set()
        for name in ("parallel_benchmark_1slot", "parallel_benchmark_2slots", "parallel_benchmark_4slots",
                     "parallel_example_plan"):
            plan, units = parse_plan((configs / f"{name}.json").read_text(encoding="utf-8"))
            names |= {unit.name for unit in units}
            if name.startswith("parallel_benchmark"):
                self.assertEqual([(u.first_world_seed, u.episodes, u.agent_seed) for u in units],
                                 [(30000, 5, 0), (30005, 5, 1), (30010, 5, 2), (30015, 5, 3)])
        # The benchmark runs can be reported together: no two share a folder name.
        # Three benchmark plans of four chunks, and the example's two training runs and three four-chunk tests.
        self.assertEqual(len(names), 3 * 4 + 2 + 3 * 4)


if __name__ == "__main__":
    unittest.main()

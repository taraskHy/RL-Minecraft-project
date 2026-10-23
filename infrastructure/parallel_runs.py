"""Run several live natural-resource jobs at once, each on its own loopback port.

A plan file (JSON, normally committed under ``configs/``) lists training and
evaluation jobs. Every job runs through the same ``resource-train`` or
``resource-demo`` command as a run started by hand, in a new folder under
``runs/<plan name>/``, with ``--port`` and ``--parallel-slots`` added. At most
``slots`` runs execute at once, each on a different port.

The scheduler refuses uncommitted code and a busy port (it never stops a
process it did not start), never reuses, replaces, or deletes a folder, and
never kills a run: on Ctrl+C it starts nothing new and waits while every
running run stops its own server and saves its world. A run cut short by an
invalid episode is continued as the masking experiment declared: in a new
``-partN`` folder from the next world seed, with the invalid episode still
counted as attempted. ``schedule.json`` records when each run started and
stopped, on which port, and how many runs were going at the time.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable

from minecraft_rl.resources import CONDITIONS
from . import server as managed_server
from .resource_server import MAX_PARALLEL_SLOTS, PORTS

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
# No trailing dot: Windows silently drops it from folder names.
NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,62}[A-Za-z0-9_-])?$")
# The code a live run reads; it must be committed before a schedule starts.
CODE_PATHS = ("minecraft_bridge", "infrastructure", "minecraft_rl", "configs", "scripts")
PLAN_KEYS = {"schema", "name", "description", "slots", "first_port", "max_continuations", "interleave",
             "evaluate_after_all_training", "defaults", "jobs"}
SHARED_KEYS = {"agent_seed", "max_steps", "pause"}
JOB_KEYS = {
    "train": {"name", "kind", "condition", "algorithm", "policy", "episodes", "first_world_seed",
              "epsilon", "epsilon_end", "pretrain"} | SHARED_KEYS,
    "evaluate": {"name", "kind", "condition", "policy", "policy_from", "random_baseline", "episodes",
                 "first_world_seed", "chunks"} | SHARED_KEYS,
}
# A run that ended any of these ways leaves its dependents unstarted.
UNFINISHED = ("failed", "gave_up", "blocked", "interrupted", "not_started")


@dataclass
class Unit:
    """One run folder and its continuations: a training job, or one seed chunk of an evaluation job."""
    name: str
    job: dict
    episodes: int
    first_world_seed: int
    depends_on: str | None = None
    status: str = "waiting"
    parts: list[dict] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _integer(value, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{label} must be an integer from {low} to {high}")
    return value


def _number(value, label: str, low: float, high: float) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not low <= value <= high):
        raise ValueError(f"{label} must be a number from {low:g} to {high:g}")
    return float(value)


def _flag(value, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false")
    return value


def _split(episodes: int, chunks: int) -> list[int]:
    size, extra = divmod(episodes, chunks)
    return [size + (index < extra) for index in range(chunks)]


def epsilon_at(start: float, end: float, episode: int, total: int) -> float:
    """The declared linear schedule's exploration rate for episode ``episode`` of ``total``."""
    value = start + (end - start) * (episode - 1) / max(1, total - 1)
    return min(start, max(end, value))


def _policy_path(value, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: policy must be a file path")
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_file():
        raise ValueError(f"{label}: policy file not found: {path}")
    return str(path)


def _saved_policy(path: str, label: str) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label}: cannot read the policy {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: {path} is not a resource policy")
    return payload


def parse_plan(text: str) -> tuple[dict, list[Unit]]:
    """Validate a plan and expand it into run units; nothing is touched."""
    plan = json.loads(text)
    if not isinstance(plan, dict):
        raise ValueError("A plan must be a JSON object")
    unknown = sorted(set(plan) - PLAN_KEYS)
    if unknown:
        raise ValueError(f"Unknown plan keys: {', '.join(unknown)}")
    if plan.get("schema") != SCHEMA:
        raise ValueError(f"Plan schema must be {SCHEMA}")
    if not isinstance(plan.get("name"), str) or not NAME.match(plan["name"]):
        raise ValueError("Plan name must be 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit")
    slots = _integer(plan.get("slots"), "slots", 1, MAX_PARALLEL_SLOTS)
    _integer(plan.get("first_port", PORTS.start), "first_port", PORTS.start, PORTS.stop - slots)
    _integer(plan.get("max_continuations", 5), "max_continuations", 0, 20)
    interleave = _flag(plan.get("interleave", False), "interleave")
    _flag(plan.get("evaluate_after_all_training", False), "evaluate_after_all_training")
    defaults = plan.get("defaults", {})
    if not isinstance(defaults, dict) or set(defaults) - SHARED_KEYS:
        raise ValueError(f"defaults may set only {', '.join(sorted(SHARED_KEYS))}")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("A plan needs a non-empty list of jobs")
    per_job: list[list[Unit]] = []
    trained: set[str] = set()
    for index, raw in enumerate(jobs, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"Job {index} must be a JSON object")
        job = {**defaults, **raw}
        name = job.get("name")
        if not isinstance(name, str) or not NAME.match(name) or re.search(r"-part\d+$", name, re.IGNORECASE):
            raise ValueError(f"Job {index} needs a name of 1-64 letters, digits, '.', '_' or '-' that does not end "
                             "in -partN (continuation folders use that suffix)")
        kind = job.get("kind")
        if kind not in JOB_KEYS:
            raise ValueError(f"{name}: kind must be 'train' or 'evaluate'")
        unknown = sorted(set(job) - JOB_KEYS[kind])
        if unknown:
            raise ValueError(f"{name}: unknown keys for a {kind} job: {', '.join(unknown)}")
        episodes = _integer(job.get("episodes"), f"{name}: episodes", 1, 1000)
        _integer(job.get("first_world_seed"), f"{name}: first_world_seed", -(2**63), 2**63 - episodes)
        _integer(job.get("agent_seed"), f"{name}: agent_seed", -(2**63), 2**63 - 1)
        _integer(job.get("max_steps"), f"{name}: max_steps", 1, 1000)
        _number(job.get("pause"), f"{name}: pause", 0, 10)
        condition = job.get("condition")
        if condition is not None and condition not in CONDITIONS:
            raise ValueError(f"{name}: condition must be one of {', '.join(CONDITIONS)}")
        saved = {}
        if job.get("policy") is not None:
            job["policy"] = _policy_path(job["policy"], name)
            saved = _saved_policy(job["policy"], name)
            # A policy carries its own condition (legacy checkpoints ran masked) and algorithm.
            if condition is not None and saved.get("condition", "masked") != condition:
                raise ValueError(f"{name}: the policy was trained under condition {saved.get('condition', 'masked')}, "
                                 f"not {condition}")
        if kind == "train":
            if job.get("algorithm") not in ("q_learning", "sarsa"):
                raise ValueError(f"{name}: algorithm must be q_learning or sarsa")
            if saved and saved.get("algorithm") != job["algorithm"]:
                raise ValueError(f"{name}: the policy is a {saved.get('algorithm')} policy, not {job['algorithm']}")
            if condition is None and job.get("policy") is None:
                raise ValueError(f"{name}: a new training job needs a condition")
            start = _number(job.get("epsilon"), f"{name}: epsilon", 0, 1)
            _number(job.get("epsilon_end"), f"{name}: epsilon_end", 0, start)
            if _integer(job.get("pretrain", 0), f"{name}: pretrain", 0, 100_000) and job.get("policy"):
                raise ValueError(f"{name}: pretrain cannot be combined with a policy")
            per_job.append([Unit(name, job, episodes, job["first_world_seed"])])
            trained.add(name)
            continue
        if "random_baseline" in job:
            _flag(job["random_baseline"], f"{name}: random_baseline")
        sources = [job.get("policy") is not None, job.get("policy_from") is not None, job.get("random_baseline") is True]
        if sum(sources) != 1:
            raise ValueError(f"{name}: give exactly one of policy, policy_from, or random_baseline: true")
        if job.get("random_baseline") and condition not in ("masked", "unmasked"):
            raise ValueError(f"{name}: a random baseline needs condition masked or unmasked; unmasked_history "
                             "policies are compared with the unmasked baseline")
        if job.get("policy_from") is not None and job["policy_from"] not in trained:
            raise ValueError(f"{name}: policy_from must name a training job listed earlier in the plan")
        chunks = _integer(job.get("chunks", 1), f"{name}: chunks", 1, episodes)
        units, seed = [], job["first_world_seed"]
        for chunk, size in enumerate(_split(episodes, chunks), 1):
            units.append(Unit(name if chunks == 1 else f"{name}-c{chunk}", job, size, seed,
                              depends_on=job.get("policy_from")))
            seed += size
        per_job.append(units)
    if interleave:
        # Round-robin over jobs: the first chunk of every job, then every second chunk, and so on.
        ordered = [units[r] for r in range(max(map(len, per_job))) for units in per_job if r < len(units)]
    else:
        ordered = [unit for units in per_job for unit in units]
    # Case-folded: the file system may not distinguish abc from ABC.
    names = [unit.name.casefold() for unit in ordered]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError("Run names must be distinct (ignoring case): " + ", ".join(repeated))
    return plan, ordered


def load_plan(path) -> tuple[dict, list[Unit]]:
    return parse_plan(Path(path).read_text(encoding="utf-8"))


def _command(unit: Unit, *, folder: Path, port: int, slots: int, done: int, policy: Path | None,
             first: bool) -> list[str]:
    job = unit.job
    if job["kind"] == "train":
        arguments = ["resource-train", "--algorithm", job["algorithm"]]
        if job.get("condition"):
            arguments += ["--condition", job["condition"]]
        if policy is not None:
            arguments += ["--policy", str(policy)]
        if first and job.get("pretrain"):
            arguments += ["--pretrain", str(job["pretrain"])]
        # A continuation resumes the declared schedule at its next episode.
        start = epsilon_at(job["epsilon"], job["epsilon_end"], done + 1, unit.episodes)
        arguments += ["--epsilon", repr(start), "--epsilon-end", repr(float(job["epsilon_end"]))]
    elif job.get("random_baseline"):
        arguments = ["resource-demo", "--random-baseline", "--condition", job["condition"]]
    else:
        arguments = ["resource-demo", "--policy", str(policy)]
        if job.get("condition"):
            arguments += ["--condition", job["condition"]]
    arguments += ["--episodes", str(unit.episodes - done), "--world-seed", str(unit.first_world_seed + done),
                  "--seed", str(job["agent_seed"]), "--max-steps", str(job["max_steps"]),
                  "--pause", repr(float(job["pause"])), "--out", str(folder),
                  "--port", str(port), "--parallel-slots", str(slots)]
    return [sys.executable, "-m", "infrastructure", *arguments]


def _episodes(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob("episode-*") if path.is_dir())


def _attempted(unit: Unit, out: Path) -> int:
    return sum(len(_episodes(out / part["folder"])) for part in unit.parts)


def _last_status(folder: Path) -> str | None:
    episodes = _episodes(folder)
    if not episodes:
        return None
    try:
        status = json.loads((episodes[-1] / "outcome.json").read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError, AttributeError):
        return None
    return status if isinstance(status, str) else None


def _latest_policy(unit: Unit, out: Path) -> Path | None:
    for part in reversed(unit.parts):
        candidate = out / part["folder"] / "policy.json"
        if candidate.is_file():
            return candidate
    return None


class _Child:
    """One launched run whose output goes to its own log file."""

    def __init__(self, command: list[str], log_path: Path):
        self._log = log_path.open("w", encoding="utf-8")
        try:
            # Same console as the scheduler, so Ctrl+C reaches every run and each
            # one stops its own server cleanly. Nothing here ever kills a run.
            self._process = subprocess.Popen(
                command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        except BaseException:
            self._log.close()
            raise

    def poll(self) -> int | None:
        return self._process.poll()

    def close(self) -> None:
        self._log.close()


def _git(*arguments: str) -> str:
    try:
        result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("resource-parallel records the committed code revision and needs git on PATH") from exc
    if result.returncode:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def code_state(plan_path: Path) -> dict:
    """The commit the runs read, any uncommitted code, and whether the plan file itself is committed."""
    state = {"head": _git("rev-parse", "HEAD"),
             "uncommitted_code": _git("status", "--porcelain", "--", *CODE_PATHS).splitlines()}
    plan, root = Path(plan_path).resolve(), ROOT.resolve()
    state["plan_committed"] = False
    if plan.is_relative_to(root):
        relative = plan.relative_to(root).as_posix()
        state["plan_committed"] = bool(_git("ls-files", "--", relative)) and not _git("status", "--porcelain", "--", relative)
    return state


@contextmanager
def _machine_lock():
    """Allow one schedule per machine; the operating system releases the lock if this process dies."""
    path = ROOT / ".local" / "resource-parallel.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("Another resource-parallel schedule is running on this machine; wait for it to finish") from exc
    try:
        yield
    finally:
        try:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def _write(path: Path, value: dict) -> None:
    # The record is replaced only after a complete write.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _describe(unit: Unit) -> str:
    job = unit.job
    if job["kind"] == "train":
        return f"{job['algorithm']} training, condition {job.get('condition') or 'of the policy'}"
    if job.get("random_baseline"):
        return f"random baseline, condition {job['condition']}"
    return f"final policy of {unit.depends_on}" if unit.depends_on else f"policy {job['policy']}"


def _print_plan(plan: dict, units: list[Unit], ports: list[int], out: Path) -> None:
    print(f"Plan {plan['name']}: {len(units)} runs, at most {plan['slots']} at once on ports "
          f"{ports[0]}-{ports[-1]}. Folder: {out}")
    for unit in units:
        last = unit.first_world_seed + unit.episodes - 1
        after = f", after {unit.depends_on}" if unit.depends_on else ""
        print(f"  {unit.name:<36} {unit.job['kind']:<8} {unit.episodes:>4} episodes, seeds "
              f"{unit.first_world_seed}-{last}: {_describe(unit)}{after}")
    if plan.get("evaluate_after_all_training"):
        print("No evaluation starts before every training run has ended.")
    print("First command of each run (the port, a trained policy, and any continuation are filled in at launch):")
    for unit in units:
        if unit.job.get("policy"):
            policy = Path(unit.job["policy"])
        else:
            policy = Path(f"<final policy.json of {unit.depends_on}>") if unit.depends_on else None
        command = _command(unit, folder=out / unit.name, port=ports[0], slots=plan["slots"], done=0,
                           policy=policy, first=True)
        line = subprocess.list2cmdline(["python", *command[1:]])
        print("  " + line.replace(f"--port {ports[0]}", "--port <slot port>"))


class _Schedule:
    def __init__(self, plan_path: Path, plan_bytes: bytes, plan: dict, units: list[Unit], ports: list[int],
                 out: Path, state: dict, launcher: Callable[[list[str], Path], Any], poll_seconds: float):
        self.plan_bytes, self.plan, self.units, self.ports, self.out = plan_bytes, plan, units, ports, out
        self.plan_path, self.launcher, self.poll_seconds = plan_path, launcher, poll_seconds
        self.max_continuations = plan.get("max_continuations", 5)
        self.wait_for_training = plan.get("evaluate_after_all_training", False)
        self.by_name = {unit.name: unit for unit in units}
        self.queue = list(units)
        self.free = list(range(len(ports)))
        self.running: dict[int, tuple[Unit, dict, Any]] = {}
        self.record = {
            "schema": SCHEMA,
            "evidence": "Scheduler record for live natural-resource runs; every run folder holds its own evidence",
            "plan_name": plan["name"], "plan_file": str(Path(plan_path).resolve()),
            "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "slots": len(ports), "ports": ports, "max_continuations": self.max_continuations,
            "interleave": plan.get("interleave", False), "evaluate_after_all_training": self.wait_for_training,
            "code_at_start": state, "code_at_end": None, "code_changed_during_schedule": None,
            "started_at": _now(), "finished_at": None, "status": "running", "max_running_observed": 0,
            "units": [],
        }

    def save(self) -> None:
        self.record["units"] = [
            {"name": u.name, "job": u.job["name"], "kind": u.job["kind"], "episodes": u.episodes,
             "first_world_seed": u.first_world_seed, "depends_on": u.depends_on, "status": u.status,
             "attempted": _attempted(u, self.out), "parts": u.parts} for u in self.units]
        _write(self.out / "schedule.json", self.record)

    def say(self, message: str) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

    def _policy_for(self, unit: Unit) -> tuple[Path | None, str | None]:
        job = unit.job
        if job["kind"] == "train":
            if unit.parts:
                policy = _latest_policy(unit, self.out)
                return policy, None if policy else "the previous part saved no policy.json"
            return (Path(job["policy"]) if job.get("policy") else None), None
        if job.get("random_baseline"):
            return None, None
        if unit.depends_on:
            policy = _latest_policy(self.by_name[unit.depends_on], self.out)
            return policy, None if policy else f"{unit.depends_on} saved no policy.json"
        return Path(job["policy"]), None

    def start(self, unit: Unit) -> None:
        number = len(unit.parts)
        folder_name = unit.name if number == 0 else f"{unit.name}-part{number}"
        policy, problem = self._policy_for(unit)
        if problem:
            unit.status = "failed"
            self.say(f"{folder_name} not started: {problem}")
            self.save()
            return
        slot = self.free.pop(0)
        done = _attempted(unit, self.out)
        command = _command(unit, folder=self.out / folder_name, port=self.ports[slot], slots=len(self.ports),
                           done=done, policy=policy, first=number == 0)
        log_path = self.out / "logs" / f"{folder_name}.log"
        part = {"folder": folder_name, "slot": slot + 1, "port": self.ports[slot], "episodes": unit.episodes - done,
                "first_world_seed": unit.first_world_seed + done, "policy": str(policy) if policy else None,
                "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest() if policy else None,
                "command": command, "log": log_path.relative_to(self.out).as_posix(), "attempted_before": done,
                "running_at_start": len(self.running) + 1, "started_at": _now(), "finished_at": None,
                "exit_code": None, "attempted_after": None, "last_status": None}
        unit.parts.append(part)
        try:
            child = self.launcher(command, log_path)
        except OSError as exc:
            self.free.insert(0, slot)
            part.update(finished_at=_now(), last_status=f"launch failed: {exc}")
            unit.status = "failed"
            self.say(f"{folder_name} could not be launched: {exc}")
            self.save()
            return
        unit.status = "running"
        self.running[slot] = (unit, part, child)
        self.record["max_running_observed"] = max(self.record["max_running_observed"], len(self.running))
        self.say(f"start {folder_name}: {part['episodes']} episodes from world seed {part['first_world_seed']} "
                 f"on port {part['port']} ({len(self.running)} of {len(self.ports)} slots busy)")
        self.save()

    def launch_ready(self) -> bool:
        launched = False
        training_open = any(u.job["kind"] == "train" and u.status in ("waiting", "running") for u in self.units)
        for unit in list(self.queue):
            if unit.depends_on:
                dependency = self.by_name[unit.depends_on]
                if dependency.status in UNFINISHED:
                    unit.status = "blocked"
                    self.queue.remove(unit)
                    self.say(f"{unit.name} will not run: {unit.depends_on} ended {dependency.status}")
                    self.save()
                    launched = True
                    continue
                if dependency.status != "done":
                    continue
            if unit.job["kind"] == "evaluate" and self.wait_for_training and training_open:
                continue
            if not self.free:
                break
            self.queue.remove(unit)
            self.start(unit)
            launched = True
        return launched

    def reap(self, *, stopping: bool) -> None:
        for slot, (unit, part, child) in list(self.running.items()):
            code = child.poll()
            if code is None:
                continue
            child.close()
            del self.running[slot]
            self.free.append(slot)
            self.free.sort()
            attempted = _attempted(unit, self.out)
            last = _last_status(self.out / part["folder"])
            part.update(finished_at=_now(), exit_code=code, attempted_after=attempted, last_status=last)
            new = attempted - part["attempted_before"]
            if code == 0 and attempted >= unit.episodes:
                unit.status = "done"
            elif stopping or last == "interrupted":
                unit.status = "interrupted"
            elif code != 0 and new > 0 and last == "invalid":
                # The declared continuation: a new folder from the next world seed.
                if len(unit.parts) <= self.max_continuations:
                    unit.status = "waiting"
                    self.queue.insert(0, unit)
                else:
                    unit.status = "gave_up"
            else:
                unit.status = "failed"
            self.say(f"{part['folder']} exited with code {code}: {attempted}/{unit.episodes} episodes attempted, "
                     f"{'continuing after an invalid episode' if unit.status == 'waiting' else unit.status}")
            self.save()

    def drain(self, timeout: float = 600) -> None:
        deadline = time.monotonic() + timeout
        while self.running and time.monotonic() < deadline:
            try:
                time.sleep(self.poll_seconds)
                self.reap(stopping=True)
            except KeyboardInterrupt:
                self.say("Second interrupt: the scheduler stops waiting; each run still stops its own server.")
                break
        for unit, part, child in self.running.values():
            part["last_status"] = "still running when the scheduler exited"
            unit.status = "interrupted"
            child.close()

    def run(self) -> int:
        self.out.mkdir(parents=True)
        (self.out / "logs").mkdir()
        (self.out / "plan.json").write_bytes(self.plan_bytes)
        self.save()
        self.say(f"Schedule {self.plan['name']}: {len(self.units)} runs, at most {len(self.ports)} at once on ports "
                 f"{self.ports[0]}-{self.ports[-1]}. Record: {self.out / 'schedule.json'}")
        interrupted = False
        try:
            while self.queue or self.running:
                launched = self.launch_ready()
                if not self.running:
                    if not launched:
                        break
                    continue
                time.sleep(self.poll_seconds)
                self.reap(stopping=False)
        except KeyboardInterrupt:
            interrupted = True
            self.say("Interrupted: no new run starts. Waiting while the running runs save their worlds and stop.")
            self.drain()
        finally:
            for unit in self.units:
                if unit.status == "waiting":
                    unit.status = "not_started"
            try:
                end = code_state(self.plan_path)
                start = self.record["code_at_start"]
                self.record["code_at_end"] = end
                self.record["code_changed_during_schedule"] = end["head"] != start["head"] or bool(end["uncommitted_code"])
            except RuntimeError as exc:
                self.record["code_at_end"] = {"error": str(exc)}
            complete = all(unit.status == "done" for unit in self.units)
            self.record["status"] = "interrupted" if interrupted else "completed" if complete else "finished_with_failures"
            self.record["finished_at"] = _now()
            self.save()
        print(f"Schedule {self.plan['name']} {self.record['status']}:")
        for unit in self.units:
            continuations = len(unit.parts) - 1
            note = f", {continuations} continuation(s)" if continuations > 0 else ""
            print(f"  {unit.name:<36} {unit.status:<12} {_attempted(unit, self.out)}/{unit.episodes} attempted{note}")
        if self.record["code_changed_during_schedule"]:
            print("WARNING: the code changed while the schedule ran; these runs do not share one revision.")
        return 130 if interrupted else 0 if self.record["status"] == "completed" else 1


def run_schedule(plan_path, *, dry_run: bool = False, launcher: Callable[[list[str], Path], Any] | None = None,
                 poll_seconds: float = 2.0) -> int:
    """Validate a plan, then (unless ``dry_run``) run its jobs at most ``slots`` at a time."""
    plan_path = Path(plan_path)
    plan_bytes = plan_path.read_bytes()
    plan, units = parse_plan(plan_bytes.decode("utf-8"))
    first_port = plan.get("first_port", PORTS.start)
    ports = list(range(first_port, first_port + plan["slots"]))
    out = ROOT / "runs" / plan["name"]
    if out.exists():
        raise ValueError(f"Schedule folder already exists: {out}. Choose a new plan name; runs are never reused")
    if dry_run:
        _print_plan(plan, units, ports, out)
        return 0
    with _machine_lock():
        state = code_state(plan_path)
        if state["uncommitted_code"]:
            raise RuntimeError("Commit the code and the plan before a live schedule; uncommitted: "
                               + "; ".join(state["uncommitted_code"]))
        for port in ports:
            # Refuses a busy port; never stops whatever holds it.
            managed_server._check_port(port)
        schedule = _Schedule(plan_path, plan_bytes, plan, units, ports, out, state, launcher or _Child, poll_seconds)
        return schedule.run()

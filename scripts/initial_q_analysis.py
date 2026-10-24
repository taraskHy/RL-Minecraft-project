"""Declared analysis of configs/initial_q_experiment.json, read only from retained files.

Usage: python scripts/initial_q_analysis.py SCHEDULE_FOLDER
SCHEDULE_FOLDER is the retained resource-parallel folder (results/.../iq-experiment), whose nested run
folders hold every episode. Prints Markdown: the per-agent test results, the pooled comparison of the
two initial values per algorithm by the declared rule, the mechanism measures, each policy against
the uniform random baseline, and training successes per ten episodes. Nothing is written.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.parallel_runs import schedule_run_folders  # noqa: E402
from minecraft_rl.resource_report import wilson_interval  # noqa: E402
from minecraft_rl.resources import ACTION_NAMES  # noqa: E402

RUN = re.compile(r"^iq-(q|sarsa)-(zero|pess)-s(\d+)-(train|test)(?:-part\d+)?$")
ALGORITHMS = {"q": "q_learning", "sarsa": "sarsa"}
ARMS = {"zero": 0.0, "pess": -1.0}
LOOP_TAIL = 20


def _episodes(run: Path):
    for folder in sorted(run.glob("episode-*")):
        outcome_path = folder / "outcome.json"
        if not folder.is_dir() or not outcome_path.is_file():
            continue
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        transitions_path = folder / "transitions.jsonl"
        transitions = ([json.loads(line) for line in transitions_path.read_text(encoding="utf-8").splitlines() if line]
                       if transitions_path.is_file() else [])
        yield outcome, transitions


def _interval(k: int, n: int) -> str:
    low, high = wilson_interval(k, n) if n else (None, None)
    return "n/a" if low is None else f"{low:.2f}–{high:.2f}"


def _verdict(a: tuple[int, int], b: tuple[int, int]) -> str:
    """The declared rule: an effect only when the pooled Wilson intervals do not overlap."""
    if not a[1] or not b[1]:
        return "not compared: no measured episodes"
    low_a, high_a = wilson_interval(*a)
    low_b, high_b = wilson_interval(*b)
    if low_a > high_b:
        return "effect shown: initial Q -1 better"
    if high_a < low_b:
        return "effect shown: initial Q -1 worse"
    return "no effect shown: the intervals overlap"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("schedule", type=Path)
    args = parser.parse_args(argv)
    runs, missing = schedule_run_folders(args.schedule)
    tests, training, random_runs = defaultdict(list), defaultdict(list), []
    for run in runs:
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        if run.name.startswith("iq-random-test"):
            random_runs.append(run)
            continue
        match = RUN.match(run.name)
        if not match:
            raise ValueError(f"Unexpected run folder in the schedule: {run.name}")
        short, arm, seed, kind = match.groups()
        key = (ALGORITHMS[short], ARMS[arm], int(seed))
        # The recorded learner must be the one the name claims; a mismatch stops the analysis.
        if (metadata.get("algorithm"), metadata.get("initial_q")) != (key[0], key[1]):
            raise ValueError(f"{run.name} records {metadata.get('algorithm')} with initial_q "
                             f"{metadata.get('initial_q')!r}, not {key[0]} with {key[1]}")
        (tests if kind == "test" else training)[key].append(run)

    lines = ["# Initial-Q experiment: declared analysis", "",
             "**REAL MINECRAFT NATURAL RESOURCE TASK; NOT FULL SURVIVAL COMPLETION**", ""]
    if missing:
        lines += [f"Recorded in the schedule but holding no run: {', '.join(missing)}", ""]

    per_agent, pooled = [], defaultdict(lambda: {"k": 0, "n": 0, "not_measured": 0, "decisions": 0, "untried": 0,
                                                 "outside": 0, "no_progress": 0, "loops": 0, "agents": []})
    for key in sorted(tests):
        algorithm, initial_q, seed = key
        k = n = not_measured = decisions = untried = outside = no_progress = loops = 0
        for run in sorted(tests[key]):
            for outcome, transitions in _episodes(run):
                if outcome.get("status") != "completed":
                    not_measured += 1
                    continue
                n += 1
                k += int(bool(outcome.get("success")))
                for t in transitions:
                    decisions += 1
                    action = ACTION_NAMES.index(t["action"])
                    untried += int(t["q_before"][action] == initial_q)
                    outside += int(action not in (t.get("masked_actions") or []))
                    no_progress += int(t["state"][:9] == t["next_state"][:9])
                tail = transitions[-LOOP_TAIL:]
                if (not outcome.get("success") and len(tail) == LOOP_TAIL
                        and all(t["state"][:9] == t["next_state"][:9] for t in tail)):
                    loops += 1
        row = {"algorithm": algorithm, "initial_q": initial_q, "seed": seed, "k": k, "n": n,
               "not_measured": not_measured, "decisions": decisions, "untried": untried, "outside": outside,
               "no_progress": no_progress, "loops": loops}
        per_agent.append(row)
        cell = pooled[(algorithm, initial_q)]
        for field in ("k", "n", "not_measured", "decisions", "untried", "outside", "no_progress", "loops"):
            cell[field] += row[field]
        cell["agents"].append(f"{k}/{n}")

    def share(part: int, whole: int) -> str:
        return f"{100 * part / whole:.0f}%" if whole else "n/a"

    lines += ["## Test, per agent (worlds 40000–40019)", "",
              "| Algorithm | Initial Q | Agent seed | Successes | Wilson 95% | Not measured | Loop failures | Untried choices | Outside the mask | No progress |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in per_agent:
        lines.append(f"| {r['algorithm']} | {r['initial_q']:g} | {r['seed']} | {r['k']}/{r['n']} | {_interval(r['k'], r['n'])} | "
                     f"{r['not_measured']} | {r['loops']} | {share(r['untried'], r['decisions'])} | "
                     f"{share(r['outside'], r['decisions'])} | {share(r['no_progress'], r['decisions'])} |")

    lines += ["", "## Pooled comparison, the declared rule", "",
              "Each arm pools its three agents' test episodes; an effect is claimed only when the Wilson intervals do not overlap.", "",
              "| Algorithm | Initial Q 0 | Initial Q -1 | Agents (0) | Agents (-1) | Loop failures (0 / -1) | Untried choices (0 / -1) | Verdict |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for algorithm in ("q_learning", "sarsa"):
        zero, pess = pooled[(algorithm, 0.0)], pooled[(algorithm, -1.0)]
        lines.append(f"| {algorithm} | {zero['k']}/{zero['n']} ({_interval(zero['k'], zero['n'])}) | "
                     f"{pess['k']}/{pess['n']} ({_interval(pess['k'], pess['n'])}) | {', '.join(zero['agents']) or 'none'} | "
                     f"{', '.join(pess['agents']) or 'none'} | {zero['loops']} / {pess['loops']} | "
                     f"{share(zero['untried'], zero['decisions'])} / {share(pess['untried'], pess['decisions'])} | "
                     f"{_verdict((pess['k'], pess['n']), (zero['k'], zero['n']))} |")

    random_k = random_n = 0
    for run in random_runs:
        for outcome, _ in _episodes(run):
            if outcome.get("status") == "completed":
                random_n += 1
                random_k += int(bool(outcome.get("success")))
    lines += ["", f"Uniform random baseline on the same worlds: {random_k}/{random_n} ({_interval(random_k, random_n)}). "
              "Each policy's comparison with it is in the resource-report comparison table.", ""]

    lines += ["## Training successes per ten episodes (worlds 100–149)", "",
              "| Algorithm | Initial Q | Agent seed | Attempted | Successes | Per ten |", "| --- | --- | --- | --- | --- | --- |"]
    for key in sorted(training):
        algorithm, initial_q, seed = key
        by_seed = {}
        for run in sorted(training[key]):
            for outcome, _ in _episodes(run):
                if isinstance(outcome.get("world_seed"), int):
                    by_seed[outcome["world_seed"]] = outcome
        tens = defaultdict(int)
        for world, outcome in by_seed.items():
            tens[(world - 100) // 10] += int(bool(outcome.get("success")))
        lines.append(f"| {algorithm} | {initial_q:g} | {seed} | {len(by_seed)} | "
                     f"{sum(int(bool(o.get('success'))) for o in by_seed.values())} | "
                     f"{', '.join(str(tens[i]) for i in range(5))} |")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Declared analysis of configs/initial_q_experiment.json, read only from retained files.

Usage: python scripts/initial_q_analysis.py SCHEDULE_FOLDER
SCHEDULE_FOLDER is the retained resource-parallel folder (results/.../iq-experiment), whose nested run
folders hold every episode. Prints Markdown: every unit's status, the per-agent test results, the
comparison of the two initial values per algorithm by the declared rule (paired on the agent and
world seeds both arms completed, with the conservative check), the mechanism measures per arm, the
uniform random baseline, and training successes per ten episodes. Nothing is written.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.parallel_runs import schedule_record_path, schedule_run_folders  # noqa: E402
from minecraft_rl.resource_report import classify_failure, wilson_interval  # noqa: E402
from minecraft_rl.resources import ACTION_NAMES  # noqa: E402

RUN = re.compile(r"^iq-(q|sarsa)-(zero|pess)-s(\d+)-(train|test)(?:-part\d+)?$")
RANDOM = re.compile(r"^iq-random-test(?:-part\d+)?$")
# The filler keeps the second slot busy until the last analysed test ends; it is never analysed.
FILLER = re.compile(r"^iq-filler-")
ALGORITHMS = {"q": "q_learning", "sarsa": "sarsa"}
ARMS = {"zero": 0.0, "pess": -1.0}
AGENT_SEEDS = (0, 1, 2)
TEST_WORLDS = range(40000, 40020)
LOOP_TAIL = 20


def _episodes(run: Path, metadata: dict):
    """Every episode folder of a run, with its world seed; a folder without outcome.json is unfinished."""
    for folder in sorted(run.glob("episode-*")):
        if not folder.is_dir():
            continue
        outcome_path = folder / "outcome.json"
        outcome = (json.loads(outcome_path.read_text(encoding="utf-8")) if outcome_path.is_file()
                   else {"status": "unfinished", "success": None})
        transitions_path = folder / "transitions.jsonl"
        transitions = ([json.loads(line) for line in transitions_path.read_text(encoding="utf-8").splitlines() if line]
                       if transitions_path.is_file() else [])
        world = outcome.get("world_seed")
        if not isinstance(world, int) or isinstance(world, bool):
            world = metadata["first_world_seed"] + int(folder.name.split("-", 1)[1]) - 1
        yield world, outcome, transitions


def _events(transition: dict) -> list[str]:
    after = transition.get("after") if isinstance(transition.get("after"), dict) else {}
    return [part for part in str(after.get("event") or "").split(";") if part]


def failure_kind(outcome: dict, transitions: list[dict]) -> str:
    """For a completed failed episode: canopy stranding, no wood, a loop of unchanged decisions, or other."""
    tail = transitions[-LOOP_TAIL:]
    if any(event.startswith("canopy_descent_refused") for t in tail for event in _events(t)):
        return "canopy"
    if classify_failure(outcome, transitions) == "no_wood_collected":
        return "no_wood"
    if len(tail) == LOOP_TAIL and all(t["state"][:9] == t["next_state"][:9] for t in tail):
        return "loop"
    return "other"


def _interval(k: int, n: int) -> str:
    interval = wilson_interval(k, n)
    return "n/a" if interval is None else f"{interval[0]:.2f}–{interval[1]:.2f}"


def _verdict(pess: tuple[int, int], zero: tuple[int, int]) -> str:
    """The declared rule: an effect only when the Wilson intervals do not overlap."""
    if not pess[1] or not zero[1]:
        return "not compared: no seed completed by both arms"
    low_p, high_p = wilson_interval(*pess)
    low_z, high_z = wilson_interval(*zero)
    if low_p > high_z:
        return "effect shown: initial Q -1 better"
    if high_p < low_z:
        return "effect shown: initial Q -1 worse"
    return "no effect shown: the intervals overlap"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("schedule", type=Path)
    args = parser.parse_args(argv)
    record = json.loads(schedule_record_path(args.schedule).read_text(encoding="utf-8"))
    runs, missing = schedule_run_folders(args.schedule)
    tests, training, random_runs = defaultdict(list), defaultdict(list), []
    for run in runs:
        if FILLER.match(run.name):
            continue
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        if RANDOM.match(run.name):
            random_runs.append((run, metadata))
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
        (tests if kind == "test" else training)[key].append((run, metadata))

    lines = ["# Initial-Q experiment: declared analysis", "",
             "**REAL MINECRAFT NATURAL RESOURCE TASK; NOT FULL SURVIVAL COMPLETION**", "",
             f"Schedule status: {record.get('status')}.", "",
             "| Unit | Status | Attempted | Planned |", "| --- | --- | --- | --- |"]
    unfinished_units = []
    for unit in record.get("units") or []:
        lines.append(f"| {unit.get('name')} | {unit.get('status')} | {unit.get('attempted')} | {unit.get('episodes')} |")
        if unit.get("status") != "done" and not FILLER.match(str(unit.get("name"))):
            unfinished_units.append(unit.get("name"))
    if missing:
        lines += ["", f"Recorded in the schedule but holding no run: {', '.join(missing)}"]
    if unfinished_units:
        lines += ["", f"**Incomplete:** these units did not finish: {', '.join(unfinished_units)}. Their missing "
                  "episodes count as not measured below."]

    per_agent = []
    outcomes = defaultdict(dict)          # (algorithm, initial_q) -> {(agent seed, world): success or None}
    mechanisms = defaultdict(lambda: defaultdict(int))
    for key in sorted(tests):
        algorithm, initial_q, seed = key
        row = defaultdict(int, {"algorithm": algorithm, "initial_q": initial_q, "seed": seed})
        for run, metadata in sorted(tests[key], key=lambda item: item[0].name):
            for world, outcome, transitions in _episodes(run, metadata):
                completed = outcome.get("status") == "completed"
                outcomes[(algorithm, initial_q)][(seed, world)] = bool(outcome.get("success")) if completed else None
                if not completed:
                    row["not_measured"] += 1
                    continue
                row["n"] += 1
                row["k"] += int(bool(outcome.get("success")))
                if not outcome.get("success"):
                    row[failure_kind(outcome, transitions)] += 1
                for t in transitions:
                    row["decisions"] += 1
                    action = ACTION_NAMES.index(t["action"])
                    row["untried"] += int(t["q_before"][action] == initial_q)
                    row["outside"] += int(action not in (t.get("masked_actions") or []))
                    row["no_progress"] += int(t["state"][:9] == t["next_state"][:9])
        per_agent.append(row)
        for field in ("decisions", "untried", "outside", "no_progress", "loop", "canopy", "no_wood", "other"):
            mechanisms[(algorithm, initial_q)][field] += row[field]

    def share(part: int, whole: int) -> str:
        return f"{100 * part / whole:.0f}%" if whole else "n/a"

    lines += ["", "## Test, per agent (worlds 40000–40019)", "",
              "Failure kinds: canopy (stranded, a refused descent in the last 20 decisions), no wood, loop (the last "
              "20 decisions changed none of the nine resource_v1 fields), other.", "",
              "| Algorithm | Initial Q | Agent seed | Successes | Wilson 95% | Not measured | Canopy | No wood | Loop | Other | Untried choices | Outside the mask | No progress |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in per_agent:
        lines.append(f"| {r['algorithm']} | {r['initial_q']:g} | {r['seed']} | {r['k']}/{r['n']} | {_interval(r['k'], r['n'])} | "
                     f"{r['not_measured']} | {r['canopy']} | {r['no_wood']} | {r['loop']} | {r['other']} | "
                     f"{share(r['untried'], r['decisions'])} | {share(r['outside'], r['decisions'])} | "
                     f"{share(r['no_progress'], r['decisions'])} |")

    planned = {(seed, world) for seed in AGENT_SEEDS for world in TEST_WORLDS}
    lines += ["", "## The two initial values compared, by the declared rule", "",
              "Each algorithm's arms are compared on the (agent seed, world seed) episodes both completed, pooled over "
              "the three agents. An effect is claimed only when the Wilson intervals do not overlap. When an arm lost "
              "episodes, the conservative check counts every not-measured episode against the claimed direction "
              "(as a failure for the arm claimed better, a success for the other) over all 60 planned episodes.", "",
              "| Algorithm | Episodes compared | Initial Q 0 | Initial Q -1 | Per agent (0) | Per agent (-1) | Not measured (0 / -1) | Verdict | Conservative check |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for algorithm in ("q_learning", "sarsa"):
        zero, pess = outcomes[(algorithm, 0.0)], outcomes[(algorithm, -1.0)]
        common = sorted(p for p in planned if zero.get(p) is not None and pess.get(p) is not None)
        k_zero, k_pess = sum(zero[p] for p in common), sum(pess[p] for p in common)
        lost_zero = [p for p in planned if zero.get(p) is None]
        lost_pess = [p for p in planned if pess.get(p) is None]
        verdict = _verdict((k_pess, len(common)), (k_zero, len(common)))
        conservative = "not needed: no episode lost"
        if lost_zero or lost_pess:
            succ_zero = sum(1 for p in planned if zero.get(p))
            succ_pess = sum(1 for p in planned if pess.get(p))
            check = None
            if verdict.endswith("better"):
                check = _verdict((succ_pess, len(planned)), (succ_zero + len(lost_zero), len(planned)))
            elif verdict.endswith("worse"):
                check = _verdict((succ_pess + len(lost_pess), len(planned)), (succ_zero, len(planned)))
            conservative = ("not needed: no effect claimed" if check is None else "the claim holds" if check == verdict
                            else f"the claim does not hold ({check.replace('effect shown: ', '')})")
        agents = {value: ", ".join(f"{r['k']}/{r['n']}" for r in per_agent
                                   if (r["algorithm"], r["initial_q"]) == (algorithm, value)) or "none"
                  for value in (0.0, -1.0)}
        lines.append(f"| {algorithm} | {len(common)} | {k_zero}/{len(common)} ({_interval(k_zero, len(common))}) | "
                     f"{k_pess}/{len(common)} ({_interval(k_pess, len(common))}) | {agents[0.0]} | {agents[-1.0]} | "
                     f"{len(lost_zero)} / {len(lost_pess)} | {verdict} | {conservative} |")

    lines += ["", "## Mechanism measures per arm", "",
              "| Algorithm | Initial Q | Canopy | No wood | Loop | Other | Untried choices | Outside the mask | No progress |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for algorithm in ("q_learning", "sarsa"):
        for value in (0.0, -1.0):
            m = mechanisms[(algorithm, value)]
            lines.append(f"| {algorithm} | {value:g} | {m['canopy']} | {m['no_wood']} | {m['loop']} | {m['other']} | "
                         f"{share(m['untried'], m['decisions'])} | {share(m['outside'], m['decisions'])} | "
                         f"{share(m['no_progress'], m['decisions'])} |")

    random_k = random_n = random_lost = 0
    for run, metadata in random_runs:
        for _, outcome, _ in _episodes(run, metadata):
            if outcome.get("status") == "completed":
                random_n += 1
                random_k += int(bool(outcome.get("success")))
            else:
                random_lost += 1
    lines += ["", f"Uniform random baseline on the same worlds: {random_k}/{random_n} ({_interval(random_k, random_n)}), "
              f"{random_lost} not measured. Each policy's comparison with it is in the resource-report comparison table.", ""]

    lines += ["## Training successes per ten episodes (worlds 100–149)", "",
              "| Algorithm | Initial Q | Agent seed | Attempted | Not measured | Successes | Per ten |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for key in sorted(training):
        algorithm, initial_q, seed = key
        by_world = {}
        for run, metadata in sorted(training[key], key=lambda item: item[0].name):
            for world, outcome, _ in _episodes(run, metadata):
                by_world[world] = outcome
        tens = defaultdict(int)
        for world, outcome in by_world.items():
            tens[(world - 100) // 10] += int(bool(outcome.get("success")))
        not_measured = sum(1 for o in by_world.values() if o.get("status") != "completed")
        lines.append(f"| {algorithm} | {initial_q:g} | {seed} | {len(by_world)} | {not_measured} | "
                     f"{sum(int(bool(o.get('success'))) for o in by_world.values())} | "
                     f"{', '.join(str(tens[i]) for i in range(5))} |")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

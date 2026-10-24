"""Summaries of retained natural-resource runs for the report.

Reads the evidence folders written by ``infrastructure.resource_session`` and
produces honestly labeled counts: success rate with a Wilson interval,
decisions, task time, deaths, invalid episodes, failure causes, and how much
of each decision the action mask made. Experiment conditions, exploratory
training, frozen evaluation, and random baselines are summarized as separate
groups and never pooled; each frozen learned policy is compared with the random
baseline that has the same action mask, step limit, and world seeds. Nothing
here infers a result that the saved outcome files do not contain.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean

from .artifacts import prepare_output, write_csv, write_json
from .resources import ACTION_NAMES, CONDITIONS

EVIDENCE = "REAL MINECRAFT NATURAL RESOURCE TASK; NOT FULL SURVIVAL COMPLETION"
MODES = ("online_training", "frozen_evaluation")
FAILURE_CAUSES = ("death", "no_wood_collected", "table_not_placed", "pickaxe_not_crafted")
NOTES = (
    "Success means a wooden pickaxe was crafted from naturally gathered wood and confirmed held; "
    "it is not full Survival or an Ender Dragon result.",
    "Rates are computed over completed episodes only; invalid or interrupted episodes are counted "
    "separately and never treated as policy failures or successes.",
    "The Wilson 95% interval describes these episodes for one checkpoint or baseline under one "
    "step limit; it does not measure variation across independently trained agents.",
    "Returns from online training include exploratory actions and shaped rewards; compare them "
    "only with other training runs, not with frozen evaluation.",
    "The feasibility mask and scripted helpers supply substantial task knowledge; compare learned "
    "policies against the random baseline of the same condition (masked random with the mask, uniform "
    "random over all eight actions without it) before claiming a benefit.",
    "Conditions are never pooled. A benefit is claimed only when a learned policy's Wilson interval lies "
    "entirely above its baseline's; overlapping intervals mean no benefit was shown. Fisher's exact test "
    "is reported as a supplement, not as the decision rule.",
    "Runs recorded before experiment conditions existed are reported as the masked condition, which is how "
    "they ran.",
    "Runs with a different parallel setup (live episodes at once on one machine) are separate groups and are "
    "never compared, because task time and the helpers' time budgets are wall-clock. Parallel runs are "
    "grouped and compared only within one schedule, whose plan rules (interleaved arms, evaluations only "
    "after all training) make the test arms share the load; that schedule's schedule.json records the "
    "concurrency each run actually had. Runs recorded before resource-parallel existed ran one episode at "
    "a time and are reported with one slot.",
)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> list[float] | None:
    """Wilson score interval for a binomial proportion; None without trials."""
    if (isinstance(successes, bool) or isinstance(trials, bool) or not isinstance(successes, int)
            or not isinstance(trials, int) or trials < 0 or not 0 <= successes <= trials):
        raise ValueError("successes and trials must be integers with 0 <= successes <= trials")
    if trials == 0:
        return None
    p = successes / trials
    center = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return [max(0.0, center - half), min(1.0, center + half)]


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON objects in {path}")
            rows.append(row)
    return rows


def _counts(observation) -> dict:
    counts = observation.get("inventory_counts") if isinstance(observation, dict) else None
    return counts if isinstance(counts, dict) else {}


def classify_failure(outcome: dict, transitions: list[dict]) -> str:
    """Name the furthest stage an unsuccessful episode reached, from saved data only."""
    status = outcome.get("status")
    if status != "completed":
        return f"{status or 'unknown'}_run"
    if outcome.get("success"):
        return "success"
    if outcome.get("death"):
        return "death"
    final = outcome.get("final_observation") if isinstance(outcome.get("final_observation"), dict) else {}
    observations = [t.get("after") for t in transitions] + [final]
    # Sticks alone are not evidence of chopping: leaf blocks cleared during a
    # canopy descent can drop them. Planks and tables only come from logs here.
    had_wood = any(
        any(_counts(o).get(field, 0) for field in ("logs", "planks", "tables"))
        or (isinstance(o, dict) and o.get("table_placed"))
        for o in observations if isinstance(o, dict))
    if not had_wood:
        return "no_wood_collected"
    if not final.get("table_placed"):
        return "table_not_placed"
    return "pickaxe_not_crafted"


def fisher_exact(a_successes: int, a_trials: int, b_successes: int, b_trials: int) -> float:
    """Two-sided Fisher exact test for two success counts (sum of tables no more likely than observed)."""
    for value in (a_successes, a_trials, b_successes, b_trials):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("counts must be nonnegative integers")
    if a_successes > a_trials or b_successes > b_trials or not a_trials or not b_trials:
        raise ValueError("each group needs trials and at most as many successes as trials")
    total = a_trials + b_trials
    successes = a_successes + b_successes

    def probability(x: int) -> float:
        return math.comb(successes, x) * math.comb(total - successes, a_trials - x) / math.comb(total, a_trials)

    observed = probability(a_successes)
    low, high = max(0, successes - b_trials), min(successes, a_trials)
    return min(1.0, sum(p for p in (probability(x) for x in range(low, high + 1)) if p <= observed * (1 + 1e-9)))


def _decision_counts(transitions: list[dict]) -> dict:
    """How much of each decision the mask made, from the saved transitions."""
    counts = {"decisions": 0, "single_allowed_decisions": 0, "masked_out_decisions": 0, "no_progress_decisions": 0}
    for transition in transitions:
        allowed = transition.get("allowed_actions")
        if not isinstance(allowed, list):
            continue
        counts["decisions"] += 1
        counts["single_allowed_decisions"] += int(len(allowed) == 1)
        # Before conditions existed every run was masked, so the allowed set was the mask.
        masked = transition.get("masked_actions", allowed)
        action = transition.get("action")
        if action in ACTION_NAMES and isinstance(masked, list) and ACTION_NAMES.index(action) not in masked:
            counts["masked_out_decisions"] += 1
        state, following = transition.get("state"), transition.get("next_state")
        if isinstance(state, list) and isinstance(following, list) and state[:9] == following[:9]:
            counts["no_progress_decisions"] += 1
    return counts


def _world_seed(outcome: dict, transitions: list[dict], metadata: dict, number: int | None):
    """The episode's world seed: recorded, else from its first transition, else from the run's first seed.

    The session gives episode n the world seed first_world_seed + n - 1, so an
    episode that failed before any transition still keeps its seed, and the
    comparison can match seed sets that include not-measured episodes.
    """
    seed = outcome.get("world_seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    if transitions and isinstance(transitions[0].get("world_seed"), int):
        return transitions[0]["world_seed"]
    first = metadata.get("first_world_seed")
    if isinstance(first, int) and not isinstance(first, bool) and isinstance(number, int):
        return first + number - 1
    return None


def _event_parts(event) -> list[str]:
    # Bridge events may join several notes with ';', for example
    # 'cleared_2_leaf_layers;descended_canopy;log_reached'.
    return [part for part in str(event or "").split(";") if part]


def load_run(run_directory: Path) -> dict:
    """Read one run folder: metadata plus every episode with an outcome file."""
    run_directory = Path(run_directory)
    metadata = _read_json(run_directory / "metadata.json")
    # Arena, simulator, and watch runs write the same folder layout; only
    # natural-resource output may be counted under this report's banner.
    if metadata.get("evidence") != EVIDENCE or metadata.get("mode") not in MODES:
        raise ValueError(
            f"{run_directory} is not a natural-resource run folder (evidence={metadata.get('evidence')!r}, "
            f"mode={metadata.get('mode')!r}); resource-report summarizes only resource-train and resource-demo output")
    condition = metadata.get("condition") or "masked"
    if condition not in CONDITIONS:
        raise ValueError(f"{run_directory} records an unknown condition: {condition!r}")
    execution = metadata.get("execution") if isinstance(metadata.get("execution"), dict) else {}
    parallel_slots = execution.get("parallel_slots", 1)
    if isinstance(parallel_slots, bool) or not isinstance(parallel_slots, int) or parallel_slots < 1:
        raise ValueError(f"{run_directory} records an invalid parallel_slots: {parallel_slots!r}")
    schedule_id = execution.get("schedule_id")
    if schedule_id is not None and not isinstance(schedule_id, str):
        raise ValueError(f"{run_directory} records an invalid schedule_id: {schedule_id!r}")
    episodes = []
    for folder in sorted(run_directory.glob("episode-*")):
        outcome_path = folder / "outcome.json"
        if not folder.is_dir():
            continue
        transitions = _read_jsonl(folder / "transitions.jsonl")
        if outcome_path.is_file():
            outcome = _read_json(outcome_path)
        else:
            # The session writes outcome.json for completed, invalid, and interrupted
            # episodes; a folder without one was cut off before any verdict.
            outcome = {"status": "unfinished", "success": None, "death": None,
                       "recorded_steps": len(transitions), "error": "no outcome file"}
        events: Counter[str] = Counter()
        recoveries = []
        for transition in transitions:
            after = transition.get("after") if isinstance(transition.get("after"), dict) else {}
            events.update(_event_parts(after.get("event")))
            if isinstance(after.get("recovery"), dict):
                recoveries.append(after["recovery"])
        descents = [r for r in recoveries if r.get("type") == "canopy_descent"]
        escapes = ([r for r in recoveries if r.get("type") == "leaf_escape"]
                   + [r["escape"] for r in recoveries if isinstance(r.get("escape"), dict)])
        try:
            number = int(folder.name.split("-", 1)[1])
        except (IndexError, ValueError):
            number = None
        steps = outcome.get("steps")
        if not isinstance(steps, int):
            steps = outcome.get("recorded_steps") if isinstance(outcome.get("recorded_steps"), int) else len(transitions)
        episodes.append({
            "run": run_directory.name, "episode": number,
            "world_seed": _world_seed(outcome, transitions, metadata, number),
            "condition": condition, "mode": metadata.get("mode"), "algorithm": metadata.get("algorithm"),
            "baseline": metadata.get("baseline"),
            "checkpoint_sha256": metadata.get("initial_policy_sha256"), "max_steps": metadata.get("max_steps"),
            "pause": metadata.get("pause_between_decisions"), "parallel_slots": parallel_slots,
            "schedule_id": schedule_id,
            # One short label for the exact helper and session sources the run started from.
            "source_revision": (hashlib.sha256(json.dumps(metadata["source_sha256"], sort_keys=True).encode()).hexdigest()[:12]
                                if isinstance(metadata.get("source_sha256"), dict) else None),
            "simulator_pretraining": sum(int(metadata.get(key) or 0) for key in
                                         ("pretrain_simulator_episodes", "checkpoint_pretrain_simulator_episodes")),
            "status": outcome.get("status"), "success": outcome.get("success"),
            "death": outcome.get("death"), "truncated": outcome.get("truncated"),
            "steps": steps, "seconds": outcome.get("seconds"), "return": outcome.get("return"),
            "failure_cause": classify_failure(outcome, transitions),
            "final_event": transitions[-1].get("after", {}).get("event") if transitions else None,
            "events": dict(events),
            "canopy_layers_cleared": sum(int(r.get("layers_cleared", 0) or 0) for r in descents),
            "leaf_wall_blocks_cleared": sum(len(e.get("blocks_dug") or []) for e in escapes),
            "recovery_damage": sum(float(r.get("damage", 0) or 0) for r in descents),
            **_decision_counts(transitions),
            "error": outcome.get("error"),
        })
    return {"run": run_directory.name, "path": str(run_directory), "metadata": metadata, "episodes": episodes}


def _mean(values) -> float | None:
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return mean(values) if values else None


def _group_key(episode: dict) -> tuple:
    # Conditions are never pooled. Training runs are pooled per algorithm; frozen
    # evaluations only per evaluated checkpoint. A different step limit is always
    # a different group, and so is a different number of parallel slots; parallel runs
    # are also grouped by schedule, so two schedules never pool their loads.
    checkpoint = episode["checkpoint_sha256"] if episode["mode"] == "frozen_evaluation" else None
    schedule = (episode["schedule_id"] or "") if episode["parallel_slots"] > 1 else ""
    return (episode["condition"], episode["mode"], episode["algorithm"], episode["baseline"] or "",
            checkpoint or "", episode["max_steps"], int(episode["simulator_pretraining"] > 0),
            episode["parallel_slots"], schedule)


def _fraction(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _summarize_group(key: tuple, rows: list[dict]) -> dict:
    condition, mode, algorithm, baseline, checkpoint, max_steps, pretrained, parallel_slots, _schedule = key
    completed = [r for r in rows if r["status"] == "completed"]
    successes = [r for r in completed if r["success"]]
    events: Counter[str] = Counter()
    for row in rows:
        events.update(row["events"])
    decisions = sum(r["decisions"] for r in rows)
    return {
        "condition": condition, "action_mask": CONDITIONS[condition][0], "state_version": CONDITIONS[condition][1],
        "mode": mode, "algorithm": algorithm, "baseline": baseline or None,
        "checkpoint_sha256": checkpoint or None, "max_steps": max_steps, "simulator_pretrained": bool(pretrained),
        "parallel_slots": parallel_slots,
        "schedules": sorted({r["schedule_id"] or "" for r in rows}),
        "pauses": sorted({r["pause"] for r in rows if r["pause"] is not None}),
        "source_revisions": sorted({r["source_revision"] for r in rows if r["source_revision"]}),
        "runs": sorted({r["run"] for r in rows}),
        "world_seeds": sorted({r["world_seed"] for r in rows if isinstance(r["world_seed"], int)}),
        "completed_seed_outcomes": sorted([r["world_seed"], bool(r["success"])] for r in completed
                                          if isinstance(r["world_seed"], int)),
        "episodes_attempted": len(rows), "episodes_completed": len(completed),
        "episodes_invalid": len(rows) - len(completed),
        "successes": len(successes),
        "success_rate": len(successes) / len(completed) if completed else None,
        "success_wilson_95": wilson_interval(len(successes), len(completed)),
        "deaths": sum(1 for r in completed if r["death"]),
        "truncations": sum(1 for r in completed if r["truncated"]),
        "mean_decisions_all": _mean(r["steps"] for r in completed),
        "mean_decisions_success": _mean(r["steps"] for r in successes),
        "mean_task_seconds_all": _mean(r["seconds"] for r in completed),
        "mean_task_seconds_success": _mean(r["seconds"] for r in successes),
        "mean_return": _mean(r["return"] for r in completed),
        "return_label": ("exploratory shaped training return" if mode == "online_training"
                         else "frozen-policy shaped return"),
        # Causes are policy failures of completed episodes only; invalid,
        # interrupted, and unfinished episodes are listed apart, never as failures.
        "failure_causes": dict(Counter(r["failure_cause"] for r in completed if r["failure_cause"] in FAILURE_CAUSES)),
        "not_measured": dict(Counter(r["status"] or "unknown" for r in rows if r["status"] != "completed")),
        "event_counts": dict(events.most_common()),
        "canopy_layers_cleared": sum(r["canopy_layers_cleared"] for r in rows),
        "leaf_wall_blocks_cleared": sum(r["leaf_wall_blocks_cleared"] for r in rows),
        "recovery_damage": sum(r["recovery_damage"] for r in rows),
        "decisions": decisions,
        "single_allowed_fraction": _fraction(sum(r["single_allowed_decisions"] for r in rows), decisions),
        "masked_out_fraction": _fraction(sum(r["masked_out_decisions"] for r in rows), decisions),
        "no_progress_fraction": _fraction(sum(r["no_progress_decisions"] for r in rows), decisions),
    }


def _verdict(learned: list[float], baseline: list[float]) -> str:
    if learned[0] > baseline[1]:
        return "benefit shown: the learned interval lies entirely above the baseline's"
    if learned[1] < baseline[0]:
        return "worse than random: the learned interval lies entirely below the baseline's"
    return "no benefit shown: the intervals overlap"


def _same_setup(a: dict, b: dict) -> bool:
    # Same step limit, the same attempted world seeds, one equal pause, one equal helper revision,
    # and the same number of parallel slots; parallel runs only within one schedule, where every
    # arm shared the same load.
    return (a["max_steps"] == b["max_steps"] and a["world_seeds"] == b["world_seeds"]
            and a["parallel_slots"] == b["parallel_slots"]
            and (a["parallel_slots"] == 1 or (len(a["schedules"]) == 1 and a["schedules"] == b["schedules"]
                                               and a["schedules"][0] != ""))
            and len(a["pauses"]) <= 1 and a["pauses"] == b["pauses"]
            and len(a["source_revisions"]) <= 1 and a["source_revisions"] == b["source_revisions"])


def _paired(first: dict, second: dict) -> dict:
    """Success counts on the seeds both groups completed, the seeds each lost, and the verdict inputs."""
    a, b = dict(map(tuple, first["completed_seed_outcomes"])), dict(map(tuple, second["completed_seed_outcomes"]))
    common = sorted(set(a) & set(b))
    a_successes, b_successes = sum(a[s] for s in common), sum(b[s] for s in common)
    lost_a, lost_b = sorted(set(first["world_seeds"]) - set(a)), sorted(set(second["world_seeds"]) - set(b))
    row = {"seeds_compared": len(common), "learned_successes": a_successes, "learned_completed": len(common),
           "learned_wilson_95": wilson_interval(a_successes, len(common)),
           "baseline_successes": b_successes, "baseline_completed": len(common),
           "baseline_wilson_95": wilson_interval(b_successes, len(common)),
           "learned_lost_seeds": lost_a, "baseline_lost_seeds": lost_b,
           "fisher_p": fisher_exact(a_successes, len(common), b_successes, len(common)) if common else None,
           "conservative_verdict": None}
    if lost_a or lost_b:
        # Sensitivity check: every not-measured episode counts against the first group.
        attempted = len(first["world_seeds"])
        pessimistic = sum(a.values())
        optimistic = sum(b.values()) + sum(1 for s in second["world_seeds"] if s not in b)
        row["conservative_verdict"] = _verdict(wilson_interval(pessimistic, attempted), wilson_interval(optimistic, attempted))
    return row


_EMPTY_PAIR = {"seeds_compared": 0, "learned_successes": None, "learned_completed": None, "learned_wilson_95": None,
               "baseline_successes": None, "baseline_completed": None, "baseline_wilson_95": None,
               "learned_lost_seeds": [], "baseline_lost_seeds": [], "fisher_p": None, "conservative_verdict": None}


def compare_to_baselines(groups: list[dict]) -> list[dict]:
    """Pair each frozen learned policy with random baselines of the same mask and setup, seed by seed."""
    baselines = [g for g in groups if g["mode"] == "frozen_evaluation" and g["algorithm"] == "random"]
    comparisons = []
    for group in groups:
        if group["mode"] != "frozen_evaluation" or group["algorithm"] not in ("q_learning", "sarsa"):
            continue
        entry = {"condition": group["condition"], "algorithm": group["algorithm"],
                 "checkpoint_sha256": group["checkpoint_sha256"], "max_steps": group["max_steps"],
                 "world_seeds": group["world_seeds"], "simulator_pretrained": group["simulator_pretrained"],
                 "parallel_slots": group["parallel_slots"],
                 # Every row carries the same keys, so the CSV columns never depend on row order.
                 "baseline": None, "baseline_condition": None, **_EMPTY_PAIR}
        matches = [b for b in baselines if b["action_mask"] == group["action_mask"] and _same_setup(group, b)]
        if not matches:
            comparisons.append({**entry, "verdict": "not compared: no random baseline with the same action mask, "
                                "step limit, world seeds, pause, helper revision, and parallel setup"})
            continue
        for baseline in matches:
            row = {**entry, "baseline": baseline["baseline"], "baseline_condition": baseline["condition"],
                   **_paired(group, baseline)}
            if not row["seeds_compared"]:
                row["verdict"] = "not compared: no seed was completed by both"
            else:
                row["verdict"] = _verdict(row["learned_wilson_95"], row["baseline_wilson_95"])
                if group["simulator_pretrained"]:
                    row["verdict"] = "simulator-pretrained checkpoint, not a live-learning result; " + row["verdict"]
            comparisons.append(row)
    return comparisons


def compare_history(groups: list[dict]) -> list[dict]:
    """The declared test of the history bit: unmasked_history against unmasked, per algorithm."""
    comparisons = []
    for history in groups:
        if (history["condition"], history["mode"]) != ("unmasked_history", "frozen_evaluation") or history["algorithm"] == "random":
            continue
        for plain in groups:
            if ((plain["condition"], plain["mode"], plain["algorithm"]) != ("unmasked", "frozen_evaluation", history["algorithm"])
                    or not _same_setup(history, plain)):
                continue
            row = {"algorithm": history["algorithm"], "parallel_slots": history["parallel_slots"],
                   "history_checkpoint": history["checkpoint_sha256"],
                   "unmasked_checkpoint": plain["checkpoint_sha256"], **_paired(history, plain)}
            if not row["seeds_compared"]:
                row["verdict"] = "not compared: no seed was completed by both"
            else:
                with_bit, without = row["learned_wilson_95"], row["baseline_wilson_95"]
                if with_bit[0] > without[1]:
                    row["verdict"] = "effect of the bit shown: the interval with the bit lies entirely above the one without"
                elif with_bit[1] < without[0]:
                    row["verdict"] = "worse with the bit: the interval with the bit lies entirely below the one without"
                else:
                    row["verdict"] = "no effect of the bit shown: the intervals overlap"
            comparisons.append(row)
    return comparisons


def summarize_runs(run_directories) -> dict:
    """Summarize one or more run folders without pooling training with evaluation."""
    paths = [Path(d) for d in run_directories]
    if not paths:
        raise ValueError("At least one run folder is required")
    # Episodes are labeled by folder name, so a folder passed twice would be
    # double-counted and two same-named folders could not be told apart.
    resolved = [os.path.normcase(str(p.resolve())) for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Each run folder may be passed only once; repeating one would double-count its episodes")
    repeated = sorted(name for name, count in Counter(p.name for p in paths).items() if count > 1)
    if repeated:
        raise ValueError("Run folders must have distinct names because episodes are labeled by folder name: "
                         + ", ".join(repeated))
    runs = [load_run(p) for p in paths]
    episodes = [e for run in runs for e in run["episodes"]]
    if not episodes:
        raise ValueError("No episode folders with results found in: " + ", ".join(run["path"] for run in runs))
    groups: dict[tuple, list[dict]] = {}
    for episode in episodes:
        groups.setdefault(_group_key(episode), []).append(episode)
    summarized = [_summarize_group(key, rows) for key, rows in sorted(groups.items(), key=lambda item: str(item[0]))]
    return {
        "evidence": EVIDENCE,
        "runs": [{"run": run["run"], "path": run["path"],
                  "condition": run["metadata"].get("condition") or "masked",
                  "mode": run["metadata"].get("mode"), "algorithm": run["metadata"].get("algorithm"),
                  "baseline": run["metadata"].get("baseline"),
                  "initial_policy_sha256": run["metadata"].get("initial_policy_sha256"),
                  "episodes_requested": run["metadata"].get("episodes_requested"),
                  "first_world_seed": run["metadata"].get("first_world_seed"),
                  "max_steps": run["metadata"].get("max_steps"),
                  "source_sha256": run["metadata"].get("source_sha256"),
                  "episodes_found": len(run["episodes"])} for run in runs],
        "groups": summarized,
        "comparisons": compare_to_baselines(summarized),
        "history_comparisons": compare_history(summarized),
        "episodes": episodes,
        "notes": list(NOTES),
    }


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _percent(value) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def _interval(interval) -> str:
    return f"{_fmt(interval[0])}–{_fmt(interval[1])}" if interval else "n/a"


def _label(group: dict) -> str:
    label = f"{group.get('condition', 'masked')} / {group['mode']} / {group['algorithm']}"
    if group["baseline"]:
        label += f" / {group['baseline']}"
    if group.get("checkpoint_sha256"):
        label += f" / checkpoint {group['checkpoint_sha256'][:12]}"
    if group.get("max_steps") is not None:
        label += f" / max_steps {group['max_steps']}"
    if group.get("simulator_pretrained"):
        label += " / simulator-pretrained"
    if group.get("parallel_slots", 1) > 1:
        label += f" / {group['parallel_slots']} parallel slots"
        schedules = [s for s in group.get("schedules", []) if s]
        # Plan name and plan digest; the full id, with its start time, is in summary.json.
        label += f" in schedule {':'.join(schedules[0].split(':')[:2])}" if len(schedules) == 1 else " in no single schedule"
    return label


def _policy_cell(comparison: dict) -> str:
    checkpoint = comparison.get("checkpoint_sha256")
    return f"{comparison['algorithm']} {checkpoint[:12]}" if checkpoint else comparison["algorithm"]


def render_markdown(summary: dict) -> str:
    lines = ["# Natural-resource run summary", "", f"**{summary['evidence']}**", ""]
    lines += ["| Group | Attempted | Completed | Invalid | Successes | Success rate | Wilson 95% | Deaths | Truncated | Mean decisions (all / success) | Mean task s (all / success) | Canopy layers cleared |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for group in summary["groups"]:
        interval = group["success_wilson_95"]
        lines.append("| " + " | ".join([
            _label(group), str(group["episodes_attempted"]), str(group["episodes_completed"]), str(group["episodes_invalid"]),
            str(group["successes"]), _fmt(group["success_rate"]),
            f"{_fmt(interval[0])}–{_fmt(interval[1])}" if interval else "n/a",
            str(group["deaths"]), str(group["truncations"]),
            f"{_fmt(group['mean_decisions_all'], 1)} / {_fmt(group['mean_decisions_success'], 1)}",
            f"{_fmt(group['mean_task_seconds_all'], 1)} / {_fmt(group['mean_task_seconds_success'], 1)}",
            str(group["canopy_layers_cleared"])]) + " |")
    lines.append("")
    lines += ["## Decisions", "",
              "How much each decision was left to the policy: the share with a single allowed action, the share of "
              "choices the disclosed mask would have forbidden, and the share that changed none of the nine "
              "resource_v1 fields.", "",
              "| Group | Decisions | Single allowed action | Choices outside the mask | No progress |",
              "| --- | --- | --- | --- | --- |"]
    for group in summary["groups"]:
        lines.append(f"| {_label(group)} | {group['decisions']} | {_percent(group['single_allowed_fraction'])} | "
                     f"{_percent(group['masked_out_fraction'])} | {_percent(group['no_progress_fraction'])} |")
    lines.append("")
    comparisons = summary.get("comparisons") or []
    if comparisons:
        lines += ["## Comparison with the random baseline", "",
                  "Each frozen learned policy against the random baseline with the same action mask, step limit, world "
                  "seeds, pause, helper revision, and parallel setup (slots, and for parallel runs one schedule), "
                  "counted on the seeds both completed. A benefit is claimed only "
                  "when the learned interval lies entirely above the baseline's. Fisher's exact p is two-sided and "
                  "supplementary.", "",
                  "| Condition | Policy | Baseline | Slots | Seeds compared | Learned successes (Wilson 95%) | Baseline successes (Wilson 95%) | Fisher exact p | Verdict |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for c in comparisons:
            if c["baseline"] is None:
                lines.append(f"| {c['condition']} | {_policy_cell(c)} | none | {c['parallel_slots']} | 0 | n/a | n/a | "
                             f"n/a | {c['verdict']} |")
                continue
            learned = f"{c['learned_successes']}/{c['learned_completed']} ({_interval(c['learned_wilson_95'])})"
            base = f"{c['baseline_successes']}/{c['baseline_completed']} ({_interval(c['baseline_wilson_95'])})"
            p = "n/a" if c["fisher_p"] is None else f"{c['fisher_p']:.3g}"
            lines.append(f"| {c['condition']} | {_policy_cell(c)} | {c['baseline']} ({c['baseline_condition']}) | "
                         f"{c['parallel_slots']} | {c['seeds_compared']} | {learned} | {base} | {p} | {c['verdict']} |")
        lines.append("")
        for c in comparisons:
            if c["learned_lost_seeds"] or c["baseline_lost_seeds"]:
                lines.append(f"- {c['condition']} / {_policy_cell(c)} / {c['parallel_slots']} slot(s): seeds not "
                             "measured for the learned policy "
                             f"{c['learned_lost_seeds'] or 'none'}, for the baseline {c['baseline_lost_seeds'] or 'none'}; "
                             f"counting every not-measured episode against the learned policy: {c['conservative_verdict']}.")
        lines.append("")
    history = summary.get("history_comparisons") or []
    if history:
        lines += ["## Effect of the history bit", "",
                  "Each unmasked_history policy against the unmasked policy of the same algorithm on the seeds both "
                  "completed. The bit is credited with an effect only when the history interval lies entirely above "
                  "the unmasked one; with one trained agent per condition this compares two checkpoints, not two "
                  "state designs.", "",
                  "| Policy | Slots | Seeds compared | With the bit (Wilson 95%) | Without (Wilson 95%) | Fisher exact p | Verdict |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for h in history:
            p = "n/a" if h["fisher_p"] is None else f"{h['fisher_p']:.3g}"
            lines.append(f"| {h['algorithm']} | {h['parallel_slots']} | {h['seeds_compared']} | "
                         f"{h['learned_successes']}/{h['learned_completed']} "
                         f"({_interval(h['learned_wilson_95'])}) | {h['baseline_successes']}/{h['baseline_completed']} "
                         f"({_interval(h['baseline_wilson_95'])}) | {p} | {h['verdict']} |")
        lines.append("")
    for group in summary["groups"]:
        lines.append(f"## {_label(group)}")
        lines.append("")
        lines.append(f"Runs: {', '.join(group['runs'])}. World seeds: {', '.join(map(str, group['world_seeds'])) or 'none recorded'}.")
        if group.get("checkpoint_sha256"):
            lines.append(f"Evaluated checkpoint (policy.json SHA-256): {group['checkpoint_sha256']}.")
        lines.append(f"Mean {group['return_label']}: {_fmt(group['mean_return'], 3)}.")
        lines.append(f"Leaf clearing: {group['canopy_layers_cleared']} canopy layers, "
                     f"{group['leaf_wall_blocks_cleared']} wall blocks, {_fmt(group['recovery_damage'], 0)} fall damage in total.")
        causes = group["failure_causes"]
        lines.append("Failure causes (completed episodes only): "
                     + (", ".join(f"{name} ×{count}" for name, count in sorted(causes.items())) if causes else "none") + ".")
        not_measured = group["not_measured"]
        lines.append("Not measured (invalid, interrupted, or unfinished; excluded from every rate): "
                     + (", ".join(f"{name} ×{count}" for name, count in sorted(not_measured.items())) if not_measured else "none") + ".")
        top = list(group["event_counts"].items())[:8]
        lines.append("Most frequent step events: " + (", ".join(f"{name} ×{count}" for name, count in top) if top else "none") + ".")
        lines.append("")
    lines.append("## Episodes")
    lines.append("")
    lines += ["| Run | Episode | Seed | Status | Success | Death | Steps | Task s | Failure cause | Final event |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in summary["episodes"]:
        lines.append("| " + " | ".join(map(str, [e["run"], e["episode"], e["world_seed"], e["status"], e["success"], e["death"],
                                               e["steps"], _fmt(e["seconds"], 1), e["failure_cause"], e["final_event"]])) + " |")
    lines.append("")
    lines.append("## Reading these numbers")
    lines.append("")
    lines += [f"- {note}" for note in summary["notes"]]
    lines.append("")
    return "\n".join(lines)


def write_report(summary: dict, output: str | Path) -> Path:
    """Write summary.json, episodes.csv, and report.md into a new or empty folder."""
    out = prepare_output(output)
    write_json(out / "summary.json", summary)
    write_csv(out / "episodes.csv", [{**e, "events": json.dumps(e["events"], sort_keys=True)}
                                     for e in summary["episodes"]])
    for name in ("comparisons", "history_comparisons"):
        if summary.get(name):
            write_csv(out / f"{name}.csv", [{key: json.dumps(value) if isinstance(value, list) else value
                                             for key, value in row.items()} for row in summary[name]])
    (out / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    return out

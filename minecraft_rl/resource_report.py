"""Summaries of retained natural-resource runs for the report.

Reads the evidence folders written by ``infrastructure.resource_session`` and
produces honestly labeled counts: success rate with a Wilson interval,
decisions, task time, deaths, invalid episodes, and failure causes. Exploratory
training, frozen evaluation, and the masked random baseline are summarized as
separate groups and never pooled. Nothing here infers a result that the saved
outcome files do not contain.
"""

from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
from statistics import mean

from .artifacts import prepare_output, write_csv, write_json

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
    "policies against the masked random baseline with the same helpers before claiming a benefit.",
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
        try:
            number = int(folder.name.split("-", 1)[1])
        except (IndexError, ValueError):
            number = None
        steps = outcome.get("steps")
        if not isinstance(steps, int):
            steps = outcome.get("recorded_steps") if isinstance(outcome.get("recorded_steps"), int) else len(transitions)
        episodes.append({
            "run": run_directory.name, "episode": number,
            "world_seed": outcome.get("world_seed", transitions[0].get("world_seed") if transitions else None),
            "mode": metadata.get("mode"), "algorithm": metadata.get("algorithm"),
            "baseline": metadata.get("baseline"),
            "checkpoint_sha256": metadata.get("initial_policy_sha256"), "max_steps": metadata.get("max_steps"),
            "status": outcome.get("status"), "success": outcome.get("success"),
            "death": outcome.get("death"), "truncated": outcome.get("truncated"),
            "steps": steps, "seconds": outcome.get("seconds"), "return": outcome.get("return"),
            "failure_cause": classify_failure(outcome, transitions),
            "final_event": transitions[-1].get("after", {}).get("event") if transitions else None,
            "events": dict(events),
            "canopy_layers_cleared": sum(int(r.get("layers_cleared", 0) or 0) for r in descents),
            "recovery_damage": sum(float(r.get("damage", 0) or 0) for r in descents),
            "error": outcome.get("error"),
        })
    return {"run": run_directory.name, "path": str(run_directory), "metadata": metadata, "episodes": episodes}


def _mean(values) -> float | None:
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return mean(values) if values else None


def _group_key(episode: dict) -> tuple:
    # Training runs are pooled per algorithm; frozen evaluations are pooled only
    # per evaluated checkpoint. A different step limit is always a different group.
    checkpoint = episode["checkpoint_sha256"] if episode["mode"] == "frozen_evaluation" else None
    return (episode["mode"], episode["algorithm"], episode["baseline"] or "", checkpoint or "", episode["max_steps"])


def _summarize_group(key: tuple, rows: list[dict]) -> dict:
    mode, algorithm, baseline, checkpoint, max_steps = key
    completed = [r for r in rows if r["status"] == "completed"]
    successes = [r for r in completed if r["success"]]
    events: Counter[str] = Counter()
    for row in rows:
        events.update(row["events"])
    return {
        "mode": mode, "algorithm": algorithm, "baseline": baseline or None,
        "checkpoint_sha256": checkpoint or None, "max_steps": max_steps,
        "runs": sorted({r["run"] for r in rows}),
        "world_seeds": sorted({r["world_seed"] for r in rows if isinstance(r["world_seed"], int)}),
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
        "recovery_damage": sum(r["recovery_damage"] for r in rows),
    }


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
    return {
        "evidence": EVIDENCE,
        "runs": [{"run": run["run"], "path": run["path"],
                  "mode": run["metadata"].get("mode"), "algorithm": run["metadata"].get("algorithm"),
                  "baseline": run["metadata"].get("baseline"),
                  "initial_policy_sha256": run["metadata"].get("initial_policy_sha256"),
                  "episodes_requested": run["metadata"].get("episodes_requested"),
                  "first_world_seed": run["metadata"].get("first_world_seed"),
                  "max_steps": run["metadata"].get("max_steps"),
                  "source_sha256": run["metadata"].get("source_sha256"),
                  "episodes_found": len(run["episodes"])} for run in runs],
        "groups": [_summarize_group(key, rows) for key, rows in sorted(groups.items(), key=lambda item: str(item[0]))],
        "episodes": episodes,
        "notes": list(NOTES),
    }


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _label(group: dict) -> str:
    label = f"{group['mode']} / {group['algorithm']}"
    if group["baseline"]:
        label += f" / {group['baseline']}"
    if group.get("checkpoint_sha256"):
        label += f" / checkpoint {group['checkpoint_sha256'][:12]}"
    if group.get("max_steps") is not None:
        label += f" / max_steps {group['max_steps']}"
    return label


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
    for group in summary["groups"]:
        lines.append(f"## {_label(group)}")
        lines.append("")
        lines.append(f"Runs: {', '.join(group['runs'])}. World seeds: {', '.join(map(str, group['world_seeds'])) or 'none recorded'}.")
        if group.get("checkpoint_sha256"):
            lines.append(f"Evaluated checkpoint (policy.json SHA-256): {group['checkpoint_sha256']}.")
        lines.append(f"Mean {group['return_label']}: {_fmt(group['mean_return'], 3)}.")
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
    (out / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    return out

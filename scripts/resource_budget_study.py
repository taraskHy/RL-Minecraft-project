"""SIMULATOR ONLY: how many training episodes each resource condition needs.

Trains tabular Q-learning and SARSA in the abstract macro simulator under the
live schedule (epsilon 0.6 to 0.1 linearly over the training episodes,
60-decision limit, alpha 0.2, gamma 0.97), then evaluates each frozen greedy
policy against the condition's random baseline on fixed simulator seeds.
Nothing here is Minecraft evidence; it only informs the live episode budget.

Usage: python scripts/resource_budget_study.py --out results/resource_budget_study
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minecraft_rl.artifacts import prepare_output, provenance, write_csv, write_json  # noqa: E402
from minecraft_rl.resources import (  # noqa: E402
    CONDITIONS, ResourceAgent, ResourceSimConfig, ResourceSimulator, encode_state,
    state_from_observation, train_simulator,
)

LABEL = "SIMULATOR ONLY; NOT MINECRAFT EVIDENCE"
BUDGETS = (10, 20, 30, 50, 100, 300)
EVAL_EPISODES = 200
EVAL_SEED_BASE = 5_000_000


def evaluate(agent: ResourceAgent, condition: str, episodes: int, seed: int) -> dict:
    action_mask, state_version = CONDITIONS[condition]
    env = ResourceSimulator(ResourceSimConfig(action_mask=action_mask))
    successes, steps = 0, []
    for index in range(episodes):
        observation = env.reset(EVAL_SEED_BASE + seed * 10_000 + index)
        base = state_from_observation(observation)
        state = encode_state(base, state_version)
        while True:
            observation, _, terminated, truncated, info = env.step(agent.act(state, 0.0))
            following = state_from_observation(observation)
            state, base = encode_state(following, state_version, following == base), following
            if terminated or truncated:
                break
        successes += int(info["success"])
        steps.append(info["steps"])
    return {"success_rate": successes / episodes, "mean_steps": statistics.mean(steps)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20, help="Independent agent seeds per cell")
    args = parser.parse_args(argv)
    out = prepare_output(args.out)
    rows = []
    for condition in CONDITIONS:
        action_mask, state_version = CONDITIONS[condition]
        for seed in range(args.seeds):
            baseline = ResourceAgent("random", seed=seed, action_mask=action_mask, state_version=state_version)
            rows.append({"condition": condition, "algorithm": "random", "budget": 0, "seed": seed,
                         **evaluate(baseline, condition, EVAL_EPISODES, seed), "label": LABEL})
        for algorithm in ("q_learning", "sarsa"):
            for budget in BUDGETS:
                for seed in range(args.seeds):
                    agent, _ = train_simulator(budget, seed=seed, algorithm=algorithm, epsilon_start=0.6,
                                               epsilon_end=0.1, condition=condition, schedule="linear")
                    rows.append({"condition": condition, "algorithm": algorithm, "budget": budget, "seed": seed,
                                 **evaluate(agent, condition, EVAL_EPISODES, seed), "label": LABEL})
        print(f"{condition}: done", flush=True)
    write_csv(out / "cells.csv", rows)
    summary = []
    for condition in CONDITIONS:
        for algorithm in ("random", "q_learning", "sarsa"):
            for budget in ((0,) if algorithm == "random" else BUDGETS):
                cell = [r for r in rows if (r["condition"], r["algorithm"], r["budget"]) == (condition, algorithm, budget)]
                rates = [r["success_rate"] for r in cell]
                summary.append({"condition": condition, "algorithm": algorithm, "training_episodes": budget,
                                "agent_seeds": len(cell), "mean_success": statistics.mean(rates),
                                "min_success": min(rates), "max_success": max(rates),
                                "mean_steps": statistics.mean(r["mean_steps"] for r in cell)})
    # provenance() hashes minecraft_rl and configs; the script's own hash is added, and the label corrected.
    write_json(out / "summary.json", {"label": LABEL, "eval_episodes_per_agent": EVAL_EPISODES,
                                      "provenance": {**provenance(), "environment": "resource_macro_simulator",
                                                     "evidence": LABEL,
                                                     "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                                      "schedule": "epsilon 0.6 to 0.1 linear over training episodes; max 60 decisions",
                                      "simulator": "ResourceSimConfig defaults (search 0.75, chop 0.9, macro 0.95)",
                                      "rows": summary})
    lines = [f"# Resource budget study", "", f"**{LABEL}**", "",
             "| Condition | Policy | Training episodes | Mean success | Range over seeds | Mean decisions |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in summary:
        lines.append(f"| {row['condition']} | {row['algorithm']} | {row['training_episodes']} | {row['mean_success']:.2f} | "
                     f"{row['min_success']:.2f}–{row['max_success']:.2f} | {row['mean_steps']:.1f} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

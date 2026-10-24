"""SIMULATOR ONLY: how the initial Q value changes unmasked learning at the live budget.

Trains tabular Q-learning and SARSA in the abstract macro simulator, unmasked,
under the live schedule (epsilon 0.6 to 0.1 linearly over 50 training
episodes, 60-decision limit, alpha 0.2, gamma 0.97), from several initial Q
values, then evaluates each frozen greedy policy on fixed simulator seeds.
Besides success it counts "untried choices": frozen decisions that picked an
action never updated in that state, the mechanism behind the live loops at
87673b8. Nothing here is Minecraft evidence; it only informs the choice of a
pessimistic initial value for the live experiment.

Usage: python scripts/initial_q_study.py --out results/initial_q_study
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minecraft_rl.artifacts import prepare_output, provenance, write_csv, write_json  # noqa: E402
from minecraft_rl.resources import (  # noqa: E402
    CONDITIONS, ResourceAgent, ResourceSimConfig, ResourceSimulator, encode_state,
    state_from_observation, train_simulator,
)

LABEL = "SIMULATOR ONLY; NOT MINECRAFT EVIDENCE"
INITIAL_QS = (0.0, -0.1, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -5.0)
BUDGET = 50
CONDITION = "unmasked"
EVAL_EPISODES = 200
EVAL_SEED_BASE = 6_000_000


def evaluate(agent: ResourceAgent, episodes: int, seed: int, config: ResourceSimConfig) -> dict:
    action_mask, state_version = CONDITIONS[CONDITION]
    env = ResourceSimulator(config)
    successes, steps, decisions, untried = 0, [], 0, 0
    for index in range(episodes):
        observation = env.reset(EVAL_SEED_BASE + seed * 10_000 + index)
        base = state_from_observation(observation)
        state = encode_state(base, state_version)
        while True:
            action = agent.act(state, 0.0)
            row = agent.q.get(state)
            untried += int(row is None or row[action] == agent.initial_q)
            decisions += 1
            observation, _, terminated, truncated, info = env.step(action)
            following = state_from_observation(observation)
            state, base = encode_state(following, state_version, following == base), following
            if terminated or truncated:
                break
        successes += int(info["success"])
        steps.append(info["steps"])
    return {"success_rate": successes / episodes, "mean_steps": statistics.mean(steps),
            "untried_choice_fraction": untried / decisions}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20, help="Independent agent seeds per cell")
    parser.add_argument("--search-success", type=float, default=ResourceSimConfig.search_success,
                        help="Chance that SEARCH finds a log (lower is harsher, closer to the live failures)")
    parser.add_argument("--chop-success", type=float, default=ResourceSimConfig.chop_success)
    args = parser.parse_args(argv)
    out = prepare_output(args.out)
    config = ResourceSimConfig(action_mask=CONDITIONS[CONDITION][0], search_success=args.search_success,
                               chop_success=args.chop_success)
    rows = []
    for algorithm in ("q_learning", "sarsa"):
        for initial_q in INITIAL_QS:
            for seed in range(args.seeds):
                agent, records = train_simulator(BUDGET, seed=seed, algorithm=algorithm, epsilon_start=0.6,
                                                 epsilon_end=0.1, condition=CONDITION, schedule="linear",
                                                 initial_q=initial_q, config=config)
                rows.append({"algorithm": algorithm, "initial_q": initial_q, "seed": seed,
                             "training_successes": sum(r["success"] for r in records),
                             **evaluate(agent, EVAL_EPISODES, seed, config), "label": LABEL})
        print(f"{algorithm}: done", flush=True)
    write_csv(out / "cells.csv", rows)
    summary = []
    for algorithm in ("q_learning", "sarsa"):
        for initial_q in INITIAL_QS:
            cell = [r for r in rows if (r["algorithm"], r["initial_q"]) == (algorithm, initial_q)]
            rates = [r["success_rate"] for r in cell]
            summary.append({"algorithm": algorithm, "initial_q": initial_q, "agent_seeds": len(cell),
                            "mean_success": statistics.mean(rates), "min_success": min(rates),
                            "max_success": max(rates),
                            "seeds_below_0_9": sum(rate < 0.9 for rate in rates),
                            "mean_training_successes": statistics.mean(r["training_successes"] for r in cell),
                            "mean_steps": statistics.mean(r["mean_steps"] for r in cell),
                            "mean_untried_choice_fraction": statistics.mean(r["untried_choice_fraction"] for r in cell)})
    write_json(out / "summary.json", {"label": LABEL, "condition": CONDITION, "training_episodes": BUDGET,
                                      "eval_episodes_per_agent": EVAL_EPISODES,
                                      "provenance": {**provenance(), "environment": "resource_macro_simulator",
                                                     "evidence": LABEL,
                                                     "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                                      "schedule": "epsilon 0.6 to 0.1 linear over training episodes; max 60 decisions",
                                      "simulator": f"search {config.search_success}, chop {config.chop_success}, "
                                                   f"macro {config.macro_success}",
                                      "rows": summary})
    lines = ["# Initial-Q study", "", f"**{LABEL}**", "",
             f"Unmasked, {BUDGET} training episodes, {args.seeds} agent seeds per cell, {EVAL_EPISODES} frozen "
             f"evaluation episodes per agent; simulator search {config.search_success}, chop {config.chop_success}, "
             f"macro {config.macro_success}. An untried choice is a frozen decision that picked an action never "
             "updated in that state.", "",
             "| Policy | Initial Q | Mean success | Range over seeds | Seeds below 0.9 | Training successes (of 50) | Mean decisions | Untried choices |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in summary:
        lines.append(f"| {row['algorithm']} | {row['initial_q']:g} | {row['mean_success']:.3f} | "
                     f"{row['min_success']:.2f}–{row['max_success']:.2f} | {row['seeds_below_0_9']} | "
                     f"{row['mean_training_successes']:.1f} | {row['mean_steps']:.1f} | "
                     f"{100 * row['mean_untried_choice_fraction']:.1f}% |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

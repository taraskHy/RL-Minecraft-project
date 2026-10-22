"""Select hyperparameters on validation only, then test every selected seed."""

from dataclasses import asdict
from itertools import product
from pathlib import Path
from time import perf_counter

from .agents import TabularAgent
from .artifacts import learning_curve, prepare_output, provenance, write_csv, write_json
from .environment import EnvConfig
from .training import TrainingConfig, aggregate_seed_summaries, evaluate, summarize, train


def run_experiment(spec: dict, output: str | Path) -> dict:
    required = {"episodes", "training_seeds", "algorithms", "reward_modes", "alphas",
                "gammas", "decay_fractions", "validation_episodes", "test_episodes",
                "max_steps", "slip_probability", "validation_seed_start", "test_seed_start"}
    if set(spec) != required:
        raise ValueError(f"Config keys must be exactly: {sorted(required)}")
    for key in ("episodes", "validation_episodes", "test_episodes", "max_steps"):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("training_seeds", "algorithms", "reward_modes", "alphas", "gammas", "decay_fractions"):
        if not isinstance(spec[key], list) or not spec[key] or len(set(spec[key])) != len(spec[key]):
            raise ValueError(f"{key} must be a nonempty list without duplicates")
    if not all(type(s) is int for s in spec["training_seeds"]):
        raise ValueError("training_seeds must be integers")
    if max(spec["validation_episodes"], spec["test_episodes"]) * len(spec["training_seeds"]) >= 1_000_000:
        raise ValueError("Evaluation seed ranges would overlap")
    count = len(spec["training_seeds"])
    for key in ("validation_seed_start", "test_seed_start"):
        if type(spec[key]) is not int or spec[key] < 1_000_000:
            raise ValueError(f"{key} must be an integer >= 1000000, outside training seeds")
    val_start, test_start = spec["validation_seed_start"], spec["test_seed_start"]
    val_end = val_start + count * spec["validation_episodes"]
    test_end = test_start + count * spec["test_episodes"]
    if max(val_start, test_start) < min(val_end, test_end):
        raise ValueError("Validation and test seed ranges overlap")
    configs = []
    for algorithm, reward, alpha, gamma, decay in product(
            spec["algorithms"], spec["reward_modes"], spec["alphas"],
            spec["gammas"], spec["decay_fractions"]):
        tc = TrainingConfig(algorithm=algorithm, episodes=spec["episodes"], alpha=alpha,
                            gamma=gamma, decay_fraction=decay)
        ec = EnvConfig(reward_mode=reward, gamma=gamma, max_steps=spec["max_steps"],
                       slip_probability=spec["slip_probability"])
        configs.append((tc, ec))
    out = prepare_output(output)
    write_json(out / "config.json", spec)
    write_json(out / "provenance.json", provenance())
    candidates = []
    for index, (tc, ec) in enumerate(configs):
        validations = []
        paths = []
        for seed_index, seed in enumerate(spec["training_seeds"]):
            run_tc = TrainingConfig(**{**asdict(tc), "seed": seed})
            run = out / f"candidate_{index:03d}" / f"seed_{seed}"
            run.mkdir(parents=True)
            started = perf_counter()
            agent, records = train(run_tc, ec)
            training_seconds = perf_counter() - started
            agent.save(run / "policy.json", {"training": asdict(run_tc), "env": asdict(ec)})
            write_csv(run / "training.csv", records)
            learning_curve(run / "learning_curve.svg", records, f"{tc.algorithm} / {ec.reward_mode}")
            write_json(run / "run_metadata.json", {"training": asdict(run_tc), "env": asdict(ec),
                       "training_seconds": training_seconds, "total_training_steps": records[-1]["total_steps"]})
            val_seeds = [spec["validation_seed_start"] + seed_index * spec["validation_episodes"] + i
                         for i in range(spec["validation_episodes"])]
            val = evaluate(agent, ec, val_seeds)
            write_csv(run / "validation.csv", val)
            validations.append(summarize(val))
            paths.append(str(run.relative_to(out)))
        candidate = {"candidate": index, "training": asdict(tc), "env": asdict(ec),
                     "runs": paths, "validation_by_seed": validations,
                     "validation": aggregate_seed_summaries(validations)}
        candidates.append(candidate)
        print(f"Validated {index+1}/{len(configs)}: {tc.algorithm} {ec.reward_mode}, "
              f"alpha={tc.alpha}, gamma={tc.gamma}, decay={tc.decay_fraction}, "
              f"success={candidate['validation']['mean_success_rate']:.1%}", flush=True)
    write_json(out / "validation_summary.json", candidates)

    # One configuration per algorithm/reward pair. Never select the best random seed.
    selected = []
    for algorithm, reward in product(spec["algorithms"], spec["reward_modes"]):
        group = [c for c in candidates if c["training"]["algorithm"] == algorithm
                 and c["env"]["reward_mode"] == reward]
        best = max(group, key=lambda c: (c["validation"]["mean_success_rate"],
                                        c["validation"]["mean_base_return"], -c["candidate"]))
        selected.append(best)
    # Write the frozen selection BEFORE accessing test episodes.
    write_json(out / "selected_configs.json", selected)
    final = []
    for candidate in selected:
        summaries = []
        for seed_index, run_path in enumerate(candidate["runs"]):
            run = out / run_path
            agent = TabularAgent.load(run / "policy.json")
            test_seeds = [spec["test_seed_start"] + seed_index * spec["test_episodes"] + i
                          for i in range(spec["test_episodes"])]
            rows = evaluate(agent, EnvConfig(**candidate["env"]), test_seeds)
            write_csv(run / "test.csv", rows)
            summaries.append(summarize(rows))
        final.append({"algorithm": candidate["training"]["algorithm"],
                      "reward_mode": candidate["env"]["reward_mode"],
                      "candidate": candidate["candidate"],
                      "selected_training": candidate["training"],
                      "policy_paths": [str(Path(p) / "policy.json") for p in candidate["runs"]],
                      "test_by_training_seed": summaries,
                      "test": aggregate_seed_summaries(summaries)})
    random_env = EnvConfig(reward_mode="sparse", gamma=spec["gammas"][0],
                           max_steps=spec["max_steps"], slip_probability=spec["slip_probability"])
    random_seeds = [spec["test_seed_start"] + i for i in range(spec["test_episodes"] * len(spec["training_seeds"]))]
    random_rows = evaluate(TabularAgent("random"), random_env, random_seeds)
    write_csv(out / "random_test.csv", random_rows)
    result = {
        "evidence": "SIMULATOR ONLY; not real Minecraft or an Ender Dragon result",
        "evaluation": "Held-out episode RNG seeds on the SAME fixed map; not unseen-world generalization",
        "selection": "Validation success, then base return; all seeds retained; final test after selection",
        "comparison_metric": "success and base return; shaped reward returns are not comparable across conditions",
        "methods": final, "random_baseline": summarize(random_rows),
    }
    write_json(out / "summary.json", result)
    table = [{"algorithm": r["algorithm"], "reward_mode": r["reward_mode"], **r["test"]} for r in final]
    write_csv(out / "comparison.csv", table)
    return result

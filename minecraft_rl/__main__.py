"""Run with python -m minecraft_rl --help from the repository root."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .agents import TabularAgent
from .artifacts import learning_curve, prepare_output, provenance, write_csv, write_json
from .environment import ACTION_NAMES, CraftingEnv, EnvConfig
from .experiments import run_experiment
from .training import TrainingConfig, evaluate, summarize, train


def parser():
    p = argparse.ArgumentParser(description="Minecraft RL course foundation (simulator + optional development arena)")
    sub = p.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train", help="Train a tabular crafting policy in the simulator")
    training.add_argument("--algorithm", choices=["q_learning", "sarsa"], default="q_learning")
    training.add_argument("--episodes", type=int, default=15000)
    training.add_argument("--alpha", type=float, default=0.2)
    training.add_argument("--gamma", type=float, default=0.97)
    training.add_argument("--reward", choices=["sparse", "shaped"], default="shaped")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--slip", type=float, default=0.05)
    training.add_argument("--max-steps", type=int, default=100)
    training.add_argument("--out", default="runs/first_training")
    evaluation = sub.add_parser("evaluate", help="Evaluate a frozen saved policy without learning")
    evaluation.add_argument("--policy", required=True)
    evaluation.add_argument("--episodes", type=int, default=200)
    evaluation.add_argument("--seed-start", type=int, default=1_000_000)
    evaluation.add_argument("--out", default="runs/evaluation")
    demo = sub.add_parser("demo", help="Print one simulator trajectory")
    demo.add_argument("--policy", required=True)
    demo.add_argument("--seed", type=int, default=1_000_000)
    experiment = sub.add_parser("experiment", help="Compare algorithms, rewards, and hyperparameters")
    experiment.add_argument("--config", default="configs/smoke.json")
    experiment.add_argument("--out", default="runs/smoke")
    live = sub.add_parser("live", help="DEVELOPMENT ARENA ONLY: run a frozen policy in local Minecraft")
    live.add_argument("--policy", required=True)
    live.add_argument("--dev-arena", action="store_true", help="Allow replacing the documented arena region and resetting the bot")
    live.add_argument("--host", default="127.0.0.1")
    live.add_argument("--port", type=int, default=25565)
    live.add_argument("--username", default="RLCourseBot")
    live.add_argument("--seed", type=int, default=1_000_000)
    live.add_argument("--origin", nargs=3, type=int, default=(0, 64, 0), metavar=("X", "Y", "Z"))
    live.add_argument("--out", default="runs/live_arena")
    return p


def load_env_config(policy_path):
    payload = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    return EnvConfig(**payload.get("metadata", {}).get("env", {}))


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "train":
            tc = TrainingConfig(algorithm=args.algorithm, episodes=args.episodes, alpha=args.alpha,
                                gamma=args.gamma, seed=args.seed)
            ec = EnvConfig(reward_mode=args.reward, gamma=args.gamma,
                           slip_probability=args.slip, max_steps=args.max_steps)
            out = prepare_output(args.out)
            write_json(out / "metadata.json", {**provenance(), "training": asdict(tc), "env": asdict(ec)})
            agent, rows = train(tc, ec)
            agent.save(out / "policy.json", {"training": asdict(tc), "env": asdict(ec)})
            write_csv(out / "training.csv", rows)
            learning_curve(out / "learning_curve.svg", rows, f"{tc.algorithm} / {ec.reward_mode}")
            evaluation = evaluate(agent, ec, list(range(1_000_000, 1_000_200)))
            write_csv(out / "validation.csv", evaluation)
            summary = {"evidence": "SIMULATOR ONLY", "split": "validation preview, not a final test",
                       **summarize(evaluation)}
            write_json(out / "summary.json", summary)
            print(json.dumps(summary, indent=2))
        elif args.command == "evaluate":
            if args.episodes <= 0:
                raise ValueError("episodes must be positive")
            agent = TabularAgent.load(args.policy)
            ec = load_env_config(args.policy)
            out = prepare_output(args.out)
            rows = evaluate(agent, ec, list(range(args.seed_start, args.seed_start + args.episodes)))
            write_csv(out / "evaluation.csv", rows)
            summary = {"evidence": "SIMULATOR ONLY", **summarize(rows)}
            write_json(out / "summary.json", summary)
            write_json(out / "metadata.json", {**provenance(), "policy": str(Path(args.policy).resolve()),
                                               "env": asdict(ec), "seed_start": args.seed_start})
            print(json.dumps(summary, indent=2))
        elif args.command == "demo":
            agent = TabularAgent.load(args.policy, seed=args.seed + 4817)
            env = CraftingEnv(load_env_config(args.policy))
            state = env.reset(seed=args.seed)
            print("SIMULATOR ONLY: wooden pickaxe milestone\n" + env.render())
            while True:
                action = agent.act(state)
                state, reward, terminated, truncated, info = env.step(action)
                print(f"{info['steps']:3d} {ACTION_NAMES[action]:15s} reward={reward:+.3f} "
                      f"state={state} event={info['event']}")
                if terminated or truncated:
                    print(env.render())
                    print(f"Success={info['success']}; death={info['death']}; timeout={truncated}")
                    break
        elif args.command == "experiment":
            spec = json.loads(Path(args.config).read_text(encoding="utf-8"))
            summary = run_experiment(spec, args.out)
            print(json.dumps(summary, indent=2))
        elif args.command == "live":
            if not args.dev_arena:
                raise ValueError("Live support is a DEVELOPMENT ARENA, not fresh survival. Read minecraft_bridge/README.md, then explicitly pass --dev-arena.")
            from .live import MinecraftEnv
            agent = TabularAgent.load(args.policy, seed=args.seed + 4817)
            ec = load_env_config(args.policy)
            out = prepare_output(args.out)
            write_json(out / "metadata.json", {"evidence": "REAL GAME DEVELOPMENT ARENA; NOT FRESH SURVIVAL",
                                               "env": asdict(ec), "seed": args.seed, "policy": args.policy})
            rows = []
            with MinecraftEnv(ec, host=args.host, port=args.port, username=args.username,
                              dev_arena=True, origin=tuple(args.origin)) as env:
                state = env.reset(seed=args.seed)
                while True:
                    action = agent.act(state)
                    state, reward, terminated, truncated, info = env.step(action)
                    rows.append({"step": info["steps"], "action": ACTION_NAMES[action],
                                 "state": str(state), "reward": reward, "event": info["event"],
                                 "success": info["success"], "death": info["death"]})
                    write_csv(out / "trajectory.csv", rows)
                    print(rows[-1], flush=True)
                    if terminated or truncated:
                        write_json(out / "outcome.json", {"terminated": terminated, "truncated": truncated, **info})
                        break
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        p.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())

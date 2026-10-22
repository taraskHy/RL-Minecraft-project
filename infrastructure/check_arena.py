r"""Opt-in real-game integration check; scripted diagnostics are not a policy.

Run: .venv\Scripts\python.exe -m infrastructure.check_arena --dev-arena
Starts and stops only this project's disposable arena server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .__main__ import DEFAULT_POLICY, ROOT, new_output
from .runtime import runtime_paths
from .server import running_server, server_memory_mb
from minecraft_rl.__main__ import load_env_config
from minecraft_rl.agents import TabularAgent
from minecraft_rl.artifacts import provenance
from minecraft_rl.environment import Action, CraftingEnv
from minecraft_rl.live import MinecraftEnv


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-arena", action="store_true", required=True)
    args = parser.parse_args()
    output = Path(new_output("arena-integration"))
    output.mkdir(parents=True)
    source_record = provenance()
    source_record.update(environment="minecraft-java-development-arena", evidence="REAL MINECRAFT DEVELOPMENT ARENA ONLY")
    for file in [Path(__file__), ROOT / "infrastructure" / "server.py", ROOT / "minecraft_bridge" / "bridge.js", DEFAULT_POLICY]:
        source_record["source_sha256"][file.relative_to(ROOT).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    report = {"evidence": "REAL MINECRAFT DEVELOPMENT ARENA ONLY",
              "scripted_checks_are_not_learned_behavior": True,
              "server_memory_mb": server_memory_mb(), "checks": [], "passed": False,
              "provenance": source_record}
    report_path = output / "checks.json"
    print(f"Integration results: {report_path}", flush=True)

    def record(name, **data):
        report["checks"].append({"check": name, **data})
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(name, data, flush=True)

    config = load_env_config(DEFAULT_POLICY)
    simulator = CraftingEnv(config)
    # Choose a diagnostic start adjacent to the log and lava. This is not a
    # held-out evaluation seed or a trained strategy.
    diagnostic_seed = next(seed for seed in range(1000) if simulator.reset(seed=seed)[:2] == (2, 2))
    node = runtime_paths()["node"]
    require(node is not None, "Run setup.cmd to install Node first")
    try:
        with running_server("arena", mirror_output=True):
            with MinecraftEnv(config, node=str(node), dev_arena=args.dev_arena) as env:
                initial = env.reset(seed=diagnostic_seed)
                record("reset", seed=diagnostic_seed, state=initial)
                state, _, terminated, truncated, info = env.step(Action.EAST)
                require(state == initial and info["event"] == "blocked" and not terminated and not truncated,
                        "An unbroken log did not block movement")
                record("unbroken_log_blocks_movement", observation=info)
                state, _, terminated, truncated, info = env.step(Action.SOUTH)
                require(terminated and not truncated and info["death"] and info["health"] <= 0,
                        "Lava did not produce an observed terminal death")
                record("real_lava_death", observation=info)
                state = env.reset(seed=1_000_000)
                require(state[2:] == (3, 0, 0, 0), "Reset after death did not restore initial resources")
                record("reset_after_death", state=state)
                policy = TabularAgent.load(DEFAULT_POLICY, seed=1_004_817)
                while True:
                    action = policy.act(state)
                    state, reward, terminated, truncated, info = env.step(action)
                    record("frozen_policy_step", action=int(action), reward=reward, observation=info)
                    if terminated or truncated:
                        require(info["success"] and info["inventory"].get("wooden_pickaxe", 0) >= 1,
                                "Frozen policy did not produce an observed wooden pickaxe")
                        break
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Integration checks passed; the dedicated server saved and stopped.", flush=True)


if __name__ == "__main__":
    main()

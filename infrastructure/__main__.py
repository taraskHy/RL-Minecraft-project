"""Local PC launcher. Learning code stays in minecraft_rl; setup stays here."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "results" / "baseline" / "demo_policy.json"
EULA_URL = "https://www.minecraft.net/en-us/eula"


def local_environment() -> dict[str, str]:
    """Use installed project runtimes even from a plain Windows terminal."""
    env = os.environ.copy()
    node_dir = ROOT / ".local" / "runtime" / "node"
    java_dir = ROOT / ".local" / "runtime" / "java"
    env["PATH"] = os.pathsep.join((str(node_dir), str(java_dir / "bin"), env.get("PATH", "")))
    env["JAVA_HOME"] = str(java_dir)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run(command: list[str | Path]) -> int:
    return subprocess.call([str(value) for value in command], cwd=ROOT, env=local_environment())


def new_output(label: str) -> str:
    # Microseconds avoid collisions when commands are repeated rapidly.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return str(ROOT / "runs" / f"{label}-{stamp}")


def has_option(arguments: list[str], name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in arguments)


def learning_command(command: str, arguments: list[str]) -> int:
    values = list(arguments)
    if command == "demo" and not has_option(values, "--policy"):
        values += ["--policy", str(DEFAULT_POLICY)]
    if command in ("train", "evaluate", "experiment") and not has_option(values, "--out"):
        output = new_output(command)
        values += ["--out", output]
        print(f"Results: {output}", flush=True)
    if command == "evaluate" and not has_option(values, "--policy"):
        values += ["--policy", str(DEFAULT_POLICY)]
    print("Backend: crafting SIMULATOR (CPU).", flush=True)
    return run([sys.executable, "-m", "minecraft_rl", command, *values])


def doctor() -> int:
    from .runtime import describe_runtime, runtime_paths
    from .server import eula_accepted, server_memory_mb

    print(f"Project: {ROOT}\nPython: {sys.version.split()[0]} ({sys.executable})")
    print("Simulator: available; it uses the CPU and needs no external Python packages.")
    print(f"Minecraft server Java heap: 512 MB initial / {server_memory_mb()} MB maximum.")
    paths = runtime_paths()
    missing = []
    for name, details in describe_runtime().items():
        print(f"{name}: {details['path']}")
        print(f"  {details['error'] or details['version']}")
        if details["error"]:
            missing.append(name)
    if not missing:
        print("Checking installed Minecraft bridge imports...")
        if run([paths["node"], ROOT / "minecraft_bridge" / "runtime-smoke.js"]):
            missing.append("bridge dependencies")
    print(f"Minecraft EULA decision saved: {'yes' if eula_accepted() else 'no (run.cmd accept-eula)'}")
    gpu = shutil.which("nvidia-smi", path=local_environment()["PATH"])
    if gpu:
        result = subprocess.run([gpu, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=15)
        print("NVIDIA GPU: " + (result.stdout.strip() or result.stderr.strip()))
    else:
        print("NVIDIA GPU: nvidia-smi not found. This does not block the CPU simulator or Java server.")
    print("Full Survival agent / Ender Dragon policy: not implemented. Netherite/CUDA: not installed.")
    print("Doctor checks dependencies; it does not claim a successful game episode.")
    return 1 if missing else 0


def checks() -> int:
    from .runtime import runtime_paths

    commands = [[sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                [sys.executable, "-m", "unittest", "discover", "-s", "minecraft_bridge", "-p", "test_live.py", "-v"]]
    node = runtime_paths().get("node")
    if node and Path(node).is_file():
        commands += [[node, "--test", "minecraft_bridge/bridge.test.js", "minecraft_bridge/survival-probe.test.js", "minecraft_bridge/resource-bridge.test.js"],
                     [node, "--check", "minecraft_bridge/resource-bridge.js"],
                     [node, "--check", "minecraft_bridge/survival-probe.js"],
                     [node, "minecraft_bridge/runtime-smoke.js"]]
    else:
        print("Node checks require setup.cmd. Running Python checks only.", flush=True)
    for command in commands:
        result = run(command)
        if result:
            return result
    print("All available checks passed. Game connection requires a separate live run.")
    return 0


def setup() -> int:
    from .runtime import setup_runtime
    from .server import prepare_server

    setup_runtime()
    for profile in ("arena", "survival"):
        prepare_server(profile)
    print("\nSetup complete. Minecraft has not been started.")
    print("Run run.cmd demo for the simulator, or run.cmd accept-eula before a Minecraft server.")
    return doctor()


def accept_agreement(arguments: list[str]) -> int:
    p = argparse.ArgumentParser(prog="run.cmd accept-eula", description="Record your Minecraft server EULA decision locally.")
    p.add_argument("--agree", action="store_true", help="Explicitly confirm that you read and agree to the Minecraft EULA")
    args = p.parse_args(arguments)
    print(f"Minecraft server EULA: {EULA_URL}")
    if not args.agree:
        try:
            answer = input("After reading it, type I AGREE to accept, or press Enter to cancel: ").strip()
        except EOFError:
            answer = ""
        if answer != "I AGREE":
            print("No agreement recorded. The simulator remains available.")
            return 1
    from .server import accept_eula
    accept_eula()
    print("Your acceptance was recorded locally. You can now start the project servers.")
    return 0


def arena(arguments: list[str]) -> int:
    p = argparse.ArgumentParser(prog="run.cmd arena", description="Start the dedicated arena server, run one saved policy, save results, then stop the server.")
    p.add_argument("--dev-arena", action="store_true", help="Allow resetting the dedicated development world; required")
    p.add_argument("--policy", default=str(DEFAULT_POLICY))
    p.add_argument("--seed", type=int, default=1_000_000)
    p.add_argument("--out")
    args = p.parse_args(arguments)
    if not args.dev_arena:
        p.error("pass --dev-arena to enable the documented development-world resets (see minecraft_bridge/README.md)")
    if not Path(args.policy).is_file():
        p.error(f"policy does not exist: {args.policy}")
    from .server import running_server
    output = args.out or new_output("minecraft-arena")
    print("REAL MINECRAFT: wooden-pickaxe DEVELOPMENT ARENA, not fresh-world Survival.", flush=True)
    print(f"Results: {output}", flush=True)
    with running_server("arena", mirror_output=True) as server:
        print(f"Minecraft Java 1.21.4 client can join 127.0.0.1:{server['port']} while this runs.", flush=True)
        return run([sys.executable, "-m", "minecraft_rl", "live", "--dev-arena", "--policy", args.policy,
                    "--port", str(server["port"]), "--seed", str(args.seed), "--out", output])


def survival_check(arguments: list[str]) -> int:
    p = argparse.ArgumentParser(prog="run.cmd survival-check", description="Verify a real bot connection to the separate ordinary Survival world; no learned policy or game commands.")
    p.add_argument("--seconds", type=int, default=10)
    args = p.parse_args(arguments)
    if not 1 <= args.seconds <= 120:
        p.error("--seconds must be between 1 and 120")
    from .runtime import runtime_paths
    from .server import running_server
    node = runtime_paths().get("node")
    if not node or not Path(node).is_file():
        raise RuntimeError("Run setup.cmd first to install Node and bridge dependencies.")
    output = Path(new_output("survival-connection"))
    print("SURVIVAL CONNECTION CHECK: the bot will observe its spawn without moving or issuing commands.", flush=True)
    with running_server("survival", mirror_output=True) as server:
        output.mkdir(parents=True)
        print(f"Observations: {output / 'observations.jsonl'}", flush=True)
        with (output / "observations.jsonl").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run([str(node), str(ROOT / "minecraft_bridge" / "survival-probe.js"),
                                          str(server["port"]), str(args.seconds)], cwd=ROOT,
                                         env=local_environment(), stdout=log, timeout=args.seconds + 60)
                result = process.returncode
            except subprocess.TimeoutExpired:
                result = 2
                message = "Survival probe exceeded its deadline and was stopped."
                log.write(json.dumps({"event": "error", "error": message}) + "\n")
                print(message, file=sys.stderr)
        print((output / "observations.jsonl").read_text(encoding="utf-8"), end="", flush=True)
        (output / "result.json").write_text(json.dumps({"exit_code": result,
            "evidence": "Minecraft Survival connection only; no learned behavior or dragon victory",
            "world_reused_if_present": True}, indent=2) + "\n", encoding="utf-8")
        return result


def watch(arguments: list[str]) -> int:
    p = argparse.ArgumentParser(prog="run.cmd watch", description="Wait for you to join the arena, place you in spectator mode, and run the saved crafting policy.")
    p.add_argument("--dev-arena", action="store_true", help="Allow resetting the dedicated development world; required")
    p.add_argument("--viewer", required=True, help="Your Minecraft Java player name")
    p.add_argument("--policy", default=str(DEFAULT_POLICY))
    p.add_argument("--seed", type=int, default=1_000_000)
    p.add_argument("--join-timeout", type=float, default=300, help="Seconds to wait for your connection, up to 600")
    p.add_argument("--once", action="store_true", help="Stop the server after one episode instead of offering replay")
    args = p.parse_args(arguments)
    if not args.dev_arena:
        p.error("pass --dev-arena to enable resets in the dedicated development world")
    from .watch import run_watch
    return run_watch(viewer=args.viewer, policy=Path(args.policy), seed=args.seed,
                     join_timeout=args.join_timeout, once=args.once)


def resources(arguments: list[str], *, evaluate: bool = False) -> int:
    name = "resource-demo" if evaluate else "resource-train"
    p = argparse.ArgumentParser(prog=f"run.cmd {name}", description=
        "Watch a frozen policy in a fresh natural world." if evaluate else
        "Learn Q-learning/SARSA from actual Minecraft transitions in fresh natural worlds.")
    p.add_argument("--viewer", help="Your Java player name; waits for you before each episode")
    p.add_argument("--episodes", type=int, default=1 if evaluate else 5)
    p.add_argument("--world-seed", type=int, default=42, help="First world seed; each episode increments it")
    p.add_argument("--seed", type=int, default=0, help="Learner random seed")
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--policy", type=Path, required=evaluate, help="Resource checkpoint to resume or evaluate; arena policies are incompatible")
    p.add_argument("--out", type=Path, help="New output folder inside runs/")
    p.add_argument("--pause", type=float, default=.5, help="Extra seconds between decisions")
    if not evaluate:
        p.add_argument("--algorithm", choices=("q_learning", "sarsa"), default="q_learning")
        p.add_argument("--epsilon", type=float, default=.6, help="Initial exploration probability")
        p.add_argument("--epsilon-end", type=float, default=.1)
        p.add_argument("--pretrain", type=int, default=0, help="Optional explicitly labeled simulator warm-up episodes; default learns from scratch")
    args = p.parse_args(arguments)
    from .resource_session import run_resources
    return run_resources(output=args.out or Path(new_output(name)), episodes=args.episodes,
                         world_seed=args.world_seed, seed=args.seed, viewer=args.viewer,
                         max_steps=args.max_steps, policy=args.policy, evaluate=evaluate, pause=args.pause,
                         algorithm=getattr(args, "algorithm", "q_learning"),
                         epsilon=getattr(args, "epsilon", 0), epsilon_end=getattr(args, "epsilon_end", 0),
                         pretrain=getattr(args, "pretrain", 0))


def menu() -> int:
    options = {
        "1": ["setup"], "2": ["demo"], "3": ["train"], "4": ["experiment"],
        "5": ["accept-eula"], "6": ["arena", "--dev-arena"],
        "7": ["survival-check"], "8": ["server", "survival"],
        "9": ["doctor"], "10": ["test"],
    }
    while True:
        print("\nMinecraft RL course project\n"
              "  1  Set up local dependencies\n"
              "  2  Watch the saved SIMULATOR policy (text)\n"
              "  3  Train a new SIMULATOR policy\n"
              "  4  Run the small SIMULATOR comparison\n"
              "  5  Read / accept the Minecraft server EULA\n"
              "  6  Run Minecraft DEVELOPMENT ARENA (resets its test world)\n"
              "  7  Check real Survival bot connection (no learned behavior)\n"
              "  8  Start ordinary Survival server; Ctrl+C saves and stops\n"
              "  9  Check this PC\n"
              " 10  Run tests\n"
              " 11  Watch the crafting bot in Minecraft (spectator)\n"
              " 12  Watch actual resource training in fresh Survival worlds\n"
              "  0  Exit")
        try:
            choice = input("Choose: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if choice == "0":
            return 0
        if choice in ("11", "12"):
            try:
                viewer = input("Your Minecraft Java player name: ").strip()
            except (EOFError, KeyboardInterrupt):
                return 0
            status = main(["watch", "--dev-arena", "--viewer", viewer] if choice == "11" else
                          ["resource-train", "--viewer", viewer])
            print(f"\nCommand finished with exit code {status}.")
            continue
        if choice not in options:
            print("Choose a number from the menu.")
            continue
        status = main(options[choice])
        print(f"\nCommand finished with exit code {status}.")


def main(argv: list[str] | None = None) -> int:
    # Launchers work from any directory. Relative command arguments are project-relative.
    os.chdir(ROOT)
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        return menu()
    command, arguments = values[0], values[1:]
    try:
        if command in ("train", "demo", "experiment", "evaluate"):
            return learning_command(command, arguments)
        if command == "accept-eula":
            return accept_agreement(arguments)
        if command == "arena":
            return arena(arguments)
        if command == "survival-check":
            return survival_check(arguments)
        if command == "watch":
            return watch(arguments)
        if command in ("resource-train", "resource-demo"):
            return resources(arguments, evaluate=command == "resource-demo")
        if command == "server":
            p = argparse.ArgumentParser(prog="run.cmd server")
            p.add_argument("profile", choices=("arena", "survival"))
            args = p.parse_args(arguments)
            from .server import run_server
            return run_server(args.profile)
        if arguments:
            raise ValueError(f"Unexpected arguments for {command}: {' '.join(arguments)}")
        if command == "setup":
            return setup()
        if command == "doctor":
            return doctor()
        if command == "test":
            return checks()
        if command in ("help", "--help", "-h"):
            print("Usage: run.cmd [command] [options]\n"
                  "  setup                       Install project-local runtimes and server\n"
                  "  doctor / test               Check dependencies / run automated checks\n"
                  "  train / demo / evaluate     CPU simulator; forwards minecraft_rl options\n"
                  "  experiment                  CPU simulator comparison; accepts --config\n"
                  "  accept-eula                 Record your Minecraft EULA decision\n"
                  "  arena --dev-arena           Run saved policy in a managed test world\n"
                  "  watch --dev-arena --viewer NAME  Wait for you to join and watch the bot\n"
                  "  resource-train --viewer NAME    Watch actual training in fresh natural worlds\n"
                  "  resource-demo --policy PATH     Evaluate a frozen resource policy in a fresh world\n"
                  "  survival-check              Check bot connection to ordinary Survival\n"
                  "  server arena|survival       Keep a server running until Ctrl+C\n"
                  "Without a command, opens the menu. Read docs/windows_setup.md.")
            return 0
        raise ValueError(f"Unknown command: {command}. Run run.cmd help.")
    except KeyboardInterrupt:
        print("\nInterrupted. Any managed server is being stopped.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

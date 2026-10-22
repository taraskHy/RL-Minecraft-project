"""Watch a saved policy in the disposable Minecraft crafting arena.

The viewer uses spectator mode. Only the arena server owned by this command is
changed; the separate ordinary Survival world is never opened here.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from minecraft_rl.__main__ import load_env_config
from minecraft_rl.agents import TabularAgent
from minecraft_rl.artifacts import prepare_output, write_csv, write_json
from minecraft_rl.environment import ACTION_NAMES
from minecraft_rl.live import MinecraftEnv

from .runtime import runtime_paths
from .server import running_server


ROOT = Path(__file__).resolve().parents[1]
BOT_NAME = "RLCourseBot"
EVIDENCE = "REAL MINECRAFT DEVELOPMENT ARENA; NOT FRESH-WORLD SURVIVAL"


def validate_watch_options(viewer: str, join_timeout: float) -> None:
    if (not isinstance(viewer, str)
            or re.fullmatch(r"[A-Za-z0-9_]{1,16}", viewer) is None
            or viewer.casefold() == BOT_NAME.casefold()):
        raise ValueError("Viewer must be your Minecraft username (1-16 letters, digits or underscores), and cannot be RLCourseBot")
    if (isinstance(join_timeout, bool) or not isinstance(join_timeout, (int, float))
            or not math.isfinite(join_timeout) or not 1 <= join_timeout <= 600):
        raise ValueError("join_timeout must be a finite number from 1 to 600 seconds")


def _check_server(server: dict[str, Any]) -> None:
    if server["process"].poll() is not None:
        raise RuntimeError(f"The arena server stopped. Read {server['log_file']}")


def _require_viewer(env: MinecraftEnv, server: dict[str, Any], viewer: str,
                    *, spectator: bool = True) -> None:
    _check_server(server)
    player = env.players().get(viewer)
    if player is None:
        raise RuntimeError(f"Viewer {viewer} disconnected; the watched episode was stopped")
    if spectator and player.get("gamemode") != 3:
        raise RuntimeError(f"Viewer {viewer} is no longer in spectator mode; the watched episode was stopped")


def _wait_for_viewer(env: MinecraftEnv, server: dict[str, Any], viewer: str,
                     timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        _check_server(server)
        if viewer in env.players():
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Viewer {viewer} did not join the arena within {timeout:g} seconds; no policy actions ran")
        time.sleep(0.25)


def _position_viewer(env: MinecraftEnv, server: dict[str, Any], viewer: str) -> None:
    """Check server evidence for both spectator mode and this teleport."""
    _require_viewer(env, server, viewer, spectator=False)
    log_file = Path(server["log_file"])
    offset = log_file.stat().st_size
    process = server["process"]
    if process.stdin is None:
        raise RuntimeError("The managed arena server has no command input")
    process.stdin.write(
        f"gamemode spectator {viewer}\n"
        f"effect give {viewer} minecraft:night_vision 600 0 true\n"
        f"tp {viewer} 2.5 66 4.5 180 50\n")
    process.stdin.flush()
    acknowledgement = re.compile(r"\bTeleported " + re.escape(viewer) + r" to ")
    deadline = time.monotonic() + 15
    while True:
        _check_server(server)
        player = env.players().get(viewer)
        if player is None:
            raise RuntimeError(f"Viewer {viewer} disconnected while the viewing position was being prepared")
        # Ignore acknowledgements from earlier episodes. Player entities can be
        # absent for spectators, so their position is not inferred from one.
        with log_file.open("rb") as log:
            log.seek(offset)
            fresh_log = log.read().decode("utf-8", errors="replace")
        if player.get("gamemode") == 3 and acknowledgement.search(fresh_log):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Could not verify the viewer's spectator mode and teleport; no policy actions ran")
        time.sleep(0.25)


def _countdown(env: MinecraftEnv, server: dict[str, Any], viewer: str) -> None:
    for remaining in range(5, 0, -1):
        _require_viewer(env, server, viewer)
        print(f"Watch RLCourseBot: starting in {remaining}...", flush=True)
        time.sleep(1)
    _require_viewer(env, server, viewer)


def run_watch(*, viewer: str, policy: Path, seed: int = 1_000_000,
              join_timeout: float = 300, once: bool = False) -> int:
    """Run a watched episode, optionally replaying until the user quits.

    Callers must expose this command as an explicit development-arena action.
    Each replay uses the next seed and reloads the frozen policy; no Q values
    are updated. ``once`` exits after one episode for automated verification.
    """
    validate_watch_options(viewer, join_timeout)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(once, bool):
        raise ValueError("once must be a boolean")
    policy = Path(policy).resolve()
    # Validate the complete policy and environment before launching anything.
    initial_agent = TabularAgent.load(policy, seed=seed + 4817)
    config = load_env_config(policy)
    node = runtime_paths()["node"]
    if node is None or not Path(node).is_file():
        raise RuntimeError("Run setup.cmd first to install the local Node runtime")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = prepare_output(ROOT / "runs" / f"watch-{stamp}")
    common = {"evidence": EVIDENCE, "viewer": viewer, "viewer_mode": "spectator",
              "policy": str(policy), "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
              "algorithm": initial_agent.algorithm, "env": asdict(config),
              "learning_during_demo": False,
              "decision_method": "Frozen saved table; unseen states use the agent's ordinary tie-breaking, with no success guarantee",
              "scripted_support": "Development-world reset, spectator placement, and physical execution of movement/dig/craft actions"}
    write_json(output / "metadata.json", common)
    print(f"{EVIDENCE}\nSaved policy: {policy}\nResults: {output}", flush=True)
    with running_server("arena", mirror_output=True) as server:
        print(f"In Minecraft Java 1.21.4, join 127.0.0.1:{server['port']} as {viewer}.\n"
              "You will be placed in spectator mode above the crafting arena.\n"
              "This command leaves the separate Survival server and its world alone.", flush=True)
        with MinecraftEnv(config, port=server["port"], node=str(node), dev_arena=True) as env:
            episode = 0
            while True:
                episode += 1
                episode_seed = seed + episode - 1
                folder = prepare_output(output / f"episode-{episode:03d}")
                write_json(folder / "metadata.json", {**common, "seed": episode_seed,
                           "connection": env.connection_info})
                rows: list[dict[str, Any]] = []
                agent = TabularAgent.load(policy, seed=episode_seed + 4817)
                try:
                    state = env.reset(seed=episode_seed)
                    write_json(folder / "reset.json", env.observe())
                    print(f"Arena ready. Waiting up to {join_timeout:g}s for {viewer}...", flush=True)
                    _wait_for_viewer(env, server, viewer, join_timeout)
                    _position_viewer(env, server, viewer)
                    _countdown(env, server, viewer)
                    with (folder / "observations.jsonl").open("w", encoding="utf-8", buffering=1) as observations:
                        while True:
                            _require_viewer(env, server, viewer)
                            before = state
                            action = agent.act(state)
                            state, reward, terminated, truncated, info = env.step(action)
                            record = {"before_state": before, "action": ACTION_NAMES[action],
                                      "state": state, "reward": reward, "terminated": terminated,
                                      "truncated": truncated, "observation": info}
                            observations.write(json.dumps(record, allow_nan=False) + "\n")
                            rows.append({"step": info["steps"], "action": ACTION_NAMES[action],
                                         "state": json.dumps(state), "reward": reward,
                                         "event": info["event"], "success": info["success"],
                                         "death": info["death"]})
                            write_csv(folder / "trajectory.csv", rows)
                            print(f"{info['steps']:3d} {ACTION_NAMES[action]:15s} state={state} event={info['event']}", flush=True)
                            _require_viewer(env, server, viewer)
                            if terminated or truncated:
                                write_json(folder / "outcome.json", {"status": "completed",
                                           "evidence": EVIDENCE, "terminated": terminated,
                                           "truncated": truncated, **info})
                                print(f"Episode finished: pickaxe={info['success']}; death={info['death']}; "
                                      f"timeout={truncated}. Results: {folder}", flush=True)
                                if info["success"] and info.get("held_item") == "wooden_pickaxe":
                                    print("RLCourseBot is holding the wooden pickaxe. You can fly closer to inspect it.", flush=True)
                                break
                except BaseException as exc:
                    write_json(folder / "outcome.json", {
                        "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "invalid",
                        "evidence": EVIDENCE, "success": None, "death": None,
                        "recorded_steps": len(rows), "error": f"{type(exc).__name__}: {exc}",
                        "note": "This incomplete run is not a success or a measured policy failure."})
                    raise
                if once:
                    return 0
                print("The arena remains open so you can inspect it.", flush=True)
                while True:
                    try:
                        answer = input("Press Enter to replay with the next seed, or type q to save and stop: ").strip().lower()
                    except EOFError:
                        return 0
                    if answer == "q":
                        return 0
                    if not answer:
                        break
                    print("Press Enter to replay, or type q to stop.", flush=True)

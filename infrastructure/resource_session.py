"""Observable online learning in fresh Survival worlds, one world per episode."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
import time

from minecraft_rl.artifacts import prepare_output, write_csv, write_json
from minecraft_rl.resource_live import ResourceBridge
from minecraft_rl.resources import (
    ACTION_NAMES, ResourceAgent, allowed_actions, reward_transition,
    state_from_observation, train_simulator,
)
from .resource_server import fresh_resource_server
from .runtime import runtime_paths
from .watch import validate_watch_options

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = "REAL MINECRAFT NATURAL RESOURCE TASK; NOT FULL SURVIVAL COMPLETION"


def _viewer_name(viewer: str | None) -> None:
    if viewer is not None:
        validate_watch_options(viewer, 300)
        if viewer.casefold() == "rlresourcebot":
            raise ValueError("The spectator name cannot be RLResourceBot")


class Viewer:
    """Only move the named spectator; never alter the learner or its resources."""

    def __init__(self, server: dict, name: str | None):
        self.server, self.name = server, name
        self._lock = threading.Lock()

    def console(self, command: str) -> None:
        with self._lock:
            process = self.server["process"]
            if process.poll() is not None or not process.stdin:
                raise RuntimeError("The resource server stopped")
            process.stdin.write(command + "\n")
            process.stdin.flush()

    def require(self, bridge: ResourceBridge) -> None:
        if self.name and bridge.players().get(self.name, {}).get("gamemode") != 3:
            raise RuntimeError("The viewer disconnected or left Spectator mode; stopping this run")

    def prepare(self, bridge: ResourceBridge) -> None:
        if not self.name:
            return
        print(f"Join Minecraft Java 1.21.4 at 127.0.0.1:25567 as {self.name}. "
              "Waiting up to 5 minutes before learning starts.", flush=True)
        deadline = time.monotonic() + 300
        while self.name not in bridge.players():
            if time.monotonic() >= deadline:
                raise RuntimeError("Viewer did not join; no learning actions ran")
            time.sleep(.25)
        self.console(f"gamemode spectator {self.name}")
        self.console(f"effect give {self.name} minecraft:night_vision 1800 0 true")
        self.follow()
        deadline = time.monotonic() + 15
        while bridge.players().get(self.name, {}).get("gamemode") != 3:
            if time.monotonic() >= deadline:
                raise RuntimeError("Spectator mode could not be verified")
            time.sleep(.25)
        self.show("Ready. Training starts in 5 seconds.", chat=True)
        for _ in range(5):
            self.require(bridge)
            time.sleep(1)

    def follow(self) -> None:
        if self.name:
            self.console(f"execute at RLResourceBot run tp {self.name} ~ ~6 ~8 180 35")

    def show(self, message: str, *, chat: bool = False) -> None:
        if self.name:
            component = json.dumps({"text": message, "color": "aqua"})
            self.console(f"tellraw {self.name} {component}" if chat else
                         f"title {self.name} actionbar {component}")

    @contextmanager
    def following(self):
        stop = threading.Event()
        errors = []

        def follow_loop():
            while not stop.wait(2):
                try:
                    self.follow()
                except (OSError, RuntimeError) as exc:
                    errors.append(exc)
                    return

        thread = threading.Thread(target=follow_loop, daemon=True)
        if self.name:
            thread.start()
        try:
            yield
        finally:
            stop.set()
            if self.name:
                thread.join(timeout=3)
        if errors:
            raise RuntimeError("Spectator follow failed") from errors[0]


def _save_policy(agent: ResourceAgent, path: Path, metadata: dict) -> None:
    # A complete checkpoint replaces the prior one only after its write finishes.
    temporary = path.with_suffix(".tmp")
    agent.save(temporary, metadata=metadata)
    temporary.replace(path)


def recovery_summary(recovery) -> str:
    """Compact per-decision note for steps.csv; the full record stays in transitions.jsonl."""
    if not isinstance(recovery, dict):
        return ""
    def picked_note(record) -> str:
        picked = record.get("items_picked_up") if isinstance(record, dict) else None
        if not isinstance(picked, dict) or not picked:
            return ""
        return ",picked_up=" + "+".join(f"{name}:{count}" for name, count in sorted(picked.items()))

    if recovery.get("type") == "leaf_escape":
        return (f"leaf_escape:blocks={len(recovery.get('blocks_dug') or [])},direction={recovery.get('direction')}"
                f",escaped={recovery.get('escaped')},stopped={recovery.get('stopped') or '-'}{picked_note(recovery)}")
    escape = recovery.get("escape")
    escape_note = (f",escape_blocks={len(escape.get('blocks_dug') or [])},escaped={escape.get('escaped')}"
                   f"{picked_note(escape).replace(',picked_up=', ',escape_picked_up=')}"
                   if isinstance(escape, dict) else "")
    start, end = recovery.get("start_y"), recovery.get("end_y")
    descent = round(start - end, 2) if isinstance(start, (int, float)) and isinstance(end, (int, float)) else None
    return (f"{recovery.get('type', 'recovery')}:layers={recovery.get('layers_cleared', 0)}"
            f",descent={descent},damage={recovery.get('damage', 0)},stopped={recovery.get('stopped') or '-'}"
            f"{picked_note(recovery)}{escape_note}")


def run_resources(*, output: Path, episodes: int = 5, world_seed: int = 0,
                  seed: int = 0, algorithm: str = "q_learning", viewer: str | None = None,
                  max_steps: int = 80, epsilon: float = .6, epsilon_end: float = .1,
                  policy: Path | None = None, pretrain: int = 0, evaluate: bool = False,
                  pause: float = .5, random_baseline: bool = False) -> int:
    """Train from observed transitions, or explicitly evaluate a frozen policy.

    The masked random baseline is a frozen evaluation that keeps the same
    feasibility mask and execution helpers but picks uniformly among allowed
    actions; it measures what the helpers alone achieve. Generated worlds and
    all observations remain under the output folder. Nothing overwrites the
    existing arena or ordinary Survival world.
    """
    _viewer_name(viewer)
    if not 1 <= episodes <= 1000 or not 1 <= max_steps <= 1000:
        raise ValueError("episodes and max_steps must be between 1 and 1000")
    if not 0 <= epsilon_end <= epsilon <= 1 or not 0 <= pause <= 10:
        raise ValueError("Require 0 <= epsilon_end <= epsilon <= 1 and 0 <= pause <= 10")
    if not 0 <= pretrain <= 100_000 or (pretrain and policy):
        raise ValueError("pretrain must be 0..100000 and cannot be combined with a policy")
    if type(world_seed) is not int or not -(2**63) <= world_seed <= 2**63 - episodes:
        raise ValueError("All episode world seeds must fit signed 64-bit integers")
    if random_baseline and (policy or pretrain or not evaluate):
        raise ValueError("The masked random baseline is a frozen evaluation without a policy or simulator pretraining")
    if evaluate and (not (policy or random_baseline) or pretrain):
        raise ValueError("Frozen evaluation requires --policy or --random-baseline and forbids pretraining")
    node = runtime_paths()["node"]
    if not node or not Path(node).is_file():
        raise RuntimeError("Run setup.cmd first")
    output = output.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()) or output == (ROOT / "runs").resolve():
        raise ValueError("Resource output must be a new folder inside this project's runs directory")
    if policy:
        agent = ResourceAgent.load(policy, seed=seed)
    else:
        agent = ResourceAgent(algorithm="random" if random_baseline else algorithm, seed=seed)
    if not evaluate and agent.algorithm == "random":
        raise ValueError("A random policy can be evaluated, but cannot be trained")
    output = prepare_output(output)
    mode = "frozen_evaluation" if evaluate else "online_training"
    metadata = {"evidence": EVIDENCE, "mode": mode, "algorithm": agent.algorithm,
                "baseline": "masked_random" if agent.algorithm == "random" else None,
                "episodes_requested": episodes, "first_world_seed": world_seed,
                "agent_seed": seed, "max_steps": max_steps, "epsilon_start": epsilon,
                "epsilon_end": epsilon_end, "reward_mode": "shaped", "gamma": agent.gamma,
                "alpha": agent.alpha, "pretrain_simulator_episodes": pretrain,
                "pause_between_decisions": pause,
                "time_semantics": "seconds measures task time after viewer readiness; elapsed_wall_seconds also includes world startup and viewer join",
                "initial_policy": str(policy.resolve()) if policy else None,
                "initial_policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest() if policy else None,
                "viewer": viewer, "actions": list(ACTION_NAMES),
                "state_note": "Compact partial observation; not a complete Minecraft MDP state",
                "scripted_support": "Feasibility masks; bounded pathfinding, digging, pickup, recipes, placement, equip; "
                                    "bounded leaf clearing inside SEARCH (canopy descent beneath the feet, foliage escape "
                                    "beside the bot) and inside PLACE_TABLE (the same foliage escape when walled in with no "
                                    "placement surface); spectator camera only",
                "recovery_helpers": "When the bot stands on tree leaves, SEARCH may clear at most six leaf blocks directly "
                                    "beneath its feet, one at a time, only after scanning the column below for a landing "
                                    "within a recorded safe fall (no damage, or at most two points while health stays above "
                                    "ten, measured to the landing block's collision surface). Each cleared block, fall, and "
                                    "health change is saved in the transition's recovery field. Leaves broken by hand follow "
                                    "the vanilla loot table (about 5% sapling, 2% one or two sticks, 0.5% apple on oak and dark "
                                    "oak); pickups collected during a descent or an escape are listed in that record as "
                                    "items_picked_up and count in the inventory like any other natural item. When the bridge's "
                                    "movement rules report no legal move from the current cell, SEARCH (or PLACE_TABLE with no "
                                    "placement surface) may clear at most two "
                                    "leaf blocks beside the bot (feet and head height) towards a floored exit; never a log, "
                                    "dirt, or stone; recorded as leaf_escape (alone, or as the escape field of the descent "
                                    "record that led to it). Apart from these leaf blocks, no block is dug outside CHOP",
                "reset": "Each episode generates a separate new normal Survival world; bot has no operator permissions",
                "seed_note": "World seed controls terrain; server timing and spawn selection are not deterministic"}
    sources = [ROOT / "minecraft_bridge" / "resource-bridge.js", ROOT / "minecraft_bridge" / "pnpm-lock.yaml",
               ROOT / "minecraft_rl" / "resources.py", Path(__file__),
               ROOT / "minecraft_rl" / "resource_live.py", ROOT / "infrastructure" / "resource_server.py"]
    metadata["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(output / "metadata.json", metadata)
    print(f"{EVIDENCE}\nMode: {mode}\nResults and worlds: {output}", flush=True)
    if agent.algorithm == "random":
        print("MASKED RANDOM BASELINE: same feasibility mask and execution helpers, uniformly random "
              "allowed actions, no learning. It measures what the helpers alone achieve.", flush=True)
    if pretrain:
        print(f"First: {pretrain} SIMULATOR episodes; these are not Minecraft training.", flush=True)
        agent, records = train_simulator(pretrain, seed=seed, algorithm=algorithm)
        write_csv(output / "simulator_training.csv", records)
    _save_policy(agent, output / "initial_policy.json", metadata)
    _save_policy(agent, output / "policy.json", metadata)
    summaries = []
    for episode in range(1, episodes + 1):
        folder = output / f"episode-{episode:04d}"
        folder.mkdir()
        current_epsilon = 0.0 if evaluate else epsilon + (epsilon_end - epsilon) * (episode - 1) / max(1, episodes - 1)
        rows = []
        total_reward = 0.0
        started = time.monotonic()
        success = death = False
        completed = False
        try:
            with fresh_resource_server(world_seed=world_seed + episode - 1, run_directory=output,
                                       episode=episode, mirror_output=False) as server:
                with ResourceBridge(node=node, port=server["port"]) as bridge:
                    camera = Viewer(server, viewer)
                    camera.prepare(bridge)
                    observation = bridge.observe()
                    if any(observation["inventory_counts"].values()):
                        raise RuntimeError("Fresh-start inventory was not empty")
                    state = state_from_observation(observation)
                    write_json(folder / "start.json", {"connection": bridge.connection, "observation": observation})
                    task_started = time.monotonic()
                    action = agent.act(state, epsilon=current_epsilon)
                    with (folder / "transitions.jsonl").open("w", encoding="utf-8", buffering=1) as log:
                        for step in range(1, max_steps + 1):
                            camera.require(bridge)
                            before_q = list(agent.values(state))
                            camera.show(f"{'EVAL' if evaluate else 'TRAIN'} episode {episode}/{episodes} | step {step} | {ACTION_NAMES[action]} | epsilon {current_epsilon:.2f}")
                            with camera.following():
                                next_observation = bridge.step(action)
                            next_state = state_from_observation(next_observation)
                            success, death = next_observation["success"], next_observation["death"]
                            terminated = success or death
                            truncated = step == max_steps and not terminated
                            reward, rewards = reward_transition(observation, next_observation, gamma=agent.gamma,
                                                               success=success, death=death)
                            # A time limit truncates sampling, so SARSA still samples its bootstrap action.
                            next_action = None if terminated else agent.act(next_state, epsilon=current_epsilon)
                            td_error = None if evaluate else agent.update(state, action, reward, next_state,
                                                                         terminated, next_action=next_action)
                            total_reward += reward
                            record = {"episode": episode, "step": step, "world_seed": world_seed + episode - 1,
                                      "mode": mode, "epsilon": current_epsilon, "state": state,
                                      "allowed_actions": list(allowed_actions(state)), "action": ACTION_NAMES[action],
                                      "reward": reward, **rewards, "next_state": next_state,
                                      "terminated": terminated, "truncated": truncated, "td_error": td_error,
                                      "q_before": before_q, "q_after": list(agent.values(state)),
                                      "before": observation, "after": next_observation}
                            log.write(json.dumps(record, allow_nan=False) + "\n")
                            rows.append({"step": step, "action": ACTION_NAMES[action], "reward": reward,
                                         "td_error": td_error, "q_before": before_q[action],
                                         "q_after": agent.values(state)[action], "event": next_observation["event"],
                                         "success": success, "death": death,
                                         "recovery": recovery_summary(next_observation.get("recovery"))})
                            write_csv(folder / "steps.csv", rows)
                            if not evaluate:
                                _save_policy(agent, output / "policy.json", {**metadata, "last_episode": episode, "last_step": step})
                            update = "frozen" if evaluate else f"Q {before_q[action]:+.3f}->{agent.values(state)[action]:+.3f}; TD {td_error:+.3f}"
                            message = f"E{episode} step {step}: {ACTION_NAMES[action]} | r={reward:+.3f} | {update} | {next_observation['event']}"
                            print(message, flush=True)
                            camera.show(message, chat=True)
                            camera.require(bridge)
                            if terminated or truncated:
                                break
                            observation, state, action = next_observation, next_state, next_action
                            if pause:
                                time.sleep(pause)
                    summary = {"episode": episode, "world_seed": world_seed + episode - 1,
                               "status": "completed", "mode": mode, "success": success, "death": death,
                               "truncated": truncated, "steps": len(rows), "return": total_reward,
                               "epsilon": current_epsilon, "seconds": time.monotonic() - task_started,
                               "elapsed_wall_seconds": time.monotonic() - started}
                    write_json(folder / "outcome.json", {**summary, "final_observation": next_observation})
                    summaries.append(summary)
                    write_csv(output / "episodes.csv", summaries)
                    completed = True
                    message = f"Episode {episode} finished: pickaxe={success}, death={death}, steps={len(rows)}."
                    print(message, flush=True)
                    camera.show(message, chat=True)
                    if viewer:
                        print("You can inspect the bot. In this terminal press Enter to save this world and continue.", flush=True)
                        try:
                            input()
                        except EOFError:
                            pass
            if viewer and episode < episodes:
                print("Next fresh world is starting. Reconnect to 127.0.0.1:25567 when prompted.", flush=True)
        except BaseException as exc:
            write_json(folder / ("session_error.json" if completed else "outcome.json"), {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "invalid",
                       "success": None, "death": None, "recorded_steps": len(rows),
                       "error": f"{type(exc).__name__}: {exc}", "note": "Not counted as a measured policy success or failure"})
            raise
    write_json(output / "summary.json", {"evidence": EVIDENCE, "mode": mode, "algorithm": agent.algorithm,
               "baseline": metadata["baseline"],
               "episodes": len(summaries), "successes": sum(row["success"] for row in summaries),
               "note": "Exploratory training results are not frozen-policy evaluation" if not evaluate else "Small evaluation sample; does not establish broad generalization"})
    print(f"Saved checkpoint: {output / 'policy.json'}", flush=True)
    return 0

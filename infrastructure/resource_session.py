"""Observable online learning in fresh Survival worlds, one world per episode."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from minecraft_rl.artifacts import prepare_output, write_csv, write_json
from minecraft_rl.resource_live import ResourceBridge
from minecraft_rl.resources import (
    ACTION_NAMES, CONDITIONS, MASKED, ResourceAgent, allowed_actions, encode_state, reward_transition,
    state_from_observation, train_simulator,
)
from .resource_server import MAX_PARALLEL_SLOTS, PORT, current_owner, fresh_resource_server, live_run_lock, validate_port
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
        print(f"Join Minecraft Java 1.21.4 at 127.0.0.1:{self.server['port']} as {self.name}. "
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


CONDITION_NOTES = {
    "masked": "Disclosed feasibility/budget mask recipe_and_budget_v1 removes impossible and redundant actions; state resource_v1",
    "unmasked": "No action mask: all eight actions are allowed in every state; an infeasible action fails as an observed "
                "transition, costs its real time and the -0.01 step penalty, and changes nothing; state resource_v1",
    "unmasked_history": "No action mask, as in unmasked; state resource_v2 adds last_action_failed, 1 when the previous "
                        "decision left the nine resource_v1 fields unchanged",
}


def _run_resources(*, output: Path, episodes: int = 5, world_seed: int = 0,
                  seed: int = 0, algorithm: str = "q_learning", viewer: str | None = None,
                  max_steps: int = 80, epsilon: float = .6, epsilon_end: float = .1,
                  policy: Path | None = None, pretrain: int = 0, evaluate: bool = False,
                  pause: float = .5, random_baseline: bool = False, condition: str | None = None,
                  port: int = PORT, parallel_slots: int = 1, schedule_id: str | None = None) -> int:
    """Train from observed transitions, or explicitly evaluate a frozen policy.

    ``condition`` names the experiment condition (masked, unmasked, or
    unmasked_history); a loaded policy carries its own, and a different
    explicit condition is refused. The random baseline is a frozen evaluation
    with the same execution helpers that picks uniformly among the condition's
    allowed actions: the masked random baseline under the mask, the uniform
    random baseline over all eight actions without it. It measures what the
    helpers achieve without learning. Generated worlds and all observations
    remain under the output folder. Nothing overwrites the existing arena or
    ordinary Survival world.

    ``port`` is the loopback port of this run's servers. ``parallel_slots`` is
    the declared number of live episodes running at once on this machine
    (resource-parallel sets it; 1 means alone). It is recorded because task
    time and the helpers' wall-clock budgets depend on it. ``schedule_id``
    names the resource-parallel schedule a run belongs to; ``run_resources``
    checks it and holds the machine-wide live-run lock for runs started by hand.
    """
    if condition is not None and condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {', '.join(CONDITIONS)}")
    if random_baseline and condition == "unmasked_history":
        raise ValueError("The uniform random baseline does not read the state; evaluate it with condition unmasked "
                         "and compare unmasked_history policies against that run")
    _viewer_name(viewer)
    validate_port(port)
    if (isinstance(parallel_slots, bool) or not isinstance(parallel_slots, int)
            or not 1 <= parallel_slots <= MAX_PARALLEL_SLOTS):
        raise ValueError(f"parallel_slots must be an integer from 1 to {MAX_PARALLEL_SLOTS}")
    if viewer is not None and parallel_slots > 1:
        raise ValueError("A viewer run waits for you between episodes; run it alone with parallel_slots 1")
    if parallel_slots > 1 and schedule_id is None:
        raise ValueError("parallel_slots above 1 is set by resource-parallel, together with its schedule_id")
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
        if condition is not None and condition != agent.condition:
            raise ValueError(f"The policy was trained under condition {agent.condition}, not {condition}")
        # Simulator pretraining behind a checkpoint stays visible in every later run.
        prior = json.loads(policy.read_text(encoding="utf-8")).get("metadata") or {}
        checkpoint_pretrain = sum(int(prior.get(key) or 0) for key in
                                  ("pretrain_simulator_episodes", "checkpoint_pretrain_simulator_episodes"))
    else:
        checkpoint_pretrain = 0
        action_mask, state_version = CONDITIONS[condition or "masked"]
        agent = ResourceAgent(algorithm="random" if random_baseline else algorithm, seed=seed,
                              action_mask=action_mask, state_version=state_version)
    if not evaluate and agent.algorithm == "random":
        raise ValueError("A random policy can be evaluated, but cannot be trained")
    output = prepare_output(output)
    mode = "frozen_evaluation" if evaluate else "online_training"
    baseline = None
    if agent.algorithm == "random":
        baseline = "masked_random" if agent.action_mask == MASKED else "uniform_random"
    metadata = {"evidence": EVIDENCE, "mode": mode, "algorithm": agent.algorithm,
                "baseline": baseline, "condition": agent.condition, "action_mask": agent.action_mask,
                "state_version": agent.state_version, "condition_note": CONDITION_NOTES[agent.condition],
                "episodes_requested": episodes, "first_world_seed": world_seed,
                "agent_seed": seed, "max_steps": max_steps, "epsilon_start": epsilon,
                "epsilon_end": epsilon_end, "reward_mode": "shaped", "gamma": agent.gamma,
                "alpha": agent.alpha, "pretrain_simulator_episodes": pretrain,
                "checkpoint_pretrain_simulator_episodes": checkpoint_pretrain,
                "pause_between_decisions": pause,
                "time_semantics": "seconds measures task time after viewer readiness; elapsed_wall_seconds also includes world startup and viewer join",
                "initial_policy": str(policy.resolve()) if policy else None,
                "initial_policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest() if policy else None,
                "viewer": viewer, "actions": list(ACTION_NAMES),
                "state_note": "Compact partial observation; not a complete Minecraft MDP state",
                "scripted_support": ("Feasibility mask recipe_and_budget_v1; " if agent.action_mask == MASKED else
                                     "No action mask (all eight actions allowed; infeasible ones fail as observed transitions); ") +
                                    "bounded pathfinding, digging, pickup, recipes, placement, equip; "
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
                "seed_note": "World seed controls terrain; server timing and spawn selection are not deterministic",
                "execution": {"server_port": port, "parallel_slots": parallel_slots, "schedule_id": schedule_id,
                              "note": "parallel_slots is the declared number of live episodes run at once on this "
                                      "machine (1 = alone). Task time and the helpers' time budgets are wall-clock, "
                                      "so the report compares only runs with equal values, and parallel runs only "
                                      "within one schedule"}}
    sources = [ROOT / "minecraft_bridge" / "resource-bridge.js", ROOT / "minecraft_bridge" / "pnpm-lock.yaml",
               ROOT / "minecraft_rl" / "resources.py", Path(__file__),
               ROOT / "minecraft_rl" / "resource_live.py", ROOT / "infrastructure" / "resource_server.py"]
    metadata["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(output / "metadata.json", metadata)
    print(f"{EVIDENCE}\nMode: {mode}\nResults and worlds: {output}", flush=True)
    print(f"Condition: {agent.condition} ({CONDITION_NOTES[agent.condition]})", flush=True)
    if baseline == "masked_random":
        print("MASKED RANDOM BASELINE: same feasibility mask and execution helpers, uniformly random "
              "allowed actions, no learning. It measures what the helpers alone achieve.", flush=True)
    elif baseline == "uniform_random":
        print("UNIFORM RANDOM BASELINE: no action mask, same execution helpers, uniformly random choice among "
              "all eight actions, no learning.", flush=True)
    if pretrain:
        print(f"First: {pretrain} SIMULATOR episodes; these are not Minecraft training.", flush=True)
        agent, records = train_simulator(pretrain, seed=seed, algorithm=algorithm, condition=agent.condition)
        write_csv(output / "simulator_training.csv", records)
    _save_policy(agent, output / "initial_policy.json", metadata)
    _save_policy(agent, output / "policy.json", metadata)
    summaries = []
    for episode in range(1, episodes + 1):
        # Every episode runs the code its metadata describes; a changed source stops the run before a world starts.
        if {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources} != metadata["source_sha256"]:
            raise RuntimeError("This run's helper or session sources changed since it started; no further episode "
                               "runs on code that its metadata.json does not describe")
        folder = output / f"episode-{episode:04d}"
        folder.mkdir()
        current_epsilon = 0.0 if evaluate else epsilon + (epsilon_end - epsilon) * (episode - 1) / max(1, episodes - 1)
        rows = []
        total_reward = 0.0
        started = time.monotonic()
        started_at = _utc_now()
        success = death = False
        completed = False
        try:
            with fresh_resource_server(world_seed=world_seed + episode - 1, run_directory=output,
                                       episode=episode, mirror_output=False, port=port) as server:
                with ResourceBridge(node=node, port=server["port"]) as bridge:
                    camera = Viewer(server, viewer)
                    camera.prepare(bridge)
                    observation = bridge.observe()
                    if any(observation["inventory_counts"].values()):
                        raise RuntimeError("Fresh-start inventory was not empty")
                    base = state_from_observation(observation)
                    state = encode_state(base, agent.state_version)
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
                            next_base = state_from_observation(next_observation)
                            # resource_v2's history bit: this decision changed none of the nine fields.
                            next_state = encode_state(next_base, agent.state_version, next_base == base)
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
                                      "mode": mode, "condition": agent.condition, "epsilon": current_epsilon, "state": state,
                                      "allowed_actions": list(agent.allowed(state)),
                                      # What the disclosed mask would allow here, in every condition.
                                      "masked_actions": list(allowed_actions(base)), "action": ACTION_NAMES[action],
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
                            observation, base, state, action = next_observation, next_base, next_state, next_action
                            if pause:
                                time.sleep(pause)
                    summary = {"episode": episode, "world_seed": world_seed + episode - 1,
                               "status": "completed", "mode": mode, "success": success, "death": death,
                               "truncated": truncated, "steps": len(rows), "return": total_reward,
                               "epsilon": current_epsilon, "seconds": time.monotonic() - task_started,
                               "elapsed_wall_seconds": time.monotonic() - started,
                               "started_at": started_at, "finished_at": _utc_now()}
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
                print(f"Next fresh world is starting. Reconnect to 127.0.0.1:{port} when prompted.", flush=True)
        except BaseException as exc:
            write_json(folder / ("session_error.json" if completed else "outcome.json"), {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "invalid",
                       "episode": episode, "world_seed": world_seed + episode - 1,
                       "started_at": started_at, "stopped_at": _utc_now(),
                       "success": None, "death": None, "recorded_steps": len(rows),
                       "error": f"{type(exc).__name__}: {exc}", "note": "Not counted as a measured policy success or failure"})
            raise
    write_json(output / "summary.json", {"evidence": EVIDENCE, "mode": mode, "algorithm": agent.algorithm,
               "baseline": metadata["baseline"], "condition": agent.condition,
               "episodes": len(summaries), "successes": sum(row["success"] for row in summaries),
               "note": "Exploratory training results are not frozen-policy evaluation" if not evaluate else "Small evaluation sample; does not establish broad generalization"})
    print(f"Saved checkpoint: {output / 'policy.json'}", flush=True)
    return 0


def run_resources(*, schedule_id: str | None = None, **settings) -> int:
    """Train or evaluate as ``_run_resources`` describes, never sharing the machine unrecorded.

    A run started by hand holds the machine-wide live-run lock for its whole
    duration, so it never overlaps a schedule or another live run. A
    resource-parallel run carries its schedule's id, which is accepted only
    while that schedule holds the lock.
    """
    if schedule_id is None:
        output = settings.get("output")
        with live_run_lock(f"run {Path(output).name if output else 'unnamed'} (pid {os.getpid()})"):
            return _run_resources(schedule_id=None, **settings)
    if not isinstance(schedule_id, str) or not 1 <= len(schedule_id) <= 200 or not schedule_id.isprintable():
        raise ValueError("schedule_id must be 1 to 200 printable characters")
    if current_owner() != f"schedule {schedule_id}":
        raise RuntimeError("schedule_id is set only by resource-parallel for its own runs, while that schedule runs")
    return _run_resources(schedule_id=schedule_id, **settings)

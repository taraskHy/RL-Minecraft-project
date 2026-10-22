"""Optional real Minecraft adapter for the first *development arena* milestone.

This module does not implement fresh-world survival or an Ender Dragon agent.
It never starts/downloads a Minecraft server. Development resets deliberately
modify a small region of an existing, dedicated local test world.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import random
import shutil
import subprocess
import threading
from typing import Any

from .environment import (
    EnvConfig, INITIAL_LOG_MASK, LAVA, LOG_POSITIONS, N_ACTIONS, SIZE, State,
    potential,
)


class LiveBridgeError(RuntimeError):
    """The real game could not produce a valid task transition."""


class MinecraftEnv:
    """CraftingEnv-compatible adapter to an explicitly enabled local arena.

    Set dev_arena=True only for a disposable test world. The bridge uses
    actual inventory and position; it never substitutes a simulator response.
    EnvConfig.slip_probability is a simulator parameter and is not applied to
    Minecraft. Game physics supplies its own transition uncertainty.
    """

    def __init__(
        self,
        config: EnvConfig | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 25565,
        username: str = "RLCourseBot",
        dev_arena: bool = False,
        node: str = "node",
        origin: tuple[int, int, int] = (0, 64, 0),
        timeout: float = 45.0,
    ) -> None:
        self.config = config if config is not None else EnvConfig()
        if not isinstance(self.config, EnvConfig):
            raise TypeError("config must be an EnvConfig")
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("The development bridge supports localhost only")
        if not isinstance(dev_arena, bool):
            raise TypeError("dev_arena must be a bool")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be a positive finite number")
        if (len(origin) != 3 or any(isinstance(v, bool) or not isinstance(v, int)
                                    for v in origin)):
            raise ValueError("origin must contain three integers")
        executable = shutil.which(node)
        if executable is None:
            raise LiveBridgeError("Node.js >=22 is required for live Minecraft; the simulator does not need Node")
        bridge = Path(__file__).resolve().parent.parent / "minecraft_bridge" / "bridge.js"
        command = [executable, str(bridge), "--host", host, "--port", str(port),
                   "--username", username, "--origin", *(str(v) for v in origin)]
        if dev_arena:
            command.append("--dev-arena")
        self.dev_arena = dev_arena
        self.timeout = timeout
        self._state: State | None = None
        self._rng = random.Random()
        self._steps = 0
        self._needs_reset = True
        self._next_id = 0
        self._closed = False
        self._responses: queue.Queue[str | None] = queue.Queue()
        # Inherit stderr so startup and connection errors stay visible while
        # stdout remains a strict JSON-lines channel. No shell is involved.
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        try:
            self.connection_info = self._request("status")
        except Exception:
            self.close()
            raise

    @property
    def state(self) -> State:
        if self._state is None:
            raise RuntimeError("Call reset() before reading the state")
        return self._state

    def _read_output(self) -> None:
        assert self._process.stdout is not None
        try:
            for line in self._process.stdout:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def _request(self, operation: str, **values: Any) -> dict[str, Any]:
        if self._closed:
            raise LiveBridgeError("Minecraft bridge is closed")
        self._next_id += 1
        request_id = self._next_id
        request = {"id": request_id, "op": operation, **values}
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise LiveBridgeError("Minecraft bridge exited; inspect its stderr and install minecraft_bridge dependencies") from exc
        try:
            line = self._responses.get(timeout=self.timeout)
        except queue.Empty as exc:
            self.close()
            raise LiveBridgeError("Minecraft bridge timed out and was stopped; no valid transition was recorded") from exc
        if line is None:
            raise LiveBridgeError("Minecraft bridge exited; check Node dependencies, local server, and version (see minecraft_bridge/README.md)")
        try:
            response = json.loads(line)
        except (TypeError, json.JSONDecodeError) as exc:
            self.close()
            raise LiveBridgeError("Bridge returned non-JSON output; session closed") from exc
        if not isinstance(response, dict) or response.get("id") != request_id:
            self.close()
            raise LiveBridgeError("Bridge response ID mismatch; session closed")
        if response.get("ok") is not True:
            raise LiveBridgeError(str(response.get("error", "Unknown bridge error")))
        result = response.get("result")
        if not isinstance(result, dict):
            self.close()
            raise LiveBridgeError("Bridge did not return an observation object")
        return result

    @staticmethod
    def _observed_state(observation: dict[str, Any]) -> State:
        values = observation.get("state")
        if (not isinstance(values, list) or len(values) != 6
                or any(isinstance(v, bool) or not isinstance(v, int) for v in values)):
            raise LiveBridgeError("Game observation must contain six integer state values")
        x, z, mask, logs, planks, sticks = values
        if not (0 <= x < SIZE and 0 <= z < SIZE and 0 <= mask <= INITIAL_LOG_MASK):
            raise LiveBridgeError("Game observation is outside the task state space")
        if min(logs, planks, sticks) < 0:
            raise LiveBridgeError("Game reported negative inventory")
        if any((x, z) == cell and mask & (1 << i) for i, cell in enumerate(LOG_POSITIONS)):
            raise LiveBridgeError("Game reported a position inside an unbroken log")
        for field in ("success", "death"):
            if not isinstance(observation.get(field), bool):
                raise LiveBridgeError(f"Game observation is missing boolean {field}")
        return tuple(values)

    def reset(self, seed: int | None = None) -> State:
        if not self.dev_arena:
            raise LiveBridgeError("reset changes a dedicated test world; explicitly set dev_arena=True or pass --dev-arena")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("seed must be an integer or None")
            self._rng.seed(seed)
        safe_cells = [(x, z) for z in range(SIZE) for x in range(SIZE)
                      if (x, z) != LAVA and (x, z) not in LOG_POSITIONS]
        start = self._rng.choice(safe_cells)
        self._needs_reset = True
        observation = self._request("reset", start=start)
        state = self._observed_state(observation)
        if state != (*start, INITIAL_LOG_MASK, 0, 0, 0) or observation["success"] or observation["death"]:
            raise LiveBridgeError("Server did not produce the requested initial state")
        self._state = state
        self._steps = 0
        self._needs_reset = False
        return self.state

    def step(self, action: int) -> tuple[State, float, bool, bool, dict]:
        if self._needs_reset:
            raise RuntimeError("Call reset() before stepping a new episode")
        if (isinstance(action, bool) or not isinstance(action, int)
                or not 0 <= action < N_ACTIONS):
            raise ValueError(f"action must be an integer from 0 to {N_ACTIONS - 1}")
        previous = self.state
        try:
            observation = self._request("step", action=int(action))
            state = self._observed_state(observation)
        except Exception:
            self._needs_reset = True
            raise
        self._state = state
        self._steps += 1
        success, death = observation["success"], observation["death"]
        terminated = success or death
        truncated = self._steps >= self.config.max_steps and not terminated
        base_reward = -0.01 + (10.0 if success else -10.0 if death else 0.0)
        shaping_reward = 0.0
        if self.config.reward_mode == "shaped":
            next_potential = 0.0 if terminated else potential(state)
            shaping_reward = self.config.gamma * next_potential - potential(previous)
        self._needs_reset = terminated or truncated
        info = {**observation, "steps": self._steps, "base_reward": base_reward,
                "shaping_reward": shaping_reward, "simulator_slip_applied": False}
        return state, base_reward + shaping_reward, terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        self._reader.join(timeout=1)
        if self._process.stdout:
            self._process.stdout.close()

    def __enter__(self) -> MinecraftEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

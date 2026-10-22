"""Tabular TD control, with the update equations kept visible for teaching."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .environment import ACTION_NAMES, N_ACTIONS, State


class TabularAgent:
    def __init__(self, algorithm: str = "q_learning", alpha: float = 0.2,
                 gamma: float = 0.97, seed: int = 0):
        if algorithm not in ("q_learning", "sarsa", "random"):
            raise ValueError("algorithm must be q_learning, sarsa, or random")
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not math.isfinite(gamma) or not 0 <= gamma < 1:
            raise ValueError("gamma must be in [0, 1)")
        self.algorithm, self.alpha, self.gamma = algorithm, alpha, gamma
        self.rng = random.Random(seed)
        self.q: dict[State, list[float]] = {}

    def values(self, state: State) -> list[float]:
        # Reading an unseen state must not change the policy during evaluation.
        return self.q.get(state, [0.0] * N_ACTIONS)

    def act(self, state: State, epsilon: float = 0.0) -> int:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if self.algorithm == "random" or self.rng.random() < epsilon:
            return self.rng.randrange(N_ACTIONS)
        values = self.values(state)
        best = max(values)
        # Random ties avoid an accidental preference for NORTH in unseen states.
        return self.rng.choice([a for a, value in enumerate(values) if value == best])

    def update(self, state: State, action: int, reward: float,
               next_state: State, terminated: bool,
               next_action: int | None = None) -> float:
        if self.algorithm == "random":
            raise ValueError("A random baseline does not learn")
        if not 0 <= action < N_ACTIONS or not math.isfinite(reward):
            raise ValueError("invalid action or reward")
        if terminated:
            bootstrap = 0.0
        elif self.algorithm == "q_learning":
            bootstrap = max(self.values(next_state))
        else:
            if next_action is None or not 0 <= next_action < N_ACTIONS:
                raise ValueError("SARSA needs the next behavior action")
            bootstrap = self.values(next_state)[next_action]
        row = self.q.setdefault(state, [0.0] * N_ACTIONS)
        td_error = reward + self.gamma * bootstrap - row[action]
        row[action] += self.alpha * td_error
        return td_error

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        payload = {
            "format_version": 1, "environment": "crafting_arena_v1",
            "algorithm": self.algorithm, "alpha": self.alpha, "gamma": self.gamma,
            "actions": list(ACTION_NAMES), "metadata": metadata or {},
            "q": [{"state": list(s), "values": v} for s, v in sorted(self.q.items())],
        }
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> TabularAgent:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (payload.get("format_version") != 1
                or payload.get("environment") != "crafting_arena_v1"
                or payload.get("actions") != list(ACTION_NAMES)):
            raise ValueError("Incompatible policy file or action order")
        agent = cls(payload["algorithm"], payload["alpha"], payload["gamma"], seed)
        for entry in payload["q"]:
            state, values = tuple(entry["state"]), entry["values"]
            if (len(state) != 6 or any(type(x) is not int or x < 0 for x in state)
                    or len(values) != N_ACTIONS
                    or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in values)
                    or state in agent.q):
                raise ValueError("Invalid Q table")
            agent.q[state] = list(values)
        return agent

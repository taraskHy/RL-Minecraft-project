"""A small, fully observable crafting task for learning tabular RL.

This grid is an educational simulator, not Minecraft and not a model of the
Ender Dragon fight. A state contains (x, z, remaining_log_mask, logs, planks,
sticks). Bits in the mask say which of LOG_POSITIONS still contain a log.
The workbench is provided so the first experiment has only eight actions.
"""

from dataclasses import dataclass
from enum import IntEnum
import math
import random


State = tuple[int, ...]
SIZE = 5
LOG_POSITIONS = ((1, 1), (3, 2))
LAVA = (2, 3)
WORKBENCH = (4, 4)
INITIAL_LOG_MASK = (1 << len(LOG_POSITIONS)) - 1


class Action(IntEnum):
    NORTH = 0  # z decreases
    SOUTH = 1  # z increases
    EAST = 2   # x increases
    WEST = 3   # x decreases
    CHOP = 4
    CRAFT_PLANKS = 5
    CRAFT_STICKS = 6
    CRAFT_PICKAXE = 7


N_ACTIONS = len(Action)
ACTION_NAMES = tuple(action.name for action in Action)
MOVEMENT = ((0, -1), (0, 1), (1, 0), (-1, 0))


@dataclass(frozen=True)
class EnvConfig:
    reward_mode: str = "shaped"
    gamma: float = 0.97
    slip_probability: float = 0.05
    max_steps: int = 100

    def __post_init__(self) -> None:
        if self.reward_mode not in ("shaped", "sparse"):
            raise ValueError("reward_mode must be 'shaped' or 'sparse'")
        for name in ("gamma", "slip_probability"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be a finite number from 0 to 1")
        if (isinstance(self.max_steps, bool)
                or not isinstance(self.max_steps, int) or self.max_steps < 1):
            raise ValueError("max_steps must be a positive integer")


def potential(state: State) -> float:
    """Observable progress used only for potential-based reward shaping.

    Inventory conditions are deliberately reversible: crafting can consume
    planks or sticks and reduce potential. Paying a permanent bonus whenever a
    threshold is crossed would let the learner collect the bonus repeatedly.
    The environment handles terminal potential (zero) separately.
    """
    _, _, remaining_log_mask, _, planks, sticks = state
    return float(sum((remaining_log_mask != INITIAL_LOG_MASK,
                      planks >= 3, sticks >= 2)))


class CraftingEnv:
    """Eight-action MDP with an external episode time limit.

    reset(seed) returns a State. step(action) returns
    (next_state, reward, terminated, truncated, info). Lava and a crafted
    pickaxe terminate the task. max_steps only truncates sampling, so a learner
    must still bootstrap its value estimate at that boundary.
    """

    def __init__(self, config: EnvConfig = EnvConfig()) -> None:
        if not isinstance(config, EnvConfig):
            raise TypeError("config must be an EnvConfig")
        self.config = config
        self._rng = random.Random()
        self._state: State | None = None
        self._steps = 0
        self._needs_reset = True

    @property
    def state(self) -> State:
        if self._state is None:
            raise RuntimeError("Call reset() before reading the state")
        return self._state

    @state.setter
    def state(self, value: State) -> None:
        """Set a valid observation for demonstrations or test fixtures.

        This does not reset the episode counter or reopen a finished episode.
        Normal agent interaction should only use reset() and step().
        """
        if (not isinstance(value, tuple) or len(value) != 6
                or any(isinstance(v, bool) or not isinstance(v, int)
                       for v in value)):
            raise ValueError("state must be a tuple of six integers")
        x, z, mask, logs, planks, sticks = value
        if not (0 <= x < SIZE and 0 <= z < SIZE):
            raise ValueError("state coordinates must be inside the grid")
        if not 0 <= mask <= INITIAL_LOG_MASK or min(logs, planks, sticks) < 0:
            raise ValueError("invalid log mask or negative inventory")
        if any((x, z) == position and mask & (1 << index)
               for index, position in enumerate(LOG_POSITIONS)):
            raise ValueError("the agent cannot stand inside a remaining log")
        self._state = value

    def reset(self, seed: int | None = None) -> State:
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("seed must be an integer or None")
            self._rng.seed(seed)
        safe_cells = [(x, z) for z in range(SIZE) for x in range(SIZE)
                      if (x, z) != LAVA and (x, z) not in LOG_POSITIONS]
        x, z = self._rng.choice(safe_cells)
        self._state = (x, z, INITIAL_LOG_MASK, 0, 0, 0)
        self._steps = 0
        self._needs_reset = False
        return self.state

    def step(self, action: int) -> tuple[State, float, bool, bool, dict]:
        if self._needs_reset:
            raise RuntimeError("Call reset() before stepping a new episode")
        if (isinstance(action, bool) or not isinstance(action, int)
                or not 0 <= action < N_ACTIONS):
            raise ValueError(f"action must be an integer from 0 to {N_ACTIONS - 1}")

        previous_state = self.state
        x, z, mask, logs, planks, sticks = previous_state
        success = False
        death = False

        if action < 4:
            direction = action
            if self._rng.random() < self.config.slip_probability:
                direction = self._rng.randrange(4)
            dx, dz = MOVEMENT[direction]
            target = (x + dx, z + dz)
            blocked_by_log = any(target == position and mask & (1 << index)
                                 for index, position in enumerate(LOG_POSITIONS))
            if (0 <= target[0] < SIZE and 0 <= target[1] < SIZE
                    and not blocked_by_log):
                x, z = target
                death = target == LAVA
                event = "lava" if death else "moved"
            else:
                event = "blocked"
        elif action == Action.CHOP:
            event = "no_adjacent_log"
            for index, (log_x, log_z) in enumerate(LOG_POSITIONS):
                if mask & (1 << index) and abs(x - log_x) + abs(z - log_z) == 1:
                    mask &= ~(1 << index)
                    logs += 1
                    event = "chopped"
                    break
        elif action == Action.CRAFT_PLANKS:
            event = "not_enough_logs"
            if logs >= 1:
                logs -= 1
                planks += 4
                event = "crafted_planks"
        elif action == Action.CRAFT_STICKS:
            event = "not_enough_planks"
            if planks >= 2:
                planks -= 2
                sticks += 4
                event = "crafted_sticks"
        else:  # CRAFT_PICKAXE
            if (x, z) != WORKBENCH:
                event = "not_at_workbench"
            elif planks < 3 or sticks < 2:
                event = "missing_pickaxe_materials"
            else:
                planks -= 3
                sticks -= 2
                success = True
                event = "crafted_pickaxe"

        self._state = (x, z, mask, logs, planks, sticks)
        self._steps += 1
        terminated = success or death
        truncated = self._steps >= self.config.max_steps and not terminated
        base_reward = -0.01 + (10.0 if success else -10.0 if death else 0.0)
        shaping_reward = 0.0
        if self.config.reward_mode == "shaped":
            # A time limit is not a terminal MDP state: keep its potential.
            next_potential = 0.0 if terminated else potential(self.state)
            shaping_reward = self.config.gamma * next_potential - potential(previous_state)
        self._needs_reset = terminated or truncated
        info = {"success": success, "death": death, "event": event,
                "base_reward": base_reward, "shaping_reward": shaping_reward,
                "steps": self._steps}
        return self.state, base_reward + shaping_reward, terminated, truncated, info

    def render(self) -> str:
        """Return a text view; north is up and east is right."""
        x, z, mask, logs, planks, sticks = self.state
        rows = []
        for row in range(SIZE):
            cells = []
            for column in range(SIZE):
                position = (column, row)
                symbol = "."
                if position == LAVA:
                    symbol = "L"
                elif position == WORKBENCH:
                    symbol = "W"
                if any(position == log_position and mask & (1 << index)
                       for index, log_position in enumerate(LOG_POSITIONS)):
                    symbol = "T"
                if position == (x, z):
                    symbol = "A"
                cells.append(symbol)
            rows.append(" ".join(cells))
        rows.append(f"logs={logs} planks={planks} sticks={sticks} steps={self._steps}")
        rows.append("A=agent T=log L=lava W=workbench")
        return "\n".join(rows)

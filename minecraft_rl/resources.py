"""Small tabular controller for the natural wood-to-pickaxe milestone.

The policy selects eight macro actions. Navigation, digging, and recipes are
hand-written helpers, not learned motor control. Three named experiment
conditions differ only in what the policy must decide:

- ``masked``: the disclosed feasibility/budget mask removes impossible and
  redundant actions (state ``resource_v1``).
- ``unmasked``: all eight actions are always allowed; an infeasible one fails
  as an ordinary observed transition (state ``resource_v1``).
- ``unmasked_history``: unmasked, with state ``resource_v2``, which adds one
  bit saying whether the previous decision left the nine ``resource_v1``
  fields unchanged.

The bounded observation omits terrain and most history: it is a partial
observation, not a claim that natural Minecraft is a finite fully observed MDP.

ResourceAgent learns over one stage specification (``StageSpec``): this wood
stage by default, or another stage such as the stone stage in ``stone.py``.

ResourceSimulator is a CPU-only teaching abstraction with correct resource
accounting and random macro failures. Its results are NOT Minecraft results.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
import json
import math
from pathlib import Path
import random


class ResourceAction(IntEnum):
    SEARCH = 0
    CHOP = 1
    CRAFT_PLANKS = 2
    CRAFT_STICKS = 3
    CRAFT_TABLE = 4
    PLACE_TABLE = 5
    APPROACH_TABLE = 6
    CRAFT_PICKAXE = 7


ACTION_NAMES = tuple(action.name for action in ResourceAction)
N_ACTIONS = len(ResourceAction)
State = tuple[int, ...]
STATE_FIELDS = ("logs", "planks", "sticks", "tables", "pickaxes",
                "nearby_log", "nearby_table", "table_placed", "health_bin")
STATE_CAPS = (3, 9, 2, 1, 1, 1, 1, 1, 2)
INVENTORY_FIELDS = STATE_FIELDS[:5]
POLICY_ENVIRONMENT = "resource_v1"
HISTORY_ENVIRONMENT = "resource_v2"
STATE_VERSIONS = {
    POLICY_ENVIRONMENT: (STATE_FIELDS, STATE_CAPS),
    HISTORY_ENVIRONMENT: (STATE_FIELDS + ("last_action_failed",), STATE_CAPS + (1,)),
}
MASKED = "recipe_and_budget_v1"
UNMASKED = "none"
ACTION_MASKS = (MASKED, UNMASKED)
# Each condition fixes the action mask and the state version together.
CONDITIONS = {
    "masked": (MASKED, POLICY_ENVIRONMENT),
    "unmasked": (UNMASKED, POLICY_ENVIRONMENT),
    "unmasked_history": (UNMASKED, HISTORY_ENVIRONMENT),
}


def condition_name(action_mask: str, state_version: str) -> str:
    """The experiment condition for a mask and state version; raises if unsupported."""
    for name, pair in CONDITIONS.items():
        if pair == (action_mask, state_version):
            return name
    raise ValueError(f"Unsupported condition: action_mask={action_mask!r}, state_version={state_version!r}")


def _number(value, name: str, low: float, high: float) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{name} must be a finite number from {low} to {high}")
    return value


def _validate_state(state: State, state_version: str = POLICY_ENVIRONMENT) -> None:
    if state_version not in STATE_VERSIONS:
        raise ValueError(f"Unknown state version: {state_version!r}")
    caps = STATE_VERSIONS[state_version][1]
    if (not isinstance(state, tuple) or len(state) != len(caps)
            or any(type(v) is not int or not 0 <= v <= cap
                   for v, cap in zip(state, caps))):
        raise ValueError("Invalid bounded resource state")


def encode_state(base_state: State, state_version: str = POLICY_ENVIRONMENT,
                 last_action_failed: bool = False) -> State:
    """The learning state for a version from the nine resource_v1 fields.

    ``resource_v2`` appends ``last_action_failed``: 1 when the previous
    decision left all nine resource_v1 fields unchanged (an infeasible action,
    a refused helper, or a search that found nothing new), 0 at an episode's
    start and after any change.
    """
    _validate_state(base_state)
    if type(last_action_failed) is not bool:
        raise ValueError("last_action_failed must be a boolean")
    if state_version == POLICY_ENVIRONMENT:
        return base_state
    if state_version == HISTORY_ENVIRONMENT:
        return base_state + (int(last_action_failed),)
    raise ValueError(f"Unknown state version: {state_version!r}")


def actions_for(state: State, action_mask: str = MASKED,
                state_version: str = POLICY_ENVIRONMENT) -> tuple[int, ...]:
    """Allowed actions in a condition: the disclosed mask, or all eight."""
    _validate_state(state, state_version)
    if action_mask == UNMASKED:
        return tuple(range(N_ACTIONS))
    if action_mask != MASKED:
        raise ValueError(f"Unknown action mask: {action_mask!r}")
    return allowed_actions(state[:len(STATE_CAPS)])


def state_from_observation(observation: dict) -> State:
    """Encode authoritative counts and flags without inventing missing values.

    Logs/planks/sticks/tables/pickaxes cap at 3/9/2/1/1. Health is zero,
    one to ten health points, or over ten points (bins 0/1/2). Food and
    position are validated and retained in full trajectories, but omitted
    from this deliberately small learning state.
    """
    if not isinstance(observation, dict):
        raise ValueError("Resource observation must be a dictionary")
    inventory = observation.get("inventory_counts")
    if not isinstance(inventory, dict):
        raise ValueError("Missing resource inventory_counts")
    values = []
    for field, cap in zip(INVENTORY_FIELDS, STATE_CAPS):
        value = inventory.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"inventory_counts.{field} must be a nonnegative integer")
        values.append(min(value, cap))
    for field in ("nearby_log", "nearby_table", "table_placed"):
        value = observation.get(field)
        if type(value) is not bool:
            raise ValueError(f"{field} must be a boolean")
        values.append(int(value))
    health = _number(observation.get("health"), "health", 0, 20)
    _number(observation.get("food"), "food", 0, 20)
    position = observation.get("position")
    if not isinstance(position, dict):
        raise ValueError("Missing position")
    for axis in ("x", "y", "z"):
        _number(position.get(axis), f"position.{axis}", -32_000_000, 32_000_000)
    values.append(0 if health == 0 else 1 if health <= 10 else 2)
    return tuple(values)


def allowed_actions(state: State) -> tuple[int, ...]:
    """Disclosed rule-based feasibility/resource-budget mask, not craft order.

    Several orders remain possible: acquire wood first, convert it early,
    craft sticks before the table, or place the table before gathering the
    remaining wood. The mask removes impossible recipes, redundant tables,
    and collecting more wood than this milestone needs. It does not choose
    among the remaining actions. SEARCH is the fallback when no productive
    action exists (for example an already terminal state).
    """
    _validate_state(state)
    logs, planks, sticks, tables, pickaxes, near_log, near_table, placed, health = state
    if pickaxes or health == 0:
        return (int(ResourceAction.SEARCH),)
    plank_need = 3 + (0 if sticks >= 2 else 2) + (0 if tables or placed else 4)
    logs_needed = math.ceil(max(0, plank_need - planks) / 4)
    actions = []
    if logs < logs_needed:
        actions.append(ResourceAction.CHOP if near_log else ResourceAction.SEARCH)
    if logs and planks < plank_need:
        actions.append(ResourceAction.CRAFT_PLANKS)
    if planks >= 2 and sticks < 2:
        actions.append(ResourceAction.CRAFT_STICKS)
    if planks >= 4 and not tables and not placed:
        actions.append(ResourceAction.CRAFT_TABLE)
    if tables and not placed:
        actions.append(ResourceAction.PLACE_TABLE)
    if placed and not near_table:
        actions.append(ResourceAction.APPROACH_TABLE)
    if placed and near_table and planks >= 3 and sticks >= 2:
        actions.append(ResourceAction.CRAFT_PICKAXE)
    return tuple(int(a) for a in actions) or (int(ResourceAction.SEARCH),)


def potential(observation: dict) -> float:
    """Bounded reversible material and crafting progress, never a paid bonus.

    Four planks are one log; two sticks cost one plank. A table carries four
    planks' material value whether held or placed. Terminal potential is
    handled separately by reward_transition so success/death never retains
    a future shaping bonus.
    """
    state_from_observation(observation)
    counts = observation["inventory_counts"]
    has_table = bool(counts["tables"] or observation["table_placed"])
    equivalent_planks = (4 * counts["logs"] + counts["planks"]
                         + counts["sticks"] / 2 + 4 * int(has_table))
    return (0.1 * min(9, equivalent_planks)
            + 0.3 * min(3, counts["planks"]) / 3
            + 0.3 * int(counts["sticks"] >= 2)
            + 0.4 * int(has_table)
            + 0.2 * int(observation["table_placed"]))


def reward_transition(previous_observation: dict, observation: dict, *,
                      gamma: float = 0.97, reward_mode: str = "shaped",
                      success: bool = False, death: bool = False) -> tuple[float, dict]:
    """Return task reward and its components; truncation is not termination."""
    _number(gamma, "gamma", 0, 1)
    if gamma == 1:
        raise ValueError("gamma must be less than one")
    if reward_mode not in ("sparse", "shaped"):
        raise ValueError("reward_mode must be sparse or shaped")
    if type(success) is not bool or type(death) is not bool or (success and death):
        raise ValueError("success and death must be mutually exclusive booleans")
    state_from_observation(previous_observation)
    state_from_observation(observation)
    base = -0.01 + (10.0 if success else -10.0 if death else 0.0)
    shaping = 0.0
    if reward_mode == "shaped":
        following = 0.0 if success or death else potential(observation)
        shaping = gamma * following - potential(previous_observation)
    return base + shaping, {"base_reward": base, "shaping_reward": shaping}


@dataclass(frozen=True, eq=False)
class StageSpec:
    """One stage's policy contract: its macro actions, state versions, and conditions.

    ``state_versions`` maps each environment name to its state fields and
    caps; ``conditions`` maps each condition to its (action mask,
    environment); ``actions_for(state, action_mask, environment)`` returns a
    condition's allowed actions. A policy file records the environment,
    action names, fields, and caps, so an agent refuses another stage's file.
    ``consistent(state, environment)``, when given, says whether the stage's
    encoder can produce a state within the caps; loading refuses a policy
    file that holds any other state.
    """

    name: str
    actions: tuple[str, ...]
    state_versions: Mapping[str, tuple[tuple[str, ...], tuple[int, ...]]]
    conditions: Mapping[str, tuple[str, str]]
    actions_for: Callable[[State, str, str], tuple[int, ...]]
    default_condition: str
    consistent: Callable[[State, str], bool] | None = None

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def condition_name(self, action_mask: str, state_version: str) -> str:
        for name, pair in self.conditions.items():
            if pair == (action_mask, state_version):
                return name
        raise ValueError(f"Unsupported {self.name} condition: action_mask={action_mask!r}, "
                         f"state_version={state_version!r}")

    def validate(self, state: State, state_version: str) -> None:
        if not isinstance(state_version, str) or state_version not in self.state_versions:
            raise ValueError(f"Unknown state version: {state_version!r}")
        caps = self.state_versions[state_version][1]
        if (not isinstance(state, tuple) or len(state) != len(caps)
                or any(type(v) is not int or not 0 <= v <= cap
                       for v, cap in zip(state, caps))):
            raise ValueError("Invalid bounded resource state")


WOOD_STAGE = StageSpec("wood", ACTION_NAMES, STATE_VERSIONS, CONDITIONS, actions_for, "masked")


class ResourceAgent:
    """Q-learning or SARSA with epsilon-greedy selection over a condition's allowed actions.

    ``stage`` fixes the actions, state versions, and conditions (the wood
    stage by default); ``action_mask`` and ``state_version`` default to the
    stage's default condition.

    ``initial_q`` is the value of every action in a state before its first
    update (default 0). With potential-based shaping, an action that fails in
    place converges towards -0.01 / (1 - gamma) - Phi(s), about -0.33 - Phi(s),
    so untried actions at ``initial_q`` look worse than a tried, failing one
    only where ``initial_q`` is below that value: the effect depends on the
    state's potential, not on the step cost alone.
    """

    def __init__(self, algorithm: str = "q_learning", alpha: float = 0.2,
                 gamma: float = 0.97, seed: int = 0, *, action_mask: str | None = None,
                 state_version: str | None = None, initial_q: float = 0.0,
                 stage: StageSpec = WOOD_STAGE):
        if not isinstance(stage, StageSpec):
            raise TypeError("stage must be a StageSpec")
        if algorithm not in ("q_learning", "sarsa", "random"):
            raise ValueError("algorithm must be q_learning, sarsa, or random")
        _number(alpha, "alpha", 0, 1)
        _number(gamma, "gamma", 0, 1)
        if alpha == 0 or gamma == 1:
            raise ValueError("Require 0 < alpha <= 1 and 0 <= gamma < 1")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        _number(initial_q, "initial_q", -100, 100)
        default_mask, default_version = stage.conditions[stage.default_condition]
        action_mask = default_mask if action_mask is None else action_mask
        state_version = default_version if state_version is None else state_version
        self.stage = stage
        self.condition = stage.condition_name(action_mask, state_version)
        self.action_mask, self.state_version = action_mask, state_version
        self.algorithm, self.alpha, self.gamma = algorithm, alpha, gamma
        self.initial_q = float(initial_q)
        self.rng = random.Random(seed)
        self.q: dict[State, list[float]] = {}

    def allowed(self, state: State) -> tuple[int, ...]:
        return self.stage.actions_for(state, self.action_mask, self.state_version)

    def values(self, state: State) -> list[float]:
        self.stage.validate(state, self.state_version)
        return self.q.get(state, [self.initial_q] * self.stage.n_actions)

    def act(self, state: State, epsilon: float = 0.0) -> int:
        _number(epsilon, "epsilon", 0, 1)
        actions = self.allowed(state)
        if self.algorithm == "random" or self.rng.random() < epsilon:
            return self.rng.choice(actions)
        values = self.values(state)
        best = max(values[a] for a in actions)
        return self.rng.choice([a for a in actions if values[a] == best])

    def update(self, state: State, action: int, reward: float,
               next_state: State, terminated: bool,
               next_action: int | None = None) -> float:
        self.stage.validate(state, self.state_version)
        self.stage.validate(next_state, self.state_version)
        if self.algorithm == "random":
            raise ValueError("A random baseline does not learn")
        if (isinstance(action, bool) or not isinstance(action, int)
                or action not in self.allowed(state)):
            raise ValueError("Action is outside the state's allowed actions")
        if (isinstance(reward, bool) or not isinstance(reward, (int, float))
                or not math.isfinite(reward) or type(terminated) is not bool):
            raise ValueError("Reward must be finite and terminated must be boolean")
        if terminated:
            bootstrap = 0.0
        elif self.algorithm == "q_learning":
            bootstrap = max(self.values(next_state)[a] for a in self.allowed(next_state))
        else:
            if (isinstance(next_action, bool) or not isinstance(next_action, int)
                    or next_action not in self.allowed(next_state)):
                raise ValueError("SARSA needs the next sampled, allowed behavior action")
            bootstrap = self.values(next_state)[next_action]
        row = self.q.setdefault(state, [self.initial_q] * self.stage.n_actions)
        td_error = reward + self.gamma * bootstrap - row[action]
        row[action] += self.alpha * td_error
        return td_error

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        fields, caps = self.stage.state_versions[self.state_version]
        # Wood policies keep their original layout, which has no stage key. The stage is matched by
        # name, as load matches it: a copied or unpickled agent holds a new StageSpec object.
        stage = {} if self.stage.name == WOOD_STAGE.name else {"stage": self.stage.name}
        payload = {
            "format_version": 1, **stage, "environment": self.state_version,
            "algorithm": self.algorithm, "alpha": self.alpha, "gamma": self.gamma,
            "initial_q": self.initial_q,
            "actions": list(self.stage.actions), "state_fields": list(fields),
            "state_caps": list(caps), "action_mask": self.action_mask,
            "condition": self.condition, "metadata": metadata or {},
            "q": [{"state": list(s), "values": v} for s, v in sorted(self.q.items())],
        }
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, seed: int = 0, *, stage: StageSpec = WOOD_STAGE) -> ResourceAgent:
        """Load a policy saved by ``stage``; a policy from any other stage is refused."""
        if not isinstance(stage, StageSpec):
            raise TypeError("stage must be a StageSpec")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("environment") if isinstance(payload, dict) else None
        if isinstance(payload, dict):
            # A policy without a stage key predates stages, so it is a wood policy.
            saved_stage = payload.get("stage", WOOD_STAGE.name)
            if saved_stage != stage.name:
                raise ValueError(f"This policy was saved by the {saved_stage} stage (environment {version!r}); "
                                 f"a {stage.name}-stage agent cannot load it")
        versions = stage.state_versions
        if (not isinstance(payload, dict) or payload.get("format_version") != 1
                or not isinstance(version, str) or version not in versions
                or payload.get("actions") != list(stage.actions)
                or payload.get("state_fields") != list(versions[version][0])
                or payload.get("state_caps") != list(versions[version][1])
                or (payload.get("action_mask"), version) not in stage.conditions.values()):
            raise ValueError("Incompatible resource policy, state encoding, action order, or condition")
        try:
            # Policies saved before initial_q was recorded all started from 0.
            agent = cls(payload["algorithm"], payload["alpha"], payload["gamma"], seed,
                        action_mask=payload["action_mask"], state_version=version,
                        initial_q=payload.get("initial_q", 0.0), stage=stage)
            if not isinstance(payload["q"], list):
                raise ValueError("Invalid Q table")
            for entry in payload["q"]:
                state, values = tuple(entry["state"]), entry["values"]
                stage.validate(state, version)
                if stage.consistent is not None and not stage.consistent(state, version):
                    raise ValueError(f"Invalid Q table: the {stage.name} stage never produces "
                                     f"the state {list(state)}")
                if (not isinstance(values, list) or len(values) != stage.n_actions
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in values)
                        or state in agent.q):
                    raise ValueError("Invalid Q table")
                agent.q[state] = list(values)
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid resource policy contents") from error
        return agent


@dataclass(frozen=True)
class ResourceSimConfig:
    gamma: float = 0.97
    reward_mode: str = "shaped"
    max_steps: int = 60
    search_success: float = 0.75
    chop_success: float = 0.9
    macro_success: float = 0.95
    action_mask: str = MASKED

    def __post_init__(self):
        _number(self.gamma, "gamma", 0, 1)
        if self.gamma == 1 or self.reward_mode not in ("sparse", "shaped"):
            raise ValueError("Invalid discount or reward mode")
        if self.action_mask not in ACTION_MASKS:
            raise ValueError("Unknown action mask")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        for field in ("search_success", "chop_success", "macro_success"):
            _number(getattr(self, field), field, 0, 1)


class ResourceSimulator:
    """Abstract macro outcomes, not Minecraft terrain/physics or a benchmark.

    Every reset begins empty. SEARCH probabilistically finds another log;
    CHOP consumes that local log and adds one item. Recipes consume actual
    counts, including all four sticks created from two planks. The simulator
    keeps uncapped inventories internally, so state binning never generates
    or destroys materials. It assumes additional trees can be found and does
    not simulate mobs, hunger, death, terrain, or real navigation cost.
    """

    def __init__(self, config: ResourceSimConfig = ResourceSimConfig()):
        if not isinstance(config, ResourceSimConfig):
            raise TypeError("config must be ResourceSimConfig")
        self.config = config
        self._rng = random.Random()
        self._observation: dict | None = None
        self._steps = 0
        self._done = True

    def observe(self) -> dict:
        if self._observation is None:
            raise RuntimeError("Call reset before observing")
        return {**self._observation,
                "inventory_counts": dict(self._observation["inventory_counts"]),
                "position": dict(self._observation["position"])}

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            if type(seed) is not int:
                raise ValueError("seed must be an integer or None")
            self._rng.seed(seed)
        self._steps, self._done = 0, False
        self._observation = {
            "inventory_counts": dict.fromkeys(INVENTORY_FIELDS, 0),
            "nearby_log": self._rng.random() < 0.5,
            "nearby_table": False, "table_placed": False,
            "health": 20, "food": 20,
            "position": {"x": 0.0, "y": 64.0, "z": 0.0},
            "backend": "resource-macro-simulator",
        }
        return self.observe()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("Call reset before stepping a new episode")
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < N_ACTIONS:
            raise ValueError("Invalid resource action")
        before = self.observe()
        observation = self._observation
        assert observation is not None
        counts = observation["inventory_counts"]
        event = "unavailable"
        chance = (self.config.search_success if action == ResourceAction.SEARCH
                  else self.config.chop_success if action == ResourceAction.CHOP
                  else self.config.macro_success)
        if self.config.action_mask == MASKED and action not in allowed_actions(state_from_observation(before)):
            event = "masked_action"
        elif self._rng.random() >= chance:
            event = "macro_failed"
        elif action == ResourceAction.SEARCH:
            observation["nearby_log"] = True
            observation["nearby_table"] = False
            observation["position"]["x"] += 4
            event = "found_log"
        elif action == ResourceAction.CHOP and observation["nearby_log"]:
            counts["logs"] += 1
            observation["nearby_log"] = False
            event = "collected_log"
        elif action == ResourceAction.CRAFT_PLANKS and counts["logs"] >= 1:
            counts["logs"] -= 1
            counts["planks"] += 4
            event = "crafted_planks"
        elif action == ResourceAction.CRAFT_STICKS and counts["planks"] >= 2:
            counts["planks"] -= 2
            counts["sticks"] += 4
            event = "crafted_sticks"
        elif action == ResourceAction.CRAFT_TABLE and counts["planks"] >= 4:
            counts["planks"] -= 4
            counts["tables"] += 1
            event = "crafted_table"
        elif action == ResourceAction.PLACE_TABLE and counts["tables"] >= 1 and not observation["table_placed"]:
            counts["tables"] -= 1
            observation["table_placed"] = observation["nearby_table"] = True
            event = "placed_table"
        elif action == ResourceAction.APPROACH_TABLE and observation["table_placed"]:
            observation["nearby_table"] = True
            observation["nearby_log"] = False
            event = "approached_table"
        elif (action == ResourceAction.CRAFT_PICKAXE and counts["planks"] >= 3
              and counts["sticks"] >= 2 and observation["nearby_table"]):
            counts["planks"] -= 3
            counts["sticks"] -= 2
            counts["pickaxes"] += 1
            event = "crafted_pickaxe"
        self._steps += 1
        success = counts["pickaxes"] > 0
        terminated = success
        truncated = self._steps >= self.config.max_steps and not terminated
        self._done = terminated or truncated
        following = self.observe()
        reward, components = reward_transition(before, following, gamma=self.config.gamma,
                                               reward_mode=self.config.reward_mode, success=success)
        return following, reward, terminated, truncated, {
            **components, "success": success, "death": False,
            "event": event, "steps": self._steps,
            "backend": "resource-macro-simulator",
        }


def train_simulator(episodes: int, seed: int = 0, *, algorithm: str = "q_learning",
                    alpha: float = 0.2, gamma: float = 0.97,
                    epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                    config: ResourceSimConfig | None = None, condition: str = "masked",
                    schedule: str = "three_quarters",
                    initial_q: float = 0.0) -> tuple[ResourceAgent, list[dict]]:
    """Optional, explicitly labeled simulation pretraining; never game evidence.

    ``schedule`` "three_quarters" decays epsilon over the first 75% of
    episodes (the original pretraining schedule); "linear" decays it over all
    episodes exactly as live training does.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition!r}")
    if schedule not in ("three_quarters", "linear"):
        raise ValueError("schedule must be three_quarters or linear")
    action_mask, state_version = CONDITIONS[condition]
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    _number(epsilon_start, "epsilon_start", 0, 1)
    _number(epsilon_end, "epsilon_end", 0, 1)
    if epsilon_end > epsilon_start:
        raise ValueError("epsilon_end must not exceed epsilon_start")
    if algorithm not in ("q_learning", "sarsa"):
        raise ValueError("Training requires q_learning or sarsa")
    agent = ResourceAgent(algorithm, alpha, gamma, seed, action_mask=action_mask, state_version=state_version,
                          initial_q=initial_q)
    config = config or ResourceSimConfig(gamma=gamma, action_mask=action_mask)
    if config.gamma != gamma:
        raise ValueError("Learning and shaping discounts must match")
    if config.action_mask != action_mask:
        raise ValueError("The simulator's action mask must match the condition")
    environment = ResourceSimulator(config)
    seed_rng = random.Random(seed + 8731)
    records = []
    total_steps = 0
    for episode in range(episodes):
        episode_seed = seed_rng.randrange(1_000_000)
        observation = environment.reset(episode_seed)
        base = state_from_observation(observation)
        state = encode_state(base, state_version)
        if schedule == "linear":
            fraction = episode / max(1, episodes - 1)
        else:
            fraction = min(1.0, episode / max(1, episodes * 0.75 - 1))
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        action = agent.act(state, epsilon)
        episode_return = base_return = 0.0
        while True:
            observation, reward, terminated, truncated, info = environment.step(action)
            following_base = state_from_observation(observation)
            following = encode_state(following_base, state_version, following_base == base)
            next_action = None if terminated else agent.act(following, epsilon)
            agent.update(state, action, reward, following, terminated, next_action)
            episode_return += reward
            base_return += info["base_reward"]
            total_steps += 1
            if terminated or truncated:
                break
            state, base, action = following, following_base, next_action
        records.append({
            "episode": episode + 1, "episode_seed": episode_seed,
            "epsilon": epsilon, "return": episode_return, "base_return": base_return,
            "steps": info["steps"], "total_steps": total_steps,
            "success": int(info["success"]), "death": 0,
            "truncated": int(truncated), "q_states": len(agent.q),
            "condition": condition, "backend": "resource-macro-simulator",
        })
    return agent, records

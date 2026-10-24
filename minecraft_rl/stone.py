"""Stone stage of the natural stone-pickaxe milestone (state ``stone_v1``).

After the frozen wood policy crafts and holds a wooden pickaxe, a second small
tabular learner chooses among five scripted stone macro actions, bridge
indices 8 to 12. Finding, digging, mining, walking, and crafting are
hand-written helpers in the bridge; the learner chooses decisions, not motor
control. All five actions are allowed in every state (condition
``stone_unmasked``). ``stone_reference_v1`` is a diagnostic mask that is
recorded but never used to choose or update an action.

StoneSimulator is a CPU-only abstraction for choosing budgets. Its results are
NOT Minecraft results.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
import itertools
import random

from .resources import UNMASKED, ResourceAgent, StageSpec, State, _number


class StoneAction(IntEnum):
    SEARCH_STONE = 0
    DIG_DOWN = 1
    MINE_STONE = 2
    RETURN_TO_TABLE = 3
    CRAFT_STONE_PICKAXE = 4


STONE_ACTION_NAMES = tuple(action.name for action in StoneAction)
# The bridge appends the stone actions after its eight wood actions.
FIRST_BRIDGE_ACTION = 8
STONE_ENVIRONMENT = "stone_v1"
STONE_FIELDS = ("stone_items", "nearby_stone", "stone_visible", "nearby_table", "health_bin")
STONE_CAPS = (3, 1, 1, 1, 2)
STONE_CONDITION = "stone_unmasked"
STONE_REFERENCE_MASK = "stone_reference_v1"
# Each stone pickaxe recipe (all 3x3, so a table is needed) takes three of one of these and two sticks.
STONE_MATERIALS = ("cobblestone", "cobbled_deepslate", "blackstone")
PICKAXE_MATERIALS = 3
PICKAXE_STICKS = 2
STEP_COST = -0.01
SUCCESS_REWARD = 10.0
DEATH_REWARD = -10.0
BACKEND = "stone-macro-simulator"
SIMULATOR_LABEL = "SIMULATOR ONLY; NOT MINECRAFT EVIDENCE"


def bridge_action(action: int) -> int:
    """The bridge index (8 to 12) of a stone action (0 to 4)."""
    if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < len(StoneAction):
        raise ValueError("Invalid stone action")
    return FIRST_BRIDGE_ACTION + action


def stone_actions_for(state: State, action_mask: str = UNMASKED,
                      state_version: str = STONE_ENVIRONMENT) -> tuple[int, ...]:
    """All five stone actions in every state: the stone stage has no action mask."""
    STONE_STAGE.validate(state, state_version)
    if action_mask != UNMASKED:
        raise ValueError(f"Unknown stone action mask: {action_mask!r}")
    return tuple(range(len(StoneAction)))


STONE_STAGE = StageSpec("stone", STONE_ACTION_NAMES, {STONE_ENVIRONMENT: (STONE_FIELDS, STONE_CAPS)},
                        {STONE_CONDITION: (UNMASKED, STONE_ENVIRONMENT)}, stone_actions_for, STONE_CONDITION)


def consistent(state: State) -> bool:
    """Stone in reach is always visible: nearby_stone=1 with stone_visible=0 never occurs."""
    STONE_STAGE.validate(state, STONE_ENVIRONMENT)
    return not (state[1] and not state[2])


# 4 x 2 x 2 x 2 x 3 = 96 nominal states, 64 with health above 0, 48 of them consistent.
STONE_LIVE_STATES = tuple(state for state in itertools.product(*(range(cap + 1) for cap in STONE_CAPS))
                          if state[4] > 0 and consistent(state))


def _health_bin(health) -> int:
    # The wood state's bins: zero, one to ten points, or over ten points.
    health = _number(health, "health", 0, 20)
    return 0 if health == 0 else 1 if health <= 10 else 2


def stone_state_from_observation(observation: dict) -> State:
    """Encode ``stone_v1`` from exactly five observation values.

    These are inventory_counts.stone_materials (capped at 3), nearby_stone,
    stone_visible, nearby_table, and health (bins 0/1/2). Nothing else is
    read, in particular nothing under ``diagnostic_only``. The bridge sets
    stone_visible whenever nearby_stone is set, so an observation with stone
    in reach but not visible is rejected rather than repaired.
    """
    if not isinstance(observation, dict):
        raise ValueError("Stone observation must be a dictionary")
    inventory = observation.get("inventory_counts")
    if not isinstance(inventory, dict):
        raise ValueError("Missing resource inventory_counts")
    materials = inventory.get("stone_materials")
    if type(materials) is not int or materials < 0:
        raise ValueError("inventory_counts.stone_materials must be a nonnegative integer")
    flags = []
    for field in ("nearby_stone", "stone_visible", "nearby_table"):
        value = observation.get(field)
        if type(value) is not bool:
            raise ValueError(f"{field} must be a boolean")
        flags.append(int(value))
    if flags[0] and not flags[1]:
        raise ValueError("stone_visible must be true whenever nearby_stone is true")
    return (min(materials, STONE_CAPS[0]), *flags, _health_bin(observation.get("health")))


def stone_state_potential(state: State) -> float:
    """Phi_stone of a ``stone_v1`` state, from 0 to 1.1."""
    STONE_STAGE.validate(state, STONE_ENVIRONMENT)
    items, near_stone, _, near_table, _ = state
    return 0.3 * items + 0.1 * near_stone * int(items < 3) + 0.2 * near_table * int(items >= 3)


def stone_potential(observation: dict) -> float:
    """Phi_stone of an observation: bounded stone progress, never a paid bonus.

    0.3 per material up to three, 0.1 for stone in reach while short of
    three, and 0.2 for being at the table with three. Terminal potential is handled by stone_reward_transition, so success and
    death never keep a future shaping bonus.
    """
    return stone_state_potential(stone_state_from_observation(observation))


def stone_reward_transition(previous_observation: dict, observation: dict, *,
                            gamma: float = 0.97, reward_mode: str = "shaped",
                            success: bool = False, death: bool = False) -> tuple[float, dict]:
    """The wood reward's structure with Phi_stone; truncation is not termination.

    The base reward is -0.01 per decision, +10 on a confirmed held stone
    pickaxe, and -10 on death.
    """
    _number(gamma, "gamma", 0, 1)
    if gamma == 1:
        raise ValueError("gamma must be less than one")
    if reward_mode not in ("sparse", "shaped"):
        raise ValueError("reward_mode must be sparse or shaped")
    if type(success) is not bool or type(death) is not bool or (success and death):
        raise ValueError("success and death must be mutually exclusive booleans")
    stone_state_from_observation(previous_observation)
    stone_state_from_observation(observation)
    base = STEP_COST + (SUCCESS_REWARD if success else DEATH_REWARD if death else 0.0)
    shaping = 0.0
    if reward_mode == "shaped":
        following = 0.0 if success or death else stone_potential(observation)
        shaping = gamma * following - stone_potential(previous_observation)
    return base + shaping, {"base_reward": base, "shaping_reward": shaping}


def stone_reference_actions(state: State) -> tuple[int, ...]:
    """The diagnostic mask ``stone_reference_v1``; never used to choose an action.

    Search or dig with fewer than three materials and no stone in reach; mine
    with fewer than three and stone in reach; return with three away from the
    table; craft with three at the table. Reports measure the share of
    choices outside it.
    """
    STONE_STAGE.validate(state, STONE_ENVIRONMENT)
    items, near_stone, _, near_table, _ = state
    if items < 3:
        if near_stone:
            return (int(StoneAction.MINE_STONE),)
        return (int(StoneAction.SEARCH_STONE), int(StoneAction.DIG_DOWN))
    return (int(StoneAction.CRAFT_STONE_PICKAXE),) if near_table else (int(StoneAction.RETURN_TO_TABLE),)


HANDOFF_REQUIREMENTS = ("wooden_pickaxe_not_crafted", "wooden_pickaxe_not_held", "sticks_short",
                        "table_not_placed", "table_not_in_reach", "stone_materials_already_held")


def handoff_failures(observation: dict) -> tuple[str, ...]:
    """The unmet stone-stage start requirements, in HANDOFF_REQUIREMENTS order; empty when stage 2 may start.

    Stage 2 starts on the observation after the wood action whose event is
    crafted_wooden_pickaxe, when the wooden pickaxe is counted and held, at
    least two sticks remain, the own table is placed and in reach, and fewer
    than three stone materials are held. A missing or invalid field raises:
    an incomplete observation is not an unmet requirement.
    """
    if not isinstance(observation, dict):
        raise ValueError("Hand-off observation must be a dictionary")
    inventory = observation.get("inventory_counts")
    if not isinstance(inventory, dict):
        raise ValueError("Missing resource inventory_counts")
    counts = {}
    for field in ("pickaxes", "sticks", "stone_materials"):
        value = inventory.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"inventory_counts.{field} must be a nonnegative integer")
        counts[field] = value
    for field in ("table_placed", "nearby_table"):
        if type(observation.get(field)) is not bool:
            raise ValueError(f"{field} must be a boolean")
    if not isinstance(observation.get("event"), str):
        raise ValueError("event must be a string")
    held = observation.get("held_item")
    if held is not None and not isinstance(held, str):
        raise ValueError("held_item must be a string or null")
    unmet = (observation["event"] != "crafted_wooden_pickaxe",
             held != "wooden_pickaxe" or counts["pickaxes"] < 1,
             counts["sticks"] < PICKAXE_STICKS,
             not observation["table_placed"],
             not observation["nearby_table"],
             counts["stone_materials"] >= PICKAXE_MATERIALS)
    return tuple(name for name, failed in zip(HANDOFF_REQUIREMENTS, unmet) if failed)


@dataclass(frozen=True)
class StoneSimConfig:
    """SIMULATOR ONLY: the hidden-terrain and macro-outcome probabilities of StoneSimulator.

    q_s and q_d default to the design's central corner (0.5 and 0.75; the
    corners are 0.2/0.5/0.8 and 0.6/0.75/0.9). The DIG_DOWN, damage, death,
    and craft values are the design's estimates. p_v, p_nav, p_block, p_rev,
    p_mine, p_ret, damage, and face_max are grid-only parameters; their
    defaults are guesses, to be replaced by rates calibrated from the survey
    and the helper probes.
    """

    gamma: float = 0.97
    reward_mode: str = "shaped"
    max_steps: int = 30
    q_s: float = 0.5  # surface stone in the world
    p_v: float = 0.5  # stone visible at the start, given surface stone (guess)
    q_d: float = 0.75  # good digging ground
    p_nav: float = 0.8  # SEARCH_STONE reaches a visible target (guess)
    p_block: float = 0.5  # a failed approach leaves no unblocked target visible (guess)
    p_rev: float = 0.3  # exploring reveals stone, only with surface stone (guess)
    p_mine: float = 0.9  # MINE_STONE collects one material (guess)
    p_ret: float = 0.9  # RETURN_TO_TABLE reaches the table (guess)
    dig_good: float = 0.8  # DIG_DOWN reaches stone in good ground
    dig_bad: float = 0.1  # DIG_DOWN reaches stone in bad ground
    p_damage: float = 0.03  # a staircase dig costs health
    p_death: float = 0.003  # a staircase dig kills the bot
    damage: int = 4  # health points lost to damage (guess)
    p_craft: float = 0.95  # the craft is confirmed
    face_max: int = 6  # a reached face has U{1..face_max} minable blocks

    def __post_init__(self):
        _number(self.gamma, "gamma", 0, 1)
        if self.gamma == 1 or self.reward_mode not in ("sparse", "shaped"):
            raise ValueError("Invalid discount or reward mode")
        for name in ("max_steps", "damage", "face_max"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.damage > 20:
            raise ValueError("damage must be at most 20 health points")
        for field in fields(self):
            if field.name.startswith(("q_", "p_", "dig_")):
                _number(getattr(self, field.name), field.name, 0, 1)
        if self.p_damage + self.p_death > 1:
            raise ValueError("p_damage + p_death must not exceed 1")


class StoneSimulator:
    """SIMULATOR ONLY: abstract stone macro outcomes, not Minecraft terrain, physics, or evidence.

    Every reset is the hand-off state: no stone materials, 3 planks and 2
    sticks, the wooden pickaxe held, the own table placed and in reach,
    health 20, and no stone in reach. Hidden terrain is drawn per episode:
    surface stone (q_s), stone visible at the start given surface stone
    (p_v), and good digging ground (q_d). Materials come only from MINE_STONE;
    the craft takes 3 materials and 2 sticks at the table. A refused action
    changes nothing. A reached face holds U{1..face_max} blocks and is left
    behind on a return to the table. It decides budgets; it is never evidence.
    """

    def __init__(self, config: StoneSimConfig = StoneSimConfig()):
        if not isinstance(config, StoneSimConfig):
            raise TypeError("config must be StoneSimConfig")
        self.config = config
        self._rng = random.Random()
        self._observation: dict | None = None
        self._terrain: dict | None = None
        self._face = 0
        self._target = False
        self._steps = 0
        self._done = True

    def observe(self) -> dict:
        if self._observation is None:
            raise RuntimeError("Call reset before observing")
        return {**self._observation,
                "inventory_counts": dict(self._observation["inventory_counts"]),
                "position": dict(self._observation["position"])}

    def hidden_terrain(self) -> dict:
        """This episode's hidden draws, for tests and diagnostics; the learner never sees them."""
        if self._terrain is None:
            raise RuntimeError("Call reset before reading the terrain")
        return dict(self._terrain)

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            if type(seed) is not int:
                raise ValueError("seed must be an integer or None")
            self._rng.seed(seed)
        config = self.config
        # Three draws in every episode, so one probability never shifts the others' streams.
        draws = [self._rng.random() for _ in range(3)]
        surface = draws[0] < config.q_s
        self._terrain = {"surface_stone": surface, "visible_at_start": surface and draws[1] < config.p_v,
                         "dig_good": draws[2] < config.q_d}
        self._face, self._target = 0, self._terrain["visible_at_start"]
        self._steps, self._done = 0, False
        self._observation = {
            "inventory_counts": {"logs": 0, "planks": 3, "sticks": 2, "tables": 0, "pickaxes": 1,
                                 "stone_materials": 0, "stone_pickaxes": 0},
            "nearby_stone": False, "stone_visible": self._target,
            "nearby_table": True, "table_placed": True,
            "health": 20, "food": 20, "held_item": "wooden_pickaxe",
            "position": {"x": 0.0, "y": 64.0, "z": 0.0},
            "backend": BACKEND,
        }
        return self.observe()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("Call reset before stepping a new episode")
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < len(StoneAction):
            raise ValueError("Invalid stone action")
        before = self.observe()
        observation, config, rng = self._observation, self.config, self._rng
        assert observation is not None and self._terrain is not None
        counts = observation["inventory_counts"]
        if action == StoneAction.SEARCH_STONE:
            event = self._search()
        elif action == StoneAction.DIG_DOWN:
            if self._face:
                event = "stone_already_reachable"
            elif rng.random() >= (config.dig_good if self._terrain["dig_good"] else config.dig_bad):
                event = "stair_refused"
            else:
                self._face = rng.randint(1, config.face_max)
                observation["nearby_table"] = False
                event = "stair_steps;stone_exposed"
                hazard = rng.random()
                if hazard < config.p_death:
                    observation["health"] = 0
                elif hazard < config.p_death + config.p_damage:
                    observation["health"] = max(0, observation["health"] - config.damage)
                    event += ";took_damage"
        elif action == StoneAction.MINE_STONE:
            if not self._face:
                event = "no_reachable_stone"
            elif rng.random() >= config.p_mine:
                event = "stone_unreachable"
            else:
                counts["stone_materials"] += 1
                self._face -= 1
                event = "cobblestone_collected"
        elif action == StoneAction.RETURN_TO_TABLE:
            if observation["nearby_table"]:
                event = "table_reached"
            elif rng.random() >= config.p_ret:
                event = "table_path_blocked"
            else:
                observation["nearby_table"] = True
                self._face = 0
                self._target = self._terrain["surface_stone"] and rng.random() < config.p_v
                event = "table_reached"
        else:
            if not observation["nearby_table"]:
                event = "not_at_own_table"
            elif counts["stone_materials"] < PICKAXE_MATERIALS or counts["sticks"] < PICKAXE_STICKS:
                event = "missing_materials"
            elif rng.random() >= config.p_craft:
                event = "craft_not_confirmed"
            else:
                counts["stone_materials"] -= PICKAXE_MATERIALS
                counts["sticks"] -= PICKAXE_STICKS
                counts["stone_pickaxes"] += 1
                observation["held_item"] = "stone_pickaxe"
                event = "crafted_stone_pickaxe"
        observation["nearby_stone"] = self._face > 0
        observation["stone_visible"] = self._face > 0 or self._target
        self._steps += 1
        success = counts["stone_pickaxes"] > 0 and observation["held_item"] == "stone_pickaxe"
        death = observation["health"] == 0
        terminated = success or death
        truncated = self._steps >= config.max_steps and not terminated
        self._done = terminated or truncated
        following = self.observe()
        reward, components = stone_reward_transition(before, following, gamma=config.gamma,
                                                     reward_mode=config.reward_mode, success=success, death=death)
        return following, reward, terminated, truncated, {
            **components, "success": success, "death": death,
            "event": event, "steps": self._steps, "backend": BACKEND,
        }

    def _search(self) -> str:
        observation, config, rng = self._observation, self.config, self._rng
        assert observation is not None and self._terrain is not None
        if self._face:
            return "stone_already_reachable"
        # Any search walks away from the table.
        observation["nearby_table"] = False
        if self._target:
            if rng.random() < config.p_nav:
                self._face = rng.randint(1, config.face_max)
                return "stone_reached"
            if rng.random() < config.p_block:
                self._target = False
            return "stone_path_blocked"
        if self._terrain["surface_stone"] and rng.random() < config.p_rev:
            self._target = True
            return "explored;stone_visible"
        return "explored"


def linear_epsilon(episode: int, episodes: int, start: float, end: float) -> float:
    """Live training's schedule: ``start`` in the first episode (index 0) to ``end`` in the last."""
    return start + (end - start) * episode / max(1, episodes - 1)


def train_stone_simulator(episodes: int, seed: int = 0, *, algorithm: str = "q_learning",
                          alpha: float = 0.2, gamma: float = 0.97,
                          epsilon_start: float = 0.6, epsilon_end: float = 0.1,
                          config: StoneSimConfig | None = None,
                          initial_q: float = 0.0) -> tuple[ResourceAgent, list[dict]]:
    """SIMULATOR ONLY: train a stone learner; never game evidence.

    It uses the live agent class, reward function, and linear epsilon
    schedule. A step limit truncates without terminating, so updates still
    bootstrap. Each record counts the choices outside stone_reference_v1, a
    diagnostic that never feeds back into a choice.
    """
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    _number(epsilon_start, "epsilon_start", 0, 1)
    _number(epsilon_end, "epsilon_end", 0, 1)
    if epsilon_end > epsilon_start:
        raise ValueError("epsilon_end must not exceed epsilon_start")
    if algorithm not in ("q_learning", "sarsa"):
        raise ValueError("Training requires q_learning or sarsa")
    agent = ResourceAgent(algorithm, alpha, gamma, seed, initial_q=initial_q, stage=STONE_STAGE)
    config = config or StoneSimConfig(gamma=gamma)
    if not isinstance(config, StoneSimConfig):
        raise TypeError("config must be StoneSimConfig")
    if config.gamma != gamma:
        raise ValueError("Learning and shaping discounts must match")
    environment = StoneSimulator(config)
    seed_rng = random.Random(seed + 8731)
    records = []
    total_steps = 0
    for episode in range(episodes):
        episode_seed = seed_rng.randrange(1_000_000)
        state = stone_state_from_observation(environment.reset(episode_seed))
        epsilon = linear_epsilon(episode, episodes, epsilon_start, epsilon_end)
        action = agent.act(state, epsilon)
        episode_return = base_return = 0.0
        outside = 0
        while True:
            outside += int(action not in stone_reference_actions(state))
            observation, reward, terminated, truncated, info = environment.step(action)
            following = stone_state_from_observation(observation)
            # A time limit truncates sampling, so SARSA still samples its bootstrap action.
            next_action = None if terminated else agent.act(following, epsilon)
            agent.update(state, action, reward, following, terminated, next_action)
            episode_return += reward
            base_return += info["base_reward"]
            total_steps += 1
            if terminated or truncated:
                break
            state, action = following, next_action
        records.append({
            "episode": episode + 1, "episode_seed": episode_seed,
            "epsilon": epsilon, "return": episode_return, "base_return": base_return,
            "steps": info["steps"], "total_steps": total_steps,
            "success": int(info["success"]), "death": int(info["death"]),
            "truncated": int(truncated), "q_states": len(agent.q),
            "outside_reference_choices": outside, **environment.hidden_terrain(),
            "condition": agent.condition, "backend": BACKEND,
        })
    return agent, records


def evaluate_stone_policy(agent: ResourceAgent, episodes: int, seed: int,
                          config: StoneSimConfig | None = None) -> dict:
    """SIMULATOR ONLY: frozen greedy play (epsilon 0, no updates) on seeds seed, seed+1, ...

    An untried choice picks an action never updated in that state, as in
    scripts/initial_q_study.py.
    """
    if not isinstance(agent, ResourceAgent) or agent.stage is not STONE_STAGE:
        raise ValueError("A stone evaluation needs a stone-stage ResourceAgent")
    if type(episodes) is not int or episodes < 1 or type(seed) is not int:
        raise ValueError("episodes must be a positive integer and seed an integer")
    environment = StoneSimulator(config or StoneSimConfig(gamma=agent.gamma))
    successes = deaths = decisions = untried = outside = 0
    for index in range(episodes):
        state = stone_state_from_observation(environment.reset(seed + index))
        while True:
            action = agent.act(state, 0.0)
            row = agent.q.get(state)
            untried += int(row is None or row[action] == agent.initial_q)
            outside += int(action not in stone_reference_actions(state))
            decisions += 1
            observation, _, terminated, truncated, info = environment.step(action)
            state = stone_state_from_observation(observation)
            if terminated or truncated:
                break
        successes += int(info["success"])
        deaths += int(info["death"])
    return {"episodes": episodes, "success_rate": successes / episodes, "death_rate": deaths / episodes,
            "mean_decisions": decisions / episodes, "untried_choice_fraction": untried / decisions,
            "outside_reference_fraction": outside / decisions, "label": SIMULATOR_LABEL}

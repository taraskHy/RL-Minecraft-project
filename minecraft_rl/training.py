"""Training and frozen-policy evaluation share a small environment interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from statistics import mean, stdev

from .agents import TabularAgent
from .environment import CraftingEnv, EnvConfig


@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str = "q_learning"
    episodes: int = 15000
    alpha: float = 0.2
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_fraction: float = 0.75
    seed: int = 0

    def __post_init__(self):
        if self.algorithm not in ("q_learning", "sarsa"):
            raise ValueError("Training requires q_learning or sarsa")
        if type(self.episodes) is not int or self.episodes <= 0:
            raise ValueError("episodes must be a positive integer")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1")
        if not 0 < self.decay_fraction <= 1:
            raise ValueError("decay_fraction must be in (0, 1]")
        TabularAgent(self.algorithm, self.alpha, self.gamma, self.seed)

    def epsilon(self, episode: int) -> float:
        fraction = min(1.0, episode / max(1, self.episodes * self.decay_fraction - 1))
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


def train(config: TrainingConfig, env_config: EnvConfig):
    if config.gamma != env_config.gamma:
        raise ValueError("Learning and potential shaping must use the same gamma")
    env = CraftingEnv(env_config)
    agent = TabularAgent(config.algorithm, config.alpha, config.gamma, config.seed)
    seed_rng = random.Random(config.seed + 8731)
    records = []
    total_steps = 0
    for episode in range(config.episodes):
        episode_seed = seed_rng.randrange(1_000_000)
        state = env.reset(seed=episode_seed)
        epsilon = config.epsilon(episode)
        action = agent.act(state, epsilon)
        episode_return = base_return = 0.0
        while True:
            next_state, reward, terminated, truncated, info = env.step(action)
            # A time limit is an external cutoff, not a terminal MDP state.
            # SARSA still samples its next action to bootstrap at that cutoff.
            next_action = None if terminated else agent.act(next_state, epsilon)
            agent.update(state, action, reward, next_state, terminated, next_action)
            episode_return += reward
            base_return += info["base_reward"]
            total_steps += 1
            if terminated or truncated:
                break
            state, action = next_state, next_action
        records.append({
            "episode": episode + 1, "episode_seed": episode_seed,
            "epsilon": epsilon, "return": episode_return, "base_return": base_return,
            "steps": info["steps"], "total_steps": total_steps,
            "success": int(info["success"]), "death": int(info["death"]),
            "truncated": int(truncated), "q_states": len(agent.q),
        })
    return agent, records


def evaluate(agent: TabularAgent, env_config: EnvConfig, seeds: list[int]):
    if not seeds:
        raise ValueError("Evaluation needs at least one episode")
    env = CraftingEnv(env_config)
    records = []
    for seed in seeds:
        # Separate RNG: validation cannot alter the subsequent training trajectory.
        policy = TabularAgent(agent.algorithm, agent.alpha, agent.gamma, seed + 4817)
        policy.q = agent.q  # read-only: act never inserts rows or changes values
        state = env.reset(seed=seed)
        episode_return = base_return = 0.0
        unseen = 0
        while True:
            unseen += int(state not in policy.q)
            action = policy.act(state, epsilon=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            base_return += info["base_reward"]
            if terminated or truncated:
                break
        records.append({
            "episode_seed": seed, "return": episode_return, "base_return": base_return,
            "steps": info["steps"], "success": int(info["success"]),
            "death": int(info["death"]), "truncated": int(truncated),
            "unseen_state_steps": unseen,
        })
    return records


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("No episodes to summarize")
    n = len(records)
    p = mean(row["success"] for row in records)
    # Wilson interval describes these evaluation episodes, not training-seed uncertainty.
    z = 1.96
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    successes = [r["steps"] for r in records if r["success"]]
    return {
        "episodes": n, "success_rate": p,
        "success_wilson_95": [center - half, center + half],
        "death_rate": mean(row["death"] for row in records),
        "truncation_rate": mean(row["truncated"] for row in records),
        "mean_steps": mean(row["steps"] for row in records),
        "mean_success_steps": mean(successes) if successes else None,
        "mean_base_return": mean(row["base_return"] for row in records),
        "mean_training_reward_return": mean(row["return"] for row in records),
    }


def aggregate_seed_summaries(summaries: list[dict]) -> dict:
    rates = [s["success_rate"] for s in summaries]
    return {"training_seeds": len(rates), "mean_success_rate": mean(rates),
            "std_success_rate_across_training_seeds": stdev(rates) if len(rates) > 1 else None,
            "mean_base_return": mean(s["mean_base_return"] for s in summaries),
            "mean_death_rate": mean(s["death_rate"] for s in summaries)}

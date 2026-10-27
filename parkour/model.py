"""Actor-critic network with a factored categorical policy.

The actor outputs one categorical distribution per action component (move,
strafe, jump, sprint, yaw step); the action is sampled component-wise. The
critic is a separate MLP. Observations are standardised with running
statistics that are frozen in evaluation and exported with the weights, so
the JavaScript runner (bridge/policy.js) reproduces the exact same forward
pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from .env import ACTION_DIMS, OBS_DIM


class RunningNorm(nn.Module):
    def __init__(self, dim: int, clip: float = 5.0, eps: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(eps, dtype=torch.float64))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.to(torch.float64)
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta.pow(2) * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.var + 1e-8).to(x.dtype)
        return ((x - self.mean.to(x.dtype)) / std).clamp(-self.clip, self.clip)


def mlp(sizes, act=nn.ELU, out_gain=1.0):
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        nn.init.orthogonal_(lin.weight, gain=out_gain if last else 2 ** 0.5)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if not last:
            layers.append(act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden=(512, 256, 128), action_dims=ACTION_DIMS):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden = tuple(hidden)
        self.action_dims = tuple(action_dims)
        self.norm = RunningNorm(obs_dim)
        self.actor = mlp((obs_dim, *hidden, sum(action_dims)), out_gain=0.01)
        self.critic = mlp((obs_dim, *hidden, 1), out_gain=1.0)

    def logits(self, obs_n: torch.Tensor):
        return torch.split(self.actor(obs_n), self.action_dims, dim=-1)

    def value(self, obs_n: torch.Tensor) -> torch.Tensor:
        return self.critic(obs_n).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        obs_n = self.norm(obs)
        logits = self.logits(obs_n)
        acts, logp = [], 0.0
        for lg in logits:
            dist = torch.distributions.Categorical(logits=lg)
            a = lg.argmax(-1) if deterministic else dist.sample()
            acts.append(a)
            logp = logp + dist.log_prob(a)
        return torch.stack(acts, -1), logp, self.value(obs_n)

    def evaluate(self, obs_n: torch.Tensor, actions: torch.Tensor):
        logits = self.logits(obs_n)
        logp, ent = 0.0, 0.0
        for i, lg in enumerate(logits):
            dist = torch.distributions.Categorical(logits=lg)
            logp = logp + dist.log_prob(actions[:, i])
            ent = ent + dist.entropy()
        return logp, ent, self.value(obs_n)

    def export(self, path: Path, meta: dict | None = None) -> None:
        """Weights and normaliser as JSON for bridge/policy.js."""
        layers = []
        for m in self.actor:
            if isinstance(m, nn.Linear):
                layers.append({"w": m.weight.detach().cpu().double().tolist(), "b": m.bias.detach().cpu().double().tolist()})
        data = {
            "format": "parkour-policy-v1", "obs_dim": self.obs_dim, "action_dims": list(self.action_dims),
            "activation": "elu", "norm_mean": self.norm.mean.cpu().tolist(),
            "norm_std": torch.sqrt(self.norm.var + 1e-8).cpu().tolist(), "norm_clip": self.norm.clip,
            "layers": layers, "meta": meta or {},
        }
        Path(path).write_text(json.dumps(data))

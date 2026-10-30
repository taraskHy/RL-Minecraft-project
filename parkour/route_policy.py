"""The route policy: scores the platforms in view (parkour.route_env tokens) and values the scene.

A pointer network over a set: each token goes through a shared MLP, a small transformer encoder lets
the tokens see each other (no positional encoding, so the result does not depend on token order),
and a linear head gives one logit per token; tokens that are not selectable get -1e9. The critic has
its own encoder and pools the tokens (masked mean and max) into one value.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from .route_env import NEG, TOKEN_DIM


class SetEncoder(nn.Module):
    def __init__(self, d: int, layers: int, heads: int):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(TOKEN_DIM, d), nn.ELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, feats: torch.Tensor, inview: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.enc(self.embed(feats), src_key_padding_mask=~inview))


class RoutePolicy(nn.Module):
    def __init__(self, d: int = 128, layers: int = 2, heads: int = 4):
        super().__init__()
        self.config = {"d": d, "layers": layers, "heads": heads}
        self.actor = SetEncoder(d, layers, heads)
        self.pi = nn.Linear(d, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        self.critic = SetEncoder(d, layers, heads)
        self.v = nn.Sequential(nn.Linear(2 * d, d), nn.ELU(), nn.Linear(d, 1))

    def logits(self, feats, ok, inview):
        lg = self.pi(self.actor(feats, inview)).squeeze(-1)
        return torch.where(ok, lg, torch.full_like(lg, NEG))

    def value(self, feats, inview):
        h = self.critic(feats, inview)
        w = inview.to(h.dtype)[..., None]
        mean = (h * w).sum(1) / w.sum(1).clamp(min=1)
        mx = torch.where(inview[..., None], h, torch.full_like(h, -1e4)).max(1).values
        return self.v(torch.cat((mean, mx), -1)).squeeze(-1)

    def act(self, feats, ok, inview, deterministic: bool = False):
        lg = self.logits(feats, ok, inview)
        dist = torch.distributions.Categorical(logits=lg)
        a = lg.argmax(-1) if deterministic else dist.sample()
        return a, dist.log_prob(a), self.value(feats, inview)

    def evaluate(self, feats, ok, inview, actions):
        lg = self.logits(feats, ok, inview)
        dist = torch.distributions.Categorical(logits=lg)
        return dist.log_prob(actions), dist.entropy(), self.value(feats, inview), lg

    def export(self, path, meta: dict | None = None) -> None:
        """The actor as JSON for bridge/router.js (float64 lists)."""
        def t(x):
            return x.detach().cpu().double().tolist()

        def lin(m):
            return {"w": t(m.weight), "b": t(m.bias)}
        a = self.actor
        layers = []
        for L in a.enc.layers:
            layers.append({"in_w": t(L.self_attn.in_proj_weight), "in_b": t(L.self_attn.in_proj_bias),
                           "out": lin(L.self_attn.out_proj), "ln1": {"w": t(L.norm1.weight), "b": t(L.norm1.bias)},
                           "ln2": {"w": t(L.norm2.weight), "b": t(L.norm2.bias)},
                           "ff1": lin(L.linear1), "ff2": lin(L.linear2)})
        data = {"format": "parkour-router-v1", "token_dim": TOKEN_DIM, "heads": self.config["heads"],
                "d": self.config["d"], "embed": [lin(a.embed[0]), lin(a.embed[2])], "layers": layers,
                "out_norm": {"w": t(a.out_norm.weight), "b": t(a.out_norm.bias)}, "pi": lin(self.pi),
                "ln_eps": 1e-5, "meta": meta or {}}
        Path(path).write_text(json.dumps(data))


def load_router(path, device) -> RoutePolicy:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = RoutePolicy(**ck["net"]).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net

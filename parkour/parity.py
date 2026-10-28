"""Compare the batched simulator with the real prismarine-physics engine.

The JavaScript side (bridge/physics_parity.js) runs prismarine-physics 1.11.1,
the engine Mineflayer moves bots with, on the same blocks and control
sequences. Agreement here means a policy trained in the simulator meets the
same physics when it drives a Mineflayer bot on a real server.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import physics
from .paths import BRIDGE, NODE


@dataclass
class ParityCase:
    blocks: np.ndarray  # (B, 3) int, world cells
    pos: tuple
    yaw: np.ndarray  # (T,)
    controls: np.ndarray  # (T, 6) bool
    ice: np.ndarray | None = None  # (I, 3) int, cells of ``blocks`` that are ice
    shapes: np.ndarray | None = None  # (M, 4) int, cells of ``blocks`` with a collision shape: x, y, z, shape id


def random_case(rng: np.random.Generator, ticks: int = 120, size: int = 14, ice: float = 0.0,
                shapes: float = 0.0) -> ParityCase:
    """A random block field with a player on top of a block and sticky random inputs; with ``ice``
    > 0, that share of the raised blocks (and always the start block) is ice; with ``shapes`` > 0,
    that share of the other raised blocks has a random collision shape of physics.shape_table()."""
    blocks = set()
    floor_y = 6
    o = 26  # offset so a safety floor fits around the field at non-negative cells
    # A safety floor four blocks down keeps falls short (the collision window
    # assumes moves under 3.2 blocks per tick, which the parkour env enforces).
    for x in range(0, size + 2 * o):
        for z in range(0, size + 2 * o):
            blocks.add((x, 1, z))
    # Islands of floor, with gaps, so falls and edge landings happen.
    for _ in range(rng.integers(3, 8)):
        x0, z0 = rng.integers(0, size - 3, size=2) + o
        w, d = rng.integers(1, 4, size=2)
        y = floor_y + rng.integers(-2, 3)
        for x in range(x0, x0 + w):
            for z in range(z0, z0 + d):
                blocks.add((int(x), int(y), int(z)))
    # Pillars, steps and ceilings.
    for _ in range(rng.integers(5, 25)):
        x, z = rng.integers(0, size, size=2) + o
        y = floor_y + rng.integers(-2, 5)
        blocks.add((int(x), int(y), int(z)))
    raised = sorted(b for b in blocks if b[1] > 1)
    start = raised[rng.integers(len(raised))]
    for dy in (1, 2):  # head room above the start block
        blocks.discard((start[0], start[1] + dy, start[2]))
    pos = (start[0] + rng.uniform(0.2, 0.8), float(start[1] + 1), start[2] + rng.uniform(0.2, 0.8))

    yaw = np.empty(ticks)
    controls = np.zeros((ticks, 6), dtype=bool)
    y = rng.uniform(-np.pi, np.pi)
    c = np.zeros(6, dtype=bool)
    for t in range(ticks):
        if rng.random() < 0.25:
            c = rng.random(6) < np.array([0.7, 0.15, 0.25, 0.25, 0.35, 0.6])
        if rng.random() < 0.3:
            y += rng.normal(0.0, 0.4)
        yaw[t] = y
        controls[t] = c
    cells = None
    if ice > 0:  # drawn last, so cases without ice are the same as before
        cells = np.array([b for b in raised if b == start or rng.random() < ice], dtype=np.int64)
    shaped = None
    if shapes > 0:  # drawn after that, for the same reason
        icy = set() if cells is None else {tuple(int(v) for v in b) for b in cells}
        n_shapes = len(physics.shape_table()[0])
        shaped = np.array([(*b, int(rng.integers(1, n_shapes))) for b in raised  # not the start block's head room
                           if b != start and b in blocks and b not in icy and rng.random() < shapes],
                          dtype=np.int64).reshape(-1, 4)
    return ParityCase(np.array(sorted(blocks), dtype=np.int64), pos, yaw, controls, cells, shaped)


def run_js(cases: list[ParityCase]) -> list[list[dict]]:
    payload = {"cases": [{
        "blocks": c.blocks.tolist(), "ice": [] if c.ice is None else c.ice.tolist(),
        "shapes": [] if c.shapes is None else [[*b[:3].tolist(), physics.shape_table()[0][b[3]]] for b in c.shapes],
        "pos": list(c.pos), "vel": [0.0, 0.0, 0.0], "onGround": False,
        "jumpTicks": 0,
        "ticks": [{"yaw": float(c.yaw[t]), "c": [int(v) for v in c.controls[t]]} for t in range(len(c.yaw))],
    } for c in cases]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        name = f.name
    try:
        out = subprocess.run([str(NODE), str(BRIDGE / "physics_parity.js"), name], capture_output=True,
                             text=True, check=True, cwd=str(BRIDGE))
    finally:
        os.unlink(name)
    return json.loads(out.stdout)["cases"]


def run_torch(cases: list[ParityCase], device: str = "cpu") -> list[dict]:
    """Simulate all cases as one batch. Returns arrays of shape (N, T, ...)."""
    n = len(cases)
    extent = np.max([c.blocks.max(axis=0) for c in cases], axis=0) + 8
    solid = torch.zeros(n, *extent.tolist(), dtype=torch.bool)
    for i, c in enumerate(cases):
        b = torch.as_tensor(c.blocks)
        solid[i, b[:, 0], b[:, 1], b[:, 2]] = True
    slip = None
    if any(c.ice is not None and len(c.ice) for c in cases):
        slip = torch.full(solid.shape, physics.DEFAULT_SLIPPERINESS, dtype=torch.float64)
        for i, c in enumerate(cases):
            if c.ice is not None and len(c.ice):
                b = torch.as_tensor(c.ice)
                slip[i, b[:, 0], b[:, 1], b[:, 2]] = physics.ICE_SLIPPERINESS
        slip = slip.to(device)
    shape = None
    if any(c.shapes is not None and len(c.shapes) for c in cases):
        shape = torch.zeros(solid.shape, dtype=torch.uint8)
        for i, c in enumerate(cases):
            if c.shapes is not None and len(c.shapes):
                b = torch.as_tensor(c.shapes)
                shape[i, b[:, 0], b[:, 1], b[:, 2]] = b[:, 3].to(torch.uint8)
        shape = shape.to(device)
    world = physics.World(solid.to(device), slip, shape)
    course = torch.arange(n, device=device)
    state = physics.PlayerState.at(torch.tensor([c.pos for c in cases], dtype=torch.float64, device=device),
                                   torch.zeros(n, dtype=torch.float64, device=device))
    ticks = len(cases[0].yaw)
    yaw = torch.as_tensor(np.stack([c.yaw for c in cases]), dtype=torch.float64, device=device)
    ctrl = torch.as_tensor(np.stack([c.controls for c in cases]), device=device)
    rec = {k: [] for k in ("pos", "vel", "onGround", "jumpTicks", "collH", "collV")}
    for t in range(ticks):
        state.yaw = yaw[:, t]
        state = physics.simulate(world, course, state, ctrl[:, t], check=True)
        rec["pos"].append(state.pos)
        rec["vel"].append(state.vel)
        rec["onGround"].append(state.on_ground)
        rec["jumpTicks"].append(state.jump_ticks)
        rec["collH"].append(state.collided_h)
        rec["collV"].append(state.collided_v)
    return {k: torch.stack(v, 1).cpu().numpy() for k, v in rec.items()}


def compare(cases: list[ParityCase], device: str = "cpu") -> dict:
    js = run_js(cases)
    py = run_torch(cases, device)
    jpos = np.array([[s["pos"] for s in case] for case in js])
    jvel = np.array([[s["vel"] for s in case] for case in js])
    flags = {k: np.array([[s[k] for s in case] for case in js]) for k in ("onGround", "jumpTicks", "collH", "collV")}
    pos_err = np.abs(jpos - py["pos"]).max(axis=-1)
    vel_err = np.abs(jvel - py["vel"]).max(axis=-1)
    flag_ok = np.ones_like(pos_err, dtype=bool)
    for k, v in flags.items():
        flag_ok &= (v == py[k])
    # First divergence per case (pos error above 1e-9 or any flag mismatch).
    bad = (pos_err > 1e-9) | (vel_err > 1e-9) | ~flag_ok
    first_bad = [int(np.argmax(b)) if b.any() else -1 for b in bad]
    return {
        "cases": len(cases),
        "ticks": int(pos_err.size),
        "max_pos_err": float(pos_err.max()),
        "max_vel_err": float(vel_err.max()),
        "flag_mismatch_ticks": int((~flag_ok).sum()),
        "exact_pos_ticks": int((pos_err == 0).sum()),
        "diverged_cases": int(sum(1 for f in first_bad if f >= 0)),
        "first_bad": first_bad,
    }

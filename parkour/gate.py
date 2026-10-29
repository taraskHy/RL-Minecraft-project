"""Skill gate: at the start and at every landing, a learned choice of the frozen expert that plays next.

HANDOFF.md section 3 ("After those two, if it pays: a skill gate", and the rules fixed on 29 Oct at 20:15 and
20:20). Opt-in: nothing else in the project imports this module.

    python -m parkour.gate train --run NAME [--experts configs/gate_experts.json] [--iterations 150] [--save-at 50,100]
    python -m parkour.gate train --run NAME-swap1 --init-from runs/NAME/checkpoints/best.pt --experts NEW.json --swap
    python -m parkour.gate eval --checkpoint runs/NAME/checkpoints/best.pt --split validation [--per-level 100]
                                [--fixed-slot SLOT] [--experts FILE --swap] [--out FILE]

Experts are slots. ``configs/gate_experts.json`` lists slots {slot, skill, checkpoint[, sha1]}: checkpoint paths
relative to the repository root, the optional sha1 checked when the file is read, and skill the slot's swap
criterion of the 20:20 rule (basic, ice and extra: that group's validation; neo: the single-neo set, family 40,
strict metric, policy alone; all: the mean over the four groups). Every expert is loaded with
``parkour.train.load_frozen`` and acts deterministically, the argmax of every head, as in every evaluation (one
network per distinct checkpoint, each run on every row every tick, so that a gate fixed to one slot plays bit for
bit like ``parkour.train.evaluate`` on that expert). The gate picks a slot, never a network: a gate checkpoint
stores its slot list with each checkpoint's sha1, and loading it with another slot list is refused unless
``--swap`` is given. ``--swap`` may replace the checkpoint of any slot while the slots themselves (names, order,
skills) stay; each replacement is logged in the run's config.json under "swaps" (slot, old and new checkpoint and
sha1).

Decisions: at every episode start and whenever the progress index cp increases (a landing on a new platform),
the gate chooses a slot; that expert plays every tick until the next decision or the end of the episode.

Inputs (266 numbers): the experts' 255-number observation, then 11 features they lack (FEATURE_NAMES): the block
kind of the current platform (index cp) and of the next one (cp + 1) as one-hots over stone, ice, slime, shaped,
ladder, and whether a ceiling covers the next platform (a head-hitter runway). They are derived per platform from
the course data the CoursePool holds (``platform_tags``):
  - ice: the platform's index is in ``Course.ice`` (ice_prob platforms and ice run-ups);
  - slime: its index is in ``Course.slime`` (slime pads);
  - shaped: its index is a key of ``Course.shapes`` (slabs, soul sand, fence posts, ...);
  - ladder: it is the top of a ladder climb, i.e. a cell (x, y, z, facing) of ``Course.ladders`` lies in the
    platform's own block row y, one block before its footprint along the climb's heading ((x + hx, z + hz) inside
    the footprint, with the heading the facing encodes as ``courses._ladder`` writes it: facing 3 +x, 0 +z,
    1 -x, 2 -z); the takeoff, at the other end of the ladder, does not count;
  - several kinds on one platform (no course of skills mode has that): the first of ladder, slime, shaped, ice;
    none: stone;
  - ceiling: an obstacle cell (``Course.extra``) above the platform's footprint 2 or 3 air blocks above its top
    face, i.e. in block row y + 3 or y + 4: the rows the observation's scan reports as a ceiling to a player
    standing on that platform (``env.SCAN_CEILING_LEVELS`` above the feet cell y + 1). In generated courses only
    head-hitter roofs (``courses._ceiling``) lie there; beams, pillars and neo walls never stand over a platform.
  - The next platform's one-hot and ceiling flag are zero past the goal.
The gate has its own observation normaliser (``model.RunningNorm``, updated as parkour.train updates its own: from
the inputs of every decision during a rollout, used from the next rollout on) and two MLPs (``model.mlp``, ELU,
``--hidden`` 256,128): the actor's logits over the slots and the critic's value. Training samples the slot with its
own random generator (``--seed``), so the evaluations' environments, which reseed torch, do not reset it.

Training (``train``) is PPO on the decisions as a semi-MDP. An update collects ``--rollout`` ticks in each of
``--num-envs`` environments, as parkour.train does. A transition runs from one decision to the next (or the
episode's end); its reward is the sum of the environment's rewards over its ticks, and it is discounted per
transition: delta = R + gamma^ticks * V(next decision) - V, with gamma 0.995 per tick, and GAE lambda 0.95 over
the decisions (A = delta + gamma^ticks * lambda * A(next)). An episode's end cuts the chain: 0 after a success, a
fall or a stall with a stall penalty; after a time limit (the env's truncation) the gate's value of the final
state, as parkour.train bootstraps truncations. A decision still running when a rollout ends is carried into the
next rollout and completes there (with its stored log-probability, the behaviour policy's); the update uses the
transitions completed in its rollout and cuts GAE at the rollout's end with the value of the state there. The gate
samples its slot in training and takes the argmax in evaluation. PPO settings are parkour.train's (epochs 4,
minibatches 8, clip 0.2 with the dual clip 3, KL-adaptive learning rate, early stop at KL 0.05, entropy 0.005,
value weight 1, advantage normalisation) except two choices for a critic that starts from nothing: the value
loss is the plain squared error (``--value-clip 0``; parkour.train clips each value's change to 0.2 per update,
made for fine-tuning calibrated critics, which would hold returns of tens of reward units for hundreds of updates;
``--value-clip 0.2`` restores it), and the gradient norm is clipped for the actor and the critic separately.

Training courses are skills mode's, built by parkour.train's own code exactly as for runs/combined-distill
(``--skills basic,advanced,extra --ice-prob 0.3 --obstacle-prob 0.3 --pool-size 1024 --curriculum uniform``,
jumps 5-12, turn probability 0.25, seed 0, family 0) and exclude the same held-out courses: the skills-mode test
and validation sets and both single-neo sets (families 40 and 41). The pool is refreshed as parkour.train does
(12.5% every 20 updates); a decision running on a replaced course ends there like a time limit.

Validation is the measurement's (scripts/gate_headroom.py): ``parkour.train.skill_sets("validation", 100, 10,
("basic", "advanced", "extra"), ice=True)``, stall 200, compact ice, batches of 600 courses per group and part,
the first episode on every course; per-group success and the mean over the four groups
(``parkour.train.group_scores`` / ``macro_score``) and the share of decisions and ticks of every slot per group.
runs/NAME gets config.json, train_log.csv, eval_val_log.csv, summary.json (best = the highest four-group
validation mean, ties to the earlier; last_eval; completed) and checkpoints/best.pt, latest.pt and
updates_XXXXX.pt (``--save-at``). The test sets are played only with ``--test-once`` (off by default), once,
with best.pt.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import courses as C
from . import paths
from . import train as T
from .env import (MAX_PLATFORMS, OBS_DIM, SCAN_CEILING_LEVELS, CoursePool, EnvConfig, LevelSampler, ParkourEnv,
                  RewardConfig)
from .model import RunningNorm, mlp
from .paths import RUNS

FORMAT = "parkour-gate-v1"
KINDS = ("stone", "ice", "slime", "shaped", "ladder")
PRIORITY = ("ladder", "slime", "shaped", "ice")  # several kinds on one platform: the first of these
LADDER_HEADING = {3: (1, 0), 0: (0, 1), 1: (-1, 0), 2: (0, -1)}  # ladder facing -> climb heading (courses._ladder)
CEILING_ROWS = tuple(1 + c for c in SCAN_CEILING_LEVELS)  # block rows above a platform's row: (3, 4)
FEATURE_NAMES = tuple(f"current_{k}" for k in KINDS) + tuple(f"next_{k}" for k in KINDS) + ("next_ceiling",)
GATE_OBS_DIM = OBS_DIM + len(FEATURE_NAMES)
SKILLS = ("basic", "ice", "neo", "extra", "all")
DEFAULT_EXPERTS = "configs/gate_experts.json"
PER_LEVEL = {"validation": 100, "test": 200}  # parkour.train's val_per_level and eval_per_level


# Experts -------------------------------------------------------------------------------------------------------

def file_sha1(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def resolve(checkpoint: str) -> Path:
    """A slot's checkpoint path: absolute, or relative to the repository root."""
    p = Path(checkpoint)
    return p if p.is_absolute() else paths.ROOT / p


def read_slots(path) -> list[dict]:
    """The slot list of an experts file, with every checkpoint's sha1: [{slot, skill, checkpoint, sha1}]."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["slots"] if isinstance(data, dict) else data
    out, seen = [], set()
    for r in rows:
        name, skill, ck = str(r["slot"]), str(r["skill"]), str(r["checkpoint"])
        if name in seen:
            raise ValueError(f"{path}: slot {name} twice")
        if skill not in SKILLS:
            raise ValueError(f"{path}: slot {name} has skill {skill!r}; one of {', '.join(SKILLS)}")
        f = resolve(ck)
        if not f.is_file():
            raise FileNotFoundError(f"{path}: slot {name}: no file {f}")
        h = file_sha1(f)
        if r.get("sha1") and r["sha1"] != h:
            raise ValueError(f"{path}: slot {name}: {ck} has sha1 {h}, the file lists {r['sha1']}")
        seen.add(name)
        out.append({"slot": name, "skill": skill, "checkpoint": Path(ck).as_posix(), "sha1": h})
    if not out:
        raise ValueError(f"{path}: no slots")
    return out


def current_slots(slots: list[dict]) -> list[dict]:
    """A stored slot list, checked against the files as they are now (each sha1 must still match)."""
    for s in slots:
        h = file_sha1(resolve(s["checkpoint"]))
        if h != s["sha1"]:
            raise ValueError(f"slot {s['slot']}: {s['checkpoint']} changed since the gate stored it (sha1 {h}, "
                             f"stored {s['sha1']}); give the experts with --experts FILE --swap")
    return [dict(s) for s in slots]


def compare_slots(old: list[dict], new: list[dict], swap: bool) -> list[dict]:
    """The swaps from ``old`` (a gate checkpoint's slot list) to ``new``: none when they are equal.

    Another slot list is refused without ``swap``; with it, only the checkpoints (and their sha1) may differ: the
    slot names, their order and their skills must stay, since the gate's outputs are the slots."""
    key = lambda s: (s["slot"], s["skill"], s["checkpoint"], s["sha1"])  # noqa: E731
    if [key(s) for s in old] == [key(s) for s in new]:
        return []
    if not swap:
        raise ValueError("the gate was trained with another slot list: "
                         + json.dumps([key(s) for s in old]) + "; pass --swap to replace the experts in its slots")
    if [(s["slot"], s["skill"]) for s in old] != [(s["slot"], s["skill"]) for s in new]:
        raise ValueError("--swap replaces the expert in a slot: the slot names, their order and skills must stay "
                         f"({[s['slot'] for s in old]} -> {[s['slot'] for s in new]})")
    return [{"slot": o["slot"], "old_checkpoint": o["checkpoint"], "old_sha1": o["sha1"],
             "new_checkpoint": n["checkpoint"], "new_sha1": n["sha1"]} for o, n in zip(old, new) if key(o) != key(n)]


class Experts:
    """The slots' frozen networks (one per distinct checkpoint), each acting by the argmax of every head."""

    def __init__(self, slots: list[dict], device):
        self.slots = [dict(s) for s in slots]
        self.names = [s["slot"] for s in slots]
        nets, index, where = [], [], {}
        for s in slots:
            if s["sha1"] not in where:
                m = T.load_frozen(str(resolve(s["checkpoint"])), device)
                if m.obs_dim != OBS_DIM:
                    raise ValueError(f"slot {s['slot']}: the gate's experts use the {OBS_DIM}-number observation")
                where[s["sha1"]] = len(nets)
                nets.append(m)
            index.append(where[s["sha1"]])
        self.nets = nets
        self.index = torch.as_tensor(index, dtype=torch.int64, device=device)

    @torch.no_grad()
    def actions(self, obs: torch.Tensor) -> torch.Tensor:
        """(networks, N, 5): every network's deterministic action on every row (as ``ActorCritic.act``)."""
        return torch.stack([torch.stack([lg.argmax(-1) for lg in m.logits(m.norm(obs))], -1) for m in self.nets])

    def act(self, obs: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """(N, 5): the action of each row's slot."""
        return self.actions(obs)[self.index[slot], torch.arange(obs.shape[0], device=obs.device)]


# Features ------------------------------------------------------------------------------------------------------

def platform_tags(course: C.Course) -> list[set]:
    """The block kinds (other than stone) that apply to each platform, by the rules of the module docstring."""
    p = np.asarray(course.platforms, dtype=np.int64).reshape(-1, 5)
    tags = [set() for _ in range(len(p))]
    for k in course.ice:
        tags[int(k)].add("ice")
    for k in course.slime:
        tags[int(k)].add("slime")
    for k in course.shapes:
        tags[int(k)].add("shaped")
    for x, y, z, f in np.asarray(course.ladders, dtype=np.int64).reshape(-1, 4):
        hx, hz = LADDER_HEADING[int(f)]
        for k, (x0, x1, z0, z1, py) in enumerate(p):
            if py == y and x0 <= x + hx <= x1 and z0 <= z + hz <= z1:
                tags[k].add("ladder")
    return tags


def platform_kinds(course: C.Course) -> tuple[np.ndarray, np.ndarray]:
    """Per platform: its kind (an index into KINDS) and whether a ceiling covers it."""
    kind = np.array([KINDS.index(next((t for t in PRIORITY if t in tg), "stone")) for tg in platform_tags(course)],
                    dtype=np.int64)
    p = np.asarray(course.platforms, dtype=np.int64).reshape(-1, 5)
    e = np.asarray(course.extra, dtype=np.int64).reshape(-1, 3)
    lo, hi = CEILING_ROWS
    roof = np.array([bool(((e[:, 0] >= x0) & (e[:, 0] <= x1) & (e[:, 2] >= z0) & (e[:, 2] <= z1)
                           & (e[:, 1] >= y + lo) & (e[:, 1] <= y + hi)).any()) for x0, x1, z0, z1, y in p], dtype=bool)
    return kind, roof


class GateFeatures:
    """The 11 features for rows (course index, cp) of one CoursePool, from per-platform tables on its device."""

    def __init__(self, pool: CoursePool):
        self.device = pool.device
        width = MAX_PLATFORMS + 1  # one column past the last platform, so that cp + 1 always has an entry
        self.kind = torch.full((pool.size, width), -1, dtype=torch.int64, device=self.device)
        self.roof = torch.zeros((pool.size, width), dtype=torch.bool, device=self.device)
        for i, c in enumerate(pool.courses):
            self.set(i, c)
        self.range = torch.arange(len(KINDS), device=self.device)

    def set(self, index: int, course: C.Course) -> None:
        """(Re)build one course's row, e.g. after ``CoursePool.replace``."""
        kind, roof = platform_kinds(course)
        self.kind[index] = -1
        self.kind[index, :len(kind)] = torch.as_tensor(kind, device=self.device)
        self.roof[index] = False
        self.roof[index, :len(roof)] = torch.as_tensor(roof, device=self.device)

    def __call__(self, course: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
        cur = cp.clamp(max=MAX_PLATFORMS)
        nxt = (cp + 1).clamp(max=MAX_PLATFORMS)
        onehot = lambda k: (k[:, None] == self.range[None, :]).to(torch.float32)  # noqa: E731  (-1: all zero)
        return torch.cat((onehot(self.kind[course, cur]), onehot(self.kind[course, nxt]),
                          self.roof[course, nxt].to(torch.float32)[:, None]), -1)


def gate_input(obs: torch.Tensor, features: GateFeatures, course: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
    return torch.cat((obs, features(course, cp)), -1)


# Network -------------------------------------------------------------------------------------------------------

class Gate(nn.Module):
    """Normaliser, actor (logits over the slots) and critic (a value) on the 266 gate inputs."""

    def __init__(self, n_slots: int, hidden=(256, 128), in_dim: int = GATE_OBS_DIM):
        super().__init__()
        self.in_dim, self.hidden, self.n_slots = in_dim, tuple(int(h) for h in hidden), int(n_slots)
        self.norm = RunningNorm(in_dim)
        self.actor = mlp((in_dim, *self.hidden, self.n_slots), out_gain=0.01)
        self.critic = mlp((in_dim, *self.hidden, 1), out_gain=1.0)

    def logits(self, xn: torch.Tensor) -> torch.Tensor:
        return self.actor(xn)

    def value(self, xn: torch.Tensor) -> torch.Tensor:
        return self.critic(xn).squeeze(-1)

    def arch(self) -> dict:
        return {"in_dim": self.in_dim, "hidden": list(self.hidden), "n_slots": self.n_slots,
                "features": list(FEATURE_NAMES)}


def checkpoint(gate: Gate, slots: list[dict], iteration: int, config: dict, optimizer=None, ev=None) -> dict:
    ck = {"format": FORMAT, "gate": gate.state_dict(), "arch": gate.arch(), "slots": [dict(s) for s in slots],
          "iteration": iteration, "updates": iteration + 1, "config": config, "eval": ev}
    if optimizer is not None:
        ck["optimizer"] = optimizer.state_dict()
    return ck


def load_gate(path, device) -> tuple[Gate, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ck, dict) or ck.get("format") != FORMAT:
        raise ValueError(f"{path} is not a gate checkpoint ({FORMAT})")
    a = ck["arch"]
    if tuple(a["features"]) != FEATURE_NAMES or a["in_dim"] != GATE_OBS_DIM:
        raise ValueError(f"{path}: a gate with other inputs")
    gate = Gate(a["n_slots"], a["hidden"], a["in_dim"]).to(device)
    gate.load_state_dict(ck["gate"])
    gate.eval()
    return gate, ck


@torch.no_grad()
def masked_update(norm: RunningNorm, x: torch.Tensor, mask: torch.Tensor) -> None:
    """``RunningNorm.update`` with the rows of ``x`` in ``mask`` only, without a host synchronisation."""
    w = mask.to(torch.float64)[:, None]
    x = x.to(torch.float64)
    bc = w.sum()
    bm = (x * w).sum(0) / bc.clamp(min=1)
    bv = ((x - bm).pow(2) * w).sum(0) / bc.clamp(min=1)
    delta = bm - norm.mean
    tot = norm.count + bc
    norm.mean += delta * bc / tot
    m2 = norm.var * norm.count + bv * bc + delta.pow(2) * norm.count * bc / tot
    norm.var = m2 / tot
    norm.count = tot


# Evaluation ----------------------------------------------------------------------------------------------------

@torch.no_grad()
def play(gate: Gate | None, experts: Experts, pool: CoursePool, device, stall_ticks: int = 200, max_ticks: int = 4000,
         fixed_slot: int | None = None, trace: list | None = None) -> dict:
    """First episode on every course of ``pool``, as ``parkour.train.evaluate`` plays it, with the gate's argmax
    choosing the slot at the start and at every landing (``fixed_slot``: that slot everywhere, no gate needed).

    Also counts each course's decisions and ticks per slot (first episode). ``trace`` (a list) receives one dict
    per tick: decide, slot, cp (before the step), done."""
    n = pool.size
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=stall_ticks, max_ticks=max_ticks), sampler=None)
    obs = env.reset_all(torch.arange(n, device=device))
    features = GateFeatures(pool) if fixed_slot is None else None
    k = len(experts.names)
    slot = torch.zeros(n, dtype=torch.int64, device=device)
    decide = torch.ones(n, dtype=torch.bool, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    success = torch.zeros(n, dtype=torch.bool, device=device)
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    length = torch.zeros(n, dtype=torch.int64, device=device)
    prog = torch.zeros(n, dtype=torch.float64, device=device)
    decisions = torch.zeros(n, k, dtype=torch.int64, device=device)
    ticks = torch.zeros(n, k, dtype=torch.int64, device=device)
    for t in range(max_ticks):
        if fixed_slot is None:
            choice = gate.logits(gate.norm(gate_input(obs, features, env.course, env.cp))).argmax(-1)
        else:
            choice = torch.full_like(slot, fixed_slot)
        slot = torch.where(decide, choice, slot)
        live = torch.nn.functional.one_hot(slot, k) * (~finished)[:, None]
        decisions += live * decide[:, None]
        ticks += live
        cp = env.cp
        obs, _, _, _, _, info = env.step(experts.act(obs, slot))
        if trace is not None:
            trace.append({"decide": decide.clone(), "slot": slot.clone(), "cp": cp.clone(), "done": info["done"].clone()})
        decide = info["done"] | (env.cp > cp)
        first = info["done"] & ~finished
        success |= first & info["success"]
        fell |= first & info["fell"]
        length = torch.where(first, info["ep_len"], length)
        prog = torch.where(first, info["progress_frac"], prog)
        finished |= info["done"]
        if t % 25 == 0 and bool(finished.all()):
            break
    lv = pool.level.cpu().numpy()
    s, f, ln, pr = (x.cpu().numpy() for x in (success, fell, length, prog))
    per_level = {}
    for L in sorted(set(lv.tolist())):
        m = lv == L
        per_level[int(L)] = {"success": float(s[m].mean()), "fell": float(f[m].mean()), "progress": float(pr[m].mean()),
                             "ticks_success": float(ln[m & s].mean()) if (m & s).any() else None, "n": int(m.sum())}
    return {"success": float(s.mean()), "per_level": per_level,
            "per_course": {"success": s.tolist(), "fell": f.tolist(), "ticks": ln.tolist(), "progress": pr.tolist()},
            "decisions": decisions.cpu(), "slot_ticks": ticks.cpu()}


def merge_levels(parts: list[dict]) -> dict:
    """Per-level rows of several batches as one (as ``parkour.train.evaluate_course_batches`` merges them)."""
    out = {}
    for L in sorted({L for p in parts for L in p}):
        rows = [p[L] for p in parts if L in p]
        n = sum(r["n"] for r in rows)
        row = {key: sum(r[key] * r["n"] for r in rows) / n for key in ("success", "fell", "progress")}
        won = sum(r["n"] * r["success"] for r in rows if r.get("ticks_success") is not None)
        row["ticks_success"] = (sum(r["ticks_success"] * r["n"] * r["success"] for r in rows
                                    if r.get("ticks_success") is not None) / won) if won else None
        row["n"] = n
        out[L] = row
    return out


def validate(gate: Gate | None, experts: Experts, sets: dict, device, batch_size: int = 600, stall_ticks: int = 200,
             fixed_slot: int | None = None) -> dict:
    """The gated policy on held-out sets (``parkour.train.skill_sets``), as scripts/gate_headroom.py plays them:
    every group and part in batches of ``batch_size`` courses, compact ice, the first episode on every course."""
    t0 = time.time()
    names = experts.names
    per_level = {"standard": [], "obstacles": []}
    per_course, usage = {}, {}
    for g in T.GROUP_ORDER:
        if g not in sets:
            continue
        dec = torch.zeros(len(names), dtype=torch.int64)
        tic = torch.zeros(len(names), dtype=torch.int64)
        per_course[g] = {}
        for part in ("standard", "obstacles"):
            courses = sets[g].get(part) or []
            if not courses:
                continue
            rows = {"success": [], "ticks": [], "fell": [], "decisions": []}
            for start in range(0, len(courses), batch_size):
                pool = CoursePool(courses[start:start + batch_size], device, C.GRID, compact_ice=True)
                r = play(gate, experts, pool, device, stall_ticks, fixed_slot=fixed_slot)
                per_level[part].append(r["per_level"])
                for key in ("success", "ticks", "fell"):
                    rows[key] += r["per_course"][key]
                rows["decisions"] += r["decisions"].sum(1).tolist()
                dec += r["decisions"].sum(0)
                tic += r["slot_ticks"].sum(0)
                del pool
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
            per_course[g][part] = rows
        usage[g] = {"decisions": {s: float(v) / max(1, int(dec.sum())) for s, v in zip(names, dec.tolist())},
                    "ticks": {s: float(v) / max(1, int(tic.sum())) for s, v in zip(names, tic.tolist())},
                    "n_decisions": int(dec.sum()), "n_ticks": int(tic.sum())}
    levels = {part: merge_levels(rows) for part, rows in per_level.items()}
    groups = T.group_scores(*levels.values())
    groups = {g: groups[g] for g in T.GROUP_ORDER if g in groups}
    return {"success": T.macro_score(groups), "groups": groups, "usage": usage, "per_level": levels,
            "per_course": per_course, "courses": sum(len(p) for s in sets.values() for p in s.values()),
            "seconds": round(time.time() - t0, 1)}


def compact(ev: dict) -> dict:
    """An evaluation without its per-course lists (for checkpoints and summaries)."""
    return {k: v for k, v in ev.items() if k != "per_course"}


# Training ------------------------------------------------------------------------------------------------------

@dataclass
class GateConfig:
    run: str = "gate-dev"
    experts: str = DEFAULT_EXPERTS
    init_from: str = ""  # a gate checkpoint to start from (weights and normaliser; a new optimizer)
    swap: bool = False  # allow --init-from with experts replaced in its slots (logged in config.json)
    seed: int = 0
    device: str = "cuda"
    threads: int = 0
    num_envs: int = 8192
    rollout: int = 64  # ticks per environment and update, as parkour.train
    iterations: int = 150
    epochs: int = 4
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.995  # per tick; a transition of k ticks is discounted by gamma**k
    lam: float = 0.95  # GAE over decisions
    clip: float = 0.2
    dual_clip: float = 3.0
    ent_coef: float = 0.005
    vf_coef: float = 1.0
    value_clip: float = 0.0  # 0: plain squared error; 0.2: parkour.train's clipped value loss
    max_grad_norm: float = 1.0  # for the actor and the critic each
    target_kl: float = 0.01
    kl_stop: float = 0.05
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    hidden: tuple = (256, 128)
    # Training courses: skills mode, built by parkour.train's code as for runs/combined-distill/config.json.
    skills: str = "basic,advanced,extra"
    ice_prob: float = 0.3
    obstacle_prob: float = 0.3
    pool_size: int = 1024
    pool_refresh_every: int = 20
    pool_refresh_frac: float = 0.125
    jumps_min: int = 5
    jumps_max: int = 12
    turn_prob: float = 0.25
    curriculum: str = "uniform"
    wide_neo: bool = False
    compact_ice: bool = True
    stall_ticks: int = 200
    reward: RewardConfig = field(default_factory=RewardConfig)
    eval_per_level: int = 200  # the test sets' size (excluded from training; played only with --test-once)
    val_per_level: int = 100  # the validation sets' size (the measurement's)
    eval_jumps: int = 10
    eval_every: int = 10
    eval_batch_size: int = 600
    save_at: str = ""  # comma-separated update counts with a snapshot (0 = before the first update)
    test_once: bool = False


def course_config(cfg: GateConfig) -> T.TrainConfig:
    """The parkour.train settings that build the training courses and held-out sets."""
    return T.TrainConfig(run=cfg.run, seed=cfg.seed, device=cfg.device, skills=cfg.skills, ice_prob=cfg.ice_prob,
                         obstacle_prob=cfg.obstacle_prob, pool_size=cfg.pool_size,
                         pool_refresh_every=cfg.pool_refresh_every, pool_refresh_frac=cfg.pool_refresh_frac,
                         jumps_min=cfg.jumps_min, jumps_max=cfg.jumps_max, turn_prob=cfg.turn_prob,
                         curriculum=cfg.curriculum, wide_neo=cfg.wide_neo, compact_ice=cfg.compact_ice,
                         stall_ticks=cfg.stall_ticks, reward=cfg.reward, eval_per_level=cfg.eval_per_level,
                         val_per_level=cfg.val_per_level, eval_jumps=cfg.eval_jumps, test_once=True)


def skill_groups(skills: str) -> list[str]:
    groups = [g for g in skills.split(",") if g]
    if not groups or set(groups) - set(T.SKILL_GROUPS) or len(set(groups)) != len(groups):
        raise ValueError(f"--skills: groups {', '.join(T.SKILL_GROUPS)}, each once")
    return groups


def training_courses(tc: T.TrainConfig, device):
    """Skills mode's training pool, course factory and curriculum, built by parkour.train's own code as
    ``parkour.train.train`` builds them (the same calls in the same order)."""
    skills = skill_groups(tc.skills)
    tables = T.skill_tables(skills)
    levels = [lv for g in skills for lv in tables[g]]
    n_levels = max(gid for gid, _, _ in T.skill_plan(skills)) + 1
    held = T.skill_sets("test", tc.eval_per_level, tc.eval_jumps, skills)
    tests = [c for g in held.values() for c in g["standard"]]
    obstacle_tests = [c for g in held.values() for c in g["obstacles"]]
    vals = [c for g in T.skill_sets("validation", tc.val_per_level, tc.eval_jumps, skills).values()
            for c in g["standard"] + g["obstacles"]]
    exclude = {T.course_hash(c) for c in tests + obstacle_tests + vals}
    if "advanced" in skills:  # neos in training: none of the single-neo sets either
        from .neo_assist import single_neo_hashes
        exclude |= single_neo_hashes()
    factory = T.CourseFactory(tc, levels, exclude)
    pool = CoursePool(factory.make(tc.pool_size), device, C.GRID,
                      ice=tc.ice_prob > 0 or any(sp.ice or sp.bounce for L in levels for sp in L),
                      compact_ice=tc.compact_ice, shapes=any(bool(sp.block) or sp.ladder for L in levels for sp in L),
                      slime=any(sp.bounce for L in levels for sp in L))
    if tc.curriculum == "adaptive":
        sampler = LevelSampler(n_levels)
    elif tc.curriculum == "uniform":
        sampler = LevelSampler(n_levels, fixed=[1.0] * n_levels)
    else:
        raise ValueError(tc.curriculum)
    return pool, factory, sampler, exclude


class Collector:
    """Rollouts of the gate in a training environment, kept as semi-MDP transitions.

    Decision rows: row 0 holds the decision carried from the previous rollout (its option still running), row
    t + 1 the decisions made at tick t. A row's transition, once its option ends, has its summed reward ``E_R``,
    its ticks ``E_tau``, whether the episode ended (``E_done``) and then the value to bootstrap (``E_boot``: 0 or a
    time limit's value of the final state)."""

    def __init__(self, env: ParkourEnv, gate: Gate, experts: Experts, features: GateFeatures, rollout: int,
                 generator: torch.Generator, shadow: RunningNorm | None = None):
        self.env, self.gate, self.experts, self.features = env, gate, experts, features
        self.rollout, self.gen, self.shadow = rollout, generator, shadow
        n, dev = env.n, env.device
        rows = rollout + 1
        self.rows = torch.arange(n, device=dev)
        self.D_obs = torch.zeros(rows, n, GATE_OBS_DIM, device=dev)
        self.D_slot = torch.zeros(rows, n, dtype=torch.int64, device=dev)
        self.D_logp = torch.zeros(rows, n, device=dev)
        self.D_val = torch.zeros(rows, n, device=dev)
        self.D_valid = torch.zeros(rows, n, dtype=torch.bool, device=dev)
        self.E_R = torch.zeros(rows, n, dtype=torch.float64, device=dev)
        self.E_tau = torch.zeros(rows, n, dtype=torch.int64, device=dev)
        self.E_end = torch.zeros(rows, n, dtype=torch.bool, device=dev)
        self.E_done = torch.zeros(rows, n, dtype=torch.bool, device=dev)
        self.E_boot = torch.zeros(rows, n, device=dev)
        self.obs = env.observe()
        self.decide = torch.ones(n, dtype=torch.bool, device=dev)  # every environment starts an episode
        self.slot = torch.zeros(n, dtype=torch.int64, device=dev)
        self.cur = torch.zeros(n, dtype=torch.int64, device=dev)  # the row of each environment's running option
        self.R = torch.zeros(n, dtype=torch.float64, device=dev)
        self.tau = torch.zeros(n, dtype=torch.int64, device=dev)
        self.last_val = torch.zeros(n, device=dev)

    def _value(self, obs, course, cp):
        return self.gate.value(self.gate.norm(gate_input(obs, self.features, course, cp)))

    def _close(self, end, done, boot) -> None:
        """The options ending on rows ``end`` (``done``: with their episode) complete their transitions."""
        c, r = self.cur, self.rows
        self.E_R[c, r] = torch.where(end, self.R, self.E_R[c, r])
        self.E_tau[c, r] = torch.where(end, self.tau, self.E_tau[c, r])
        self.E_done[c, r] = torch.where(end, done, self.E_done[c, r])
        self.E_boot[c, r] = torch.where(end, boot, self.E_boot[c, r])
        self.E_end[c, r] = self.E_end[c, r] | end

    @torch.no_grad()
    def collect(self, trace: list | None = None) -> None:
        env, gate = self.env, self.gate
        keep = ~self.decide  # options still running: their decision becomes row 0
        for buf in (self.D_obs, self.D_slot, self.D_logp, self.D_val):
            buf[0] = buf[self.cur, self.rows]
        self.D_valid.zero_()
        self.D_valid[0] = keep
        self.cur = torch.where(keep, torch.zeros_like(self.cur), self.cur)
        for buf in (self.E_R, self.E_tau, self.E_end, self.E_done, self.E_boot):
            buf.zero_()
        for t in range(self.rollout):
            x = gate_input(self.obs, self.features, env.course, env.cp)
            xn = gate.norm(x)
            logits = gate.logits(xn)
            pick = torch.multinomial(torch.softmax(logits, -1), 1, generator=self.gen).squeeze(1)
            logp = torch.log_softmax(logits, -1).gather(1, pick[:, None]).squeeze(1)
            d = self.decide
            if self.shadow is not None:
                masked_update(self.shadow, x, d)
            self.D_obs[t + 1] = xn
            self.D_slot[t + 1] = pick
            self.D_logp[t + 1] = logp
            self.D_val[t + 1] = gate.value(xn)
            self.D_valid[t + 1] = d
            self.slot = torch.where(d, pick, self.slot)
            self.cur = torch.where(d, torch.full_like(self.cur, t + 1), self.cur)
            self.R = torch.where(d, torch.zeros_like(self.R), self.R)
            self.tau = torch.where(d, torch.zeros_like(self.tau), self.tau)
            cp = env.cp
            nobs, rew, _, trunc, final_obs, info = env.step(self.experts.act(self.obs, self.slot))
            self.R = self.R + rew.to(torch.float64)
            self.tau = self.tau + 1
            done = info["done"]
            end = done | (env.cp > cp)
            # A time limit (the env's truncation; parkour.train bootstraps it): the gate's value of the final state,
            # on the course and at the cp the episode ended with (``env.step`` has reset both).
            fc = info["course"]
            fcp = torch.round(info["progress_frac"] * (env.pool.n[fc] - 1).clamp(min=1)).to(torch.int64)
            vf = self._value(final_obs, fc, fcp)
            self._close(end, done, torch.where(trunc, vf, torch.zeros_like(vf)))
            if trace is not None:
                trace.append({"decide": d.clone(), "slot": self.slot.clone(), "cp": cp.clone(), "done": done.clone(),
                              "reward": rew.clone(), "trunc": trunc.clone(), "boot": vf.clone()})
            self.decide = end
            self.obs = nobs
        self.last_val = self._value(self.obs, env.course, env.cp)

    def cut(self, mask: torch.Tensor) -> None:
        """End the options running on rows ``mask`` now, like a time limit (bootstrapped with the value of the
        state the rollout ended in): their environments restart on new courses (pool refresh)."""
        self._close(mask & ~self.decide, torch.ones_like(mask), self.last_val)

    def restart(self, mask: torch.Tensor) -> None:
        """After ``env.reset_rows(mask)``: those rows start new episodes, so they decide first."""
        self.obs = self.env.observe()
        self.decide = self.decide | mask

    def batch(self, gamma: float, lam: float) -> dict:
        """The transitions completed in the last rollout, with their GAE advantages and returns."""
        rows, n = self.D_valid.shape
        dev = self.D_valid.device
        val = self.D_val.to(torch.float64)
        adv_all = torch.zeros(rows, n, dtype=torch.float64, device=dev)
        nv = self.last_val.to(torch.float64)  # the value of the next decision, in time order
        na = torch.zeros(n, dtype=torch.float64, device=dev)  # its advantage (0: unknown, the chain is cut)
        for j in reversed(range(rows)):
            valid = self.D_valid[j]
            fin = valid & self.E_end[j]
            done = self.E_done[j]
            disc = gamma ** self.E_tau[j].to(torch.float64)
            v_next = torch.where(done, self.E_boot[j].to(torch.float64), nv)
            a_next = torch.where(done, torch.zeros_like(na), na)
            adv = self.E_R[j] + disc * v_next - val[j] + disc * lam * a_next
            adv_all[j] = torch.where(fin, adv, torch.zeros_like(adv))
            nv = torch.where(valid, val[j], nv)
            na = torch.where(fin, adv, torch.where(valid, torch.zeros_like(na), na))
        m = self.D_valid & self.E_end
        return {"obs": self.D_obs[m], "act": self.D_slot[m], "logp": self.D_logp[m], "val": self.D_val[m],
                "adv": adv_all[m].to(torch.float32), "ret": (adv_all + val)[m].to(torch.float32),
                "ticks": self.E_tau[m], "done": self.E_done[m], "index": m.nonzero(),
                "decided": int(self.D_valid[1:].sum()), "running": int((self.D_valid & ~self.E_end).sum())}


def ppo_update(gate: Gate, opt, b: dict, cfg: GateConfig, lr: float) -> tuple[dict, float]:
    """PPO epochs on the completed transitions (parkour.train's update, on decisions)."""
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0}
    n = int(b["obs"].shape[0])
    nmb = max(1, min(cfg.minibatches, n))
    mb = n // nmb if n else 0
    adv = b["adv"]
    if n > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    nupd, stopped = 0, False
    for _ in range(cfg.epochs if n else 0):
        perm = torch.randperm(n, device=adv.device)
        for k in range(nmb):
            idx = perm[k * mb:(k + 1) * mb]
            obs = b["obs"][idx]
            dist = torch.distributions.Categorical(logits=gate.logits(obs))
            logp, ent, v = dist.log_prob(b["act"][idx]), dist.entropy(), gate.value(obs)
            log_ratio = (logp - b["logp"][idx]).clamp(-20.0, 20.0)
            ratio = torch.exp(log_ratio)
            kl = ((ratio - 1) - log_ratio).mean().item()
            if kl > cfg.kl_stop or not math.isfinite(kl):
                stopped = True
                break
            a = adv[idx]
            surr = torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)
            surr = torch.where(a < 0, torch.maximum(surr, cfg.dual_clip * a), surr)
            pl = -surr.mean()
            ret = b["ret"][idx]
            if cfg.value_clip > 0:
                old = b["val"][idx]
                vc = old + (v - old).clamp(-cfg.value_clip, cfg.value_clip)
                vl = 0.5 * torch.max((v - ret).pow(2), (vc - ret).pow(2)).mean()
            else:
                vl = 0.5 * (v - ret).pow(2).mean()
            el = ent.mean()
            loss = pl + cfg.vf_coef * vl - cfg.ent_coef * el
            if not torch.isfinite(loss):
                stopped = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.actor.parameters(), cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(gate.critic.parameters(), cfg.max_grad_norm)
            opt.step()
            if kl > 2.0 * cfg.target_kl:  # KL-adaptive learning rate, as parkour.train
                lr = max(cfg.lr_min, lr / 1.5)
            elif kl < 0.5 * cfg.target_kl:
                lr = min(cfg.lr_max, lr * 1.5)
            for g in opt.param_groups:
                g["lr"] = lr
            stats["pi_loss"] += pl.item()
            stats["v_loss"] += vl.item()
            stats["entropy"] += el.item()
            stats["kl"] += kl
            stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
            nupd += 1
        if stopped:
            break
    for key in stats:
        stats[key] /= max(nupd, 1)
    stats.update(lr=lr, minibatch_updates=nupd, stopped=int(stopped))
    return stats, lr


def _save(obj, path: Path) -> None:
    T._save(obj, path)


def _eval_row(it: int, updates: int, ticks: int, ev: dict, names: list) -> list:
    row = [it, updates, ticks, round(ev["success"], 4)]
    row += [round(ev["groups"].get(g, float("nan")), 4) for g in T.GROUP_ORDER]
    for g in T.GROUP_ORDER:
        row += [round(ev["usage"].get(g, {}).get("decisions", {}).get(s, 0.0), 4) for s in names]
    return row


def _summary_eval(ev: dict, it: int) -> dict:
    return {"iteration": it, "updates": it + 1, "validation_success": ev["success"], "validation_groups": ev["groups"],
            "slot_usage": ev["usage"], "seconds": ev["seconds"]}


def train(cfg: GateConfig) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device(cfg.device)
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
    out = RUNS / cfg.run
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"run {cfg.run} exists: choose a new run name")
    if cfg.iterations < 1:
        raise ValueError("--iterations: at least 1")
    skills = skill_groups(cfg.skills)
    slots = read_slots(cfg.experts)
    init, swaps = None, []
    if cfg.init_from:
        init = torch.load(cfg.init_from, map_location=dev, weights_only=False)
        if not isinstance(init, dict) or init.get("format") != FORMAT:
            raise ValueError("--init-from takes a gate checkpoint")
        if tuple(init["arch"]["features"]) != FEATURE_NAMES or init["arch"]["in_dim"] != GATE_OBS_DIM:
            raise ValueError("--init-from: a gate with other inputs")
        swaps = compare_slots(init["slots"], slots, cfg.swap)
        cfg = dataclasses.replace(cfg, hidden=tuple(init["arch"]["hidden"]))
    elif cfg.swap:
        raise ValueError("--swap replaces experts in the slots of a gate given with --init-from")
    snapshots = {int(v) for v in cfg.save_at.split(",") if v.strip()}
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict.update({"what": "parkour.gate: a skill gate over frozen experts, PPO on its decisions (semi-MDP)",
                     "slots": slots, "swaps": swaps, "features": list(FEATURE_NAMES)})
    if init is not None:
        cfg_dict.update({"init_slots": init["slots"], "init_sha1": file_sha1(cfg.init_from),
                         "init_iteration": init.get("iteration")})
    (out / "config.json").write_text(json.dumps(cfg_dict, indent=1))

    experts = Experts(slots, dev)
    tc = course_config(cfg)
    pool, factory, sampler, _ = training_courses(tc, dev)
    env = ParkourEnv(pool, EnvConfig(num_envs=cfg.num_envs, stall_ticks=cfg.stall_ticks, reward=cfg.reward),
                     sampler=sampler, seed=cfg.seed)
    features = GateFeatures(pool)
    val_sets = T.skill_sets("validation", cfg.val_per_level, cfg.eval_jumps, tuple(skills), ice=True)
    gate = Gate(len(slots), cfg.hidden).to(dev)
    if init is not None:
        gate.load_state_dict(init["gate"])
    opt = torch.optim.Adam(gate.parameters(), lr=cfg.lr, eps=1e-5)
    lr = cfg.lr
    shadow = RunningNorm(GATE_OBS_DIM).to(dev)
    shadow.load_state_dict(gate.norm.state_dict())
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg.seed)
    collector = Collector(env, gate, experts, features, cfg.rollout, gen, shadow)
    names = experts.names

    def snapshot(updates: int) -> None:
        _save(checkpoint(gate, slots, updates - 1, cfg_dict), out / "checkpoints" / f"updates_{updates:05d}.pt")

    if 0 in snapshots:
        snapshot(0)
    log_path, val_path = out / "train_log.csv", out / "eval_val_log.csv"
    log_f = open(log_path, "a", newline="")
    log_w = None
    with open(val_path, "w", newline="") as f:
        csv.writer(f).writerow(["iteration", "updates", "env_ticks", "success"] + [f"success_{g}" for g in T.GROUP_ORDER]
                               + [f"usage_{g}_{s}" for g in T.GROUP_ORDER for s in names])
    best, best_info, last = -1.0, {}, {}
    total = 0
    t_start = time.time()
    n, steps = cfg.num_envs, cfg.rollout
    for it in range(cfg.iterations):
        t0 = time.time()
        gate.eval()
        collector.collect()
        t_roll = time.time() - t0
        gate.norm.load_state_dict(shadow.state_dict())
        st = env.pop_stats()
        sampler.update(st[0], st[1])
        env.set_level_probs(sampler.probs())
        refresh = cfg.pool_refresh_every and it % cfg.pool_refresh_every == cfg.pool_refresh_every - 1
        if refresh:  # parkour.train's pool refresh; the options it cuts end in this rollout
            k = int(pool.size * cfg.pool_refresh_frac)
            idx = np.random.default_rng([2026, 2, cfg.seed, it]).choice(pool.size, k, replace=False)
            moved = torch.isin(env.course, torch.as_tensor(idx, device=dev))
            collector.cut(moved)
        b = collector.batch(cfg.gamma, cfg.lam)
        gate.train()
        stats, lr = ppo_update(gate, opt, b, cfg, lr)
        gate.eval()
        total += n * steps
        if it + 1 in snapshots:
            snapshot(it + 1)
        dt = time.time() - t0
        eps = st[0].sum()
        frac = (lambda i: round(float(st[i].sum() / eps), 4) if eps else None)  # noqa: E731
        used = torch.bincount(b["act"], minlength=len(names)).tolist()
        row = {"iteration": it, "updates": it + 1, "env_ticks": total, "minutes": round((time.time() - t_start) / 60, 2),
               "ticks_per_s": int(n * steps / dt), "rollout_s": round(t_roll, 2), "update_s": round(dt - t_roll, 2),
               "transitions": int(b["obs"].shape[0]), "decisions": b["decided"], "running": b["running"],
               "ticks_per_transition": round(float(b["ticks"].double().mean()), 2) if len(b["ticks"]) else None,
               "episodes": int(eps), "success": frac(1), "fell": frac(2), "truncated": frac(3), "ep_len": frac(4),
               "ep_return": frac(5), "progress": frac(6),
               **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in stats.items()},
               **{f"usage_{s}": round(u / max(1, sum(used)), 4) for s, u in zip(names, used)}}
        if log_w is None:
            log_w = csv.DictWriter(log_f, fieldnames=list(row.keys()))
            log_w.writeheader()
        log_w.writerow(row)
        log_f.flush()
        if it % 10 == 0:
            print(json.dumps(row), flush=True)
        if refresh:
            for i, c in zip(idx, factory.make(k)):
                pool.replace(int(i), c)
                features.set(int(i), c)
            env.rebuild_level_table()
            env.set_level_probs(sampler.probs())
            env.reset_rows(moved)
            collector.restart(moved)
        if it % cfg.eval_every == cfg.eval_every - 1 or it == cfg.iterations - 1:
            ev = validate(gate, experts, val_sets, dev, cfg.eval_batch_size, cfg.stall_ticks)
            with open(val_path, "a", newline="") as f:
                csv.writer(f).writerow(_eval_row(it, it + 1, total, ev, names))
            last = _summary_eval(ev, it)
            print(json.dumps({"validation_iteration": it, "success": ev["success"], "groups": ev["groups"],
                              "seconds": ev["seconds"]}), flush=True)
            ck = checkpoint(gate, slots, it, cfg_dict, opt, compact(ev))
            _save(ck, out / "checkpoints" / "latest.pt")
            if ev["success"] > best:  # ties keep the earlier
                best, best_info = ev["success"], last
                _save(ck, out / "checkpoints" / "best.pt")
            (out / "summary.json").write_text(json.dumps(
                {"best": best_info, "last_eval": last, "iteration": it, "updates": it + 1, "env_ticks": total,
                 "env_steps": total, "minutes": round((time.time() - t_start) / 60, 2), "completed": False,
                 "slots": slots, "swaps": swaps}, indent=1))
    log_f.close()
    summary = json.loads((out / "summary.json").read_text())
    summary["completed"] = True
    if cfg.test_once:  # once, with best.pt, after training (the main session decides when)
        tested, ck = load_gate(out / "checkpoints" / "best.pt", dev)
        ev = validate(tested, experts, T.skill_sets("test", cfg.eval_per_level, cfg.eval_jumps, tuple(skills), ice=True),
                      dev, cfg.eval_batch_size, cfg.stall_ticks)
        (out / "test_eval.json").write_text(json.dumps({"iteration": ck["iteration"], **ev}, indent=1))
        summary["test_eval"] = {"iteration": ck["iteration"], **compact(ev)}
        print(json.dumps({"selected_checkpoint_test": ck["iteration"], "success": ev["success"],
                          "groups": ev["groups"]}), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return out


# Command line --------------------------------------------------------------------------------------------------

def parse_train(argv=None) -> GateConfig:
    ap = argparse.ArgumentParser(prog="python -m parkour.gate train", description="PPO training of the skill gate.")
    base = GateConfig()
    for f in dataclasses.fields(GateConfig):
        if f.name in ("reward", "hidden"):
            continue
        v = getattr(base, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=v)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    for f in dataclasses.fields(RewardConfig):
        ap.add_argument("--reward-" + f.name, type=float, default=getattr(RewardConfig(), f.name))
    ap.add_argument("--hidden", type=str, default=",".join(map(str, base.hidden)))
    a = ap.parse_args(argv)
    kw = {f.name: getattr(a, f.name) for f in dataclasses.fields(GateConfig) if f.name not in ("reward", "hidden")}
    kw["reward"] = RewardConfig(**{f.name: getattr(a, "reward_" + f.name) for f in dataclasses.fields(RewardConfig)})
    kw["hidden"] = tuple(int(x) for x in a.hidden.split(",") if x)
    return GateConfig(**kw)


def evaluate_cmd(argv=None) -> dict:
    ap = argparse.ArgumentParser(prog="python -m parkour.gate eval",
                                 description="The gated policy on held-out sets (simulator, deterministic).")
    ap.add_argument("--checkpoint", default="", help="a gate checkpoint (its slot list is used unless --experts)")
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--per-level", type=int, default=0, help="courses per level and part (default: validation "
                                                             "100, test 200)")
    ap.add_argument("--fixed-slot", default="", help="play this slot everywhere instead of the gate's choices")
    ap.add_argument("--experts", default="", help="an experts file instead of the checkpoint's slots (needs --swap "
                                                  "if they differ); without --checkpoint, the slots for --fixed-slot")
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--skills", default="", help="groups (default: the gate's training groups)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=600)
    ap.add_argument("--stall-ticks", type=int, default=200)
    ap.add_argument("--out", default="", help="write the result (with every course's outcome) to this JSON file")
    a = ap.parse_args(argv)
    gate, ck, swaps = None, None, []
    if a.checkpoint:
        gate, ck = load_gate(a.checkpoint, a.device)
        if a.experts:
            slots = read_slots(a.experts)
            swaps = compare_slots(ck["slots"], slots, a.swap)
        else:
            slots = current_slots(ck["slots"])
    elif a.fixed_slot:
        slots = read_slots(a.experts or DEFAULT_EXPERTS)
    else:
        raise SystemExit("give --checkpoint, or --fixed-slot with the experts to play")
    experts = Experts(slots, a.device)
    fixed = None
    if a.fixed_slot:
        if a.fixed_slot not in experts.names:
            raise SystemExit(f"no slot {a.fixed_slot}: {', '.join(experts.names)}")
        fixed = experts.names.index(a.fixed_slot)
    skills = skill_groups(a.skills or (ck["config"]["skills"] if ck else GateConfig.skills))
    per_level = a.per_level or PER_LEVEL[a.split]
    sets = T.skill_sets(a.split, per_level, 10, tuple(skills), ice=True)
    ev = validate(gate, experts, sets, a.device, a.batch_size, a.stall_ticks, fixed)
    fam = ({"standard": T.VALIDATION_FAMILY, "obstacles": T.OBSTACLE_VALIDATION_FAMILY, "ice": T.ICE_VALIDATION_FAMILY}
           if a.split == "validation" else
           {"standard": T.TEST_SEED_FAMILY, "obstacles": T.OBSTACLE_TEST_FAMILY, "ice": T.ICE_TEST_FAMILY})
    result = {"what": "parkour.gate eval: the gated policy, simulator, deterministic experts and gate (argmax)",
              "split": a.split, "families": fam, "per_level_courses": per_level, "checkpoint": a.checkpoint or None,
              "checkpoint_sha1": file_sha1(a.checkpoint) if a.checkpoint else None,
              "iteration": ck.get("iteration") if ck else None, "fixed_slot": a.fixed_slot or None, "slots": slots,
              "swaps": swaps, **ev}
    print(json.dumps({"split": a.split, "fixed_slot": a.fixed_slot or None, "success": ev["success"],
                      "groups": ev["groups"], "usage": {g: u["decisions"] for g, u in ev["usage"].items()},
                      "courses": ev["courses"], "seconds": ev["seconds"]}), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=1))
    return result


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("train", "eval"):
        print(__doc__)
        raise SystemExit(2)
    if argv[0] == "train":
        train(parse_train(argv[1:]))
    else:
        evaluate_cmd(argv[1:])


if __name__ == "__main__":
    main()

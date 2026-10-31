"""The neo skill: a second network that plays only a neo's block (opt-in; HANDOFF.md section 3, rules of 31 Oct).

    python -m parkour.neo_skill bank --out runs/neo_skill_bank.pt
    run.cmd train --run neo-skill-dev --init-from runs/neo-dagger4/checkpoints/updates_00250.pt --segments runs/neo_skill_bank.pt
    python -m parkour.neo_assist --off --set single --family 40 --skill runs/neo-skill-dev/checkpoints/updates_00100.pt

The final policy runs ordinary courses at 98% but never passes a neo, and the
best network that does (``neo-dagger4``) passes about half of the neos it
reaches, and also fails the jumps onto the neo's 1x1 takeoff and off its 1x1
target more often than other jumps. The neo skill takes the whole block: in
the neo-skill mode (``neo_assist --off --skill``) the final policy plays
until it first stands on the platform before a neo's takeoff (k-1; from the
start when k is 0 or 1), the skill plays from there until it first stands
on the platform after the neo's target (k+2, or the goal), then the final
policy plays on (``neo_assist.skill_cps``). The switch is the course list's
neo detector (``neo_dagger.neo_takeoffs``), a fixed gate; no search runs.

Training (``train.py --segments BANK``) is PPO on the block alone:
``SegmentEnv`` starts every episode in the exact state in which the final
policy (deterministic) first stood on k-1 of a training single-neo course,
the held-out sets' own generator (``neo_assist.single_neo_courses``) on
families 60-69, none equal to a course of the single-neo sets (families 40,
41, 49). The bank (``bank``) holds those states, from the simulator's start
and after the bots' 10 settle ticks. An episode succeeds when the bot stands
on k, then on k+1, then reaches k+2 (or the goal): the course's goal reward
there, so the value of the block is what the skill learns. Skipping k or k+1
(``block_rule``) ends the episode as a fall: the strict metric counts a neo
only if the bot stood on its takeoff and then on its target. Falls, stalls
and the other rewards are the environment's. The course's level is its neo
type (0-31, ``neo_assist.single_neo_types``), so the training log shows
each type's success, and ``TypeSampler`` draws the types that fail more
often more often.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import courses as C
from . import physics
from .env import (FALL_MARGIN, MAX_PLATFORMS, CoursePool, EnvConfig, LevelSampler, ParkourEnv, action_to_controls,
                  on_obstacle)
from .live import load_model
from .neo_assist import (SINGLE_FRESH_TEST_FAMILY, SINGLE_FRESH_TEST_FAMILY_2, SINGLE_TEST_FAMILY, SINGLE_VALIDATION_FAMILY,
                         single_neo_courses, single_neo_hashes, single_neo_types, skill_cps)
from .neo_dagger import neo_takeoffs
from .paths import ROOT
from .train import course_hash

TRAIN_FAMILIES = tuple(range(60, 70))  # the bank's single-neo courses; nothing else uses these families
SETTLES = (0, 10)  # the simulator's start, and the bots' (parkour.live.SETTLE_TICKS)
STATE_FIELDS = ("pos", "vel", "yaw", "on_ground", "jump_ticks", "collided_h", "collided_v")


def block(course: C.Course) -> tuple[int, int, int]:
    """(k, first, end) of the course's one neo: its takeoff k, the platform the skill starts on (k-1, or 0) and
    the one it ends on (k+2, or the goal)."""
    ks = neo_takeoffs(course)
    if len(ks) != 1:
        raise ValueError(f"a block course has exactly one neo, not {len(ks)}")
    k = ks[0]
    return k, max(k - 1, 0), min(k + 2, course.n - 1)


def block_rule(cp: torch.Tensor, new_cp: torch.Tensor, k: torch.Tensor, end: torch.Tensor):
    """(success, skipped) of one step in a block: skipped when the furthest platform stood on went past the
    takeoff k without standing on it, or past the target k+1 without standing on it; success when it reached
    ``end`` without skipping."""
    skipped = ((cp < k) & (new_cp > k)) | ((cp < k + 1) & (new_cp > k + 1))
    return (new_cp >= end) & ~skipped, skipped


def training_courses(families=TRAIN_FAMILIES, per_family: int = 6400) -> tuple[list[C.Course], list[int]]:
    """Single-neo courses of the held-out sets' generator on training families, none equal to a course of the
    single-neo sets; each course's level set to its neo type (course i of a family has type i mod 32)."""
    exclude = single_neo_hashes()
    n_types = len(single_neo_types())
    out, types = [], []
    for f in families:
        if f in (SINGLE_VALIDATION_FAMILY, SINGLE_TEST_FAMILY, SINGLE_FRESH_TEST_FAMILY, SINGLE_FRESH_TEST_FAMILY_2):
            raise ValueError(f"family {f} is a held-out single-neo set")
        for i, c in enumerate(single_neo_courses(f, per_family)):
            if course_hash(c) in exclude:
                continue
            c.level = i % n_types
            out.append(c)
            types.append(i % n_types)
    return out, types


def collect_starts(model, courses: list[C.Course], device, settle: int = 0, max_ticks: int = 2000) -> list:
    """Per course, the state in which ``model`` (deterministic, as ``train.evaluate``) first stands on the block's
    first platform, or None if its first episode ends before. From the start (after ``settle`` idle ticks) when
    the block starts there. A dict of ``STATE_FIELDS``, ``cp`` (the first platform, or the takeoff if the model
    jumped over the first one) and ``last_action``."""
    pool = CoursePool(courses, device, C.GRID)
    n = len(courses)
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=200, max_ticks=max_ticks), sampler=None)
    obs = env.reset_all(torch.arange(n, device=device))
    if settle:
        idle = torch.zeros(n, 6, dtype=torch.bool, device=device)
        for _ in range(settle):
            env.state = physics.simulate(pool.world, env.course, env.state, idle)
        env.dist = env._distance(env.state.pos, env.course, env.cp)
        obs = env.observe()
    blocks = [block(c) for c in courses]
    first = torch.tensor([b[1] for b in blocks], device=device)
    k = torch.tensor([b[0] for b in blocks], device=device)
    out = [None] * n

    def take(rows):
        s = env.state
        for i in rows:
            out[i] = {"pos": s.pos[i].tolist(), "vel": s.vel[i].tolist(), "yaw": float(s.yaw[i]),
                      "on_ground": bool(s.on_ground[i]), "jump_ticks": int(s.jump_ticks[i]),
                      "collided_h": bool(s.collided_h[i]), "collided_v": bool(s.collided_v[i]),
                      "cp": int(env.cp[i]), "last_action": env.last_action[i].tolist()}

    at_start = first == 0
    take(torch.nonzero(at_start).flatten().tolist())
    done = at_start.clone()
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    with torch.no_grad():
        for _ in range(max_ticks):
            if bool((done | finished).all()):
                break
            a, _, _ = model.act(obs, deterministic=True)
            obs, _, _, _, _, info = env.step(a)
            finished |= info["done"]
            now = ~done & ~finished & (env.cp >= first)
            if bool(now.any()):
                rows = torch.nonzero(now).flatten().tolist()
                take([i for i in rows if int(env.cp[i]) <= int(k[i])])  # (past the takeoff: no block to play)
                done |= now
    return out


def build_bank(checkpoint: Path, out: Path, families=TRAIN_FAMILIES, per_family: int = 6400, device="cuda",
               batch: int = 4096) -> dict:
    """The training bank: every training course with at least one start (``collect_starts``, per ``SETTLES``)."""
    t0 = time.time()
    model = load_model(checkpoint, device)
    courses, types = training_courses(families, per_family)
    V = len(SETTLES)
    rows = []
    for lo in range(0, len(courses), batch):
        part = courses[lo:lo + batch]
        per = [collect_starts(model, part, device, settle=s) for s in SETTLES]
        for j, c in enumerate(part):
            starts = [per[v][j] for v in range(V)]
            if any(starts):
                rows.append((lo + j, starts))
        print(json.dumps({"courses": lo + len(part), "kept": len(rows), "seconds": round(time.time() - t0, 1)}),
              flush=True)
    n = len(rows)
    arr = {"pos": torch.zeros(n, V, 3, dtype=torch.float64), "vel": torch.zeros(n, V, 3, dtype=torch.float64),
           "yaw": torch.zeros(n, V, dtype=torch.float64), "on_ground": torch.zeros(n, V, dtype=torch.bool),
           "jump_ticks": torch.zeros(n, V, dtype=torch.int64), "collided_h": torch.zeros(n, V, dtype=torch.bool),
           "collided_v": torch.zeros(n, V, dtype=torch.bool), "cp": torch.zeros(n, V, dtype=torch.int64),
           "last_action": torch.zeros(n, V, 5, dtype=torch.int64), "valid": torch.zeros(n, V, dtype=torch.bool)}
    for r, (_, starts) in enumerate(rows):
        for v, st in enumerate(starts):
            if st is None:
                continue
            arr["valid"][r, v] = True
            for key in STATE_FIELDS + ("cp", "last_action"):
                arr[key][r, v] = torch.tensor(st[key], dtype=arr[key].dtype)  # (float64: the exact state)
    ck = Path(checkpoint).resolve()
    data = {"courses": [courses[i].to_json() for i, _ in rows], "types": [types[i] for i, _ in rows],
            "k": [block(courses[i])[0] for i, _ in rows], **arr,
            "meta": {"checkpoint": ck.relative_to(ROOT).as_posix() if ck.is_relative_to(ROOT) else ck.name,
                     "checkpoint_sha1": hashlib.sha1(ck.read_bytes()).hexdigest()[:12], "families": list(families),
                     "per_family": per_family, "settles": list(SETTLES), "generated": len(courses), "kept": n,
                     "starts": {str(s): int(arr["valid"][:, v].sum()) for v, s in enumerate(SETTLES)},
                     "device": str(device), "seconds": round(time.time() - t0, 1)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    print(json.dumps(data["meta"]), flush=True)
    return data["meta"]


class SegmentBank:
    """A bank file's courses and starts, handed out in order (and again from the first when all were used)."""

    def __init__(self, path, device, exclude: set = frozenset()):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.meta = d["meta"]
        self.courses = [C.Course.from_json(c) for c in d["courses"]]
        for c, t in zip(self.courses, d["types"]):
            c.level = int(t)
        held = set(exclude) | single_neo_hashes()
        if any(course_hash(c) in held for c in self.courses):
            raise ValueError("the bank holds a held-out course")
        self.k = torch.as_tensor(d["k"], dtype=torch.int64)
        self.end = torch.as_tensor([min(int(k) + 2, c.n - 1) for k, c in zip(d["k"], self.courses)], dtype=torch.int64)
        self.starts = {key: d[key] for key in STATE_FIELDS + ("cp", "last_action", "valid")}
        if not bool(self.starts["valid"].any(1).all()):
            raise ValueError("every bank course needs a start")
        self.n_types = len(single_neo_types())
        self.device = device
        self.next = 0

    def take(self, k: int) -> list[int]:
        idx = [(self.next + j) % len(self.courses) for j in range(k)]
        self.next = (self.next + k) % len(self.courses)
        return idx

    def rows(self, idx: list[int]) -> dict:
        """The env's per-slot tensors of bank rows ``idx``."""
        i = torch.as_tensor(idx, dtype=torch.int64)
        out = {key: v[i].to(self.device) for key, v in self.starts.items()}
        out["k"] = self.k[i].to(self.device)
        out["end"] = self.end[i].to(self.device)
        return out


class JumpReadout:
    """A skill whose deterministic jump reads "jump" when the jump head's probability is above ``t`` (argmax: t 0.5;
    the other heads stay argmax), for a policy that jumps at a random tick in training (HANDOFF.md, 31 Oct, 11:43)."""

    def __init__(self, model, t: float):
        self.model, self.t, self.obs_dim = model, t, model.obs_dim

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        a, logp, v = self.model.act(obs, deterministic)
        if deterministic:
            jump = self.model.logits(self.model.norm(obs))[2]
            a = a.clone()
            a[:, 2] = (torch.softmax(jump, -1)[:, 1] > self.t).to(a.dtype)
        return a, logp, v


class TypeSampler(LevelSampler):
    """Neo types (the courses' levels) drawn with weight ``floor`` + their recent failure rate (1 until a type
    has ``min_episodes`` in its window), so the types that fail more are played more."""

    def __init__(self, n_levels: int, floor: float = 0.25, min_episodes: int = 200, window: int = 2000):
        super().__init__(n_levels, window=window)
        self.floor = floor
        self.min_episodes = min_episodes

    def probs(self) -> np.ndarray:
        w = np.full(self.n_levels, self.floor + 1.0)
        for level in range(self.n_levels):
            rate, n = self._rate(level)
            if rate is not None and n >= self.min_episodes:
                w[level] = self.floor + (1.0 - rate)
        return w / w.sum()


class SegmentEnv(ParkourEnv):
    """``ParkourEnv`` whose episodes are a neo's block: they start in a bank state on the block's first
    platform and end at its last (``block_rule``); everything else is the environment's own.

    With ``base`` (a frozen policy, the one the skill hands back to: ``train.py --segment-base``) an episode goes
    on for the jump after the block: once the bot stands on k+2 the base plays (its own deterministic action
    replaces the one given to ``step``), and the episode succeeds when the bot reaches k+3 (or the goal). The
    skill's last landing is then judged by what the base makes of it. ``info["ctrl"]`` marks the rows whose given
    action was played (the trainer leaves the others out of the policy loss)."""

    def __init__(self, pool: CoursePool, config: EnvConfig, sampler=None, seed: int = 0, slots: dict | None = None,
                 base=None):
        self.slot = slots  # per pool course: the starts (P, V, ...), the takeoff k and the block's end
        self.base = base
        super().__init__(pool, config, sampler=sampler, seed=seed)

    def set_slot(self, index: int, rows: dict) -> None:
        """Pool course ``index`` now holds the block of bank row ``rows`` (``SegmentBank.rows`` of one row)."""
        for key, v in rows.items():
            self.slot[key][index] = v[0]

    def _apply_reset(self, mask: torch.Tensor, courses: torch.Tensor) -> None:
        super()._apply_reset(mask, courses)
        c = self.course
        v = torch.multinomial(self.slot["valid"][c].to(torch.float32), 1).squeeze(1)  # one of the course's starts
        start = physics.PlayerState(*(self.slot[key][c, v] for key in STATE_FIELDS))
        self.state.assign(mask, start)
        self.cp = torch.where(mask, self.slot["cp"][c, v], self.cp)
        self.last_action = torch.where(mask[:, None], self.slot["last_action"][c, v], self.last_action)
        self.dist = torch.where(mask, self._distance(self.state.pos, self.course, self.cp), self.dist)
        self.reset_memory(mask)

    def step(self, action: torch.Tensor):
        """``ParkourEnv.step`` with the block's end as the goal and a skipped platform as a fall."""
        cfg = self.cfg.reward
        kk = self.slot["k"][self.course]
        end = self.slot["end"][self.course]
        ctrl = torch.ones_like(self.cp, dtype=torch.bool)  # the given action is played
        if self.base is not None:  # the jump after the block is the base's
            end = torch.minimum(end + 1, self.pool.n[self.course] - 1)
            ctrl = self.cp < kk + 2
            if not bool(ctrl.all()):
                with torch.no_grad():
                    action = torch.where(ctrl[:, None], action, self.base.act(self.observe(), deterministic=True)[0])
        s = self.state
        yaw = s.yaw + self.yaw_steps[action[:, 4]]
        s.yaw = torch.remainder(yaw + math.pi, 2 * math.pi) - math.pi
        prev_ground = s.on_ground
        new = physics.simulate(self.pool.world, self.course, s, action_to_controls(action))
        self.state = new
        self.last_action = action
        self.ticks = self.ticks + 1
        if self.cfg.extended_obs:
            self._remember_landings(prev_ground)

        stand = self.standing_on(new, self.course)
        k = self.k_range[None, :]
        ahead = stand & (k > self.cp[:, None])
        reached = torch.where(ahead, k, torch.full_like(k, -1)).amax(dim=1)
        new_cp = torch.maximum(self.cp, reached)
        gained = (new_cp - self.cp).to(torch.float64)
        last = self.pool.n[self.course] - 1
        success, skipped = block_rule(self.cp, new_cp, kk, end)

        dist = self._distance(new.pos, self.course, new_cp)
        progress = self.dist - dist  # (a skip still earns its landings and progress, then the fall penalty)
        j = new_cp[:, None] + self.fall_range[None, :]
        jv = (j >= 0) & (j <= last[:, None])
        tops = self.pool.top[self.course[:, None], j.clamp(0, MAX_PLATFORMS - 1)]
        tops = torch.where(jv, tops, torch.full_like(tops, math.inf))
        fell = (new.pos[:, 1] < tops.amin(dim=1) - FALL_MARGIN) & ~success
        fell = fell | (on_obstacle(self.pool, new, self.course) & ~success) | skipped

        self.stall = torch.where(gained > 0, torch.zeros_like(self.stall), self.stall + 1)
        truncated = ((self.stall >= self.cfg.stall_ticks) | (self.ticks >= self.cfg.max_ticks)) & ~success & ~fell
        stalled = truncated & (self.stall >= self.cfg.stall_ticks) & (cfg.stall > 0)
        terminal = success | fell | stalled
        reward = (cfg.progress * progress + cfg.landing * gained + cfg.goal * success.to(torch.float64)
                  - cfg.fall * fell.to(torch.float64) - cfg.stall * stalled.to(torch.float64) - cfg.time)
        self.cp = new_cp
        self.dist = dist
        self.ep_return = self.ep_return + reward

        done = terminal | truncated
        final_obs = self.observe()
        level = self.pool.level[self.course]
        frac = new_cp.to(torch.float64) / last.clamp(min=1).to(torch.float64)
        d = done.to(torch.float64)
        for i, v in enumerate((d, (success & done).to(torch.float64), fell.to(torch.float64),
                               truncated.to(torch.float64), d * self.ticks, d * self.ep_return, d * frac)):
            self.stats[i].index_add_(0, level, v)
        info = {"done": done, "success": success & done, "fell": fell, "truncated": truncated,
                "ep_len": self.ticks, "ep_return": self.ep_return, "level": level, "progress_frac": frac,
                "course": self.course, "skipped": skipped, "ctrl": ctrl}
        self._apply_reset(done, self._sample_courses(self.n))
        obs = self.observe()
        return obs, reward.to(torch.float32), terminal, truncated & ~stalled, final_obs, info


def takeoff_states(skill, bank: SegmentBank, idx: list[int], device, waits=(0, 10, 20), max_ticks: int = 600,
                   air_every: int = 0, exits: bool = False, air_max: int = 12):
    """The skill's own states in the blocks of bank rows ``idx`` (its deterministic block from a bank start): on the
    takeoff at its first ground contact and ``waits`` ticks later while it still stands there; with ``air_every``,
    also every that many ticks in the air between the takeoff and the target (searched to the target); with
    ``exits``, its first ground contact on the target (searched to the platform after it). Returns the pool and the
    arrival dicts (``neo_assist._arrival``: the target is the next platform) for the neo planner's search."""
    from .neo_assist import _arrival
    pool = CoursePool([bank.courses[i] for i in idx], device, C.GRID)
    n = len(idx)
    env = SegmentEnv(pool, EnvConfig(num_envs=n, stall_ticks=200, max_ticks=max_ticks), slots=bank.rows(idx))
    obs = env.reset_all(torch.arange(n, device=device))
    k = env.slot["k"]
    end = env.slot["end"]
    rows = torch.arange(n, device=device)
    first = torch.full((n,), -1, dtype=torch.int64, device=device)
    left = torch.full((n,), -1, dtype=torch.int64, device=device)  # the tick it left the takeoff
    exited = torch.zeros(n, dtype=torch.bool, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    out = []
    with torch.no_grad():
        for t in range(max_ticks):
            a, _, _ = skill.act(obs, deterministic=True)
            obs, _, _, _, _, info = env.step(a)
            finished |= info["done"]
            on = ~finished & (env.cp == k) & env.standing_on(env.state, env.course)[rows, k]
            first = torch.where(on & (first < 0), torch.full_like(first, t), first)
            kinds = {"takeoff": on & (first >= 0) & torch.isin(t - first, torch.tensor(waits, device=device))}
            if air_every:  # the first ``air_max`` ticks of its first flight off the takeoff, while not below it
                air = ~finished & (env.cp == k) & (first >= 0) & ~env.state.on_ground
                left = torch.where(air & (left < 0), torch.full_like(left, t), left)
                high = env.state.pos[:, 1] > env.pool.top[env.course, k] - 0.5
                kinds["air"] = (air & high & (left >= 0) & ((t - left) % air_every == 0) & (t - left <= air_max)
                                & (t > left))
            if exits:
                kinds["exit"] = ~finished & ~exited & (env.cp == k + 1) & (k + 1 < end)
                exited |= kinds["exit"]
            for kind, take in kinds.items():
                out += [dict(_arrival(env, i), kind=kind) for i in torch.nonzero(take).flatten().tolist()]
            if bool(finished.all()) or (not air_every and not exits
                                        and bool((finished | ((first >= 0) & (t - first > max(waits)))).all())):
                break
    return pool, out


def dagger_pairs(skill_path: Path, bank_path: Path, out: Path, types=(16, 17, 18, 19), courses: int = 6000,
                 device="cuda", batch: int = 256, prior: float = 0.95, seed: int = 0, air_every: int = 0,
                 exits: bool = False) -> dict:
    """(observation, action) pairs from the neo planner's landings, simplified, at the skill's own takeoff states
    (``takeoff_states``) on the bank's courses of ``types`` (HANDOFF.md, 31 Oct, 11:46): ``train.py --bc-pairs``."""
    from . import neo_assist as NA
    from .neo_dagger import HORIZON, pairs, simplify
    t0 = time.time()
    skill = load_model(skill_path, device)
    bank = SegmentBank(bank_path, "cpu")
    idx = [i for i, c in enumerate(bank.courses) if c.level in types][:courses]
    bank.device = device
    pool, arrivals = takeoff_states(skill, bank, idx, device, air_every=air_every, exits=exits)
    rules = NA.Rules(pool)
    cfg = NA.AssistConfig(keep=1, prior=prior)
    found = []
    for b in range(0, len(arrivals), batch):
        part = arrivals[b:b + batch]
        pr = NA.policy_prior(skill, pool, part, device, HORIZON) if prior > 0 else None
        found += NA.search(pool, rules, part, device, cfg, seed + b, pr, HORIZON)[0]
        print(json.dumps({"searched": len(found), "of": len(arrivals), "with_landing": sum(bool(f) for f in found),
                          "seconds": round(time.time() - t0, 1)}), flush=True)
    simple = simplify(skill, pool, arrivals, found, device)
    obs, act = pairs(pool, arrivals, simple, device)
    meta = {"skill": str(skill_path), "skill_sha1": hashlib.sha1(Path(skill_path).read_bytes()).hexdigest()[:12],
            "bank": str(bank_path), "types": list(types), "courses": len(idx), "air_every": air_every, "exits": exits,
            "arrivals": len(arrivals),
            "kinds": {kd: [sum(1 for x in arrivals if x["kind"] == kd),
                           sum(1 for x, f in zip(arrivals, found) if x["kind"] == kd and f)] for kd in ("takeoff", "air", "exit")},
            "with_landing": sum(bool(f) for f in found), "simplified": sum(len(f) for f in simple),
            "pairs": int(obs.shape[0]), "prior": prior, "device": str(device), "seconds": round(time.time() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"obs": obs, "act": act, **meta}, out)
    print(json.dumps(meta), flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dagger", help="imitation pairs from the neo planner at the skill's own takeoff states")
    d.add_argument("--skill", required=True)
    d.add_argument("--bank", default=str(ROOT / "runs" / "neo_skill_bank_v1.pt"))
    d.add_argument("--out", required=True)
    d.add_argument("--types", default="16,17,18,19")
    d.add_argument("--courses", type=int, default=6000)
    d.add_argument("--batch", type=int, default=256)
    d.add_argument("--device", default="cuda")
    d.add_argument("--air-every", type=int, default=0, help="also the skill's states every N ticks in the air after the takeoff")
    d.add_argument("--exits", action="store_true", help="also its first ground contact on the target (to the next platform)")
    d.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("bank", help="the training bank of block starts")
    b.add_argument("--checkpoint", default=str(ROOT / "results" / "final_policy.pt"))
    b.add_argument("--out", default=str(ROOT / "runs" / "neo_skill_bank.pt"))
    b.add_argument("--per-family", type=int, default=6400)
    b.add_argument("--families", default=f"{TRAIN_FAMILIES[0]}-{TRAIN_FAMILIES[-1]}")
    b.add_argument("--device", default="cuda")
    b.add_argument("--batch", type=int, default=4096)
    a = ap.parse_args(argv)
    if a.cmd == "dagger":
        if Path(a.out).exists():
            raise SystemExit(f"{a.out} exists; choose a new name")
        dagger_pairs(Path(a.skill), Path(a.bank), Path(a.out), tuple(int(t) for t in a.types.split(",")), a.courses,
                     a.device, a.batch, seed=a.seed, air_every=a.air_every, exits=a.exits)
        return
    if a.cmd == "bank":
        lo, _, hi = a.families.partition("-")
        fams = tuple(range(int(lo), int(hi or lo) + 1))
        if Path(a.out).exists():
            raise SystemExit(f"{a.out} exists; choose a new name")
        build_bank(Path(a.checkpoint), Path(a.out), fams, a.per_family, a.device, a.batch)


if __name__ == "__main__":
    main()

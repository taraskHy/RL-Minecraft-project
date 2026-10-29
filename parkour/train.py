"""PPO training on procedurally generated parkour courses.

    python -m parkour.train --run NAME [--iterations 3000] [--curriculum adaptive]

Everything runs on the GPU: thousands of environments step in parallel in the
batched simulator, and the network trains on the same device. Training
courses are freshly generated as the run goes on; evaluation uses a fixed
held-out set of courses (different seeds, checked to share no course with
training) at every difficulty level. Results go to runs/NAME/.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import courses as C
from . import physics
from .env import ACTION_DIMS, OBS_DIM, OBS_DIM_EXT, CoursePool, EnvConfig, LevelSampler, ParkourEnv, RewardConfig
from .model import ActorCritic, RunningNorm
from .paths import RUNS

TEST_SEED_FAMILY = 1  # numpy SeedSequence entropy [2026, family, ...]; training uses family 0
OBSTACLE_TEST_FAMILY = 4  # held-out courses with obstacles
VALIDATION_FAMILY = 2  # model selection (best.pt); the test families are only reported
OBSTACLE_VALIDATION_FAMILY = 5
ICE_TEST_FAMILY = 16  # skills mode: basic levels with ice on 30% of platforms (16 and 17 were unused)
ICE_VALIDATION_FAMILY = 17

# Skills mode (--skills, docs/combine_policies.md): every training course comes from one specialist's own
# level table, so every jump on it is one its teacher trained on. Global level ids: basic 0-5, advanced
# 6-8, extra 9-14; the held-out ice group (basic levels with ice) is 15-20.
SKILL_GROUPS = {  # group -> (level table keywords, first and last local level, first global id)
    "basic": ({}, 0, 5, 0),
    "advanced": ({"advanced": True}, 6, 8, 6),
    "extra": ({"extra": True}, 6, 11, 9),
}
ICE_GROUP = (15, 20)
SKILL_ICE_PROB = 0.3  # the held-out ice sets, as g995-ice's training courses
GROUP_ORDER = ("basic", "ice", "advanced", "extra")  # best.pt = the highest mean over these groups (validation)


def skill_plan(groups) -> list[tuple]:
    """[(global level id, group, local level)] of the training levels of ``groups``."""
    out = []
    for g in groups:
        _, lo, hi, first = SKILL_GROUPS[g]
        out += [(first + L - lo, g, L) for L in range(lo, hi + 1)]
    return out


def level_groups() -> dict:
    """Global level id -> group, the ice group included."""
    out = {gid: g for g in SKILL_GROUPS for gid, _, _ in skill_plan([g])}
    out.update({gid: "ice" for gid in range(ICE_GROUP[0], ICE_GROUP[1] + 1)})
    return out


def skill_tables(groups) -> dict:
    return {g: C.load_levels(**SKILL_GROUPS[g][0]) for g in groups}


def skill_sets(split: str, per_level: int, n_jumps: int, groups=("basic", "advanced", "extra"), ice: bool = True) -> dict:
    """Held-out courses of skills mode: group -> {"standard": [...], "obstacles": [...]}, global level ids.

    ``split`` "validation" uses families 2 and 5 (ice 17), "test" 1 and 4 (ice 16). Each group's courses are the
    specialist's own held-out courses at its levels (same table, family and per-level seeds), relabelled."""
    if split == "validation":
        fam, ofam, ifam = VALIDATION_FAMILY, OBSTACLE_VALIDATION_FAMILY, ICE_VALIDATION_FAMILY
    elif split == "test":
        fam, ofam, ifam = TEST_SEED_FAMILY, OBSTACLE_TEST_FAMILY, ICE_TEST_FAMILY
    else:
        raise ValueError(split)
    tables = skill_tables(set(groups) | ({"basic"} if ice else set()))
    out = {}
    for g in groups:
        _, lo, hi, first = SKILL_GROUPS[g]
        parts = {"standard": held_out_courses(tables[g], per_level, n_jumps, family=fam),
                 "obstacles": held_out_courses(tables[g], per_level, n_jumps, obstacle_prob=0.5, family=ofam)}
        for k, cs in parts.items():
            parts[k] = [c for c in cs if lo <= c.level <= hi]
            for c in parts[k]:
                c.level = first + c.level - lo
        out[g] = parts
    if ice:
        cs = held_out_courses(tables["basic"], per_level, n_jumps, family=ifam, ice_prob=SKILL_ICE_PROB)
        for c in cs:
            c.level = ICE_GROUP[0] + c.level
        out["ice"] = {"standard": cs, "obstacles": []}
    return out


def group_scores(*per_level: dict) -> dict:
    """Success per group from per-level results with global ids (each course weighs the same)."""
    lg = level_groups()
    acc = {}
    for pl in per_level:
        for L, r in pl.items():
            g = lg.get(int(L))
            if g is not None:
                s, n = acc.get(g, (0.0, 0))
                acc[g] = (s + r["success"] * r["n"], n + r["n"])
    return {g: s / n for g, (s, n) in acc.items() if n}


def macro_score(groups: dict) -> float:
    present = [groups[g] for g in GROUP_ORDER if g in groups]
    return sum(present) / len(present)


def level_columns(per_level: dict, key: str, n_levels: int) -> list:
    """One CSV cell per level; blank for a level without courses (skills mode: the ice ids in obstacle sets)."""
    return [round(per_level[L][key], 4) if L in per_level else "" for L in range(n_levels)]


def load_frozen(path: str, device) -> ActorCritic:
    """A teacher: the checkpoint's network with its own observation normaliser, frozen."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    model = ActorCritic(obs_dim=ck["model"]["norm.mean"].shape[0], hidden=tuple(cfg.get("hidden", (512, 256, 128))),
                        hidden_parts=tuple(p for p in cfg.get("hide", "").split(",") if p)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def parse_teachers(spec: str, groups) -> dict:
    """"basic=CKPT,advanced=CKPT" -> {group: path}."""
    out = dict(p.split("=", 1) for p in spec.split(",") if p)
    bad = sorted(set(out) - set(groups))
    if bad:
        raise ValueError(f"--teachers names groups that are not in --skills: {bad}")
    return out


def teacher_table(teachers: dict, groups, paths: list) -> torch.Tensor:
    """Global level id -> index of its group's teacher in ``paths`` (-1: no teacher)."""
    plan = skill_plan(groups)
    table = torch.full((max(gid for gid, _, _ in plan) + 1,), -1, dtype=torch.int64)
    for gid, g, _ in plan:
        if g in teachers:
            table[gid] = paths.index(teachers[g])
    return table


def teacher_rows(env: ParkourEnv, table: torch.Tensor) -> torch.Tensor:
    """Teacher index of every environment, from the level of the course it is on now (read before ``env.step``,
    which moves finished players to new courses)."""
    return table[env.pool.level[env.course]]


def teacher_logp(teacher: ActorCritic, obs: torch.Tensor) -> torch.Tensor:
    """The teacher's log-probabilities of every head on raw observations, side by side (sum(ACTION_DIMS) numbers)."""
    return torch.cat([torch.log_softmax(lg, -1) for lg in teacher.logits(teacher.norm(obs))], -1)


def distill_kl(student_logits, teacher_lp: torch.Tensor) -> torch.Tensor:
    """Per-row KL(teacher || student), summed over the five action heads."""
    kl, o = 0.0, 0
    for lg, d in zip(student_logits, ACTION_DIMS):
        t = teacher_lp[:, o:o + d]
        kl = kl + (t.exp() * (t - torch.log_softmax(lg, -1))).sum(-1)
        o += d
    return kl


@dataclass
class TrainConfig:
    run: str = "dev"
    seed: int = 0
    device: str = "cuda"
    gpu_mem_fraction: float = 0.0  # >0: cap this process at that share of GPU memory (0.75 while the user games)
    gpu_duty: float = 1.0  # <1: sleep after each iteration so the GPU works at most this share of the time
    threads: int = 0  # >0: torch CPU threads
    num_envs: int = 16384
    rollout: int = 32
    iterations: int = 3000
    epochs: int = 4
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    # Stability (added after two catastrophic updates in the first runs; see docs/design.md):
    dual_clip: float = 3.0  # bound the objective at dual_clip * A for negative advantages
    target_kl: float = 0.01  # learning rate adapts to keep the per-minibatch KL near this
    kl_stop: float = 0.05  # skip the rest of an update once a minibatch's KL exceeds this
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    hidden: tuple = (512, 256, 128)
    pool_size: int = 8192
    pool_refresh_every: int = 20
    pool_refresh_frac: float = 0.125
    jumps_min: int = 5
    jumps_max: int = 12
    turn_prob: float = 0.25
    curriculum: str = "adaptive"  # adaptive | uniform
    eval_every: int = 50
    compact_ice: bool = False  # opt-in uint8 material grid; queried friction remains float64
    eval_batch_size: int = 0  # >0: evaluate the same fixed courses in memory-bounded batches
    test_once: bool = False  # select on validation; test only the selected checkpoint after training
    eval_per_level: int = 200
    val_per_level: int = 100
    eval_jumps: int = 10
    stall_ticks: int = 200
    reward: RewardConfig = field(default_factory=RewardConfig)
    init_from: str = ""
    hide: str = ""  # comma-separated observation parts the policy must not see, e.g. "platforms"
    extended_obs: bool = False  # far scan and landing memory (env.OBS_LAYOUT_EXT)
    # Learning from demonstrations (parkour.neo_demos): a behaviour-cloning loss on the
    # demonstrations' (observation, action) pairs next to PPO, and pool slots whose courses start
    # where a demonstration starts (takeoff, the jump, an easy platform). Slots are never refreshed.
    demos: str = ""
    bc_coef: float = 0.0
    bc_decay: bool = False  # with --bc-decay the behaviour-cloning weight falls linearly to 0 over the run
    # (the CLI makes every bool flag default to off, so all neo runs used a constant weight)
    bc_batch: int = 4096
    demo_courses: int = 0
    demo_level: int = 6
    bc_pairs: str = ""  # comma-separated .pt files of (observation, action) pairs (parkour.neo_dagger)
    wide_neo: bool = False  # training courses may place neos after wide takeoffs (courses._place)
    obstacle_prob: float = 0.0  # chance per jump of a head-hitter beam or a decoy pillar
    ice_prob: float = 0.0  # chance per platform after the start of being ice (training courses only)
    advanced: bool = False  # add levels 6-8: neos, double neos, momentum and wide jumps
    extra: bool = False  # add the head-hitter and ice run-up levels (configs/jump_levels_extra.json) after those
    save_at: str = ""  # comma-separated update counts at which to keep a snapshot (0 = untrained)
    # One network for every skill (docs/combine_policies.md), all off by default. --skills basic,advanced,extra:
    # each training course comes from one group's own level table (global level ids, SKILL_GROUPS; ice only on
    # basic courses, with --ice-prob); validation and test are each group's own held-out sets plus the ice sets
    # (families 17/16); best.pt = the highest mean over the groups basic, ice, advanced and extra. Needs
    # --test-once. --teachers basic=CKPT,advanced=CKPT,extra=CKPT with --distill-coef W adds
    # W * KL(teacher || student) on each course's group next to PPO; --distill-decay lowers W linearly to 0.
    skills: str = ""
    teachers: str = ""
    distill_coef: float = 0.0
    distill_decay: bool = False


def course_hash(c: C.Course) -> str:
    return hashlib.sha1(c.platforms.tobytes()).hexdigest()


def held_out_courses(levels, per_level: int, n_jumps: int, obstacle_prob: float = 0.0,
                     family: int = TEST_SEED_FAMILY, ice_prob: float = 0.0) -> list[C.Course]:
    out = []
    for L in range(len(levels)):
        rng = np.random.default_rng([2026, family, L])
        for i in range(per_level):
            out.append(C.generate(rng, levels, L, n_jumps, seed=i, obstacle_prob=obstacle_prob, ice_prob=ice_prob))
    return out


def obstacle_courses(levels, per_level: int, n_jumps: int) -> list[C.Course]:
    """Held-out courses where every jump has a 50% chance of an obstacle."""
    return held_out_courses(levels, per_level, n_jumps, obstacle_prob=0.5, family=OBSTACLE_TEST_FAMILY)


class CourseFactory:
    """Training courses, never equal to a held-out test course."""

    def __init__(self, cfg: TrainConfig, levels, exclude: set):
        self.cfg = cfg
        self.levels = levels
        self.exclude = exclude
        self.batch = 0
        # Skills mode: the training levels in turn, each (global id, its group's table, local level, ice chance).
        groups = [g for g in cfg.skills.split(",") if g]
        tables = skill_tables(groups)
        self.plan = [(gid, tables[g], L, cfg.ice_prob if g == "basic" else 0.0) for gid, g, L in skill_plan(groups)]

    def make(self, k: int) -> list[C.Course]:
        rng = np.random.default_rng([2026, 0, self.cfg.seed, self.batch])
        self.batch += 1
        out = []
        n_levels = len(self.levels)
        while len(out) < k:
            if self.plan:
                gid, table, L, ice = self.plan[len(out) % len(self.plan)]
            else:
                gid, table, L, ice = None, self.levels, len(out) % n_levels, self.cfg.ice_prob
            c = C.generate(rng, table, L, int(rng.integers(self.cfg.jumps_min, self.cfg.jumps_max + 1)),
                           turn_prob=self.cfg.turn_prob, obstacle_prob=self.cfg.obstacle_prob, wide_neo=self.cfg.wide_neo,
                           ice_prob=ice)
            if gid is not None:
                c.level = gid
            if c.n >= 3 and course_hash(c) not in self.exclude:
                out.append(c)
        return out


def load_demos(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())["demos"]


def demo_course(d: dict, level: int) -> C.Course:
    c = C.Course.from_json(d["course"])
    return C.Course(c.platforms, np.array(d["start"], dtype=np.float64), float(d["yaw"]), level=level,
                    specs=c.specs, extra=c.extra, name="demo")


def demo_dataset(demos: list[dict], extended: bool, device, chunk: int = 4096):
    """Observations along every demonstration (replayed in ParkourEnv) and the demonstrated actions."""
    obs_all, act_all = [], []
    for k in range(0, len(demos), chunk):
        part = demos[k:k + chunk]
        pool = CoursePool([demo_course(d, 0) for d in part], device, C.GRID)
        n = len(part)
        env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=10 ** 6, max_ticks=10 ** 6, extended_obs=extended),
                         sampler=None)
        obs = env.reset_all(torch.arange(n, device=device))
        T = max(len(d["actions"]) for d in part)
        acts = torch.zeros(n, T, 5, dtype=torch.int64, device=device)
        acts[..., 0], acts[..., 1], acts[..., 4] = 1, 1, 3
        lens = torch.as_tensor([len(d["actions"]) for d in part], device=device)
        for i, d in enumerate(part):
            acts[i, :len(d["actions"])] = torch.as_tensor(d["actions"], device=device)
        for t in range(T):
            m = lens > t
            obs_all.append(obs[m])
            act_all.append(acts[m, t])
            obs, *_ = env.step(acts[:, t])
    return torch.cat(obs_all), torch.cat(act_all)


def evaluate(model: ActorCritic, pool: CoursePool, device, stall_ticks: int, deterministic: bool = True,
             max_ticks: int = 4000, settle: int = 0, per_course: bool = False, fall_rule: str = "local") -> dict:
    """First episode on every course of ``pool``.

    ``settle`` ticks without input are simulated before the policy starts, as
    the live bot stands on the start block after its teleport.
    """
    n = pool.size
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=stall_ticks, max_ticks=max_ticks, fall_rule=fall_rule,
                                     extended_obs=model.obs_dim == OBS_DIM_EXT), sampler=None)
    obs = env.reset_all(torch.arange(n, device=device))
    if settle:
        idle = torch.zeros(n, 6, dtype=torch.bool, device=device)
        for _ in range(settle):
            env.state = physics.simulate(pool.world, env.course, env.state, idle)
        env.dist = env._distance(env.state.pos, env.course, env.cp)
        env.reset_memory(torch.ones(n, dtype=torch.bool, device=device))
        obs = env.observe()
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    success = torch.zeros(n, dtype=torch.bool, device=device)
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    length = torch.zeros(n, dtype=torch.int64, device=device)
    prog = torch.zeros(n, dtype=torch.float64, device=device)
    with torch.no_grad():
        for t in range(max_ticks):
            a, _, _ = model.act(obs, deterministic=deterministic)
            obs, _, _, _, _, info = env.step(a)
            first = info["done"] & ~finished
            success |= first & info["success"]
            fell |= first & info["fell"]
            length = torch.where(first, info["ep_len"], length)
            prog = torch.where(first, info["progress_frac"], prog)
            finished |= info["done"]
            if t % 25 == 0 and bool(finished.all()):
                break
    lv = pool.level.cpu().numpy()
    s, f, ln, pr = (t.cpu().numpy() for t in (success, fell, length, prog))
    per_level = {}
    for L in sorted(set(lv.tolist())):
        m = lv == L
        per_level[int(L)] = {"success": float(s[m].mean()), "fell": float(f[m].mean()),
                             "progress": float(pr[m].mean()), "ticks_success": float(ln[m & s].mean()) if (m & s).any() else None,
                             "n": int(m.sum())}
    out = {"success": float(s.mean()), "per_level": per_level}
    if per_course:
        out["per_course"] = {"success": s.tolist(), "fell": f.tolist(), "ticks": ln.tolist(), "progress": pr.tolist()}
    return out


def evaluate_course_batches(model, courses, device, stall_ticks, batch_size, compact_ice=False):
    """Same independent episodes, with only one batch's course grids on the GPU."""
    parts = []
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    for start in range(0, len(courses), batch_size):
        chunk = courses[start:start + batch_size]
        batch_pool = CoursePool(chunk, device, C.GRID, compact_ice=compact_ice)
        parts.append((len(chunk), evaluate(model, batch_pool, device, stall_ticks)))
        del batch_pool
        if torch.device(device).type == 'cuda':
            # Retaining many differently sized cached grids can fill VRAM even
            # though the live tensors fit. Release only unused allocator blocks.
            torch.cuda.empty_cache()
    per_level = {}
    levels = sorted({L for _, result in parts for L in result['per_level']})
    for L in levels:
        rows = [r['per_level'][L] for _, r in parts if L in r['per_level']]
        n = sum(r['n'] for r in rows)
        row = {k: sum(r[k] * r['n'] for r in rows) / n for k in ('success', 'fell', 'progress')}
        successes = sum(r['n'] * r['success'] for r in rows if r.get('ticks_success') is not None)
        row['ticks_success'] = (sum(r['ticks_success'] * r['n'] * r['success'] for r in rows
                                   if r.get('ticks_success') is not None) / successes) if successes else None
        row['n'] = n
        per_level[L] = row
    return {'success': sum(n * r['success'] for n, r in parts) / len(courses), 'per_level': per_level}


def _save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # OneDrive or an antivirus can hold the file briefly
            time.sleep(0.5)
    raise RuntimeError(f"could not replace {path}")


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device(cfg.device)
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
    if cfg.gpu_mem_fraction > 0 and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(cfg.gpu_mem_fraction,
                                                   dev.index if dev.index is not None else torch.cuda.current_device())
    out = RUNS / cfg.run
    skills = [g for g in cfg.skills.split(",") if g]
    if skills:
        if set(skills) - set(SKILL_GROUPS) or len(set(skills)) != len(skills):
            raise ValueError(f"--skills: groups {', '.join(SKILL_GROUPS)}, each once")
        if not cfg.test_once:
            raise ValueError("--skills selects on its own validation sets: use --test-once")
    teacher_paths = parse_teachers(cfg.teachers, skills) if cfg.teachers else {}
    if (cfg.teachers or cfg.distill_coef) and not (teacher_paths and cfg.distill_coef > 0):
        raise ValueError("distillation needs --skills, --teachers and --distill-coef > 0")
    resuming = bool(cfg.init_from) and Path(cfg.init_from).resolve().parent.parent == out.resolve()
    if (out / "train_log.csv").exists() and not resuming:
        raise ValueError(f"Existing run {cfg.run}: resume its own latest.pt or choose a new run name")
    old_summary = json.loads((out / "summary.json").read_text()) if resuming and (out / "summary.json").exists() else {}
    if cfg.test_once and old_summary.get("test_eval"):
        old_cfg = json.loads((out / "config.json").read_text())
        if cfg.iterations > old_cfg["iterations"]:
            raise ValueError("This run has already been tested; extend in a new run to keep one test per run")
        return out
    out.mkdir(parents=True, exist_ok=False if not cfg.init_from else True)
    (out / "checkpoints").mkdir(exist_ok=True)
    cfg_dict = dataclasses.asdict(cfg)
    (out / "config.json").write_text(json.dumps(cfg_dict, indent=1))

    if skills:
        tables = skill_tables(skills)
        levels = [lv for g in skills for lv in tables[g]]  # every table a training course may come from
        n_levels = max(gid for gid, _, _ in skill_plan(skills)) + 1  # training levels (curriculum, train_log)
        n_eval = ICE_GROUP[1] + 1  # levels of the held-out sets (the ice group's ids follow the others)
        held = skill_sets("test", cfg.eval_per_level, cfg.eval_jumps, skills)
        tests = [c for g in held.values() for c in g["standard"]]
        obstacle_tests = [c for g in held.values() for c in g["obstacles"]]
        vals = [c for g in skill_sets("validation", cfg.val_per_level, cfg.eval_jumps, skills).values()
                for c in g["standard"] + g["obstacles"]]
        test_hashes = {course_hash(c) for c in tests + obstacle_tests + vals}
    else:
        levels = C.load_levels(advanced=cfg.advanced, extra=cfg.extra)
        n_levels = n_eval = len(levels)
        tests = held_out_courses(levels, cfg.eval_per_level, cfg.eval_jumps)
        test_hashes = {course_hash(c) for c in tests}
        obstacle_tests = obstacle_courses(levels, cfg.eval_per_level, cfg.eval_jumps) if cfg.obstacle_prob > 0 else []
        test_hashes |= {course_hash(c) for c in obstacle_tests}
        vals = held_out_courses(levels, cfg.val_per_level, cfg.eval_jumps, family=VALIDATION_FAMILY)
        if cfg.obstacle_prob > 0:
            vals += held_out_courses(levels, cfg.val_per_level, cfg.eval_jumps, obstacle_prob=0.5,
                                     family=OBSTACLE_VALIDATION_FAMILY)
        test_hashes |= {course_hash(c) for c in vals}
    factory = CourseFactory(cfg, levels, test_hashes)
    train_courses = factory.make(cfg.pool_size)
    demos = load_demos(cfg.demos) if cfg.demos else []
    n_demo = min(cfg.demo_courses, len(demos)) if demos else 0
    if n_demo:
        pick = np.random.default_rng([2026, 3, cfg.seed]).choice(len(demos), n_demo, replace=False)
        train_courses[:n_demo] = [demo_course(demos[i], cfg.demo_level) for i in pick]
    pool = CoursePool(train_courses, dev, C.GRID, ice=cfg.ice_prob > 0 or any(sp.ice or sp.bounce for L in levels for sp in L), compact_ice=cfg.compact_ice,
                      shapes=any(bool(sp.block) or sp.ladder for L in levels for sp in L),
                      slime=any(sp.bounce for L in levels for sp in L))
    test_pool = None if cfg.test_once else CoursePool(tests, dev, C.GRID)
    obstacle_pool = CoursePool(obstacle_tests, dev, C.GRID) if obstacle_tests and not cfg.test_once else None
    val_pool = None if cfg.eval_batch_size else CoursePool(vals, dev, C.GRID, compact_ice=cfg.compact_ice)
    best_info = old_summary.get("best", {})

    if cfg.curriculum == "adaptive":
        sampler = LevelSampler(n_levels)
    elif cfg.curriculum == "uniform":
        sampler = LevelSampler(n_levels, fixed=[1.0] * n_levels)
    else:
        raise ValueError(cfg.curriculum)
    env = ParkourEnv(pool, EnvConfig(num_envs=cfg.num_envs, stall_ticks=cfg.stall_ticks, reward=cfg.reward,
                                     extended_obs=cfg.extended_obs),
                     sampler=sampler, seed=cfg.seed)

    model = ActorCritic(obs_dim=OBS_DIM_EXT if cfg.extended_obs else OBS_DIM, hidden=cfg.hidden,
                        hidden_parts=tuple(p for p in cfg.hide.split(",") if p)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    lr = cfg.lr
    start_it = 0
    if cfg.init_from:
        ck = torch.load(cfg.init_from, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck and resuming:
            opt.load_state_dict(ck["optimizer"])
            lr = opt.param_groups[0]["lr"]
            start_it = ck["iteration"] + 1
            if "sampler" in ck:
                sampler.load_state_dict(ck["sampler"])
                env.set_level_probs(sampler.probs())
    shadow = RunningNorm(model.obs_dim).to(dev)
    shadow.load_state_dict(model.norm.state_dict())
    demo_obs = demo_act = None
    if cfg.bc_coef > 0 and (demos or cfg.bc_pairs):
        parts_o, parts_a = [], []
        if demos:
            o, a_ = demo_dataset(demos, cfg.extended_obs, dev)
            parts_o.append(o)
            parts_a.append(a_)
        for f in (x for x in cfg.bc_pairs.split(",") if x):
            d = torch.load(f, map_location=dev, weights_only=False)
            parts_o.append(d["obs"].to(dev))
            parts_a.append(d["act"].to(dev))
        demo_obs, demo_act = torch.cat(parts_o), torch.cat(parts_a)
        print(json.dumps({"demonstrations": len(demos), "bc_pairs": int(demo_obs.shape[0]),
                          "pair_files": cfg.bc_pairs, "demo_courses": n_demo}), flush=True)

    snapshots = {int(v) for v in cfg.save_at.split(",") if v.strip()}

    def snapshot(updates: int) -> None:
        """Weights only (no optimizer), to replay the policy as it was after ``updates`` iterations."""
        _save({"model": model.state_dict(), "config": cfg_dict, "iteration": updates - 1, "updates": updates,
               "env_steps": updates * cfg.rollout * cfg.num_envs}, out / "checkpoints" / f"updates_{updates:05d}.pt")

    if 0 in snapshots and start_it == 0:
        snapshot(0)
    T, N = cfg.rollout, cfg.num_envs
    obs_buf = torch.zeros(T, N, model.obs_dim, device=dev)
    act_buf = torch.zeros(T, N, 5, dtype=torch.int64, device=dev)
    logp_buf = torch.zeros(T, N, device=dev)
    val_buf = torch.zeros(T, N, device=dev)
    rew_buf = torch.zeros(T, N, device=dev)
    done_buf = torch.zeros(T, N, device=dev)
    teachers = []
    if teacher_paths:
        paths = sorted(set(teacher_paths.values()))  # one frozen network per checkpoint
        teachers = [load_frozen(p, dev) for p in paths]
        if any(t.obs_dim != model.obs_dim for t in teachers):
            raise ValueError("every teacher must use the student's observation")
        t_table = teacher_table(teacher_paths, skills, paths).to(dev)
        tlogp_buf = torch.zeros(T, N, sum(ACTION_DIMS), device=dev)
        tmask_buf = torch.zeros(T, N, dtype=torch.bool, device=dev)
        rows = torch.arange(N, device=dev)
        print(json.dumps({"teachers": teacher_paths}), flush=True)

    log_path = out / "train_log.csv"
    eval_path = out / "eval_log.csv"
    log_f = open(log_path, "a", newline="")
    eval_f = open(eval_path, "a", newline="")
    log_w = None
    eval_w = csv.writer(eval_f)
    if eval_path.stat().st_size == 0:
        eval_w.writerow(["iteration", "env_steps", "success"] + [f"success_L{L}" for L in range(n_eval)]
                        + [f"progress_L{L}" for L in range(n_eval)] + [f"fell_L{L}" for L in range(n_eval)])
    best = best_info.get("validation_success", -1.0)
    obs = env.observe()
    if dev.type == 'cuda' and cfg.gpu_mem_fraction:
        print(json.dumps({'gpu_memory_start': {
            'allocator_fraction': torch.cuda.get_per_process_memory_fraction(dev.index if dev.index is not None else torch.cuda.current_device()),
            'allocated_mib': torch.cuda.memory_allocated(dev) / 2**20,
            'reserved_mib': torch.cuda.memory_reserved(dev) / 2**20}}), flush=True)
    total_steps = start_it * T * N
    t_start = time.time()
    busy = []  # wall time of each iteration, for --gpu-duty
    for it in range(start_it, cfg.iterations):
        t0 = time.time()
        model.eval()
        with torch.no_grad():
            for t in range(T):
                obs_n = model.norm(obs)
                shadow.update(obs)
                logits = model.logits(obs_n)
                acts, logp = [], 0.0
                for lg in logits:
                    dist = torch.distributions.Categorical(logits=lg)
                    a = dist.sample()
                    acts.append(a)
                    logp = logp + dist.log_prob(a)
                action = torch.stack(acts, -1)
                value = model.value(obs_n)
                if teachers:  # every teacher on every row (no host sync), each row keeps its group's
                    tid = teacher_rows(env, t_table)
                    tl = torch.stack([teacher_logp(tm, obs) for tm in teachers])
                    tlogp_buf[t] = tl[tid.clamp(min=0), rows]
                    tmask_buf[t] = tid >= 0
                nobs, rew, term, trunc, final_obs, info = env.step(action)
                boot = model.value(model.norm(final_obs))
                rew = rew + cfg.gamma * torch.where(trunc, boot, torch.zeros_like(boot))
                obs_buf[t] = obs_n
                act_buf[t] = action
                logp_buf[t] = logp
                val_buf[t] = value
                rew_buf[t] = rew
                done_buf[t] = (term | trunc).float()
                obs = nobs
            last_val = model.value(model.norm(obs))
        model.norm.load_state_dict(shadow.state_dict())
        st = env.pop_stats()
        sampler.update(st[0], st[1])
        env.set_level_probs(sampler.probs())

        # GAE.
        adv = torch.zeros(T, N, device=dev)
        gae = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nv = last_val if t == T - 1 else val_buf[t + 1]
            mask = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * nv * mask - val_buf[t]
            gae = delta + cfg.gamma * cfg.lam * mask * gae
            adv[t] = gae
        ret = adv + val_buf

        # PPO update.
        model.train()
        b_obs = obs_buf.view(T * N, -1)
        b_act = act_buf.view(T * N, -1)
        b_logp = logp_buf.view(-1)
        b_adv = adv.view(-1)
        b_ret = ret.view(-1)
        b_val = val_buf.view(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        mb = T * N // cfg.minibatches
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0, "bc_loss": 0.0}
        bc_w = cfg.bc_coef * ((1 - (it - start_it) / max(1, cfg.iterations - start_it)) if cfg.bc_decay else 1.0)
        if teachers:
            b_tlogp = tlogp_buf.view(T * N, -1)
            b_tmask = tmask_buf.view(-1)
            # The schedule counts from iteration 0, so a resumed run continues it.
            dist_w = cfg.distill_coef * ((1 - it / cfg.iterations) if cfg.distill_decay else 1.0)
            stats["distill_kl"] = 0.0
        nupd, stopped = 0, False
        for _ in range(cfg.epochs):
            perm = torch.randperm(T * N, device=dev)
            for k in range(cfg.minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                logp, ent, v = model.evaluate(b_obs[idx], b_act[idx])
                log_ratio = (logp - b_logp[idx]).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)
                kl = ((ratio - 1) - log_ratio).mean().item()  # policy change so far, on this minibatch
                if kl > cfg.kl_stop or not math.isfinite(kl):
                    stopped = True
                    break
                a = b_adv[idx]
                surr = torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)
                surr = torch.where(a < 0, torch.maximum(surr, cfg.dual_clip * a), surr)
                pl = -surr.mean()
                vc = b_val[idx] + (v - b_val[idx]).clamp(-cfg.clip, cfg.clip)
                vl = 0.5 * torch.max((v - b_ret[idx]).pow(2), (vc - b_ret[idx]).pow(2)).mean()
                el = ent.mean()
                loss = pl + cfg.vf_coef * vl - cfg.ent_coef * el
                if demo_obs is not None and bc_w > 0:
                    j = torch.randint(demo_obs.shape[0], (cfg.bc_batch,), device=dev)
                    bc_logp, _, _ = model.evaluate(model.norm(demo_obs[j]), demo_act[j])
                    bc = -bc_logp.mean()
                    loss = loss + bc_w * bc
                    stats["bc_loss"] += bc.item()
                if teachers:
                    m = b_tmask[idx]
                    dk = (distill_kl(model.logits(b_obs[idx]), b_tlogp[idx]) * m).sum() / m.sum().clamp(min=1)
                    loss = loss + dist_w * dk
                    stats["distill_kl"] += dk.item()
                if not torch.isfinite(loss):
                    stopped = True
                    break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                # KL-adaptive learning rate (as in rsl_rl / legged_gym).
                if kl > 2.0 * cfg.target_kl:
                    lr = max(cfg.lr_min, lr / 1.5)
                elif kl < 0.5 * cfg.target_kl:
                    lr = min(cfg.lr_max, lr * 1.5)
                for g in opt.param_groups:
                    g["lr"] = lr
                with torch.no_grad():
                    stats["pi_loss"] += pl.item()
                    stats["v_loss"] += vl.item()
                    stats["entropy"] += el.item()
                    stats["kl"] += kl
                    stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                nupd += 1
            if stopped:
                break
        for k in stats:
            stats[k] /= max(nupd, 1)
        stats["lr"] = lr
        stats["updates"] = nupd
        stats["stopped"] = int(stopped)
        if teachers:
            stats["distill_w"] = dist_w

        total_steps += T * N
        if it + 1 in snapshots:
            snapshot(it + 1)
        dt = time.time() - t0
        eps = st[0].sum()
        frac = (lambda i: round(float(st[i].sum() / eps), 4) if eps else None)
        row = {
            "iteration": it, "env_steps": total_steps, "minutes": round((time.time() - t_start) / 60, 2),
            "fps": int(T * N / dt), "episodes": int(eps),
            "success": frac(1), "fell": frac(2), "truncated": frac(3), "ep_len": frac(4), "ep_return": frac(5),
            "progress": frac(6),
            "reward_per_tick": round(rew_buf.mean().item(), 4), "frontier": sampler.frontier,
            **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in stats.items()},
        }
        rates = sampler.success_rates()
        for L in range(n_levels):
            row[f"train_success_L{L}"] = None if rates[L] is None else round(rates[L], 3)
        if log_w is None:
            log_w = csv.DictWriter(log_f, fieldnames=list(row.keys()))
            if log_path.stat().st_size == 0:
                log_w.writeheader()
        log_w.writerow(row)
        log_f.flush()
        if it % 10 == 0:
            print(json.dumps(row), flush=True)

        # Refresh part of the training pool with new courses.
        if cfg.pool_refresh_every and it % cfg.pool_refresh_every == cfg.pool_refresh_every - 1:
            k = int(pool.size * cfg.pool_refresh_frac)
            idx = n_demo + np.random.default_rng([2026, 2, cfg.seed, it]).choice(pool.size - n_demo, k, replace=False)
            for i, c in zip(idx, factory.make(k)):
                pool.replace(int(i), c)
            env.rebuild_level_table()
            env.set_level_probs(sampler.probs())
            # Players on a replaced course restart. The rollout that just ended
            # bootstrapped from their last observation, as for a time limit.
            moved = torch.isin(env.course, torch.as_tensor(idx, device=dev))
            env.reset_rows(moved)
            obs = env.observe()

        if it % cfg.eval_every == cfg.eval_every - 1 or it == cfg.iterations - 1:
            model.eval()
            ev = {}
            if not cfg.test_once:
                ev = evaluate(model, test_pool, dev, cfg.stall_ticks)
                pl = ev["per_level"]
                eval_w.writerow([it, total_steps, round(ev["success"], 4)]
                                + [round(pl[L]["success"], 4) for L in range(n_levels)]
                                + [round(pl[L]["progress"], 4) for L in range(n_levels)]
                                + [round(pl[L]["fell"], 4) for L in range(n_levels)])
                eval_f.flush()
                print(json.dumps({"eval_iteration": it, **ev}), flush=True)
            # Model selection uses the validation courses only.
            evv = (evaluate_course_batches(model, vals, dev, cfg.stall_ticks, cfg.eval_batch_size, cfg.compact_ice)
                   if cfg.eval_batch_size else evaluate(model, val_pool, dev, cfg.stall_ticks))
            score = evv["success"]
            vgroups = {}
            if skills:  # the selection score: the mean over the groups
                vgroups = {g: v for g, v in sorted(group_scores(evv["per_level"]).items(), key=lambda x: GROUP_ORDER.index(x[0]))}
                score = macro_score(vgroups)
            vpath = out / "eval_val_log.csv"
            new_file = not vpath.exists()
            with open(vpath, "a", newline="") as fv:
                wv = csv.writer(fv)
                if new_file:
                    wv.writerow(["iteration", "env_steps", "success"] + [f"success_{g}" for g in vgroups])
                wv.writerow([it, total_steps, round(score, 4)] + [round(v, 4) for v in vgroups.values()])
            ev["validation_success"] = score
            ev["validation_per_level"] = evv["per_level"]
            if vgroups:
                ev["validation_groups"] = vgroups
            print(json.dumps({"validation_iteration": it, "success": score}), flush=True)
            if obstacle_pool is not None and not cfg.test_once:
                evo = evaluate(model, obstacle_pool, dev, cfg.stall_ticks)
                po = evo["per_level"]
                opath = out / "eval_obstacles_log.csv"
                new_file = not opath.exists()
                with open(opath, "a", newline="") as fo:
                    wo = csv.writer(fo)
                    if new_file:
                        wo.writerow(["iteration", "env_steps", "success"] + [f"success_L{L}" for L in range(n_levels)])
                    wo.writerow([it, total_steps, round(evo["success"], 4)] + [round(po[L]["success"], 4) for L in range(n_levels)])
                print(json.dumps({"eval_obstacles_iteration": it, "success": evo["success"]}), flush=True)
                ev["obstacles"] = evo
            ck = {"model": model.state_dict(), "optimizer": opt.state_dict(), "iteration": it,
                  "sampler": sampler.state_dict(), "config": cfg_dict, "eval": ev}
            _save(ck, out / "checkpoints" / "latest.pt")
            if score > best:
                best = score
                best_info = {"iteration": it, "validation_success": score, "test_success": ev.get("success"),
                             "validation_per_level": evv["per_level"]}
                if vgroups:
                    best_info["validation_groups"] = vgroups
                _save(ck, out / "checkpoints" / "best.pt")
                model.export(out / "policy_best.json", {"run": cfg.run, "iteration": it, "eval": ev})
            (out / "summary.json").write_text(json.dumps({"best": best_info, "last_eval": ev,
                                                          "iteration": it, "env_steps": total_steps}, indent=1))
        if (out / "pause_requested").exists() and (it + 1) % cfg.eval_every == 0 and it + 1 < cfg.iterations:
            log_f.close()
            eval_f.close()
            (out / "pause_ready.json").write_text(json.dumps({"iteration": it, "checkpoint": "checkpoints/latest.pt"}))
            print(json.dumps({"paused_at_checkpoint": it}), flush=True)
            return out
        if cfg.gpu_duty < 1:  # leave the GPU idle for the rest of the time (another program, e.g. a game)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            # An iteration the watchdog kept suspended (a game, heat) is not busy time: cap at 3x the recent median.
            busy.append(time.time() - t0)
            work = min(busy[-1], 3 * float(np.median(busy[-20:])))
            time.sleep(work * (1 - cfg.gpu_duty) / max(cfg.gpu_duty, 0.05))
    log_f.close()
    eval_f.close()
    model.export(out / "policy_last.json", {"run": cfg.run, "iteration": cfg.iterations - 1})
    if cfg.test_once:
        selected = torch.load(out / "checkpoints" / "best.pt", map_location=dev, weights_only=False)
        model.load_state_dict(selected["model"])
        model.eval()
        tested = evaluate_course_batches(model, tests, dev, cfg.stall_ticks, cfg.eval_batch_size or len(tests), cfg.compact_ice)
        best_it = selected["iteration"]
        best_steps = (best_it + 1) * T * N
        pl = tested["per_level"]
        with open(eval_path, "a", newline="") as f:
            csv.writer(f).writerow([best_it, best_steps, round(tested["success"], 4)]
                + level_columns(pl, "success", n_eval) + level_columns(pl, "progress", n_eval)
                + level_columns(pl, "fell", n_eval))
        if obstacle_tests:
            tested["obstacles"] = evaluate_course_batches(model, obstacle_tests, dev, cfg.stall_ticks,
                                                          cfg.eval_batch_size or len(obstacle_tests), cfg.compact_ice)
            po = tested["obstacles"]["per_level"]
            with open(out / "eval_obstacles_log.csv", "a", newline="") as f:
                w = csv.writer(f)
                if f.tell() == 0:
                    w.writerow(["iteration", "env_steps", "success"] + [f"success_L{L}" for L in range(n_eval)])
                w.writerow([best_it, best_steps, round(tested["obstacles"]["success"], 4)]
                           + level_columns(po, "success", n_eval))
        if skills:  # per group: standard courses, obstacle courses, and both (the validation score's mix)
            both = group_scores(pl, tested["obstacles"]["per_level"])
            tested["groups"] = {g: {"standard": group_scores(pl).get(g),
                                    "obstacles": group_scores(tested["obstacles"]["per_level"]).get(g), "all": both[g]}
                                for g in GROUP_ORDER if g in both}
            tested["macro"] = macro_score(both)
        summary = json.loads((out / "summary.json").read_text())
        summary["best"]["test_success"] = tested["success"]
        summary["test_eval"] = {"iteration": best_it, **tested}
        summary["completed"] = True
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps({"selected_checkpoint_test": best_it, **tested}), flush=True)
    return out


def parse_args(argv=None) -> TrainConfig:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = TrainConfig()
    for f in dataclasses.fields(TrainConfig):
        if f.name in ("reward", "hidden"):
            continue
        if isinstance(getattr(base, f.name), bool):
            ap.add_argument("--" + f.name.replace("_", "-"), action="store_true")
            continue
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(getattr(base, f.name)), default=getattr(base, f.name))
    for f in dataclasses.fields(RewardConfig):
        ap.add_argument("--reward-" + f.name, type=float, default=getattr(RewardConfig(), f.name))
    ap.add_argument("--hidden", type=str, default=",".join(map(str, base.hidden)))
    a = ap.parse_args(argv)
    kw = {f.name: getattr(a, f.name) for f in dataclasses.fields(TrainConfig) if f.name not in ("reward", "hidden")}
    kw["reward"] = RewardConfig(**{f.name: getattr(a, "reward_" + f.name) for f in dataclasses.fields(RewardConfig)})
    kw["hidden"] = tuple(int(x) for x in a.hidden.split(","))
    return TrainConfig(**kw)


if __name__ == "__main__":
    train(parse_args())

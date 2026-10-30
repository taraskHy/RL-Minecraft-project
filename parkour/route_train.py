"""Train the route policy with PPO (semi-MDP: one transition per route decision).

    python -m parkour.route_train --run route-a --iterations 600

The jumper is the final policy, frozen and deterministic (as on the bot). Each tick the jumper acts in
ParkourEnv with its platform slots filled by parkour.route_env; at every route decision the route
policy samples the next platform. A transition runs from one decision of a row to its next one (or
the end of the episode); its reward is the environment's own reward summed over those ticks with
discount ``gamma`` per tick, minus ``trunc_penalty`` when the episode ends by the stall or time limit
(the evaluation counts those as failures). Advantages use GAE over decisions with discount
``gamma ** ticks``.

Optional imitation (``--bc-coef``): a cross-entropy loss towards the course's next platform (the
selectable token whose course index is the smallest above the player's progress). The course is
used for the reward and this training signal only; it is never an input. ``--bc-decay`` lowers the
weight linearly to ``--bc-floor`` (default 0) by ``--bc-until``. With ``--fill policy`` the slots after
the chosen platform are the route policy's own argmax from it; ``--fill-bc`` trains those steps by
imitation of the course's next platform after the one they extend (their actions are not PPO actions).

Checkpoints are chosen on the validation families (2 standard, 5 obstacles), never on test.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from . import courses as C
from .env import CoursePool, EnvConfig, ParkourEnv
from .live import load_model
from .planner import PlannerConfig
from .route_env import RouteConfig, RouteRunner, RouteTables, evaluate_routed, next_on_course
from .route_policy import RoutePolicy
from .train import (OBSTACLE_TEST_FAMILY, OBSTACLE_VALIDATION_FAMILY, TEST_SEED_FAMILY, VALIDATION_FAMILY,
                    CourseFactory, TrainConfig, _save, course_hash, held_out_courses)


@dataclass
class RouteTrainConfig:
    run: str = "route-dev"
    seed: int = 0
    device: str = "cuda"
    jumper: str = "results/final_policy.pt"
    iterations: int = 600
    num_envs: int = 2048
    rollout_ticks: int = 128
    pool_size: int = 2048
    pool_refresh_every: int = 25
    pool_refresh_frac: float = 0.0625
    obstacle_prob: float = 0.3
    jumps_min: int = 5
    jumps_max: int = 12
    turn_prob: float = 0.25
    gamma: float = 0.997  # per tick
    lam: float = 0.95  # per decision
    lr: float = 3e-4
    epochs: int = 4
    minibatches: int = 4
    clip: float = 0.2
    ent_coef: float = 0.01
    pg_coef: float = 1.0  # 0: no PPO policy loss (an imitation-only control)
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float = 0.02  # stop an update's remaining minibatches above 1.5x this
    trunc_penalty: float = 5.0
    keep_time_cost: bool = False  # by default the router's reward adds back the environment's time cost
    bc_coef: float = 0.0
    bc_decay: bool = False
    bc_until: int = 200
    bc_floor: float = 0.0  # with --bc-decay the weight never falls below this
    fill: str = "greedy"  # slots 2 and 3: greedy | policy
    fill_bc: float = 0.0  # with --fill policy: imitation weight on the fill steps (course's next platform)
    fill_cap: int = 32768  # fill examples kept per iteration (random subset)
    d: int = 128
    layers: int = 2
    heads: int = 4
    init_from: str = ""
    eval_every: int = 20
    eval_max_ticks: int = 4000  # as parkour.planner's evaluation
    val_per_level: int = 100
    duty: float = 0.75  # GPU pacing: sleep so that work takes at most this share of wall time
    stall_ticks: int = 200
    max_ticks: int = 3000


def held_out_hashes() -> set:
    levels = C.load_levels()
    out = set()
    for fam in (TEST_SEED_FAMILY, VALIDATION_FAMILY):
        out |= {course_hash(c) for c in held_out_courses(levels, 300, 10, family=fam)}
    for fam in (OBSTACLE_TEST_FAMILY, OBSTACLE_VALIDATION_FAMILY):
        out |= {course_hash(c) for c in held_out_courses(levels, 300, 10, obstacle_prob=0.5, family=fam)}
    return out


def validation_pools(per_level: int, device):
    levels = C.load_levels()
    std = held_out_courses(levels, per_level, 10, family=VALIDATION_FAMILY)
    obs = held_out_courses(levels, per_level, 10, obstacle_prob=0.5, family=OBSTACLE_VALIDATION_FAMILY)
    out = {}
    for name, cs in (("standard", std), ("obstacles", obs)):
        pool = CoursePool(cs, device, C.GRID)
        out[name] = (pool, RouteTables(pool))
    return out


def router_choosers(net: RoutePolicy, deterministic: bool):
    def choose(feats, ok, inview):
        a, logp, v = net.act(feats, ok, inview, deterministic=deterministic)
        return a, (logp, v)

    def fill(feats, ok, inview):
        return net.logits(feats, ok, inview).argmax(-1)
    return choose, fill


def evaluate_router(net, jumper, vals: dict, rcfg: RouteConfig, max_ticks: int = 4000) -> dict:
    net.eval()
    choose, fill = router_choosers(net, True)
    out = {}
    with torch.no_grad():
        for name, (pool, tables) in vals.items():
            r = evaluate_routed(jumper, pool, tables, choose, fill, rcfg=rcfg, stall_ticks=200, max_ticks=max_ticks)
            out[name] = {"success": r["success"], "per_level": {k: v["success"] for k, v in r["per_level"].items()}}
    return out


class Transitions:
    """Open transitions per row and the closed ones of this iteration."""

    def __init__(self, n, kt, dim, device):
        self.n, self.dev = n, device
        self.feats = torch.zeros(n, kt, dim, device=device)
        self.ok = torch.zeros(n, kt, dtype=torch.bool, device=device)
        self.inview = torch.zeros(n, kt, dtype=torch.bool, device=device)
        self.act = torch.zeros(n, dtype=torch.int64, device=device)
        self.logp = torch.zeros(n, device=device)
        self.val = torch.zeros(n, device=device)
        self.label = torch.full((n,), -1, dtype=torch.int64, device=device)
        self.acc_r = torch.zeros(n, dtype=torch.float64, device=device)
        self.acc_d = torch.ones(n, dtype=torch.float64, device=device)
        self.open = torch.zeros(n, dtype=torch.bool, device=device)
        self.last = torch.full((n,), -1, dtype=torch.int64)  # id of the row's last closed transition (CPU)
        self.closed = []  # chunks of closed transitions
        self.count = 0
        self.succ = []  # (earlier id, later id) links

    def accumulate(self, reward: torch.Tensor, gamma: float):
        o = self.open
        self.acc_r = torch.where(o, self.acc_r + self.acc_d * reward.to(torch.float64), self.acc_r)
        self.acc_d = torch.where(o, self.acc_d * gamma, self.acc_d)

    def close(self, rows: torch.Tensor, done: bool, next_val: torch.Tensor | None):
        """Close the open transitions of ``rows`` (those that have one)."""
        rows = rows[self.open[rows]]
        if rows.numel() == 0:
            return
        k = rows.numel()
        ids = torch.arange(self.count, self.count + k)
        self.count += k
        nv = torch.zeros(k, device=self.dev) if next_val is None else next_val
        self.closed.append({
            "feats": self.feats[rows].clone(), "ok": self.ok[rows].clone(), "inview": self.inview[rows].clone(),
            "act": self.act[rows].clone(), "logp": self.logp[rows].clone(), "val": self.val[rows].clone(),
            "label": self.label[rows].clone(), "r": self.acc_r[rows].float(), "disc": self.acc_d[rows].float(),
            "done": torch.full((k,), done, dtype=torch.bool, device=self.dev), "next_val": nv.float()})
        rc = rows.cpu()
        prev = self.last[rc]
        for p, i in zip(prev.tolist(), ids.tolist()):
            if p >= 0:
                self.succ.append((p, i))
        self.last[rc] = ids if not done else torch.full((k,), -1, dtype=torch.int64)
        self.open[rows] = False

    def open_new(self, rows, feats, ok, inview, act, logp, val, label):
        self.feats[rows], self.ok[rows], self.inview[rows] = feats, ok, inview
        self.act[rows], self.logp[rows], self.val[rows], self.label[rows] = act, logp, val, label
        self.acc_r[rows] = 0
        self.acc_d[rows] = 1
        self.open[rows] = True

    def drop(self, rows: torch.Tensor):
        self.open[rows] = False
        self.last[rows.cpu()] = -1

    def batch(self, lam: float):
        if not self.closed:
            return None
        b = {k: torch.cat([c[k] for c in self.closed]) for k in self.closed[0]}
        n = b["r"].shape[0]
        succ = np.full(n, -1, dtype=np.int64)
        for p, i in self.succ:
            if p < n and i < n:
                succ[p] = i
        r, disc, done = b["r"].cpu().numpy(), b["disc"].cpu().numpy(), b["done"].cpu().numpy()
        v, nv = b["val"].cpu().numpy(), b["next_val"].cpu().numpy()
        delta = r + disc * nv * (~done) - v
        adv = np.zeros(n, dtype=np.float64)
        for i in range(n - 1, -1, -1):  # successors close later, so they have larger ids
            a = delta[i]
            s = succ[i]
            if s >= 0 and not done[i]:
                a += disc[i] * lam * adv[s]
            adv[i] = a
        b["adv"] = torch.as_tensor(adv, dtype=torch.float32, device=b["r"].device)
        b["ret"] = b["adv"] + b["val"]
        # Ids restart every iteration: links to this iteration's open transitions are kept by ``last``.
        self.closed, self.succ = [], []
        self.last = torch.where(self.last >= 0, self.last - self.count, self.last)
        self.count = 0
        return b


def oracle_label(cidx: torch.Tensor, ok: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
    """Token of the course's next platform: the selectable one with the smallest course index above cp."""
    return next_on_course(cidx, ok, cp)


def train(cfg: RouteTrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.device
    out = Path("runs") / cfg.run
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    jumper = load_model(Path(cfg.jumper), dev)
    for p in jumper.parameters():
        p.requires_grad_(False)
    levels = C.load_levels()
    tcfg = TrainConfig(seed=cfg.seed, obstacle_prob=cfg.obstacle_prob, jumps_min=cfg.jumps_min,
                       jumps_max=cfg.jumps_max, turn_prob=cfg.turn_prob)
    factory = CourseFactory(tcfg, levels, held_out_hashes())
    pool = CoursePool(factory.make(cfg.pool_size), dev, C.GRID)
    tables = RouteTables(pool)
    env = ParkourEnv(pool, EnvConfig(num_envs=cfg.num_envs, stall_ticks=cfg.stall_ticks, max_ticks=cfg.max_ticks),
                     sampler=None, seed=cfg.seed)
    pcfg, rcfg = PlannerConfig(), RouteConfig(fill=cfg.fill)
    run = RouteRunner(env, tables, pcfg, rcfg)
    if cfg.fill == "policy" and cfg.fill_bc > 0:
        run.fill_log = []
    net = RoutePolicy(cfg.d, cfg.layers, cfg.heads).to(dev)
    if cfg.init_from:
        ck = torch.load(cfg.init_from, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    vals = validation_pools(cfg.val_per_level, dev)
    from .route_env import TOKEN_DIM
    tr = Transitions(cfg.num_envs, tables.kt, TOKEN_DIM, dev)
    choose, fill = router_choosers(net, False)
    fill_choose = fill if cfg.fill == "policy" else None

    def decide(rows, cell):
        res = run.decide(rows, cell, choose, fill_choose)
        if res is None:
            return
        live = res["any_ok"]
        r = res["rows"][live]
        logp, val = res["out"][0][live], res["out"][1][live]
        tr.close(r, False, val[tr.open[r]])
        label = oracle_label(res["cidx"][live], res["ok"][live], env.cp[r])
        tr.open_new(r, res["feats"][live], res["ok"][live], res["inview"][live], res["action"][live], logp, val, label)

    n = cfg.num_envs
    allrows = torch.arange(n, device=dev)
    with torch.no_grad():
        decide(allrows, run.start_cells())
    train_log = open(out / "train_log.csv", "a", newline="")
    tw = csv.writer(train_log)
    tw.writerow(["iteration", "ticks", "minutes", "decisions", "episodes", "success", "fell", "truncated",
                 "pi_loss", "v_loss", "entropy", "kl", "bc_loss", "bc_acc", "bc_coef", "lr", "updates", "work_s",
                 "fill_loss", "fill_acc", "fill_n"])
    eval_log = open(out / "eval_log.csv", "a", newline="")
    ew = csv.writer(eval_log)
    ew.writerow(["iteration", "standard", "obstacles", "mean"] + [f"std_L{L}" for L in range(6)] + [f"obs_L{L}" for L in range(6)])
    best, t0, ticks = -1.0, time.time(), 0
    for it in range(cfg.iterations):
        tw0 = time.time()
        net.eval()
        ep = np.zeros(4)
        with torch.no_grad():
            for _ in range(cfg.rollout_ticks):
                obs = run.observe()
                a, _, _ = jumper.act(obs, deterministic=True)
                _, rew, _, _, _, info = env.step(a)
                ticks += n
                done = info["done"]
                trunc = info["truncated"] & done
                r = rew.to(torch.float64) - cfg.trunc_penalty * trunc.to(torch.float64)
                if not cfg.keep_time_cost:
                    r = r + env.cfg.reward.time
                tr.accumulate(r, cfg.gamma)
                drows = done.nonzero().squeeze(1)
                if drows.numel():
                    tr.close(drows, True, None)
                    ep += [drows.numel(), float((info["success"] & done).sum()), float((info["fell"] & done).sum()),
                           float(trunc.sum())]
                    run.reset(done)
                trig, cell = run.triggers(~done)
                cell = torch.where(done[:, None], run.start_cells(), cell)
                decide((trig | done).nonzero().squeeze(1), cell)
        # Pool refresh: new courses; rows on replaced courses restart (their open transitions are dropped).
        if cfg.pool_refresh_every and (it + 1) % cfg.pool_refresh_every == 0:
            k = int(cfg.pool_size * cfg.pool_refresh_frac)
            idx = np.random.choice(cfg.pool_size, k, replace=False)
            for i, c in zip(idx.tolist(), factory.make(k)):
                pool.replace(i, c)
            tables.replace_many(pool, idx.tolist())
            hit = torch.isin(env.course, torch.as_tensor(idx, device=dev))
            if bool(hit.any()):
                tr.drop(hit.nonzero().squeeze(1))
                env.reset_rows(hit)
                run.reset(hit)
                with torch.no_grad():
                    decide(hit.nonzero().squeeze(1), run.start_cells())
        b = tr.batch(cfg.lam)
        fb = None
        if run.fill_log:
            fb = {k: torch.cat([x[k] for x in run.fill_log]) for k in run.fill_log[0]}
            run.fill_log.clear()
            if fb["label"].shape[0] > cfg.fill_cap:
                keep = torch.randperm(fb["label"].shape[0], device=dev)[:cfg.fill_cap]
                fb = {k: v[keep] for k, v in fb.items()}
        bc_w = max(cfg.bc_floor, cfg.bc_coef * max(0.0, 1 - it / cfg.bc_until)) if cfg.bc_decay else cfg.bc_coef
        stats = update(net, opt, b, cfg, bc_w, fb)
        if dev == "cuda":
            torch.cuda.synchronize()
        work = time.time() - tw0
        tw.writerow([it, ticks, round((time.time() - t0) / 60, 2), b["r"].shape[0] if b else 0, int(ep[0]),
                     round(ep[1] / max(ep[0], 1), 4), round(ep[2] / max(ep[0], 1), 4), round(ep[3] / max(ep[0], 1), 4)]
                    + [round(stats[k], 5) for k in ("pi", "v", "ent", "kl", "bc", "bc_acc")]
                    + [round(bc_w, 4), opt.param_groups[0]["lr"], stats["updates"], round(work, 2),
                       round(stats["fill"], 5), round(stats["fill_acc"], 4), 0 if fb is None else fb["label"].shape[0]])
        train_log.flush()
        if (it + 1) % cfg.eval_every == 0 or it == cfg.iterations - 1:
            res = evaluate_router(net, jumper, vals, rcfg, cfg.eval_max_ticks)
            mean = (res["standard"]["success"] + res["obstacles"]["success"]) / 2
            ew.writerow([it, res["standard"]["success"], res["obstacles"]["success"], mean]
                        + [res["standard"]["per_level"].get(L) for L in range(6)]
                        + [res["obstacles"]["per_level"].get(L) for L in range(6)])
            eval_log.flush()
            ck = {"model": net.state_dict(), "net": net.config, "config": asdict(cfg), "iteration": it, "validation": res}
            _save(ck, out / "checkpoints" / f"it{it:04d}.pt")
            if mean > best:
                best = mean
                _save(ck, out / "checkpoints" / "best.pt")
            print(f"it {it} val std {res['standard']['success']:.4f} obs {res['obstacles']['success']:.4f} best {best:.4f}",
                  flush=True)
        print(f"it {it} ep {int(ep[0])} succ {ep[1] / max(ep[0], 1):.3f} dec {b['r'].shape[0] if b else 0} "
              f"kl {stats['kl']:.4f} bc_acc {stats['bc_acc']:.3f} work {work:.1f}s", flush=True)
        if cfg.duty < 1:
            time.sleep(work * (1 / cfg.duty - 1))
    _save({"model": net.state_dict(), "net": net.config, "config": asdict(cfg), "iteration": cfg.iterations - 1},
          out / "checkpoints" / "last.pt")
    return out


def update(net, opt, b, cfg: RouteTrainConfig, bc_w: float, fb: dict | None = None) -> dict:
    stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "bc": 0.0, "bc_acc": 0.0, "fill": 0.0, "fill_acc": 0.0, "updates": 0}
    if b is None or b["r"].shape[0] < 64:
        return stats
    net.train()
    n = b["r"].shape[0]
    mb = math.ceil(n / cfg.minibatches)
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, device=b["r"].device)
        for s in range(0, n, mb):
            i = perm[s:s + mb]
            logp, ent, v, lg = net.evaluate(b["feats"][i], b["ok"][i], b["inview"][i], b["act"][i])
            adv = b["adv"][i]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = torch.exp(logp - b["logp"][i])
            pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            vl = 0.5 * (v - b["ret"][i]).pow(2).mean()
            loss = cfg.pg_coef * pg + cfg.vf_coef * vl - cfg.ent_coef * ent.mean()
            lab = b["label"][i]
            has = (lab >= 0) & (b["ok"][i].sum(1) > 1)
            bc = torch.zeros((), device=lg.device)
            if bc_w > 0 and bool(has.any()):
                bc = torch.nn.functional.cross_entropy(lg[has], lab[has])
                loss = loss + bc_w * bc
            with torch.no_grad():
                acc = (lg[has].argmax(-1) == lab[has]).float().mean() if bool(has.any()) else torch.zeros(())
            fl, facc = torch.zeros((), device=lg.device), torch.zeros(())
            if fb is not None and cfg.fill_bc > 0 and fb["label"].shape[0]:
                nf = fb["label"].shape[0]
                j = torch.randint(0, nf, (min(nf, math.ceil(nf / cfg.minibatches)),), device=lg.device)
                flg = net.logits(fb["feats"][j], fb["ok"][j], fb["inview"][j])
                fl = torch.nn.functional.cross_entropy(flg, fb["label"][j])
                loss = loss + cfg.fill_bc * fl
                with torch.no_grad():
                    facc = (flg.argmax(-1) == fb["label"][j]).float().mean()
            with torch.no_grad():
                multi = b["ok"][i].sum(1) > 1
                kl = ((ratio - 1) - torch.log(ratio))[multi].mean().item() if bool(multi.any()) else 0.0
            if kl > 1.5 * cfg.target_kl and stats["updates"] > 0:
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            for k, x in (("pi", pg.item()), ("v", vl.item()), ("ent", ent.mean().item()), ("kl", kl),
                         ("bc", bc.item()), ("bc_acc", acc.item()), ("fill", fl.item()), ("fill_acc", facc.item())):
                stats[k] += x
            stats["updates"] += 1
        else:
            continue
        break
    for k in ("pi", "v", "ent", "kl", "bc", "bc_acc", "fill", "fill_acc"):
        stats[k] /= max(stats["updates"], 1)
    return stats


def parse_args(argv=None) -> RouteTrainConfig:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(RouteTrainConfig):
        v = f.default
        if isinstance(v, bool):
            ap.add_argument("--" + f.name.replace("_", "-"), action="store_true", default=v)
        else:
            ap.add_argument("--" + f.name.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args(argv)
    return RouteTrainConfig(**{f.name: getattr(a, f.name) for f in fields(RouteTrainConfig)})


if __name__ == "__main__":
    train(parse_args())

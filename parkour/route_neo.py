"""The router with the neo planner: a chosen platform behind a wall is planned, not jumped by the policy.

    python -m parkour.route_neo --set single --family 40 --assist '{"restarts": 3}' --name NAME --tag "..."
    python -m parkour.route_neo --set standard --family 2 --assist '{"restarts": 3}' --name NAME --tag "..."

Without the course list (``parkour.route_env``), the planner's straight-path rule (``PlannerConfig.clear_path``)
drops a neo's landing: the line from the takeoff to it passes through the neo's wall. With
``RouteConfig(blocked="plan")`` every platform whose straight path is blocked stays selectable, with its head-hitter
flag, and when the router chooses one whose path is blocked by a block that is no part of a platform in view
(``route_env.wall_blocked``: a wall, pillar or beam, the blocks ``neo_dagger.neo_takeoffs`` counts on the course
list), the jump to it is a neo: the neo planner (``neo_assist.plan_once``) searches it from the bot's exact state
with the chosen platform as the target, and the actions found replace the jumper's until they have landed. A path
blocked only by other platforms (a jump past or over another platform) is the jumper's, as with "keep". With no
landing the jumper keeps control; as in ``neo_assist.evaluate_assisted``, with ``brake`` the bot first stops on the
takeoff, and with ``retries`` the search runs again while it stands there.

The trigger is the router's own: a decision (the bot stands on a new platform, ``RouteRunner``) whose first
choice is blocked by a wall. The course list only names the two platforms for the search: the platform stood on is the
request's ``cp``, the chosen one its ``target`` (perception equals the course's platforms, tests/test_route.py),
so the search lands on the chosen platform, its fall line is the lower top of the two (``floor``) and its seeds
are those of a course-list request from that platform. A choice that is no course platform (an obstacle's top:
standing on it ends the episode) is not planned. Per platform planned from: one takeoff search, then up to
``retries`` retries every ``retry_every`` ticks while the bot stands on it (at once after a brake), each only if
its plan can end before the stall limit. ``prior`` and ``approach`` need the course list and are refused;
``jumps`` stays "neo" (the blocked-path rule decides which jumps are planned). With ``plan_on`` (opt-in) a
decision made on a plan's landing is planned too, whatever it chose, and so on while plans land: the jumper
then never starts from a plan's landing state (the course-list planner's ``jumps`` "all" plans every jump). With
``plan_ahead`` (opt-in) a decision whose route crosses a wall in a later slot (the router's three slots, ``LOOK``:
``RouteRunner.decide``'s ``walled_chain``) is planned too, so the jumps that lead to a neo are the planner's as well;
on a course with no wall it plans nothing.

The bots do the same (``parkour.live --router R --assist JSON``: bridge/router.js, ``RouteAssist`` in
bridge/assist.js, the planner service); tests/test_route_neo.py checks the offline bot against this module.

Runs append one JSON row to ``results/neo_routing/validation.jsonl`` (families 40 single-neo, 2 standard,
5 obstacles) or ``test.jsonl`` (41, 1, 4) with ``neo_assist.summarize``'s counts; the details (every course,
search and router decision) go to ``results/neo_routing/details/NAME.json.gz``. Searches run on ``--search-device``
(the CPU by default, like the course-list planner's runs and the bots' planner service on this PC), the rest
on ``--device``. ``--parts N --jobs J`` splits a run into N contiguous parts, J processes at once, each with one
thread and its courses' own seeds (``course_offset``), as ``neo_assist --parts`` does; the parts are kept in
``runs/neo_routing_parts/NAME`` and merged into the one row (a rerun skips the finished parts).
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import courses as C
from . import physics
from .env import CoursePool, EnvConfig, ParkourEnv, standing_on
from .live import load_model
from .neo_assist import (AssistConfig, Rules, _arrival, brake_plan, config_meta, horizons, jump_kind, neo_done, plan_once,
                         single_neo_courses, summarize)
from .neo_dagger import neo_takeoffs
from .paths import ROOT
from .planner import PlannerConfig
from .route_env import RouteConfig, RouteRunner, RouteTables
from .train import (OBSTACLE_TEST_FAMILY, OBSTACLE_VALIDATION_FAMILY, TEST_SEED_FAMILY, VALIDATION_FAMILY,
                    held_out_courses)
from .neo_assist import SINGLE_FRESH_TEST_FAMILY, SINGLE_TEST_FAMILY, SINGLE_VALIDATION_FAMILY

K = C.MAX_PLATFORMS
OUT_DIR = ROOT / "results" / "neo_routing"
# (test, validation[, a further test set]); the single-neo sets' fresh test family 49 is the neo skill's (31 Oct)
SETS = {"single": (SINGLE_TEST_FAMILY, SINGLE_VALIDATION_FAMILY, SINGLE_FRESH_TEST_FAMILY),
        "standard": (TEST_SEED_FAMILY, VALIDATION_FAMILY), "obstacles": (OBSTACLE_TEST_FAMILY, OBSTACLE_VALIDATION_FAMILY)}


def check_config(rcfg: RouteConfig, acfg: AssistConfig) -> None:
    """The AssistConfig fields this mode cannot use (they need the course list or batched seeds)."""
    if acfg.prior > 0:
        raise ValueError("the search prior plays the policy with the course list: prior must be 0 with the router")
    if acfg.approach:
        raise ValueError("approach plans start from the platform before a known neo: approach must be 0 with the router")
    if acfg.batch != 1:
        raise ValueError("the router's searches are per request (plan_once): batch must be 1")
    if acfg.jumps != "neo":
        raise ValueError('with the router the blocked-path rule decides which jumps are planned: jumps must be "neo"')


def route_arrival(env: ParkourEnv, i: int, frm: int, target: int, seed_course: int) -> dict:
    """The search request for row i: its exact state, from course platform ``frm`` (stood on) to ``target`` (the
    router's blocked choice), a neo's, its fall line the lower top of the two."""
    top = env.pool.top[env.course[i]]
    return dict(_arrival(env, i), cp=frm, target=target, neo=True, seed_course=seed_course,
                floor=float(torch.minimum(top[frm], top[target])))


def evaluate_routed_neo(jumper, pool: CoursePool, tables: RouteTables, choose=None, fill_choose=None,
                        pcfg: PlannerConfig | None = None, rcfg: RouteConfig | None = None,
                        acfg: AssistConfig | None = None, stall_ticks: int = 200, max_ticks: int = 4000,
                        fall_rule: str = "local", record: bool = False, settle: int = 0,
                        search_pool: CoursePool | None = None, course_offset: int = 0, log=None, skill=None) -> dict:
    """First episode on every course, the jumper's slots filled by the router (``route_env.evaluate_routed``), and
    with ``rcfg.blocked == "plan"`` (and ``acfg.enabled``) every first choice blocked by a wall planned by the neo
    planner.

    ``search_pool``: the same courses on the searches' device (default ``pool``). ``course_offset``: the index of the
    pool's first course in the whole set (search seeds and event numbers, for a run in parts). Returns
    ``neo_assist.evaluate_assisted``'s fields (``per_course`` with ``neo_done``, ``events``: one per search,
    ``plans``, ``search_seconds``), and with ``record`` the router's decisions as ``evaluate_routed`` records them.

    ``skill``: the neo skill (``parkour.neo_skill``) instead of the planner (the assist off, blocked choices kept):
    a decision whose route's next jump crosses a wall hands the next two jumps to it, one whose second jump does
    (the router's second slot) the next three, so it plays from the platform before a neo's takeoff until the bot
    stands on the platform after the neo's target, as ``neo_assist.skill_cps`` does with the course list; each
    decision counts one down, and a later walled jump extends the count."""
    pcfg = pcfg or PlannerConfig()
    rcfg = rcfg or RouteConfig(blocked="plan")
    acfg = acfg or AssistConfig()
    planning = rcfg.blocked == "plan" and acfg.enabled
    if skill is not None and (planning or rcfg.blocked == "drop" or skill.obs_dim != jumper.obs_dim):
        raise ValueError("the neo skill replaces the planner: blocked choices kept, the assist off, the jumper's "
                         "observation")
    if planning:
        check_config(rcfg, acfg)
    dev = pool.device
    n = pool.size
    rows = torch.arange(n, device=dev)
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=stall_ticks, max_ticks=max_ticks, fall_rule=fall_rule),
                     sampler=None)
    env.reset_all(rows)
    if settle:
        idle = torch.zeros(n, 6, dtype=torch.bool, device=dev)
        for _ in range(settle):
            env.state = physics.simulate(pool.world, env.course, env.state, idle)
        env.dist = env._distance(env.state.pos, env.course, env.cp)
        env.reset_memory(torch.ones(n, dtype=torch.bool, device=dev))
    run = RouteRunner(env, tables, pcfg, rcfg)
    spool = search_pool if search_pool is not None else pool
    sdev = spool.device
    rules = Rules(spool) if planning else None
    hz = horizons(acfg)
    takeoffs = [neo_takeoffs(c) for c in pool.courses]  # the statistics (neo_done), never the trigger
    neo = torch.zeros(n, K, dtype=torch.bool, device=dev)
    for i, ks in enumerate(takeoffs):
        neo[i, ks] = True
    frm = torch.full((n,), -1, dtype=torch.int64, device=dev)  # course platform the last decision planned from
    want = torch.full((n,), -1, dtype=torch.int64, device=dev)  # its first choice blocked by a wall, to plan (-1: none)
    tries = torch.zeros(n, K, dtype=torch.int64, device=dev)  # searches per platform planned from
    last_try = torch.full((n,), -10**9, dtype=torch.int64, device=dev)
    left = torch.zeros(n, dtype=torch.int64, device=dev)  # decisions the neo skill still plays (``skill``)
    skill_at = torch.zeros(n, K, dtype=torch.bool, device=dev)  # the checkpoints at which it played (per course)
    H = max(hz["takeoff"], hz["retry"], acfg.brake, 1)
    plan = torch.zeros(n, H, 5, dtype=torch.int64, device=dev)
    plan_len = torch.zeros(n, dtype=torch.int64, device=dev)
    plan_pos = torch.zeros(n, dtype=torch.int64, device=dev)
    plan_t = torch.full((n,), -1, dtype=torch.int64, device=dev)  # the platform a plan ends on
    braking = torch.zeros(n, dtype=torch.bool, device=dev)
    finished = torch.zeros(n, dtype=torch.bool, device=dev)
    success = torch.zeros(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    length = torch.zeros(n, dtype=torch.int64, device=dev)
    prog = torch.zeros(n, dtype=torch.float64, device=dev)
    took = torch.zeros(n, K, dtype=torch.bool, device=dev)  # values the first episode's checkpoint took (neo_done)
    took[:, 0] = True
    reached = [set() for _ in range(n)]
    events = []
    decisions = [[] for _ in range(n)] if record else None
    plans = {"played": 0, "landed_as_planned": 0, "brakes": 0}
    search_seconds = 0.0

    def decided(res, t: int, landed: torch.Tensor | None = None) -> None:
        """A decision's platform planned from and, if its first choice is blocked by a wall (or, with ``plan_on``, the
        decision was made on a plan's landing: ``landed``; with ``plan_ahead``, a later slot of its route is), the
        platform to plan."""
        if res is None:
            return
        r = res["rows"]
        ar = torch.arange(r.numel(), device=dev)
        hk = res["cidx"][ar, res["here"]]
        act = res["action"]
        tk = torch.where(act >= 0, res["cidx"][ar, act.clamp(min=0)], torch.full_like(act, -1))
        frm[r] = hk
        plan_it = res["walled"] | (landed[r] if landed is not None else torch.zeros_like(res["walled"]))
        if acfg.plan_ahead:
            plan_it = plan_it | res["walled_chain"].any(1)
        want[r] = torch.where(plan_it & (hk >= 0) & (tk >= 0), tk, torch.full_like(tk, -1))
        if skill is not None:
            wc = res["walled_chain"]
            need = torch.where(wc[:, 1], 3, torch.where(wc[:, 0], 2, 0))
            left[r] = torch.maximum(left[r] - 1, need)
        if record:
            rects = res["rect"].tolist()
            for j, i in enumerate(r.tolist()):
                decisions[i].append([t, rects[j][res["here"][j].item()],
                                     [rects[j][p] for p in res["picks"][j].tolist() if p >= 0]])

    def searches(t: int, on: torch.Tensor, ended: torch.Tensor, retry_now: torch.Tensor) -> None:
        """The searches after tick t's step (t = -1: the start state, ``plan_start``); a plan found plays from the
        next tick on."""
        nonlocal search_seconds
        fr = frm.clamp(min=0)
        on_frm = standing_on(pool, env.state, env.course)[rows, fr] & (frm >= 0)
        n_try = tries[rows, fr]
        for kind in ("takeoff", "retry"):
            free = on & on_frm & (want >= 0) & (plan_pos >= plan_len)
            if kind == "takeoff":
                mask = free & (n_try == 0)
            else:
                mask = (free & (n_try > 0) & (n_try <= acfg.retries) & ((t - last_try >= acfg.retry_every) | retry_now)
                        & (env.stall + hz["retry"] < stall_ticks))
            for i in torch.nonzero(mask).flatten().tolist():
                c, k = int(frm[i]), int(want[i])
                arr = route_arrival(env, i, c, k, course_offset + i)
                attempt = int(tries[i, c])
                st = {}
                t0 = time.time()
                seq, rounds, restarts = plan_once(jumper, spool, rules, arr, kind, attempt, acfg, hz[kind], sdev, st)
                dt = time.time() - t0
                search_seconds += dt
                ev = {"course": course_offset + i, "kind": kind, "takeoff": c, "target": k, "tick": t + 1,
                      "after_plan": bool(ended[i]), "found": seq is not None, "plan_ticks": 0 if seq is None else len(seq),
                      "rounds": rounds, "braked": False}
                if acfg.restarts:
                    ev["restarts"] = restarts
                if acfg.corner_limit:
                    ev["corner_dropped"] = st.get("corner_dropped", 0)
                events.append(ev)
                tries[i, c] += 1
                last_try[i] = t
                if seq is not None:
                    s = torch.as_tensor(seq, dtype=torch.int64, device=dev)
                    plan[i, :len(s)] = s
                    plan_len[i] = len(s)
                    plan_pos[i] = 0
                    plan_t[i] = k
                elif acfg.brake and int(tries[i, c]) <= acfg.retries:
                    b = brake_plan(spool, rules, arr, acfg.brake, sdev)
                    if b is not None:
                        plan[i, :len(b)] = torch.as_tensor(b, device=dev)
                        plan_len[i] = len(b)
                        plan_pos[i] = 0
                        braking[i] = True
                        plans["brakes"] += 1
                        ev["braked"] = True
                if log:
                    log({"tick": t + 1, "course": course_offset + i, "kind": kind, "takeoff": c, "target": k,
                         "found": seq is not None, "seconds": round(dt, 2), "unfinished": int((~finished).sum())})

    with torch.no_grad():
        decided(run.decide(rows, run.start_cells(), choose, fill_choose), 0)
        obs = run.observe()
        if planning and acfg.plan_start:  # a start on the ground (settled): its searches before the first action
            on = env.state.on_ground.clone()
            for i in torch.nonzero(on & neo[rows, env.cp]).flatten().tolist():
                reached[i].add(int(env.cp[i]))
            none = torch.zeros(n, dtype=torch.bool, device=dev)
            searches(-1, on, none, none)
        for t in range(max_ticks):
            a, _, _ = jumper.act(obs, deterministic=True)
            if skill is not None and bool((left > 0).any()):
                a = torch.where((left > 0)[:, None], skill.act(obs, deterministic=True)[0], a)
                skill_at[rows, env.cp] |= (left > 0) & ~finished
            active = plan_pos < plan_len
            if bool(active.any()):
                queued = plan[rows, plan_pos.clamp(max=H - 1)]
                a = torch.where(active[:, None], queued, a)
                plan_pos = plan_pos + active.to(torch.int64)
            _, _, _, _, _, info = env.step(a)
            done = info["done"]
            first = done & ~finished
            success |= first & info["success"]
            fell |= first & info["fell"]
            length = torch.where(first, info["ep_len"], length)
            prog = torch.where(first, info["progress_frac"], prog)
            ended = active & (plan_pos >= plan_len)  # a plan's (or brake's) last action was just played
            retry_now = ended & braking
            braking &= ~ended
            landed = None  # plan_on: the rows whose plan has just landed on its target (their decision now is planned)
            if bool(ended.any()):
                stand = standing_on(pool, env.state, env.course)
                on_target = stand[rows, plan_t.clamp(min=0)] & ended & ~retry_now & ~first
                if planning and acfg.plan_on:
                    landed = on_target
                for i in torch.nonzero(ended & ~retry_now).flatten().tolist():
                    plans["played"] += 1
                    ok = bool(info["success"][i]) if bool(first[i]) else bool(on_target[i])
                    plans["landed_as_planned"] += int(ok)
            finished |= done
            plan_len = torch.where(done, torch.zeros_like(plan_len), plan_len)
            cp = env.cp
            took[rows, cp] = took[rows, cp] | ~finished
            on = ~finished & env.state.on_ground
            for i in torch.nonzero(on & neo[rows, cp]).flatten().tolist():
                reached[i].add(int(cp[i]))
            if bool(finished.all()):
                break
            trig, cell = run.triggers(~finished)
            decided(run.decide(trig.nonzero().squeeze(1), cell, choose, fill_choose), t + 1, landed)
            if planning:
                searches(t, on, ended, retry_now)
            obs = run.observe()
    lv = pool.level.cpu().numpy()
    s, f, ln, pr = (x.cpu().numpy() for x in (success, fell, length, prog))
    last = (pool.n - 1).cpu().numpy()
    cp_final = np.rint(pr * last).astype(int)
    took = took.cpu().numpy()
    per_course = []
    for i in range(n):
        ks = takeoffs[i]
        cps = set(np.flatnonzero(took[i]).tolist()) | {int(cp_final[i])}
        per_course.append({"success": bool(s[i]), "fell": bool(f[i]), "ticks": int(ln[i]), "progress": float(pr[i]),
                           "level": int(lv[i]), "neos": len(ks), "neos_reached": len(reached[i]),
                           "neos_passed": sum(1 for k in ks if k in reached[i] and (s[i] or cp_final[i] > k)),
                           "neo_done": neo_done(ks, cps),
                           "stopped_at": int(cp_final[i]) if not s[i] else None,
                           "stopped_at_neo": bool(not s[i] and int(cp_final[i]) in ks),
                           "stopped_at_kind": None if s[i] else jump_kind(pool.courses[i], int(cp_final[i]), ks)})
        if skill is not None:  # where the skill played (the course list's skill_cps when the router follows the course)
            per_course[-1]["skill_at"] = torch.nonzero(skill_at[i]).flatten().tolist()
    out = {"per_course": per_course, "events": events, "plans": plans, "search_seconds": round(search_seconds, 1)}
    if record:
        out["decisions"] = decisions
    return out


def course_set(name: str, family: int, per_level: int = 0) -> list[C.Course]:
    """The single-neo sets (40, 41), or the router's own sets (route_eval.course_sets): standard (2, 1) and
    obstacles (5, 4), 300 courses per level on validation and 200 on test unless ``per_level``."""
    if family not in SETS[name]:
        raise ValueError(f"--set {name} has families {SETS[name]} (test, validation, then any further test set)")
    if name == "single":
        return single_neo_courses(family)
    per_level = per_level or (200 if family == SETS[name][0] else 300)
    levels = C.load_levels()
    if name == "standard":
        return held_out_courses(levels, per_level, 10, family=family)
    return held_out_courses(levels, per_level, 10, obstacle_prob=0.5, family=family)


PARTS_DIR = ROOT / "runs" / "neo_routing_parts"


def part_bounds(n: int, k: int, parts: int) -> tuple[int, int]:
    """Courses [lo, hi) of part k (1-based) of ``parts`` contiguous parts of n courses (neo_assist.part_bounds)."""
    if not 1 <= k <= parts <= n:
        raise ValueError(f"part {k} of {parts}")
    return round((k - 1) * n / parts), round(k * n / parts)


def merge_parts(parts: list[dict]) -> dict:
    """One result, as ``evaluate_routed_neo`` returns it with ``record``, from consecutive parts of a course set."""
    out = {"per_course": [], "events": [], "decisions": [], "plans": {}, "search_seconds": 0.0}
    for p in parts:
        for key in ("per_course", "events", "decisions"):
            out[key] += p[key]
        for k, v in p["plans"].items():
            out["plans"][k] = out["plans"].get(k, 0) + v
        out["search_seconds"] = round(out["search_seconds"] + p["search_seconds"], 1)
    return out


def _part_argv(argv: list[str], k: int, n: int) -> list[str]:
    """The command line of part k of n: this run's own arguments without --parts, --jobs and --threads."""
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
            continue
        if x in ("--parts", "--jobs", "--threads"):
            skip = True
            continue
        if x.startswith(("--parts=", "--jobs=", "--threads=")):
            continue
        out.append(x)
    return out + ["--part", f"{k}/{n}", "--threads", "1"]


def _run_parts(argv: list[str], meta: dict, n_courses: int, parts: int, jobs: int, folder: Path) -> list[dict]:
    """Run the unfinished parts (at most ``jobs`` at once, one process and thread each); return every part."""
    import os
    import subprocess
    import sys
    folder.mkdir(parents=True, exist_ok=True)
    files = [folder / f"{k}of{parts}.json" for k in range(1, parts + 1)]

    def cached(k, f):
        p = json.loads(f.read_text(encoding="utf-8"))
        if p.get("meta") != meta or p.get("courses") != list(part_bounds(n_courses, k, parts)):
            raise SystemExit(f"{f} was computed with other settings or courses; keep it and use a new name")
        return p
    for k, f in enumerate(files, 1):
        if f.exists():
            cached(k, f)
    todo = [k for k, f in enumerate(files, 1) if not f.exists()]
    running = []
    try:
        while todo or running:
            while todo and len(running) < jobs:
                k = todo.pop(0)
                log = open(folder / f"{k}of{parts}.log", "a", encoding="utf-8")
                proc = subprocess.Popen([sys.executable, "-m", "parkour.route_neo"] + _part_argv(argv, k, parts),
                                        cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                running.append((k, proc, log))
            time.sleep(5)
            for item in [r for r in running if r[1].poll() is not None]:
                running.remove(item)
                item[2].close()
                if item[1].returncode != 0:
                    raise SystemExit(f"part {item[0]} of {parts} failed (see {folder / f'{item[0]}of{parts}.log'})")
    finally:
        for _, proc, log in running:
            if proc.poll() is None:
                if os.name == "nt":  # the venv's python.exe is a launcher: stop its child too
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    proc.terminate()
                proc.wait()
            log.close()
    return [cached(k, f) for k, f in enumerate(files, 1)]


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def _rel(path: Path) -> str:
    p = path.resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def main(argv=None):
    from .route_policy import load_router
    from .route_train import router_choosers
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=tuple(SETS), default="single")
    ap.add_argument("--family", type=int, default=SINGLE_VALIDATION_FAMILY)
    ap.add_argument("--per-level", type=int, default=0, help="standard and obstacles sets: courses per level "
                                                              "(default 300 on validation, 200 on test)")
    ap.add_argument("--courses", type=int, default=0, help="only the first N courses (quick checks)")
    ap.add_argument("--jumper", default="results/final_policy.pt")
    ap.add_argument("--router", default="results/routing/router_policy.pt")
    ap.add_argument("--fill", default="", choices=("", "greedy", "policy"), help="default: the router's training setting")
    ap.add_argument("--blocked", default="plan", choices=("drop", "keep", "plan"), help="RouteConfig.blocked")
    ap.add_argument("--assist", default="{}", help="JSON overrides of neo_assist.AssistConfig (the neo planner)")
    ap.add_argument("--skill", default="", help="the neo skill's checkpoint (parkour.neo_skill) instead of the planner: it "
                                                "plays each walled jump's block (the assist off); rows go to "
                                                "results/neo_routing/skill_*.jsonl")
    ap.add_argument("--settle", type=int, default=0, help="ticks without keys on the start block first (the bots: 10)")
    ap.add_argument("--fall-rule", default="local", choices=("local", "remaining"))
    ap.add_argument("--device", default="", help="the episodes and the router (default: cuda if available)")
    ap.add_argument("--search-device", default="cpu", help="the searches (the bots' planner service must use the same)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--name", required=True, help="the run's name: its details file and its row's run_id")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="", help="the row's file (default results/neo_routing/validation.jsonl or test.jsonl)")
    ap.add_argument("--details", default="", help="default results/neo_routing/details/NAME.json.gz")
    ap.add_argument("--parts", type=int, default=0, help="N: run the unfinished parts of NAME (--jobs at once, one thread "
                                                         "each), then merge them into one row")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--part", default="", help="K/N: only the K-th of N contiguous parts, seeded as in the whole run; "
                                               "writes runs/neo_routing_parts/NAME/KofN.json and no row")
    ap.add_argument("--parts-dir", default=str(PARTS_DIR))
    raw = list(argv) if argv is not None else __import__("sys").argv[1:]
    a = ap.parse_args(raw)
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.part and a.parts:
        raise SystemExit("use one of --part and --parts")
    test = a.family != SETS[a.set][1]
    courses = course_set(a.set, a.family, a.per_level)
    if a.courses:
        courses = courses[:a.courses]
    stem = "skill_" if a.skill else ""
    out = Path(a.out) if a.out else OUT_DIR / (f"{stem}test.jsonl" if test else f"{stem}validation.jsonl")
    details = Path(a.details) if a.details else OUT_DIR / "details" / f"{a.name}.json.gz"
    folder = Path(a.parts_dir) / a.name
    if details.exists() and not a.part:
        raise SystemExit(f"{details} exists: a run's name is used once")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    acfg = AssistConfig(**{"batch": 1, **json.loads(a.assist)})
    if a.skill:  # the skill replaces the planner
        if json.loads(a.assist) or a.blocked == "drop":
            raise SystemExit("--skill plays the walled jumps itself: no --assist, and blocked choices kept (plan or keep)")
        acfg = dataclasses.replace(acfg, enabled=False)
    rcfg_fill = a.fill or torch.load(a.router, map_location="cpu", weights_only=False)["config"].get("fill", "greedy")
    rcfg = RouteConfig(fill=rcfg_fill, blocked=a.blocked)
    if rcfg.blocked == "plan" and acfg.enabled:
        check_config(rcfg, acfg)
    pcfg = PlannerConfig()
    jumper_path, router_path = Path(a.jumper), Path(a.router)
    meta = {"set": a.set, "family": a.family, "split": "test" if test else "validation", "courses": len(courses),
            "jumper": _rel(jumper_path), "jumper_sha1": _sha1(jumper_path), "router": _rel(router_path),
            "router_sha1": _sha1(router_path), "router_fill": rcfg.fill, "blocked": rcfg.blocked,
            "assist": rcfg.blocked == "plan" and acfg.enabled, "config": config_meta(acfg), "planner": dataclasses.asdict(pcfg),
            "settle": a.settle, "fall_rule": a.fall_rule, "stall": 200, "max_ticks": 4000,
            "measured": f"simulator (float64 port of prismarine-physics): episodes {dev.upper() if dev == 'cpu' else 'CUDA'}, "
                        f"searches {a.search_device.upper() if a.search_device == 'cpu' else 'CUDA'}",
            "tag": a.tag, "run_id": "neo_routing/" + a.name}
    if a.courses:
        meta["first_courses_only"] = a.courses
    if a.per_level and a.set != "single":
        meta["per_level"] = a.per_level
    if a.skill:
        meta["skill"], meta["skill_sha1"] = _rel(Path(a.skill)), _sha1(Path(a.skill))
    if a.parts:
        meta["courses_sha256"] = hashlib.sha256(json.dumps([c.to_json() for c in courses], sort_keys=True).encode()).hexdigest()
        t0 = time.time()
        parts = _run_parts(raw, meta, len(courses), a.parts, a.jobs, folder)
        res = merge_parts(parts)
        row = {**meta, **summarize(res), "parts": a.parts, "seconds": round(sum(p["seconds"] for p in parts), 1),
               "wall_seconds": round(time.time() - t0, 1)}
        return _publish(row, res, out, details)
    lo, hi = 0, len(courses)
    if a.part:
        k, n = (int(v) for v in a.part.split("/"))
        lo, hi = part_bounds(len(courses), k, n)
        meta["courses_sha256"] = hashlib.sha256(json.dumps([c.to_json() for c in courses], sort_keys=True).encode()).hexdigest()
        part_file = folder / f"{k}of{n}.json"
        if part_file.exists():
            print(f"{part_file} exists")
            return None
    jumper = load_model(jumper_path, dev)
    net = load_router(router_path, dev)
    choose, fill = router_choosers(net, True)
    pool = CoursePool(courses[lo:hi], dev, C.GRID)
    spool = pool if a.search_device == dev else CoursePool(courses[lo:hi], a.search_device, C.GRID)
    t0 = time.time()
    res = evaluate_routed_neo(jumper, pool, RouteTables(pool), choose, fill, pcfg, rcfg, acfg, settle=a.settle,
                              fall_rule=a.fall_rule, record=True, search_pool=spool, course_offset=lo,
                              log=lambda d: print(json.dumps(d), flush=True),
                              skill=load_model(Path(a.skill), dev) if a.skill else None)
    if a.part:
        part_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = part_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"meta": meta, "courses": [lo, hi], **res, "seconds": round(time.time() - t0, 1)}),
                       encoding="utf-8")
        tmp.replace(part_file)
        return None
    row = {**meta, **summarize(res), "seconds": round(time.time() - t0, 1)}
    return _publish(row, res, out, details)


def _publish(row: dict, res: dict, out: Path, details: Path) -> dict:
    """Write the run's details file and append its row."""
    details.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(details, "wt", encoding="utf-8") as fh:
        json.dump({"row": row, "per_course": res["per_course"], "events": res["events"], "decisions": res["decisions"]}, fh)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))
    return row


if __name__ == "__main__":
    main()

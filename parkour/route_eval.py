"""Evaluate a route policy against the greedy planner on held-out courses (simulator).

    python -m parkour.route_eval --router runs/NAME/checkpoints/best.pt --set validation --out results/routing/router_validation_NAME
    python -m parkour.route_eval --router greedy --set validation --per-level 100

``--set validation``: families 2 (standard) and 5 (obstacles), 300 per level; ``--set test``: families 1
and 4, 200 per level, for the final report only (one run per router version, never used to choose).

Protocol (docs/MISSION_ROUTING.md section 8, as ``planner.evaluate_planned``): one episode per course,
the final policy by argmax, the router by argmax (ties: the fixed candidate order), stall 200, 4,000
ticks, settle 0, both fall rules (``local`` is the headline). Each set is written in the planner
CLI's layout, ``<out>_standard.json`` and ``<out>_obstacles.json``: per fall rule ``given_route`` (the
course list), ``planner_route`` (the greedy planner, ``planner.evaluate_planned``) and ``router_route``
(with ``failure_modes`` from ``planner.failure_modes`` on the router's events), each with its outcome
on every course in ``held_out_courses`` order. ``<out>_paired.json`` compares router and planner on
the same courses: the difference, a 95% bootstrap interval over courses, and McNemar's exact test
on the courses where exactly one of the two succeeds, per set and for both sets together.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import courses as C
from .env import CoursePool
from .live import load_model
from .planner import PlannerConfig, evaluate_planned, failure_modes
from .route_env import RouteConfig, RouteTables, evaluate_routed
from .route_policy import load_router
from .route_train import router_choosers
from .train import (OBSTACLE_TEST_FAMILY, OBSTACLE_VALIDATION_FAMILY, TEST_SEED_FAMILY, VALIDATION_FAMILY, evaluate,
                    held_out_courses)

RULES = ("local", "remaining")


def course_sets(which: str, per_level: int) -> dict:
    levels = C.load_levels()
    fam, ofam = (VALIDATION_FAMILY, OBSTACLE_VALIDATION_FAMILY) if which == "validation" else (TEST_SEED_FAMILY, OBSTACLE_TEST_FAMILY)
    return {"standard": (fam, held_out_courses(levels, per_level, 10, family=fam)),
            "obstacles": (ofam, held_out_courses(levels, per_level, 10, obstacle_prob=0.5, family=ofam))}


def paired(a, b, n_boot: int = 10000, seed: int = 0) -> dict:
    """Router (a) against planner (b), both 0/1 per course on the same courses."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(1)
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    k, n = min(n10, n01), n10 + n01
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"courses": len(d), "router": float(a.mean()), "planner": float(b.mean()), "difference": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "router_only": n10, "planner_only": n01, "mcnemar_exact_p": p}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", required=True, help="checkpoint, or 'greedy' (the planner's rule through the router's "
                                                       "runner) or 'oracle' (diagnostic: the course's next platform)")
    ap.add_argument("--fill", default="", choices=("", "greedy", "policy", "oracle"),
                    help="slots 2 and 3 (default: the checkpoint's training setting; greedy for 'greedy')")
    ap.add_argument("--set", default="validation", choices=("validation", "test"))
    ap.add_argument("--per-level", type=int, default=0, help="default: 300 on validation, 200 on test")
    ap.add_argument("--jumper", default="results/final_policy.pt")
    ap.add_argument("--rules", default="local,remaining", help="fall rules to evaluate")
    ap.add_argument("--no-baselines", action="store_true", help="skip given_route and planner_route (quick checks)")
    ap.add_argument("--out", default="", help="prefix of the output files")
    a = ap.parse_args(argv)
    per_level = a.per_level or (300 if a.set == "validation" else 200)
    rules = [r for r in a.rules.split(",") if r]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    jumper = load_model(Path(a.jumper), dev)
    choose = fill = None
    fill_mode = a.fill or "greedy"
    if a.router == "oracle":
        choose, fill = "oracle", ("oracle" if a.fill == "oracle" else None)
    elif a.router != "greedy":
        net = load_router(Path(a.router), dev)
        choose, fill = router_choosers(net, True)
        fill_mode = a.fill or torch.load(a.router, map_location="cpu", weights_only=False)["config"].get("fill", "greedy")
    rcfg = RouteConfig(fill=fill_mode if fill_mode in ("greedy", "policy") else "greedy")
    pcfg = PlannerConfig()
    measured = "simulator (GPU)" if dev == "cuda" else "simulator (CPU)"
    outs = {}
    for name, (fam, cs) in course_sets(a.set, per_level).items():
        pool = CoursePool(cs, dev, C.GRID)
        tables = RouteTables(pool)
        res = {"checkpoint": a.jumper, "router": a.router, "fill": fill_mode, "planner": pcfg.__dict__,
               "obstacles": name == "obstacles", "family": fam, "set": a.set, "courses": len(cs), "measured": measured,
               "protocol": "first episode per course, argmax, stall 200, 4000 ticks, settle 0"}
        for rule in rules:
            t0 = time.time()
            block = {}
            with torch.no_grad():
                if not a.no_baselines:
                    base = evaluate(jumper, pool, dev, 200, deterministic=True, fall_rule=rule, per_course=True)
                    block["given_route"] = {"success": base["success"], "per_level": base["per_level"],
                                            "per_course": base["per_course"]}
                    plan = evaluate_planned(jumper, pool, dev, pcfg, fall_rule=rule)
                    block["planner_route"] = {k: plan[k] for k in ("success", "per_level", "failure_modes", "per_course")}
                r = evaluate_routed(jumper, pool, tables, choose, fill, pcfg, rcfg, stall_ticks=200, max_ticks=4000,
                                    fall_rule=rule, record=True)
            s = np.asarray(r["per_course"]["success"])
            modes = failure_modes(pool, [SimpleNamespace(events=e) for e in r["events"]], s)
            block["router_route"] = {"success": r["success"], "per_level": r["per_level"], "failure_modes": modes,
                                     "per_course": r["per_course"], "plans": r["events"]}
            res[rule] = block
            line = {"set": name, "family": fam, "fall_rule": rule, "router": round(r["success"], 4), "failure_modes": modes,
                    "router_per_level": {k: round(v["success"], 3) for k, v in r["per_level"].items()}}
            if not a.no_baselines:
                line.update(given=round(block["given_route"]["success"], 4), planner=round(block["planner_route"]["success"], 4))
            print(json.dumps(line), f"{time.time() - t0:.0f}s", flush=True)
        outs[name] = res
        if a.out:
            Path(a.out + f"_{name}.json").parent.mkdir(parents=True, exist_ok=True)
            Path(a.out + f"_{name}.json").write_text(json.dumps(res))
        del pool, tables
        if dev == "cuda":
            torch.cuda.empty_cache()
    if not a.no_baselines:
        comp = {"router": a.router, "set": a.set, "measured": measured,
                "note": "router minus greedy planner on the same courses; ci95: bootstrap over courses (10,000 resamples); "
                        "mcnemar_exact_p: two-sided exact test on the courses where exactly one succeeds"}
        for rule in rules:
            ra = {n: outs[n][rule]["router_route"]["per_course"]["success"] for n in outs}
            pa = {n: outs[n][rule]["planner_route"]["per_course"]["success"] for n in outs}
            comp[rule] = {n: paired(ra[n], pa[n]) for n in outs}
            comp[rule]["combined"] = paired(sum((ra[n] for n in outs), []), sum((pa[n] for n in outs), []))
            print(json.dumps({"fall_rule": rule, **{k: {q: (round(v[q], 4) if isinstance(v[q], float) else v[q])
                                                          for q in ("difference", "ci95", "router_only", "planner_only", "mcnemar_exact_p")}
                                                      for k, v in comp[rule].items()}}), flush=True)
        if a.out:
            Path(a.out + "_paired.json").write_text(json.dumps(comp, indent=1))
    return outs


if __name__ == "__main__":
    main()

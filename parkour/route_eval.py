"""Evaluate a route policy (or the greedy rule) on held-out courses, simulator.

    python -m parkour.route_eval --router runs/NAME/checkpoints/best.pt --set validation --out results/routing/router_validation.json
    python -m parkour.route_eval --router greedy --set validation --per-level 100

``--set validation``: families 2 (standard) and 5 (obstacles); ``--set test``: families 1 and 4, for
the final report only (one run per router version, never used to choose). The jumper is the final
policy, deterministic; fall rule ``local`` unless ``--fall-rule remaining``. The JSON keeps every
course's outcome so routers can be compared course by course.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import courses as C
from .env import CoursePool
from .live import load_model
from .planner import PlannerConfig
from .route_env import RouteConfig, RouteTables, evaluate_routed
from .route_policy import load_router
from .route_train import router_choosers
from .train import OBSTACLE_TEST_FAMILY, OBSTACLE_VALIDATION_FAMILY, TEST_SEED_FAMILY, VALIDATION_FAMILY, held_out_courses


def course_sets(which: str, per_level: int) -> dict:
    levels = C.load_levels()
    fam, ofam = (VALIDATION_FAMILY, OBSTACLE_VALIDATION_FAMILY) if which == "validation" else (TEST_SEED_FAMILY, OBSTACLE_TEST_FAMILY)
    return {"standard": (fam, held_out_courses(levels, per_level, 10, family=fam)),
            "obstacles": (ofam, held_out_courses(levels, per_level, 10, obstacle_prob=0.5, family=ofam))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", required=True, help="checkpoint, or 'greedy' (the planner's rule) or 'oracle' (diagnostic)")
    ap.add_argument("--fill", default="", choices=("", "greedy", "policy", "oracle"),
                    help="slots 2 and 3 (default: the checkpoint's training setting; greedy for 'greedy')")
    ap.add_argument("--set", default="validation", choices=("validation", "test"))
    ap.add_argument("--per-level", type=int, default=300, help="validation: 300 (1,800 per set); test: 200")
    ap.add_argument("--jumper", default="results/final_policy.pt")
    ap.add_argument("--fall-rule", default="local", choices=("local", "remaining"))
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    jumper = load_model(Path(a.jumper), dev)
    choose = fill = None
    fill_mode = a.fill or "greedy"
    if a.router == "oracle":
        choose = "oracle"
        fill = "oracle" if a.fill == "oracle" else None
    elif a.router != "greedy":
        net = load_router(Path(a.router), dev)
        choose, fill = router_choosers(net, True)
        fill_mode = a.fill or torch.load(a.router, map_location="cpu", weights_only=False)["config"].get("fill", "greedy")
    rcfg = RouteConfig(fill=fill_mode if fill_mode in ("greedy", "policy") else "greedy")
    res = {"router": a.router, "fill": fill_mode, "set": a.set, "per_level": a.per_level, "jumper": a.jumper,
           "fall_rule": a.fall_rule, "measured": "simulator (GPU)" if dev == "cuda" else "simulator (CPU)"}
    for name, (fam, cs) in course_sets(a.set, a.per_level).items():
        pool = CoursePool(cs, dev, C.GRID)
        t0 = time.time()
        with torch.no_grad():
            r = evaluate_routed(jumper, pool, RouteTables(pool), choose, fill, PlannerConfig(), rcfg,
                                stall_ticks=200, max_ticks=4000, fall_rule=a.fall_rule)
        res[name] = {"family": fam, "courses": len(cs), "success": r["success"],
                     "per_level": {k: v["success"] for k, v in r["per_level"].items()},
                     "per_course": r["per_course"], "seconds": round(time.time() - t0, 1)}
        print(json.dumps({"set": name, "family": fam, "success": round(r["success"], 4),
                          "per_level": {k: round(v["success"], 3) for k, v in r["per_level"].items()}}), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()

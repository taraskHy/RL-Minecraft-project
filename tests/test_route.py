"""Routing (parkour.route_env): the GPU perception equals the planner's own."""
import math
import unittest

import numpy as np
import torch

from parkour import courses as C
from parkour import planner as P
from parkour.env import CoursePool
from parkour.route_env import RouteTables, Scene


def cpu_view(pool, i, cell, cfg):
    """The planner's targets and options from the platform containing ``cell`` (standing on it)."""
    s = pool.world.solid[i].cpu().numpy()
    gx, gy, gz = s.shape

    def is_solid(x, y, z):
        return 0 <= x < gx and 0 <= y < gy and 0 <= z < gz and bool(s[x, y, z])
    plats = P.perceive(is_solid, cell[0], cell[1] + 1, cell[2], cfg)
    targets = [p for p in plats if not P.head_hitter(p, plats)]
    here = next(p for p in targets if p.contains(tuple(cell)))
    goal = tuple(pool.centre[i, int(pool.n[i]) - 1].tolist())
    opts = P.options(targets, here, here, [], set(here.cells), goal, cfg, is_solid)
    return targets, here, opts


def compare(courses, cfg=P.PlannerConfig(), every=1):
    """Stand on every platform (its x0, z0 cell) of every course; GPU scene versus perceive/options."""
    pool = CoursePool(courses, "cpu", C.GRID)
    rows, cells = [], []
    for i, c in enumerate(courses):
        for k in range(0, c.n, every):
            x0, _, z0, _, y = (int(v) for v in c.platforms[k])
            rows.append(i)
            cells.append((x0, y, z0))
    course = torch.as_tensor(rows)
    cell = torch.as_tensor(cells)
    pos = torch.stack((cell[:, 0] + 0.5, cell[:, 1] + 1.0, cell[:, 2] + 0.5), -1).to(torch.float64)
    tables = RouteTables(pool)
    sc = Scene(pool, tables, course, pos, cell, cfg)
    K = sc.kt
    # Token of the platform stood on.
    hit = ((sc.rect[..., 4] == cell[:, None, 1]) & (sc.rect[..., 0] <= cell[:, None, 0]) & (cell[:, None, 0] <= sc.rect[..., 1])
           & (sc.rect[..., 2] <= cell[:, None, 2]) & (cell[:, None, 2] <= sc.rect[..., 3]) & sc.inview)
    here = hit.to(torch.int8).argmax(1)
    excluded = torch.zeros(len(rows), K, dtype=torch.bool)
    excluded[torch.arange(len(rows)), here] = True
    ok, cost, *_ = sc.options(here, excluded)
    bad = []
    for m, (i, cl) in enumerate(zip(rows, cells)):
        targets, h, opts = cpu_view(pool, i, cl, cfg)
        gpu = [tuple(sc.rect[m, t].tolist()) for t in range(K) if sc.inview[m, t]]
        cpu = [p.rect() for p in targets]
        if gpu != cpu:
            bad.append(("view", i, cl, gpu, cpu))
            continue
        go = {tuple(sc.rect[m, t].tolist()): float(cost[m, t]) for t in range(K) if ok[m, t]}
        co = {p.rect(): c for c, p in opts}
        if set(go) != set(co) or any(abs(go[r] - co[r]) > 1e-12 for r in go):
            bad.append(("options", i, cl, go, co))
    return len(rows), bad, tables


class RoutePerception(unittest.TestCase):
    def test_gpu_scene_equals_perceive_on_validation_courses(self):
        from parkour.train import OBSTACLE_VALIDATION_FAMILY, VALIDATION_FAMILY, held_out_courses
        levels = C.load_levels()
        cs = held_out_courses(levels, 20, 10, family=VALIDATION_FAMILY)
        cs += held_out_courses(levels, 20, 10, obstacle_prob=0.5, family=OBSTACLE_VALIDATION_FAMILY)
        n, bad, tables = compare(cs)
        self.assertGreater(n, 1000)
        self.assertEqual(bad[:3], [])


def bot_parity(case, router_ckpt=None, fill="greedy", n=24, seed=7, blocked="drop", extra=()):
    """The offline bot (bridge/router.js) and the Python evaluation make the same plans, outcomes and ticks
    (``blocked``: RouteConfig.blocked; ``extra``: more courses after the n random ones)."""
    import dataclasses
    import json
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    from parkour.live import SETTLE_TICKS, load_model
    from parkour.route_env import RouteConfig, RouteTables, evaluate_routed
    from parkour.route_policy import RoutePolicy, load_router
    from parkour.route_train import router_choosers
    from parkour.paths import NODE as node
    root = Path(__file__).resolve().parents[1]
    levels = C.load_levels()
    rng = np.random.default_rng(seed)
    cs = [C.generate(rng, levels, i % len(levels), 10, obstacle_prob=0.5 if i % 2 else 0.0, seed=i) for i in range(n)]
    cs += list(extra)
    model = load_model(root / "results" / "final_policy.pt", "cpu")
    if router_ckpt is None:
        torch.manual_seed(0)
        net = RoutePolicy().eval()
    else:
        net = load_router(router_ckpt, "cpu")
    pool = CoursePool(cs, "cpu", C.GRID)
    choose, fill_fn = router_choosers(net, True)
    with torch.no_grad():
        py = evaluate_routed(model, pool, RouteTables(pool), choose, fill_fn, rcfg=RouteConfig(fill=fill, blocked=blocked),
                             settle=SETTLE_TICKS, record=True, max_ticks=3000)
    with tempfile.TemporaryDirectory() as d:
        pol, crs, out, rj = (os.path.join(d, f) for f in ("policy.json", "courses.json", "episodes.jsonl", "router.json"))
        model.export(pol)
        net.export(rj)
        with open(crs, "w") as f:
            json.dump({"courses": [c.to_json() for c in cs]}, f)
        cfg = {**dataclasses.asdict(P.PlannerConfig()), "router": rj, "fill": fill,
               **({"blocked": blocked} if blocked != "drop" else {})}
        subprocess.run([str(node), str(root / "bridge" / "offline_bot.js"), "--policy", pol, "--courses", crs, "--out", out,
                        "--settle", str(SETTLE_TICKS), "--planner", json.dumps(cfg)],
                       check=True, cwd=str(root / "bridge"), capture_output=True)
        js = [json.loads(line) for line in open(out) if line.strip()]
    case.assertEqual(len(js), len(cs))
    for i, ep in enumerate(js):
        case.assertEqual(ep["plans"], py["events"][i], f"course {i}: plans differ")
        case.assertEqual(ep["success"], bool(py["per_course"]["success"][i]), f"course {i}")
        case.assertEqual(ep["ticks"], int(py["per_course"]["ticks"][i]), f"course {i}")
    return py


class RouteBotParity(unittest.TestCase):
    def setUp(self):
        from parkour.paths import NODE
        if not NODE.exists():
            self.skipTest("portable Node runtime not installed (run setup.cmd)")

    def test_random_router_greedy_fill(self):
        bot_parity(self, fill="greedy")

    def test_random_router_policy_fill(self):
        bot_parity(self, fill="policy", seed=8)

    def test_shipped_router(self):
        from pathlib import Path
        ck = Path(__file__).resolve().parents[1] / "results" / "routing" / "router_policy.pt"
        if not ck.exists():
            self.skipTest("no shipped router")
        py = bot_parity(self, router_ckpt=ck, fill=torch.load(ck, map_location="cpu", weights_only=False)["config"]["fill"],
                        seed=9)
        self.assertGreater(sum(py["per_course"]["success"]), 18)

    def test_shipped_router_keeping_blocked_platforms(self):
        # RouteConfig(blocked="keep"): no straight-path filter, head-hitter flags kept; on single-neo courses the
        # router then chooses the neos' landings (bridge/router.js options() with cfg.blocked).
        from pathlib import Path
        from parkour.neo_assist import SINGLE_VALIDATION_FAMILY, single_neo_courses
        from parkour.neo_dagger import neo_takeoffs
        ck = Path(__file__).resolve().parents[1] / "results" / "routing" / "router_policy.pt"
        if not ck.exists():
            self.skipTest("no shipped router")
        neos = single_neo_courses(SINGLE_VALIDATION_FAMILY, n=8)
        py = bot_parity(self, router_ckpt=ck, fill=torch.load(ck, map_location="cpu", weights_only=False)["config"]["fill"],
                        n=8, seed=10, blocked="keep", extra=neos)
        landings = [tuple(int(v) for v in c.platforms[k + 1]) for c in neos for k in neo_takeoffs(c)]
        chosen = {tuple(ch[0]) for ev in py["events"][8:] for _, _, ch in ev if ch}
        self.assertTrue(set(landings) <= chosen)  # every neo's landing was chosen at least once


if __name__ == "__main__":
    unittest.main()

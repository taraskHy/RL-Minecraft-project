"""Run a trained policy on held-out courses outside the simulator, and compare.

    python -m parkour.live --checkpoint runs/NAME/checkpoints/best.pt --out runs/live-NAME
                           [--per-level 5] [--offline] [--viewer YourMinecraftName]

Live mode starts the local void-world server (parkour.server), and the
Mineflayer bot (bridge/parkour_bot.js) plays every course in real Minecraft.
Offline mode runs the same JavaScript episode loop on prismarine-physics
without a server (bridge/offline_bot.js).

Afterwards, every episode is checked two ways:
* replay: the recorded actions are re-simulated in the Python simulator from
  the recorded start state, and the positions are compared tick by tick;
* policy: the same network plays the same course in the simulator, and the
  outcomes are compared.
Results: OUT/episodes.jsonl (every tick), OUT/summary.json, OUT/summary.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import courses as C
from . import physics
from .env import YAW_STEPS_DEG, CoursePool, action_to_controls
from .model import ActorCritic
from .paths import BRIDGE, NODE, ROOT
from .train import evaluate, held_out_courses, obstacle_courses

SETTLE_TICKS = 10


def pick_courses(per_level: int, n_jumps: int = 10, pool_per_level: int = 200, obstacles: bool = False) -> list[C.Course]:
    levels = C.load_levels()
    if obstacles:
        allc = obstacle_courses(levels, pool_per_level, n_jumps)
    else:
        allc = held_out_courses(levels, pool_per_level, n_jumps)
    out = []
    for L in range(len(levels)):
        out.extend([c for c in allc if c.level == L][:per_level])
    return out


def load_model(checkpoint: Path, device) -> ActorCritic:
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    hidden = tuple(ck["config"].get("hidden", (512, 256, 128)))
    parts = tuple(p for p in ck["config"].get("hide", "").split(",") if p)
    model = ActorCritic(obs_dim=ck["model"]["norm.mean"].shape[0], hidden=hidden, hidden_parts=parts).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def replay(courses: list[C.Course], episodes: list[dict]) -> list[dict]:
    """Re-simulate recorded actions from each recorded start state."""
    out = []
    pool = CoursePool(courses, "cpu", C.GRID)
    steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64) * (math.pi / 180)
    for ep in episodes:
        if not ep["actions"]:
            out.append({"max_pos_err": None, "ticks": 0})
            continue
        i = ep["index"]
        init = ep["init"]
        s = physics.PlayerState.at(torch.tensor([init["pos"]], dtype=torch.float64), torch.tensor([init["yaw"]], dtype=torch.float64))
        s.vel = torch.tensor([init["vel"]], dtype=torch.float64)
        s.on_ground = torch.tensor([bool(init["onGround"])])
        s.jump_ticks = torch.tensor([int(init["jumpTicks"])])
        s.collided_h = torch.tensor([bool(init["collidedH"])])
        course = torch.tensor([i])
        traj = np.array(ep["traj"])
        err = []
        for t, a in enumerate(ep["actions"]):
            if t >= len(traj):
                break
            a = torch.tensor([a])
            s.yaw = torch.remainder(s.yaw + steps[a[:, 4]] + math.pi, 2 * math.pi) - math.pi
            s = physics.simulate(pool.world, course, s, action_to_controls(a))
            err.append(float(np.abs(s.pos[0].numpy() - traj[t, :3]).max()))
        err = np.array(err)
        first = int(np.argmax(err > 1e-6)) if (err > 1e-6).any() else None
        out.append({"max_pos_err": float(err.max()), "first_divergence_tick": first, "ticks": len(err)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-level", type=int, default=5)
    ap.add_argument("--obstacles", action="store_true", help="play held-out obstacle courses")
    ap.add_argument("--courses-file", default="", help="JSON {\"courses\": [...]} to play instead of held-out courses")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--viewer", default="")
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--stall", type=int, default=200)
    ap.add_argument("--style", default="stone", choices=("stone", "rainbow"),
                    help="block looks in real Minecraft (rainbow: coloured full cubes, same physics)")
    ap.add_argument("--web-viewer", type=int, default=0, help="serve a textured 3D view of the bot on this port")
    ap.add_argument("--hold", type=int, default=0, help="extra milliseconds on the start block before each course")
    ap.add_argument("--first-person", action="store_true", help="web viewer from the bot's eyes")
    ap.add_argument("--planner", action="store_true",
                    help="no course list for the policy: a greedy planner finds platforms in the blocks around the bot")
    ap.add_argument("--planner-config", default="{}", help="JSON overrides of parkour.planner.PlannerConfig")
    ap.add_argument("--fall-rule", default="local", choices=("local", "remaining"))
    ap.add_argument("--assist", default="",
                    help="JSON overrides of parkour.neo_assist.AssistConfig ('{}': defaults): the bot plans its neos "
                         "with a Python search from its exact state, checks each plan with prismarine-physics, plays it")
    a = ap.parse_args(argv)

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if a.courses_file:
        raw = json.loads(Path(a.courses_file).read_text())["courses"]
        courses = [C.Course.from_json(d) for d in raw]
    else:
        courses = pick_courses(a.per_level, obstacles=a.obstacles)
        raw = [c.to_json() for c in courses]
    course_json = [c.to_json() for c in courses]
    assist_cfg = None
    if a.assist:
        if a.planner:
            raise SystemExit("--assist needs the course list (not --planner)")
        from .neo_assist import AssistConfig, plan_target, planned_takeoffs, specs_from_keys
        assist_cfg = AssistConfig(**{"batch": 1, **json.loads(a.assist)})
        if assist_cfg.batch != 1:
            raise SystemExit("the bots reproduce only searches with batch 1")
        if not assist_cfg.enabled:
            assist_cfg = None
    if assist_cfg is not None:
        for c, d, r in zip(courses, course_json, raw):
            if not c.specs and r.get("specs"):  # a course file: its jump types, so the simulator plans where the bot does
                c.specs = specs_from_keys(r["specs"], c.n)
            d["assist_takeoffs"] = planned_takeoffs(c, assist_cfg.jumps)
            d["assist_targets"] = [plan_target(c, k) for k in d["assist_takeoffs"]]  # past a slime pad: its exit
    (out / "courses.json").write_text(json.dumps({"courses": course_json}))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(a.checkpoint).resolve(), dev)
    model.export(out / "policy.json", {"checkpoint": str(a.checkpoint)})
    episodes_path = out / "episodes.jsonl"

    assist_args = []
    if assist_cfg is not None:
        assist_args = ["--assist", json.dumps(dataclasses.asdict(assist_cfg)), "--assist-python", sys.executable,
                       "--assist-checkpoint", str(Path(a.checkpoint).resolve()), "--assist-root", str(ROOT),
                       "--assist-device", "cuda" if torch.cuda.is_available() else "cpu"]
    planner_json = ""
    if a.planner:
        from .planner import PlannerConfig
        planner_cfg = PlannerConfig(**json.loads(a.planner_config))
        planner_json = json.dumps(dataclasses.asdict(planner_cfg))
    t0 = time.time()
    if a.offline:
        subprocess.run([str(NODE), str(BRIDGE / "offline_bot.js"), "--policy", str(out / "policy.json"),
                        "--courses", str(out / "courses.json"), "--out", str(episodes_path),
                        "--settle", str(SETTLE_TICKS), "--stall", str(a.stall), "--planner", planner_json,
                        "--fall-rule", a.fall_rule] + assist_args, check=True, cwd=str(BRIDGE))
    else:
        from .server import COURSE_ORIGIN, SPAWN_PAD, Server
        with open(out / "server.log", "w") as slog, Server(a.port, log=slog):
            cmd = [str(NODE), str(BRIDGE / "parkour_bot.js"), "--policy", str(out / "policy.json"),
                   "--courses", str(out / "courses.json"), "--out", str(episodes_path), "--port", str(a.port),
                   "--origin", ",".join(map(str, COURSE_ORIGIN)), "--pad", ",".join(map(str, SPAWN_PAD)),
                   "--stall", str(a.stall), "--style", a.style, "--web-viewer", str(a.web_viewer), "--hold", str(a.hold),
                   "--first-person", "true" if a.first_person else "false", "--planner", planner_json, "--fall-rule", a.fall_rule]
            cmd += assist_args
            if a.viewer:
                cmd += ["--viewer", a.viewer]
                print(f"Join 127.0.0.1:{a.port} in Minecraft Java 1.21.4 as {a.viewer} to watch.", flush=True)
            subprocess.run(cmd, check=True, cwd=str(BRIDGE))
    wall = time.time() - t0

    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    rep = replay(courses, episodes)
    pool = CoursePool(courses, dev, C.GRID)
    if a.planner:
        from .planner import evaluate_planned
        sim = evaluate_planned(model, pool, dev, planner_cfg, a.stall, settle=SETTLE_TICKS, fall_rule=a.fall_rule)["per_course"]
    elif assist_cfg is not None:
        from .neo_assist import evaluate_assisted
        pc = evaluate_assisted(model, pool, dev, assist_cfg, a.stall, settle=SETTLE_TICKS, fall_rule=a.fall_rule)["per_course"]
        sim = {"success": [c["success"] for c in pc], "ticks": [c["ticks"] for c in pc]}
    else:
        sim = evaluate(model, pool, dev, a.stall, deterministic=True, settle=SETTLE_TICKS, per_course=True,
                       fall_rule=a.fall_rule)["per_course"]

    rows = []
    for ep, r in zip(episodes, rep):
        i = ep["index"]
        rows.append({"index": i, "name": courses[i].name, "level": ep["level"], "result": ep["result"], "platform": ep["cp"], "n": ep["n"],
                     "ticks": ep["ticks"], "server_corrections": ep.get("server_corrections"),
                     "replay_max_pos_err": r["max_pos_err"], "replay_first_divergence": r.get("first_divergence_tick"),
                     "sim_success": bool(sim["success"][i]), "sim_ticks": int(sim["ticks"][i])})
    n_levels = max(c.level for c in courses) + 1
    by_level = {}
    for L in range(n_levels):
        rl = [r for r in rows if r["level"] == L]
        if rl:
            by_level[L] = {"n": len(rl), "success": sum(r["result"] == "success" for r in rl) / len(rl),
                           "sim_success": sum(r["sim_success"] for r in rl) / len(rl)}
    errs = [r["replay_max_pos_err"] for r in rows if r["replay_max_pos_err"] is not None]
    summary = {
        "mode": "offline (prismarine-physics, no server)" if a.offline else "live (Minecraft 1.21.4 server)",
        "checkpoint": str(a.checkpoint), "courses": len(rows), "wall_seconds": round(wall, 1),
        "success": sum(r["result"] == "success" for r in rows) / max(len(rows), 1),
        "sim_success_same_courses": sum(r["sim_success"] for r in rows) / max(len(rows), 1),
        "outcome_agreement": sum((r["result"] == "success") == r["sim_success"] for r in rows) / max(len(rows), 1),
        "replay_max_pos_err": max(errs) if errs else None,
        "replay_exact_episodes": sum(e is not None and e < 1e-6 for e in errs),
        "by_level": by_level, "episodes": rows,
    }
    if assist_cfg is not None:
        log = [x for ep in episodes for x in ep.get("assist", [])]
        summary["assist"] = {"config": dataclasses.asdict(assist_cfg), "requests": len(log),
                             "found": sum(bool(x.get("found")) for x in log), "verified": sum(bool(x.get("verified")) for x in log),
                             "played": sum(bool(x.get("played")) for x in log),
                             "planner_seconds": round(sum(x.get("seconds") or 0 for x in log), 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = [f"# {summary['mode']}", "", f"Checkpoint `{a.checkpoint}`, {len(rows)} courses "
             f"({'from ' + a.courses_file if a.courses_file else 'held-out' + (' with obstacles' if a.obstacles else '') + ', 10 jumps each'}).", "",
             "| Level | Courses | Success here | Success in simulator (same courses) |", "| --- | --- | --- | --- |"]
    for L, v in by_level.items():
        lines.append(f"| {L} | {v['n']} | {v['success']:.0%} | {v['sim_success']:.0%} |")
    lines += ["", f"Overall: {summary['success']:.0%} here, {summary['sim_success_same_courses']:.0%} in the simulator; "
              f"outcomes agree on {summary['outcome_agreement']:.0%} of courses.",
              f"Replaying the recorded actions in the simulator: largest position difference "
              f"{summary['replay_max_pos_err']:.2e} blocks; {summary['replay_exact_episodes']} of {len(errs)} episodes "
              f"within 1e-6 blocks on every tick."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

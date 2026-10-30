"""Where a real server disagrees with the bot past wall corners: every real-Minecraft run in results/games.

    python scripts/corner_scan.py results/corner_scan.json

prismarine-physics (the simulator and the bots) resolves horizontal collisions x then z; the game resolves the larger
component first. Each recorded episode is replayed in the simulator (as parkour.live's replay check does), and every
tick is checked two ways, both more than 0.25 blocks apart (the server's limit) counting as flagged:
  velocity      the tick simulated again in the game's order (physics.simulate(..., larger_first=True)), what the neo
                planner's corner_limit uses;
  displacement  the bot's own displacement re-applied in the game's order (physics._move(..., nxt - pos, ...,
                larger_first=True)), as the server's player.move does.
A displacement-flagged tick whose box starts clear of every block (not flush on a face: the server's box is 0.3F wide
and overlaps a flush face by 1.2e-8) is a predicted send-back. Each run's counts sit next to its server log's "moved
wrongly" warnings and the episodes the bot counted server corrections in. Runs with a learned router are skipped.
"""
import argparse
import gzip
import json
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour import courses as C  # noqa: E402
from parkour import physics  # noqa: E402
from parkour.env import YAW_STEPS_DEG, CoursePool, action_to_controls  # noqa: E402
from parkour.neo_assist import courses_from_file  # noqa: E402

LIMIT = 0.25


def clear_of_blocks(pos, cells) -> bool:
    """The player's box at ``pos`` (prismarine's half-width 0.3) touches no solid cell beside it."""
    x, y, z = pos
    w, h = physics.HALF_WIDTH, physics.PLAYER_HEIGHT
    gaps = [max(max(bx - (x + w), (x - w) - (bx + 1)), max(bz - (z + w), (z - w) - (bz + 1)))
            for bx, by, bz in cells if y + h > by and y < by + 1]
    return min(gaps, default=1.0) > 1e-9


def scan_run(d: Path) -> dict:
    courses = courses_from_file(json.loads((d / "courses.json").read_text())["courses"])
    log = gzip.open(d / "server.log.gz", "rt", errors="replace").read()
    eps = [json.loads(x) for x in gzip.open(d / "episodes.jsonl.gz", "rt") if x.strip()]
    pool = CoursePool(courses, "cpu", C.GRID)
    steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64) * (math.pi / 180)
    by_velocity, by_displacement, send_back = [], [], set()
    for ep in eps:
        if not ep.get("actions"):
            continue
        i, init = ep["index"], ep["init"]
        cells = [tuple(int(v) for v in e) for e in courses[i].extra]
        for p in courses[i].platforms:
            cells += [(x, int(p[4]), z) for x in range(int(p[0]), int(p[1]) + 1) for z in range(int(p[2]), int(p[3]) + 1)]
        s = physics.PlayerState.at(torch.tensor([init["pos"]], dtype=torch.float64),
                                   torch.tensor([init["yaw"]], dtype=torch.float64))
        s.vel = torch.tensor([init["vel"]], dtype=torch.float64)
        s.on_ground = torch.tensor([bool(init["onGround"])])
        s.jump_ticks = torch.tensor([int(init["jumpTicks"])])
        course = torch.tensor([i])
        for t, a in enumerate(ep["actions"][:len(ep["traj"])]):
            a = torch.tensor([a])
            s.yaw = torch.remainder(s.yaw + steps[a[:, 4]] + math.pi, 2 * math.pi) - math.pi
            controls = action_to_controls(a)
            game = physics.simulate(pool.world, course, s, controls, larger_first=True)
            nxt = physics.simulate(pool.world, course, s, controls)
            server = physics._move(pool.world, course, s.pos, nxt.pos - s.pos, s.on_ground, False, larger_first=True)[0]
            if float(torch.linalg.norm(nxt.pos[0, [0, 2]] - game.pos[0, [0, 2]])) > LIMIT:
                by_velocity.append([i, t + 1])
            if float(torch.linalg.norm(nxt.pos[0, [0, 2]] - server[0, [0, 2]])) > LIMIT:
                by_displacement.append([i, t + 1])
                if clear_of_blocks(s.pos[0].tolist(), cells):
                    send_back.add(i)
            s = nxt
    return {"episodes": len(eps), "server_warnings": len(re.findall(r"moved wrongly|moved too quickly", log)),
            "flagged_by_velocity": by_velocity, "flagged_by_displacement": by_displacement,
            "predicted_send_backs": sorted(send_back),
            "corrected_episodes": sorted(e["index"] for e in eps if e.get("server_corrections"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    torch.set_num_threads(1)
    runs = {}
    for d in sorted((ROOT / "results" / "games").iterdir()):
        s = d / "summary.json"
        if not (s.exists() and (d / "server.log.gz").exists() and (d / "episodes.jsonl.gz").exists()):
            continue
        summary = json.loads(s.read_text())
        if not str(summary.get("mode", "")).startswith("live") or summary.get("router"):
            continue
        runs[d.name] = scan_run(d)
        r = runs[d.name]
        print(d.name, r["server_warnings"], len(r["flagged_by_velocity"]), len(r["flagged_by_displacement"]),
              r["predicted_send_backs"], r["corrected_episodes"], flush=True)
    total = {"runs": len(runs),
             "warnings_equal_displacement_flags": sum(r["server_warnings"] == len(r["flagged_by_displacement"])
                                                      for r in runs.values()),
             "warnings_equal_velocity_flags": sum(r["server_warnings"] == len(r["flagged_by_velocity"])
                                                  for r in runs.values()),
             "send_backs_equal_corrections": sum(r["predicted_send_backs"] == r["corrected_episodes"]
                                                 for r in runs.values())}
    a.out.write_text(json.dumps({"measured": "simulator replay of real-Minecraft recordings (CPU)", "limit": LIMIT,
                                 "summary": total, "runs": runs}, indent=1))
    print(json.dumps(total))


if __name__ == "__main__":
    main()

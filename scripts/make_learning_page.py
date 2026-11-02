"""Build the technical page on how the agent learned ("How the Parkour Bot Learned", an artifact of 2 Nov 2026).

    .venv/Scripts/python.exe scripts/make_learning_page.py [OUT.html]     (default results/learning.html)

The page is scripts/learning_page_template.html with two data blocks filled in:

* __LEARNING__: the main run's held-out curves (results/runs/main-g995/eval_log.csv per level, eval_val_log.csv),
  every tenth row of its train_log.csv (success, curriculum frontier, entropy, return) and the reward and curriculum
  ablation table of the generated results/README.md;
* __TRACES__: the per-tick reward terms of two real-game recordings, stepped again through parkour.env.ParkourEnv
  (CPU, float64, the default RewardConfig) from the recorded start state with the recorded actions: the first
  level-5 course the final policy made in results/games/v2-live, and the hand-designed neo course of
  results/games/v2-live-showcase (it stood still until the stall rule). The replayed positions are checked against
  the recording; a difference above 1e-9 blocks stops the build.
"""
import csv
import gzip
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch  # noqa: E402

from parkour import courses as C  # noqa: E402
from parkour import physics  # noqa: E402
from parkour.env import CoursePool, EnvConfig, ParkourEnv  # noqa: E402

RUN = REPO / "results" / "runs" / "main-g995"
TRACES = [("v2-live", "level5"), ("v2-live-showcase", 3)]
ARMS = "full|sparse|nolanding|nofall|uniform|noroute"


def learning_data() -> dict:
    ev = list(csv.DictReader(open(RUN / "eval_log.csv")))
    curve = [{"it": int(r["iteration"]) + 1, "ticks": int(r["env_steps"]), "all": float(r["success"]),
              **{f"L{k}": float(r[f"success_L{k}"]) for k in range(6)}} for r in ev]
    val = [{"it": int(r["iteration"]) + 1, "all": float(r["success"])} for r in csv.DictReader(open(RUN / "eval_val_log.csv"))]
    train = [{"it": int(r["iteration"]) + 1, "ticks": int(r["env_steps"]), "success": float(r["success"]),
              "frontier": int(r["frontier"]), "entropy": float(r["entropy"]), "ret": float(r["ep_return"]),
              "len": float(r["ep_len"]), "fell": float(r["fell"])}
             for i, r in enumerate(csv.DictReader(open(RUN / "train_log.csv"))) if i % 10 == 9 or i == 0]
    abl = []
    row = re.compile(rf"\| ({ARMS}) \| (.*?) \| 3 \| ([\d.]+) \(([\d.]+)-([\d.]+)\) \| ([\d.]+) \| ([\d.]+) \(([\d.]+)-([\d.]+)\) \| ([\d.]+) \| ([\d.]+) \|")
    for line in open(REPO / "results" / "README.md", encoding="utf-8"):
        m = row.match(line)
        if m:
            g = m.groups()
            abl.append({"arm": g[0], "what": g[1], "final": float(g[2]), "fmin": float(g[3]), "fmax": float(g[4]),
                        "l5": float(g[6]), "l5min": float(g[7]), "l5max": float(g[8]), "mean": float(g[9]), "to80": float(g[10])})
    if len(abl) != 6:
        raise SystemExit(f"expected the 6 ablation arms in results/README.md, found {len(abl)}")
    return {"curve": curve, "val": val, "train": train, "ablation": abl}


def trace(run: str, which) -> dict:
    d = REPO / "results" / "games" / run
    cs = json.loads((d / "courses.json").read_text())["courses"]
    eps = [json.loads(l) for l in gzip.open(d / "episodes.jsonl.gz", "rt") if l.strip()]
    if which == "level5":
        e = next(e for e in eps if cs[e["index"]]["level"] == 5 and e["result"] == "success")
    else:
        e = eps[which]
    cj = cs[e["index"]]
    pool = CoursePool([C.Course.from_json(cj)], "cpu", C.GRID)
    env = ParkourEnv(pool, EnvConfig(num_envs=1))
    env.reset_all(torch.zeros(1, dtype=torch.int64))
    i = e["init"]
    st = physics.PlayerState.at(torch.tensor([i["pos"]], dtype=torch.float64), torch.tensor([i["yaw"]], dtype=torch.float64))
    st.vel = torch.tensor([i["vel"]], dtype=torch.float64)
    st.on_ground = torch.tensor([bool(i["onGround"])])
    st.jump_ticks = torch.tensor([int(i["jumpTicks"])], dtype=st.jump_ticks.dtype)
    st.collided_h = torch.tensor([bool(i["collidedH"])])
    env.state = st
    env.dist = env._distance(env.state.pos, env.course, env.cp)
    rows, worst = [], 0.0
    last = len(cj["platforms"]) - 1
    for t, a in enumerate(e["actions"]):
        cp0, d0 = int(env.cp[0]), float(env.dist[0])
        _, rew, _, _, _, info = env.step(torch.tensor([a], dtype=torch.int64))
        r = float(rew[0])
        succ, fell = bool(info["success"][0]), bool(info["fell"][0])
        if not bool(info["done"][0]):
            gained, prog = int(env.cp[0]) - cp0, d0 - float(env.dist[0])
            worst = max(worst, max(abs(p - q) for p, q in zip(env.state.pos[0].tolist(), e["traj"][t][:3])))
        else:  # the row has restarted inside step; the last tick's terms follow from its reward
            gained = last - cp0 if succ else 0
            prog = r - gained - 10.0 * succ + 5.0 * fell + 0.01
        rows.append({"t": t + 1, "progress": round(prog, 5), "landing": gained, "goal": 10.0 if succ else 0.0,
                     "fall": -5.0 if fell else 0.0, "time": -0.01, "total": round(r, 5), "cp": cp0 + gained,
                     "y": round(e["traj"][t][1], 3)})
        if bool(info["done"][0]):
            break
    if worst > 1e-9 or len(rows) != e["ticks"]:
        raise SystemExit(f"{run} course {e['index']}: the replay does not follow the recording ({worst:.1e} blocks, "
                         f"{len(rows)} of {e['ticks']} ticks)")
    out = {"run": run, "index": e["index"], "result": e["result"], "level": cj["level"], "name": cj.get("name") or "",
           "platforms": len(cj["platforms"]), "ticks": e["ticks"], "replay_max_pos_diff": worst,
           "return": round(sum(r["total"] for r in rows), 4), "rows": rows}
    print(f"{run} course {e['index'] + 1}: {e['result']}, {len(rows)} ticks, return {out['return']}, replay {worst:.1e} blocks")
    return out


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "learning.html"
    learning = json.dumps(learning_data()).replace("</", "<\\/")
    traces = json.dumps([trace(r, w) for r, w in TRACES]).replace("</", "<\\/")
    page = (REPO / "scripts" / "learning_page_template.html").read_text(encoding="utf-8")
    page = page.replace("__LEARNING__", learning).replace("__TRACES__", traces)
    out.write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()

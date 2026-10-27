"""Build a standalone 3D replay page from recorded episodes.

    python scripts/make_replay_viewer.py OUT.html RUN_DIR [RUN_DIR ...]

Each RUN_DIR is the output of parkour.live (courses.json and episodes.jsonl,
offline or live). The page embeds the courses and every tick's position and
yaw, and replays them with three.js.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parkour.env import YAW_STEPS_DEG  # noqa: E402


def wrap(y):
    return (y + math.pi) % (2 * math.pi) - math.pi


def episodes_from(run: Path) -> list[dict]:
    courses = json.loads((run / "courses.json").read_text())["courses"]
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    mode = summary.get("mode", "")
    out = []
    for line in (run / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        c = courses[e["index"]]
        init = e["init"]
        pos = [init["pos"]] + [row[:3] for row in e["traj"]]
        yaw = [init["yaw"]]
        for a in e["actions"][: len(e["traj"])]:
            yaw.append(wrap(yaw[-1] + math.radians(YAW_STEPS_DEG[a[4]])))
        # Platform index reached at each tick (for the HUD), from on-ground positions.
        plats = c["platforms"]
        cp, cp_at = 0, [0]
        for row in e["traj"]:
            x, y, z, on = row[0], row[1], row[2], row[6]
            if on:
                for k, (x0, x1, z0, z1, py) in enumerate(plats):
                    if k > cp and abs(y - (py + 1)) < 1e-6 and x + 0.3 > x0 and x - 0.3 < x1 + 1 and z + 0.3 > z0 and z - 0.3 < z1 + 1:
                        cp = k
            cp_at.append(cp)
        name = c.get("name") or f"Level {c['level']} course {c['seed']}"
        group = c.get("name") and "hand-designed" or f"L{c['level']}"
        out.append({
            "title": name.capitalize() if c.get("name") else name, "group": group, "result": e["result"], "cp": e["cp"],
            "n": e["n"], "ticks": e["ticks"], "platforms": plats, "mode": mode,
            "pos": [round(v, 3) for p in pos for v in p], "yaw": [round(v, 4) for v in yaw], "cpAt": cp_at,
        })
    return out


def main():
    out = Path(sys.argv[1])
    runs = [Path(p) for p in sys.argv[2:]]
    eps = []
    for r in runs:
        eps.extend(episodes_from(r))
    ok = sum(e["result"] == "success" for e in eps)
    modes = sorted({e["mode"] for e in eps if e["mode"]})
    data = {
        "subtitle": "A PPO policy plays Minecraft parkour courses it never saw in training. Pick a course to watch it, tick by tick.",
        "meta": f"{ok}/{len(eps)} courses completed · " + "; ".join(modes),
        "notes": ("Trajectories come from the bot's JavaScript stack running on prismarine-physics, the engine a Mineflayer bot "
                  "moves with in real Minecraft (<code>run.cmd offline</code>). Start block in green, goal in gold. "
                  "Space plays or pauses; the arrow keys step one game tick."),
        "episodes": eps,
    }
    html = (ROOT / "scripts" / "replay_template.html").read_text(encoding="utf-8")
    html = html.replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
    out.write_text(html, encoding="utf-8")
    print("wrote", out, f"{len(eps)} episodes, {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

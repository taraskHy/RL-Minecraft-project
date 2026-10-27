"""Record a training run's snapshots on the same courses and build a "learning progress" page.

    python scripts/make_progress_page.py RUN_NAME OUT.html [--live 0,10,50,400]

RUN_NAME must have been trained with --save-at (checkpoints/updates_*.pt). Every
snapshot plays the same 7 courses (the hand-designed "tour" and the first
held-out test course of each level 0-5) through the bot's JavaScript on
prismarine-physics. With --live, the listed snapshots also play 3 of those
courses on the real Minecraft server. Every episode is kept, failures
included, under runs/RUN_NAME-games/.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from parkour import courses as C  # noqa: E402
from parkour.train import held_out_courses  # noqa: E402
import make_replay_viewer as V  # noqa: E402

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--live", default="", help="snapshot update counts to also play in real Minecraft")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    games = ROOT / "runs" / f"{a.run}-games"
    games.mkdir(exist_ok=True)
    snaps = sorted(run.glob("checkpoints/updates_*.pt"))
    showcase = [C.Course.from_json(d) for d in json.loads((ROOT / "configs" / "showcase_courses.json").read_text())["courses"]
                if d.get("name") == "tour"]
    held = held_out_courses(C.load_levels(), 200, 10)
    per_level = [next(c for c in held if c.level == L) for L in range(6)]
    courses = showcase + per_level
    (games / "courses.json").write_text(json.dumps({"courses": [c.to_json() for c in courses]}))
    live_courses = [courses[0], courses[3], courses[5]]
    (games / "courses_live.json").write_text(json.dumps({"courses": [c.to_json() for c in live_courses]}))
    live = {int(v) for v in a.live.split(",") if v.strip()}

    episodes = []
    for snap in snaps:
        updates = int(snap.stem.split("_")[1])
        ticks = updates * 32 * 16384
        stage = "untrained" if updates == 0 else f"after {updates} updates ({ticks / 1e6:.0f} M ticks)"
        for mode, cfile in (("offline", "courses.json"), ("live", "courses_live.json")):
            if mode == "live" and updates not in live:
                continue
            dst = games / f"{mode}_{updates:05d}"
            if not (dst / "summary.json").exists():
                cmd = [PY, "-m", "parkour.live", "--checkpoint", str(snap), "--out", str(dst), "--courses-file",
                       str(games / cfile)]
                if mode == "offline":
                    cmd.append("--offline")
                subprocess.run(cmd, check=True, cwd=ROOT)
            for e in V.episodes_from(dst):
                e["group"] = stage + (" - real Minecraft" if mode == "live" else "")
                e["title"] = f"{e['title']} - {stage}" + (" (real Minecraft)" if mode == "live" else "")
                episodes.append(e)
        ok = sum(1 for e in episodes if e["group"].startswith(stage) and e["result"] == "success")
        print(f"{stage}: {ok} courses completed", flush=True)

    data = {
        "subtitle": "The same courses played by the bot at successive stages of training, from an untrained network "
                    "that falls at once to the trained policy. Every attempt is kept, failures included.",
        "meta": f"{len(snaps)} snapshots of run {a.run}",
        "notes": ("Offline episodes run the bot's JavaScript on prismarine-physics, the engine a Mineflayer bot moves "
                  "with; groups marked real Minecraft were played on the local 1.21.4 server. Start block in green, "
                  "goal in gold. Space plays or pauses; the arrow keys step one game tick."),
        "episodes": episodes,
    }
    html = (ROOT / "scripts" / "replay_template.html").read_text(encoding="utf-8")
    html = html.replace("<title>Parkour Agent Replays</title>", "<title>Parkour Learning Progress</title>")
    html = html.replace("<h1>Parkour agent replays</h1>", "<h1>Parkour learning progress</h1>")
    html = html.replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
    Path(a.out).write_text(html, encoding="utf-8")
    print("wrote", a.out, len(episodes), "episodes")


if __name__ == "__main__":
    main()

"""Build the Minecraft-style "theatre" page: ghost races and real-game replays.

    python scripts/make_theatre.py OUT.html

Ghost races: every snapshot of the progress run (runs/progress-demo-games)
plays the same course at the same time, one coloured runner per training
stage, from the untrained network to the trained one. Replays: the final
policy's episodes in real Minecraft (runs/live2-*), and any neo runs found
under runs/neo-live*.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_replay_viewer as V  # noqa: E402


def main():
    out = Path(sys.argv[1])
    courses, course_ids, runs, races, replays = [], {}, [], [], []

    def course_id(e):
        key = json.dumps([e["platforms"], e.get("extra", [])])
        if key not in course_ids:
            course_ids[key] = len(courses)
            courses.append({"platforms": e["platforms"], "extra": e.get("extra", [])})
        return course_ids[key]

    def run_of(e, label, real):
        return {"course": course_id(e), "label": label, "result": e["result"], "cp": e["cp"], "n": e["n"],
                "ticks": e["ticks"], "pos": e["pos"], "yaw": e["yaw"], "cpAt": e["cpAt"], "ground": e["ground"],
                "real": real}

    games = ROOT / "runs" / "progress-demo-games"
    stages = sorted(games.glob("offline_*"))
    by_course = {}
    for d in stages:
        updates = int(d.name.split("_")[1])
        label = "untrained" if updates == 0 else f"{updates} updates"
        for i, e in enumerate(V.episodes_from(d)):
            runs.append(run_of(e, label, False))
            by_course.setdefault(i, (e["title"], []))[1].append(len(runs) - 1)
    for i, (title, idx) in sorted(by_course.items()):
        races.append({"name": f"{title}: all {len(idx)} training stages", "course": runs[idx[0]]["course"], "runs": idx})

    for d in [ROOT / "runs" / n for n in ("live2-showcase", "live2-final", "live2-obstacles")] + sorted(ROOT.glob("runs/neo-live*")):
        if (d / "episodes.jsonl").exists():
            for e in V.episodes_from(d):
                replays.append(run_of(e, e["title"], "live" in e["mode"]))
    data = {
        "title": "Parkour bot theatre",
        "subtitle": "Ghost race: one runner per training stage on the same unseen course, from the untrained network to "
                    "the trained policy. Replays: the trained bot in real Minecraft 1.21.4.",
        "notes": ("Every runner follows a recorded trajectory tick by tick; nothing is animated by hand. Ghost-race runs "
                  "come from the bot's JavaScript on prismarine-physics (Mineflayer's own engine); replays marked real "
                  "Minecraft were played on a local server. Platforms are coloured from the green start to the gold "
                  "goal; obstacles are quartz."),
        "courses": courses, "runs": runs, "races": races, "replays": replays,
    }
    html = (ROOT / "scripts" / "replay_mc_template.html").read_text(encoding="utf-8")
    html = html.replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
    out.write_text(html, encoding="utf-8")
    print("wrote", out, f"{len(races)} races, {len(runs)} race runs, {len(replays)} replays, {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

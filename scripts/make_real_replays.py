"""Build the real-Minecraft replay page from results/games/ (recordings of parkour.live, real game only).

    .venv/Scripts/python.exe scripts/make_real_replays.py [OUT.html]     (default results/real_replays.html)

Published as the artifact "Real-Minecraft Replays" (2 Nov 2026). The page is scripts/real_replays_template.html with
the data in place of __REPLAY_DATA__; it loads three.js 0.128.0 from jsDelivr.

Every course of each listed run is included, failures too. Per tick: position, yaw (from the recorded actions, as
scripts/make_replay_viewer.py does), on-ground, the platform reached, and which network acted (the skill on the
course list's skill platforms, parkour.neo_assist.evaluate_assisted's switch); the switch is checked against the
run's own skill tick count, so a mismatch stops the build.
"""
import gzip
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "real_replays.html"
sys.path.insert(0, str(REPO))
from parkour.env import YAW_STEPS_DEG  # noqa: E402

GAMES = REPO / "results" / "games"
SETS = [
    {"id": "hand", "title": "Hand-designed", "runs": [("v2-live-showcase", "final policy, course list"),
                                                     ("film-tour-1031-033109", "final policy, learned router"),
                                                     ("film-stairs-1031-033230", "final policy, learned router"),
                                                     ("film-sprint-1031-033315", "final policy, learned router")],
     "note": "Four courses designed by hand. The course list tells the bot which platform comes next; with the learned "
             "router it finds the route itself. The fourth course is built around neo jumps, which the final policy "
             "never trained on."},
    {"id": "standard", "title": "Standard", "runs": [("v2-live", "final policy")],
     "note": "The first 60 standard test courses, levels 0 to 5 (gaps, height changes, turns), never seen in training."},
    {"id": "obstacles", "title": "Obstacles", "runs": [("v2-live-obstacles", "final policy")],
     "note": "The first 60 obstacle test courses: head-hitter beams over the flight path and decoy pillars."},
    {"id": "oneneo", "title": "One neo", "runs": [("single41-live-skill-d4", "final policy + neo skill (d4)")],
     "note": "Courses of 10 jumps with exactly one neo, a jump around the end of a wall. The neo skill, a second "
             "network, plays the neo's block; the final policy plays the rest. Played with the neo skill shipped on 31 Oct."},
    {"id": "twoneo", "title": "Two neos", "runs": [("two53-live-skill-t1", "final policy + neo skill (t1)")],
     "note": "Courses with two neos: back to back (the first neo's target is the second's takeoff), one jump apart or "
             "two apart. The neo skill retrained on two-neo blocks plays both."},
    {"id": "slime", "title": "Slime", "runs": [("slime56-live-skill-sp3c", "final policy + slime skill")],
     "note": "Slime-heavy courses: four slime bounces each, a pad and an exit up to three blocks above it. The slime "
             "skill plays each pad's block; the final policy plays the rest."},
    {"id": "water", "title": "Water", "runs": [("water101-live-skill-w1-u300", "final policy + water skill")],
     "note": "Courses with two pools each: a drop into a pool, a column to swim up or a gap to swim across, in stone "
             "basins the bot fills with water. The water skill plays each pool's block."},
    {"id": "skills", "title": "One network", "runs": [("skill-replays-live-65", "one network (combined policy)")],
     "note": "The one network that learned every skill, on fresh courses: head hitters, ice run-ups, shaped blocks "
             "(slabs, trapdoors, fence posts, end rods ...), slime bounces, ladders and the advanced levels."},
]


def wrap(y):
    return (y + math.pi) % (2 * math.pi) - math.pi


def tags_of(c):
    specs = c.get("specs") or []
    t = []
    neo = sum(1 for s in specs if "_neo" in s)
    if neo:
        t.append(f"{neo} neo" + ("s" if neo > 1 else ""))
    sl = len(c.get("slime") or [])
    if sl:
        t.append(f"{sl} slime")
    if c.get("water"):
        t.append("water")
    if c.get("ice"):
        t.append("ice")
    if c.get("shapes"):
        t.append("shaped blocks")
    if c.get("ladders"):
        t.append("ladder")
    if any("_hh" in s for s in specs):
        t.append("head hitter")
    if c.get("extra") and not neo and not c.get("water") and not c.get("ladders") and not any("_hh" in s for s in specs):
        t.append("obstacles")
    return t


def platform_reached(c, traj, shape_top):
    plats = c["platforms"]
    shapes = {int(k): v for k, v in (c.get("shapes") or {}).items()}
    tops = [p[4] + shape_top.get(shapes.get(k, "full"), 1.0) for k, p in enumerate(plats)]
    cp, out = 0, [0]
    for row in traj:
        x, y, z, on = row[0], row[1], row[2], row[6]
        if on:
            for k, (x0, x1, z0, z1, py) in enumerate(plats):
                if k > cp and abs(y - tops[k]) < 1e-6 and x + 0.3 > x0 and x - 0.3 < x1 + 1 and z + 0.3 > z0 and z - 0.3 < z1 + 1:
                    cp = k
        out.append(cp)
    return out


def main():
    shapes = {s["name"]: s["boxes"] for s in json.loads((REPO / "configs" / "block_shapes.json").read_text())["shapes"]}
    shape_top = {k: max(b[4] for b in v) for k, v in shapes.items()}
    used_shapes = set()
    out_sets = []
    for st in SETS:
        courses, eps = [], []
        made = n = same = corr = 0
        err = 0.0
        for run, who in st["runs"]:
            d = GAMES / run
            cs = json.loads((d / "courses.json").read_text())["courses"]
            summ = json.loads((d / "summary.json").read_text())
            sep = summ["episodes"]
            skill = bool(summ.get("skill")) and not summ.get("router")
            lines = [json.loads(l) for l in gzip.open(d / "episodes.jsonl.gz", "rt") if l.strip()]
            assert len(lines) == len(sep), run
            base = len(courses)
            for c in cs:
                cc = {k: c[k] for k in ("platforms", "extra", "water", "slime", "ice", "shapes", "ladders", "level", "name")
                      if c.get(k) not in (None, [], {}, "")}
                cc["level"] = c.get("level")
                cc["tags"] = tags_of(c)
                cc["jumps"] = len(c["platforms"]) - 1
                if c.get("skill_cps") and skill:
                    cc["skill"] = c["skill_cps"]
                for s in (c.get("shapes") or {}).values():
                    used_shapes.add(s)
                for lad in c.get("ladders") or []:
                    used_shapes.add("ladder_" + ("north", "east", "south", "west")[lad[3]])
                courses.append(cc)
            for e, s in zip(lines, sep):
                c = cs[e["index"]]
                init = e["init"]
                pos = [init["pos"]] + [r[:3] for r in e["traj"]]
                yaw = [init["yaw"]]
                for a in e["actions"][: len(e["traj"])]:
                    yaw.append(wrap(yaw[-1] + math.radians(YAW_STEPS_DEG[a[4]])))
                cp_at = platform_reached(c, e["traj"], shape_top)
                assert cp_at[-1] == e["cp"] or e["result"] != "success", (run, e["index"], cp_at[-1], e["cp"])
                ep = {"c": base + e["index"], "who": who, "run": run, "result": e["result"], "cp": e["cp"], "ticks": e["ticks"],
                      "pos": [round(v * 1000) for p in pos for v in p], "yaw": [round(v * 1e4) for v in yaw],
                      "ground": "".join("1" if r[6] else "0" for r in e["traj"]), "cpAt": cp_at,
                      "simSame": s.get("sim_success") == (e["result"] == "success"), "simTicks": s.get("sim_ticks"),
                      "corr": s.get("server_corrections") or 0, "err": s.get("replay_max_pos_err")}
                if skill and c.get("skill_cps"):
                    sk = set(c["skill_cps"])
                    ctrl = "".join("1" if cp_at[t] in sk else "0" for t in range(len(e["actions"])))
                    assert ctrl.count("1") == e["skill_ticks"], (run, e["index"], ctrl.count("1"), e["skill_ticks"])
                    ep["ctrl"] = ctrl
                eps.append(ep)
                n += 1
                made += e["result"] == "success"
                same += ep["simSame"]
                corr += ep["corr"]
                err = max(err, ep["err"] or 0.0)
        out_sets.append({"id": st["id"], "title": st["title"], "note": st["note"], "runs": [r for r, _ in st["runs"]],
                         "courses": courses, "episodes": eps,
                         "totals": {"made": made, "n": n, "same": same, "corr": corr, "err": err}})
        print(f"{st['id']:10s} {made:3d}/{n:3d} made, simulator outcome same on {same}/{n}, corrections {corr}, "
              f"replay err {err:.1e}")
    data = {"sets": out_sets, "shapes": {k: shapes[k] for k in sorted(used_shapes)}}
    blob = json.dumps(data, separators=(",", ":"))
    page = (REPO / "scripts" / "real_replays_template.html").read_text(encoding="utf-8").replace("__REPLAY_DATA__", blob.replace("</", "<\\/"))
    OUT.write_text(page, encoding="utf-8")
    print(f"data {len(blob) / 1e6:.2f} MB, page {OUT.stat().st_size / 1e6:.2f} MB, shapes {sorted(used_shapes)}")


if __name__ == "__main__":
    main()

"""Neo preview: optimizer proofs of double neos, and the learning bot on neo courses.

    python scripts/make_neo_demo.py CHECKPOINT OUT_DIR

1. For a few double-neo layouts, parkour.cem searches tick-by-tick action
   sequences and records the first one that lands (proof that the jump is
   possible in Minecraft's physics; open-loop, not the learned policy).
2. The policy in CHECKPOINT plays the hand-designed neo course and held-out
   level-6 courses that contain neos, through the bot's JavaScript on
   prismarine-physics (run.cmd offline).
Writes OUT_DIR/proofs.json (viewer episodes) and OUT_DIR/courses.json.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour import courses as C  # noqa: E402
from parkour.cem import demo  # noqa: E402
from parkour.train import held_out_courses  # noqa: E402

PROOFS = [
    ("Double neo, flat", C.JumpSpec(1, 0, 0, (1, 1), neo=2, side=1)),
    ("Double neo, one block up", C.JumpSpec(1, 1, 0, (1, 1), neo=2, side=1)),
    ("Double neo over a 3-block gap, landing one lower", C.JumpSpec(3, -1, 0, (1, 1), neo=2, side=1)),
]


def main():
    ck, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    proofs = []
    for title, spec in PROOFS:
        r = demo(spec)
        if r is None:
            print("no landing found for", title)
            continue
        c = r["course"]
        pos = [round(v, 3) for p in r["pos"] for v in p]
        ticks = len(r["yaw"]) - 1
        proofs.append({"title": title + " (optimizer proof)", "group": "double neo proofs", "result": "success",
                       "cp": 1, "n": 2, "ticks": ticks, "platforms": c.platforms.tolist(), "extra": c.extra.tolist(),
                       "mode": "trajectory optimizer on the simulator", "pos": pos,
                       "yaw": [round(v, 4) for v in r["yaw"]], "cpAt": [0] * ticks + [1]})
        print("proof:", title, ticks, "ticks", flush=True)
    (out / "proofs.json").write_text(json.dumps(proofs))

    levels = C.load_levels(advanced=True)
    showcase = [C.Course.from_json(d) for d in json.loads((ROOT / "configs" / "showcase_courses.json").read_text())["courses"]
                if d.get("name") == "neo"]
    held = [c for c in held_out_courses(levels, 200, 10) if c.level == 6 and any(s.neo for s in c.specs)][:12]
    courses = showcase + held
    (out / "courses.json").write_text(json.dumps({"courses": [c.to_json() for c in courses]}))
    run = out / "bot"
    subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "parkour.live", "--checkpoint", ck, "--out",
                    str(run), "--offline", "--courses-file", str(out / "courses.json")], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()

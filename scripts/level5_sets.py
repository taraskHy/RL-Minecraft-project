"""Level 5 on 2,000 courses each: family 46 (obstacles on half the jumps) and family 47 (none).

    python scripts/level5_sets.py NAME=CHECKPOINT [NAME=CHECKPOINT ...] [--out results/level5_control.json]

The validation-only sets of the 30 Oct level-5 rule (HANDOFF section 3, "Level 5, a closer look"):
level-5 courses of 10 jumps from seed family 46 (obstacle_prob 0.5) and 47 (no obstacles), generated as
``train.held_out_courses`` generates its levels; simulator, deterministic first episode, stall 200, on
CUDA when available. Prints and writes the number completed of each set, and each course's outcome for
paired comparisons. results/level5_closer_look.json has the final policy's 1,704 and 1,831.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour import courses as C  # noqa: E402
from parkour.env import CoursePool  # noqa: E402
from parkour.live import load_model  # noqa: E402
from parkour.train import evaluate  # noqa: E402

SETS = {46: 0.5, 47: 0.0}  # family: obstacle_prob
LEVEL, N, JUMPS, STALL, BATCH = 5, 2000, 10, 200, 256


def level5_courses(family: int) -> list[C.Course]:
    levels = C.load_levels()
    rng = np.random.default_rng([2026, family, LEVEL])
    return [C.generate(rng, levels, LEVEL, JUMPS, seed=i, obstacle_prob=SETS[family]) for i in range(N)]


def completed(model, courses, device) -> list[bool]:
    out = []
    for start in range(0, len(courses), BATCH):
        pool = CoursePool(courses[start:start + BATCH], device, C.GRID)
        out += evaluate(model, pool, device, STALL, per_course=True)["per_course"]["success"]
        del pool
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="NAME=PATH")
    ap.add_argument("--out", type=Path, help="JSON with each set's count and per-course outcomes")
    a = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets = {f: level5_courses(f) for f in SETS}
    res = {"measured": f"simulator ({device}), deterministic first episode, stall {STALL}",
           "sets": {str(f): f"{N} level-{LEVEL} courses, obstacle_prob {p}" for f, p in SETS.items()}, "runs": {}}
    for spec in a.checkpoints:
        name, path = spec.split("=", 1)
        model = load_model(Path(path), device)
        row = {"checkpoint": path}
        for f, cs in sets.items():
            ok = completed(model, cs, device)
            row[str(f)] = int(sum(ok))
            row[f"{f}_courses"] = [int(x) for x in ok]
        res["runs"][name] = row
        print(f"{name}: family 46 {row['46']} of {N}, family 47 {row['47']} of {N}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

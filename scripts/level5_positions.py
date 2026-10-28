"""Failure rate by jump position on extra level-5 courses.

    python scripts/level5_positions.py [--courses 3000]

Generates level-5 courses from seed family 11, which nothing else uses (for
analysis only, never for choosing a checkpoint or a setting), plays them with
the final policy in the simulator, and counts for each position in the
course (jump 0 is taken from a standing start) how often the jump there
fails, for all jumps and for the level-5 jumps alone (a level-5 course mixes
level-5 jumps with easier ones). Writes results/level5_by_position.json.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour import courses as C  # noqa: E402
from parkour.env import CoursePool  # noqa: E402
from parkour.live import load_model  # noqa: E402
from parkour.train import evaluate  # noqa: E402

ANALYSIS_FAMILY = 11


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "results" / "final_policy.pt"))
    ap.add_argument("--courses", type=int, default=3000)
    ap.add_argument("--out", default=str(ROOT / "results" / "level5_by_position.json"))
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(a.checkpoint), dev)
    levels = C.load_levels()
    level_of = {sp.key(): L for L, specs in enumerate(levels) for sp in specs}
    rng = np.random.default_rng([2026, ANALYSIS_FAMILY, 5])
    courses = [C.generate(rng, levels, 5, 10, seed=i) for i in range(a.courses)]
    pc = evaluate(model, CoursePool(courses, dev, C.GRID), dev, 200, deterministic=True, per_course=True)["per_course"]
    att, fail, att5, fail5 = Counter(), Counter(), Counter(), Counter()
    for i, c in enumerate(courses):
        n = len(c.platforms)
        reached = round(pc["progress"][i] * (n - 1))
        ok = bool(pc["success"][i])
        for j in range(n - 1 if ok else min(reached + 1, n - 1)):
            att[j] += 1
            att5[j] += level_of.get(c.specs[j].key()) == 5
        if not ok and reached < n - 1:
            fail[reached] += 1
            fail5[reached] += level_of.get(c.specs[reached].key()) == 5
    out = {"method": f"{a.courses} level-5 courses of 10 jumps from seed family {ANALYSIS_FAMILY} (analysis only), "
                     "deterministic policy, simulator; position 0 is the jump from the standing start",
           "checkpoint": str(Path(a.checkpoint).resolve().relative_to(ROOT)),
           "success": float(np.mean(pc["success"])),
           "positions": [{"jump": j, "failures": fail[j], "attempts": att[j],
                          "level5_failures": fail5[j], "level5_attempts": att5[j]} for j in sorted(att)]}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for r in out["positions"]:
        print(r)


if __name__ == "__main__":
    main()

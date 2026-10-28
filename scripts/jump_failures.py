"""Which in-course jumps the final policy fails on the held-out test sets.

    python scripts/jump_failures.py [--checkpoint results/final_policy.pt]

Plays the 1,200 standard and 1,200 obstacle test courses in the simulator
(deterministic policy, as in train.evaluate) and, for every failed course,
records the jump it failed on (the one after the last platform reached).
Writes results/jump_failures.json: per set, the failures by level and by
jump index, the failure rate of every jump type (failures / attempts), and,
for obstacle courses, the failure rate of each type with and without a
head-hitter beam over that jump. Runs on the CPU in a few minutes.
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
from parkour.train import evaluate, held_out_courses, obstacle_courses  # noqa: E402


def beam_over(c: C.Course, k: int) -> bool:
    """A block 4 above the takeoff platform k, over the jump to k + 1 (a head hitter)."""
    a, b = c.platforms[k], c.platforms[k + 1]
    x0, x1, z0, z1 = min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])
    ex = np.asarray(c.extra).reshape(-1, 3) if c.extra is not None and len(c.extra) else np.zeros((0, 3))
    return any(x0 <= e[0] <= x1 and z0 <= e[2] <= z1 and e[1] == a[4] + 4 for e in ex)


def analyse(model, courses, dev) -> dict:
    pool = CoursePool(courses, dev, C.GRID)
    pc = evaluate(model, pool, dev, 200, deterministic=True, per_course=True)["per_course"]
    attempts, fails = Counter(), Counter()
    beam_attempts, beam_fails = Counter(), Counter()
    rows = []
    for i, c in enumerate(courses):
        n = len(c.platforms)
        reached = round(pc["progress"][i] * (n - 1))  # progress_frac = platforms reached / (n - 1)
        ok = bool(pc["success"][i])
        tried = n - 1 if ok else min(reached + 1, n - 1)
        for j in range(tried):
            key = c.specs[j].key()
            attempts[key] += 1
            if beam_over(c, j):
                beam_attempts[key] += 1
        if not ok and reached < n - 1:
            key = c.specs[reached].key()
            fails[key] += 1
            beam = beam_over(c, reached)
            if beam:
                beam_fails[key] += 1
            rows.append({"course": i, "level": int(c.level), "jump": reached, "type": key,
                         "result": "fell" if pc["fell"][i] else "stalled", "beam": beam})
    worst = sorted(fails, key=lambda k: (-fails[k] / attempts[k], -fails[k]))
    return {
        "courses": len(courses), "failed": len(rows),
        "by_level": dict(sorted(Counter(r["level"] for r in rows).items())),
        "by_jump_index": dict(sorted(Counter(r["jump"] for r in rows).items())),
        "by_result": dict(Counter(r["result"] for r in rows)),
        "jump_types": [{"type": k, "failures": fails[k], "attempts": attempts[k],
                        "with_beam": {"failures": beam_fails[k], "attempts": beam_attempts[k]}} for k in worst],
        "failed_courses": rows,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "results" / "final_policy.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "jump_failures.json"))
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(a.checkpoint), dev)
    levels = C.load_levels()
    out = {"method": "held-out test sets (families 1 and 4), 200 courses per level, deterministic policy, simulator; "
                     "the failed jump is the one after the last platform reached",
           "checkpoint": str(Path(a.checkpoint).resolve().relative_to(ROOT)),
           "standard": analyse(model, held_out_courses(levels, 200, 10), dev),
           "obstacles": analyse(model, obstacle_courses(levels, 200, 10), dev)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for name in ("standard", "obstacles"):
        r = out[name]
        print(name, r["failed"], "failed of", r["courses"], "by level", r["by_level"])
        for t in r["jump_types"][:3]:
            print("  ", t)


if __name__ == "__main__":
    main()

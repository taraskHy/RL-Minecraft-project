"""How the final policy, never trained on ice, does when platforms are ice.

    python scripts/ice_preview.py [--checkpoint results/final_policy.pt] [--out results/ice_preview.json]

Plays the 1,200 standard test courses in the simulator three times: as they
are, with every platform made of ice, and with only the platforms after the
first one (so the start block stays stone). Ice uses the optional
slipperiness grid of parkour.physics (0.98, checked against prismarine-physics
in results/physics_parity_ice.json). The policy cannot see block types (its
observation has none), so this measures what ice does to a policy that
assumes stone. Writes results/ice_preview.json.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour import courses as C  # noqa: E402
from parkour import physics  # noqa: E402
from parkour.env import CoursePool  # noqa: E402
from parkour.live import load_model  # noqa: E402
from parkour.train import evaluate, held_out_courses  # noqa: E402


def icy_pool(courses, dev, skip_first: bool) -> CoursePool:
    pool = CoursePool(courses, dev, C.GRID)
    slip = torch.full(pool.world.solid.shape, physics.DEFAULT_SLIPPERINESS, dtype=torch.float64)
    for i, c in enumerate(courses):
        for k, (x0, x1, z0, z1, y) in enumerate(c.platforms.tolist()):
            if k == 0 and skip_first:
                continue
            slip[i, x0:x1 + 1, y, z0:z1 + 1] = physics.ICE_SLIPPERINESS
    pool.world = physics.World(pool.world.solid, slip.to(dev))
    return pool


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "results" / "final_policy.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "ice_preview.json"))
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(a.checkpoint), dev)
    courses = held_out_courses(C.load_levels(), 200, 10)
    out = {"method": "standard test set (family 1), 200 courses per level, deterministic, simulator",
           "checkpoint": str(Path(a.checkpoint).resolve().relative_to(ROOT)), "conditions": {}}
    for name, pool in (("stone", CoursePool(courses, dev, C.GRID)), ("all platforms ice", icy_pool(courses, dev, False)),
                       ("ice after the start block", icy_pool(courses, dev, True))):
        r = evaluate(model, pool, dev, 200, deterministic=True)
        out["conditions"][name] = {"success": round(100 * r["success"], 1),
                                   "per_level": {L: round(100 * v["success"], 1) for L, v in r["per_level"].items()},
                                   "fell": {L: round(100 * v["fell"], 1) for L, v in r["per_level"].items()}}
        print(name, out["conditions"][name], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

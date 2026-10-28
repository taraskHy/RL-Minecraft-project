"""Block shapes in the simulator against prismarine-physics: writes results/physics_parity_shapes.json.

    python scripts/shape_parity.py [--cases 300] [--ticks 200]

Random block fields (parity.random_case) where half the raised blocks other
than the start block get a random shape of configs/block_shapes.json (slabs,
stairs, soul sand, carpets, trapdoors, fence and wall posts, ...): the
simulator with World(shape=...) against prismarine-physics 1.11.1 through
Node, whose blocks are the real block states (prismarine-block, Minecraft
1.21.4), same 1e-7 collision tolerance. Also: how many trajectories the
shapes change (the same cases with full cubes), and that a shape grid of only
full cubes gives exactly the trajectories of the original code path.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour import parity, physics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cases = [parity.random_case(rng, ticks=a.ticks, shapes=0.5) for _ in range(a.cases)]
    r = parity.compare(cases)
    r.pop("first_bad")
    shaped = parity.run_torch(cases)
    n_shaped = [len(c.shapes) for c in cases]
    for c in cases:
        c.shapes = None
    cubes = parity.run_torch(cases)
    change = np.abs(shaped["pos"] - cubes["pos"]).max(axis=(1, 2))
    # The same cubes through the shaped code path (a shape grid of zeros).
    world_cls = physics.World

    class ZeroShapes(world_cls):
        def __init__(self, solid, slip=None, shape=None):
            super().__init__(solid, slip, torch.zeros(solid.shape, dtype=torch.uint8) if shape is None else shape)

    parity.physics.World = ZeroShapes
    try:
        zero = parity.run_torch(cases)
    finally:
        parity.physics.World = world_cls
    names = physics.shape_table()[0]
    out = {"method": "parkour.parity.compare with shapes: random_case(np.random.default_rng(%d), ticks=%d, shapes=0.5) x %d, "
                     "the simulator with World(shape=...) against prismarine-physics 1.11.1 through Node with the real "
                     "block states (configs/block_shapes.json, prismarine-block, Minecraft 1.21.4), same 1e-7 collision "
                     "tolerance, CPU" % (a.seed, a.ticks, a.cases),
           "shapes": list(names[1:]), "shaped_cells_per_case": round(float(np.mean(n_shaped)), 1),
           "result": r,
           "effect_of_shapes": {"cases_changed": int((change > 0).sum()), "cases": a.cases,
                                "largest_position_change_blocks": round(float(change.max()), 3)},
           "full_cube_shape_grid_identical": all(np.array_equal(zero[k], cubes[k]) for k in cubes)}
    path = ROOT / "results" / "physics_parity_shapes.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

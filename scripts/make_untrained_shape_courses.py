"""Showcase courses for the shaped blocks that no training level uses: stairs, top slab, top trapdoor, iron bars.

    python scripts/make_untrained_shape_courses.py OUT.json

The user asked on 29 October 2026 for replays of stairs too. configs/block_shapes.json (the physics) has four
shapes that no level of any table uses, so no policy has ever jumped onto them. Each gets two courses, fixed
before any outcome was seen: courses.generate on the extra group's table, on a copy whose shaped-blocks level
keeps only the specs of a trained type with nearly the same top, the block swapped for the untrained one:
stairs, top slab and top trapdoor (top 1.0) from the dirt-path specs (top 0.9375); iron bars (a thin post, top
1.0) from the end-rod specs (a thicker post, top 1.0). Stream numpy default_rng([2026, 44, k]) with k = 10..13,
continuing scripts/make_skill_replays.py's ten trained types (k = 0..9) in seed family 44; seeds 0 and 1;
10 jumps, no obstacles, level label 12. Every stairs platform then faces the direction of travel (from the
previous platform's centre to its own, the larger of the two axes), so it rises the way the bot moves: it
lands on the lower step or the upper one. Standard library and the repository's code only.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import parkour.courses as C  # noqa: E402
from parkour.train import SKILL_GROUPS, skill_tables  # noqa: E402

FAMILY = 44
SEEDS = 2
N_JUMPS = 10
SHAPED_LEVEL = 12  # global id of the shaped-blocks level
UNTRAINED = (  # k, shape, the trained type whose specs it takes (top 0.9375 or 1.0 against the shape's 1.0)
    (10, "stairs", "dirt_path"),
    (11, "top_slab", "dirt_path"),
    (12, "top_trapdoor", "dirt_path"),
    (13, "iron_bars", "end_rod"),
)


def facing(prev, cur) -> str:
    """The direction of travel from one platform's centre to the next: north is -z, east +x (Minecraft)."""
    dx = (cur[0] + cur[1]) / 2 - (prev[0] + prev[1]) / 2
    dz = (cur[2] + cur[3]) / 2 - (prev[2] + prev[3]) / 2
    if abs(dx) >= abs(dz):
        return "east" if dx > 0 else "west"
    return "south" if dz > 0 else "north"


def courses() -> list[dict]:
    table = skill_tables(["extra"])["extra"]
    _, lo, _, first = SKILL_GROUPS["extra"]
    local = SHAPED_LEVEL - first + lo
    trained = {s.block for s in table[local]}
    shapes = {s["name"] for s in json.loads((ROOT / "configs" / "block_shapes.json").read_text())["shapes"]}
    used = {s.block for lv in C.load_levels(advanced=True, extra=True) for s in lv if s.block}
    out = []
    for k, shape, base in UNTRAINED:
        name = "stairs_north" if shape == "stairs" else shape  # stairs: turned to the travel direction below
        assert base in trained and name in shapes and shape not in used and name not in used, (shape, base)
        levels = list(table)  # a copy of the table: configs/ stays as it is
        levels[local] = [replace(s, block=name) for s in table[local] if s.block == base]
        rng = np.random.default_rng([2026, FAMILY, k])
        for i in range(SEEDS):
            c = C.generate(rng, levels, local, N_JUMPS, seed=i, obstacle_prob=0.0, ice_prob=0.0)
            c.level = SHAPED_LEVEL
            assert set(c.shapes.values()) <= {name}, (shape, c.shapes)
            if shape == "stairs":
                p = c.platforms
                c.shapes = {k2: f"stairs_{facing(p[int(k2) - 1], p[int(k2)])}" for k2 in c.shapes}
            d = c.to_json()
            d.update(kind=f"untrained: {shape}", family=FAMILY, id=f"untrained-{shape}-{i}", based_on=base)
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; the course file is written once")
    cs = courses()
    a.out.write_text(json.dumps({"courses": cs}), encoding="utf-8")
    from parkour.neo_assist import courses_from_file  # the file must read back as parkour.live reads it
    back = courses_from_file(json.loads(a.out.read_text(encoding="utf-8"))["courses"])
    assert [c.shapes for c in back] == [{int(k): v for k, v in d["shapes"].items()} for d in cs] or \
        [c.shapes for c in back] == [d["shapes"] for d in cs]
    print(json.dumps([{"id": d["id"], "shapes": d["shapes"]} for d in cs], indent=1))


if __name__ == "__main__":
    main()

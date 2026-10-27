"""Write configs/showcase_courses.json: hand-designed courses for demos.

    python scripts/make_showcase.py

The courses are designed by hand from named jumps rather than drawn by the
generator, so they are unseen by construction. Play them with
    run.cmd live --checkpoint ... --courses-file configs\\showcase_courses.json --viewer NAME
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour.courses import GRID, JumpSpec, from_specs  # noqa: E402

J = JumpSpec
RIGHT, LEFT, ON = 1, -1, 0

TOUR = [  # one of everything, with turns
    (J(1, 0, 0, (3, 3)), ON), (J(2, 0, 0, (1, 1)), ON), (J(1, 1, 0, (1, 1)), RIGHT), (J(2, 0, 1, (1, 1)), ON),
    (J(3, -1, 0, (2, 2)), ON), (J(2, 1, 0, (1, 1)), LEFT), (J(3, 0, 0, (1, 1)), ON), (J(1, -2, 2, (1, 1)), ON),
    (J(4, -2, 0, (1, 1)), ON), (J(2, 1, -1, (1, 1)), ON), (J(3, 0, 0, (2, 2)), RIGHT), (J(1, -3, 0, (3, 3)), ON),
]
STAIRS = [  # climb with single blocks, then a long drop
    (J(1, 1, 0, (1, 1)), ON), (J(1, 1, 1, (1, 1)), ON), (J(2, 1, 0, (1, 1)), ON), (J(1, 1, -1, (1, 1)), RIGHT),
    (J(2, 1, 0, (1, 1)), ON), (J(0, 1, 1, (1, 1)), ON), (J(1, 1, 0, (1, 1)), LEFT), (J(2, 1, 0, (1, 1)), ON),
    (J(3, -3, 0, (1, 1)), ON), (J(4, -3, 0, (2, 2)), ON), (J(2, 0, 0, (3, 3)), ON),
]
SPRINT = [  # long flat gaps that need momentum
    (J(2, 0, 0, (1, 1)), ON), (J(3, 0, 0, (1, 1)), ON), (J(3, 0, 0, (1, 1)), ON), (J(3, 0, 1, (1, 1)), ON),
    (J(4, -1, 0, (1, 1)), ON), (J(3, 0, 0, (1, 1)), RIGHT), (J(3, 0, 0, (1, 1)), ON), (J(3, 0, -1, (1, 1)), ON),
    (J(2, 0, 0, (3, 3)), ON),
]


def check(c):
    b = c.blocks()
    assert (b >= 0).all() and (b < GRID).all(), "course leaves the grid"
    assert len({tuple(x) for x in b}) == len(b), "platforms overlap"


def main():
    courses = []
    for name, steps, start, heading in (("tour", TOUR, (4, 14, 8), 0), ("stairs", STAIRS, (6, 6, 40), 0),
                                        ("sprint", SPRINT, (4, 16, 30), 0)):
        c = from_specs(steps, start=start, heading=heading, name=name)
        check(c)
        courses.append(c.to_json())
    out = ROOT / "configs" / "showcase_courses.json"
    out.write_text(json.dumps({"courses": courses}, indent=1))
    print("wrote", out, [len(c["platforms"]) for c in courses])


if __name__ == "__main__":
    main()

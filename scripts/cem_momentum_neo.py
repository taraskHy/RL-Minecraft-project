"""Are double (and triple) neos possible with a run-up?

    python scripts/cem_momentum_neo.py

From a standing start on one block, parkour.cem finds no double neo. Real
double neos are run at speed, so here the takeoff is a runway (3 or 4 blocks
long, 1 wide) ending at the wall, and the search may start anywhere on it.
Writes configs/jump_cem_momentum.json.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parkour.cem import search  # noqa: E402
from parkour.courses import JumpSpec  # noqa: E402

SPECS = [JumpSpec(g, dy, 0, (1, 1), neo=k, side=1, wall_at=1, runway=r)
         for k in (2, 3) for r in (3, 4) for g in (1, 2) for dy in (-1, 0)]
# The wall further into the gap: air to curve in before it.
SPECS += [JumpSpec(g, dy, 0, (1, 1), neo=2, side=1, wall_at=w, runway=r)
          for r in (3, 4) for g in (2, 3) for w in range(2, g + 1) for dy in (-1, 0)]


def main():
    path = ROOT / "configs" / "jump_cem_momentum.json"
    done = {r["key"]: r for r in json.loads(path.read_text())["specs"]} if path.exists() else {}
    out = []
    t0 = time.time()
    for sp in SPECS:
        if sp.key() in done:  # keep earlier results
            out.append(done[sp.key()])
            continue
        res = search(sp, iterations=40)
        out.append({"key": sp.key(), "gap": sp.gap, "dy": sp.dy, "neo": sp.neo, "runway": sp.runway, **res})
        print(sp.key(), "POSSIBLE" if res["landed"] else "not found", flush=True)
    path.write_text(json.dumps(
        {"method": __doc__.strip().splitlines()[0], "seconds": round(time.time() - t0, 1), "specs": out}, indent=1))


if __name__ == "__main__":
    main()

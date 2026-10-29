"""Build the showcase replay page: the 45 recorded offline episodes of 29 October in 3D.

    python scripts/make_showcase_replays.py OUT.html [--ref REF | --dir DIR]

The data are the post-training showcase recordings of the two
validation-selected policies (g995-adv checkpoint 99, g995-ice checkpoint
449), exported on branch codex/advanced-ice-replay-export-20261029 under
exports/advanced-ice-replays/ (README.md there has the schema). By default
they are read from that branch with git (``git fetch origin
codex/advanced-ice-replay-export-20261029`` first); ``--dir`` reads a copy of
the folder instead. The files are checked against the export's manifest and
embedded in the page (scripts/showcase_replays_template.html), so it plays
without the PC, a GPU, Python or a Minecraft server; three.js comes from the
same CDN as the other pages. Offline prismarine-physics recordings, not
training rollouts or benchmark results: the page says so, and its statistics
are counts over these 45 episodes only. Standard library only.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts" / "showcase_replays_template.html"
BRANCH = "codex/advanced-ice-replay-export-20261029"
FOLDER = "exports/advanced-ice-replays/"
SHARED = ["schema", "tick_hz", "measurement", "purpose", "selection", "coordinates", "frame_columns", "timing",
          "action_columns", "yaw_steps_degrees"]  # must be the same in both files


def git(*args) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ref", default=f"origin/{BRANCH}", help="git ref holding the export folder")
    ap.add_argument("--dir", type=Path, help="a copy of exports/advanced-ice-replays/ instead of --ref")
    a = ap.parse_args()
    if a.dir:
        read = lambda name: (a.dir / name).read_bytes()  # noqa: E731
        commit = "unknown (read from a folder)"
    else:
        try:
            read = lambda name: git("show", f"{a.ref}:{FOLDER}{name}")  # noqa: E731
            commit = git("log", "-1", "--format=%H", a.ref, "--", FOLDER + "advanced.json", FOLDER + "ice.json").decode().strip()
        except subprocess.CalledProcessError as e:
            raise SystemExit(f"cannot read {FOLDER} from {a.ref} ({e.stderr.decode().strip()}); "
                             f"run: git fetch origin {BRANCH}") from None
    raw = {name: read(name) for name in ("advanced.json", "ice.json", "manifest.json", "verification.json")}
    manifest = json.loads(raw["manifest.json"])
    files = {}
    for name in ("advanced.json", "ice.json", "verification.json"):
        files[name] = hashlib.sha256(raw[name]).hexdigest()
        if files[name] != manifest["files"][name]["sha256"]:
            raise SystemExit(f"{name} does not match the export's manifest")
    adv, ice = json.loads(raw["advanced.json"]), json.loads(raw["ice.json"])
    for k in SHARED:
        assert adv[k] == ice[k], f"advanced.json and ice.json differ in {k}"
    payload = {k: v for k, v in adv.items() if k not in ("runs", "episodes")}
    payload.update({"data_branch": BRANCH, "data_folder": FOLDER, "data_commit": commit, "files": files,
                    "verification": json.loads(raw["verification.json"]),
                    "runs": adv["runs"] + ice["runs"], "episodes": adv["episodes"] + ice["episodes"]})
    eps = payload["episodes"]
    assert len({e["id"] for e in eps}) == len(eps) == manifest["episodes"], "episodes missing or repeated"
    assert sum(len(e["frames"]) for e in eps) == manifest["frames"]
    for e in eps:
        assert len(e["frames"]) == len(e["actions"]) + 1 == e["ticks"] + 1, e["id"]
        assert len(e["course"]["specs"]) == len(e["course"]["platforms"]) - 1, e["id"]
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    assert "</" not in data and "<!--" not in data  # safe inside the page's <script> element
    template = TEMPLATE.read_text(encoding="utf-8")
    assert template.count("__REPLAY_DATA__") == 1
    a.out.write_text(template.replace("__REPLAY_DATA__", data), encoding="utf-8")
    print(json.dumps({"out": str(a.out), "bytes": a.out.stat().st_size, "episodes": len(eps), "data_commit": commit}))


if __name__ == "__main__":
    main()

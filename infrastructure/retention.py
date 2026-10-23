"""Retain compact evidence from natural-resource run folders outside ``runs/``.

Copies the small result files unchanged, never worlds, runtimes, dependencies,
or consent files, records a SHA-256 digest of every retained file in
``retention.json``, and refuses to overwrite an existing non-empty destination.
Nothing here summarizes, edits, or interprets a result.

Usage::

    python -m infrastructure.retention RUN [RUN ...] --out results/NAME --label "..."
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

RUN_FILES = ("metadata.json", "initial_policy.json", "policy.json", "episodes.csv", "summary.json",
             "simulator_training.csv")
EPISODE_FILES = ("start.json", "transitions.jsonl", "steps.csv", "outcome.json", "session_error.json")
# A resource-parallel schedule folder: the plan as run and the scheduler's record.
SCHEDULE_FILES = ("plan.json", "schedule.json")
WORLD_FILES = ("server.properties", "ops.json")
SERVER_LOG_TAIL_LINES = 200
NOTE = ("Files were copied unchanged and hashed after copying; server_tail.log holds the last lines of "
        "each world's latest.log. Worlds, runtimes, dependencies, and the EULA consent file are not retained.")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retain(run_directories, destination, label: str, log_tail_lines: int = SERVER_LOG_TAIL_LINES) -> dict:
    """Copy the compact evidence of each run folder into ``destination`` and return the manifest."""
    if not isinstance(label, str) or not label.strip():
        raise ValueError("A non-empty evidence label is required")
    runs = [Path(d) for d in run_directories]
    if not runs:
        raise ValueError("At least one run folder is required")
    # Case-folded: the destination file system may not distinguish abc from ABC.
    repeated = sorted(name for name, count in Counter(run.name.casefold() for run in runs).items() if count > 1)
    if repeated:
        raise ValueError("Run folders must have distinct names: " + ", ".join(repeated))
    for run in runs:
        if not (run / "metadata.json").is_file() and not (run / "schedule.json").is_file():
            raise ValueError(f"Not a run or schedule folder (missing metadata.json or schedule.json): {run}")
    out = Path(destination)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Destination is not empty: {out}. Choose a new folder; retained evidence is never overwritten.")
    out.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}

    def copy(source: Path, relative: Path) -> None:
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError(f"Refusing to overwrite {target}")
        shutil.copyfile(source, target)
        digest = _sha256(target)
        if digest != _sha256(source):
            raise RuntimeError(f"Copy of {source} does not match its source")
        digests[relative.as_posix()] = digest

    for run in runs:
        for name in SCHEDULE_FILES if not (run / "metadata.json").is_file() else ():
            if (run / name).is_file():
                copy(run / name, Path(run.name) / name)
        for name in RUN_FILES:
            if (run / name).is_file():
                copy(run / name, Path(run.name) / name)
        for episode in sorted(run.glob("episode-*")):
            if not episode.is_dir():
                continue
            for name in EPISODE_FILES:
                if (episode / name).is_file():
                    copy(episode / name, Path(run.name) / episode.name / name)
        worlds = run / "worlds"
        if worlds.is_dir():
            for world in sorted(worlds.glob("episode-*")):
                if not world.is_dir():
                    continue
                for name in WORLD_FILES:
                    if (world / name).is_file():
                        copy(world / name, Path(run.name) / "worlds" / world.name / name)
                latest = world / "logs" / "latest.log"
                if latest.is_file():
                    lines = latest.read_text(encoding="utf-8", errors="replace").splitlines()
                    relative = Path(run.name) / "worlds" / world.name / "server_tail.log"
                    target = out / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("\n".join(lines[-log_tail_lines:]) + "\n", encoding="utf-8")
                    digests[relative.as_posix()] = _sha256(target)
    manifest = {
        "label": label,
        "retained_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [str(run.resolve()) for run in runs],
        "files": len(digests),
        "sha256": digests,
        "note": NOTE,
    }
    (out / "retention.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify(destination) -> list[str]:
    """Return the retained files whose current digest differs from retention.json (empty when intact)."""
    out = Path(destination)
    manifest = json.loads((out / "retention.json").read_text(encoding="utf-8"))
    return [relative for relative, digest in manifest["sha256"].items()
            if not (out / relative).is_file() or _sha256(out / relative) != digest]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m infrastructure.retention", description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="*", type=Path,
                        help="Run folders written by resource-train or resource-demo, or resource-parallel schedule folders")
    parser.add_argument("--out", required=True, type=Path, help="New or empty folder, normally under results/")
    parser.add_argument("--label", help="Honest evidence label stored in retention.json (required unless --verify)")
    parser.add_argument("--verify", action="store_true", help="Instead of copying, check --out against its retention.json")
    args = parser.parse_args(argv)
    if args.verify and (args.runs or args.label):
        parser.error("--verify takes only --out")
    if not args.verify and (not args.runs or not args.label):
        parser.error("retaining needs at least one run folder and --label")
    try:
        if args.verify:
            changed = verify(args.out)
            print("intact" if not changed else "CHANGED: " + ", ".join(changed))
            return 0 if not changed else 1
        manifest = retain(args.runs, args.out, args.label)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Retained {manifest['files']} files in {args.out}; digests in retention.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

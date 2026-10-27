"""Run a queue of training runs, a few at a time on one GPU.

    python scripts/sweep.py configs/sweep_ablation.json [--slots 2]

The sweep file lists runs as {"run": NAME, "args": [...extra train args]}.
A run whose summary shows it reached its last iteration is skipped, so the
sweep can be restarted after an interruption (unfinished runs restart from
scratch under a new name suffix).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def finished(run: str, iterations: int) -> bool:
    s = ROOT / "runs" / run / "summary.json"
    if not s.exists():
        return False
    return json.loads(s.read_text()).get("iteration", -1) >= iterations - 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--slots", type=int, default=2)
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.sweep).read_text())
    common = spec.get("common", [])
    queue = []
    for r in spec["runs"]:
        args = common + r.get("args", [])
        its = int(args[args.index("--iterations") + 1]) if "--iterations" in args else 3000
        if finished(r["run"], its):
            print("skip (finished)", r["run"], flush=True)
            continue
        name = r["run"]
        if (ROOT / "runs" / name).exists():
            k = 2
            while (ROOT / "runs" / f"{name}-r{k}").exists():
                k += 1
            name = f"{name}-r{k}"
        queue.append((name, args))
    running = []
    while queue or running:
        while queue and len(running) < a.slots:
            name, args = queue.pop(0)
            log = open(ROOT / "runs" / f"{name}.out", "w")
            cmd = [str(PY), "-m", "parkour.train", "--run", name] + args
            print(time.strftime("%H:%M:%S"), "start", name, " ".join(args), flush=True)
            running.append((name, subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(10)
        for item in list(running):
            name, proc, log = item
            if proc.poll() is not None:
                log.close()
                print(time.strftime("%H:%M:%S"), "end", name, "exit", proc.returncode, flush=True)
                running.remove(item)
    print("sweep done", flush=True)


if __name__ == "__main__":
    sys.exit(main())

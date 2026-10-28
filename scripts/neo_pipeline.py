"""A learned neo, round after round, for the PC: DAgger with the assist's search as the expert.

    python scripts/neo_pipeline.py --start runs/advanced-g995/checkpoints/best.pt --name neo995 [--rounds 3]
    python scripts/neo_pipeline.py ... --dry-run      (print the commands only)

Every round r:
  1. parkour.neo_dagger plays training neo courses (never held-out ones) with
     the current checkpoint, records the states in which the bot really
     arrives on neo takeoffs, and searches landings from them (``--expert
     assist``, search prior 0.5, training courses drawn with seed r);
  2. parkour.train fine-tunes the current checkpoint with the pairs of every
     round so far as behaviour cloning (neo-dagger3's settings on the
     discount-0.995 policy: levels 0-8, obstacles on 30% of jumps, stall
     penalty 5, bc 0.3, wide neos, learning rate 3e-4), with the
     standing-start demonstrations too when ``runs/neo_demos_train.json``
     exists (as neo-dagger3);
  3. parkour.neo_assist --off scores the round's ``best.pt`` on the 218
     validation neo courses (family 2), as the starting checkpoint was.
The next round starts from the round's checkpoint when it completed more
validation neo courses than the current one; otherwise the pipeline stops.
Test families are never used here; report the chosen checkpoint on them once
(``parkour.neo_assist --off --family 1``, ``parkour.jumpeval --advanced``).

``--gaming`` adds the training flags of HANDOFF section 6 (75% of GPU memory,
GPU busy 90% of the time, 8 CPU threads); run scripts/gpu_watchdog.py too.
Scores go to runs/{name}_validation.jsonl, one line per checkpoint.
``--smoke`` shrinks every step to seconds (a CPU check that the chain runs);
its runs land in runs/ like real ones, so move them away before
``parkour.report`` (which archives every run there into results/runs/).
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAMING = ["--gpu-mem-fraction", "0.75", "--gpu-duty", "0.9", "--threads", "8"]
TRAIN = ["--advanced", "--obstacle-prob", "0.3", "--gamma", "0.995", "--reward-stall", "5", "--bc-coef", "0.3",
         "--wide-neo", "--lr", "3e-4", "--lr-max", "3e-4"]
DEMOS = ROOT / "runs" / "neo_demos_train.json"


def run(cmd: list, dry: bool) -> None:
    print("$ " + " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    if not dry:
        subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT))


def score(checkpoint: Path, tag: str, out: Path, a) -> int:
    """Validation neo courses (family 2) the checkpoint completes alone."""
    cmd = [sys.executable, "-m", "parkour.neo_assist", "--off", "--set", "official", "--family", "2",
           "--checkpoint", checkpoint, "--tag", tag, "--out", out]
    if a.smoke:
        cmd += ["--courses", "4"]
    run(cmd, a.dry_run)
    if a.dry_run:
        return 0
    return json.loads(out.read_text().splitlines()[-1])["completed"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="checkpoint of round 0 (an advanced-level policy)")
    ap.add_argument("--name", required=True, help="prefix of the runs and pair files")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=500, help="training iterations per round")
    ap.add_argument("--gaming", action="store_true", help="the GPU settings for when the user plays (HANDOFF section 6)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny settings: checks that the chain runs (CPU)")
    a = ap.parse_args()
    runs = ROOT / "runs"
    scores = runs / f"{a.name}_validation.jsonl"
    cur = Path(a.start).resolve()
    best = score(cur, f"{a.name} round 0 (start)", scores, a)
    print(json.dumps({"round": 0, "checkpoint": str(cur), "validation_completed": best}), flush=True)
    pairs = []
    for r in range(1, a.rounds + 1):
        pair_file = runs / f"{a.name}_r{r}.pt"
        collect = [sys.executable, "-m", "parkour.neo_dagger", "--checkpoint", cur, "--out", pair_file, "--prior", "0.5",
                   "--expert", "assist", "--seed", str(r)]
        if a.smoke:
            collect += ["--courses", "8", "--samples", "256", "--iterations", "8", "--keep", "1"]
        run(collect, a.dry_run)
        if not a.dry_run:
            import torch
            n = int(torch.load(pair_file, weights_only=False)["obs"].shape[0])
            if n == 0:
                print(json.dumps({"stopped": f"round {r}: the search found no landing, so there is nothing to imitate",
                                  "chosen": str(cur)}), flush=True)
                return
        pairs.append(pair_file)
        name = f"{a.name}-r{r}"
        train = [sys.executable, "-m", "parkour.train", "--run", name, "--init-from", cur, *TRAIN,
                 "--bc-pairs", ",".join(str(p) for p in pairs), "--iterations", str(a.iterations)]
        if DEMOS.exists():
            train += ["--demos", DEMOS, "--demo-courses", "1024", "--demo-level", "6"]
        if a.gaming:
            train += GAMING
        if a.smoke:
            train += ["--device", "cpu", "--iterations", "2", "--num-envs", "256", "--rollout", "8", "--epochs", "1", "--eval-every", "1",
                      "--eval-per-level", "4", "--val-per-level", "4", "--pool-size", "256"]
        run(train, a.dry_run)
        ck = runs / name / "checkpoints" / "best.pt"
        got = score(ck, f"{name} best", scores, a)
        print(json.dumps({"round": r, "checkpoint": str(ck), "validation_completed": got, "previous": best}), flush=True)
        if not a.dry_run and got <= best:
            print(json.dumps({"stopped": f"round {r} did not beat {cur} on validation", "chosen": str(cur)}), flush=True)
            return
        cur, best = ck.resolve(), got
    print(json.dumps({"chosen": str(cur), "validation_completed": best}), flush=True)


if __name__ == "__main__":
    main()

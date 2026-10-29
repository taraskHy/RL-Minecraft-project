"""Tiny CPU training with default flags, for comparing two versions of parkour/train.py bit for bit.

    python scripts/defaults_check.py PACKAGE_ROOT OUT_DIR

PACKAGE_ROOT holds the parkour/ and configs/ folders to test (the repository, or an older commit extracted with
``git archive``). Three tiny runs (basic, advanced, extra with ice) write OUT_DIR/report.json: final weights,
train_log without its timing columns, the evaluation logs and the test result. Two versions train identically
when their reports are equal (results/skills/defaults_check.json).
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root))
import torch  # noqa: E402

from parkour import train as T  # noqa: E402

assert Path(T.__file__).resolve().is_relative_to(root.resolve()), T.__file__
out_dir.mkdir(parents=True, exist_ok=True)
T.RUNS = out_dir
original = T.held_out_courses
T.held_out_courses = lambda levels, per_level, n_jumps, obstacle_prob=0.0, family=1, **kw: original(
    levels, 2, 3, obstacle_prob=obstacle_prob, family=family, **kw)
torch.manual_seed(0)
torch.save({"model": T.ActorCritic(hidden=(16,)).state_dict()}, out_dir / "init.pt")
report = {}
for name, kw in (("basic", {}), ("advanced", {"advanced": True}), ("extra", {"extra": True, "ice_prob": 0.3})):
    cfg = T.TrainConfig(run=name, device="cpu", threads=1, num_envs=16, rollout=4, pool_size=24, pool_refresh_every=1,
                        hidden=(16,), epochs=2, minibatches=2, eval_every=2, iterations=5, obstacle_prob=0.3,
                        test_once=True, init_from=str(out_dir / "init.pt"), eval_batch_size=7, compact_ice=True, **kw)
    out = T.train(cfg)
    ck = torch.load(out / "checkpoints" / "latest.pt", weights_only=False)
    h = hashlib.sha1()
    for k, v in ck["model"].items():
        h.update(k.encode())
        h.update(v.cpu().numpy().tobytes())
    rows = list(csv.DictReader((out / "train_log.csv").open()))
    for r in rows:
        r.pop("minutes")
        r.pop("fps")
    report[name] = {"weights": h.hexdigest(), "train_log": hashlib.sha1(json.dumps(rows).encode()).hexdigest(),
                    "train_log_columns": list(rows[0]),
                    "eval_val_log": (out / "eval_val_log.csv").read_text(), "eval_log": (out / "eval_log.csv").read_text(),
                    "test": json.loads((out / "summary.json").read_text())["test_eval"]["success"]}
(out_dir / "report.json").write_text(json.dumps(report, indent=1))

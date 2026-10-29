"""Export visualization-only offline episodes without touching training or test sets.

python export_replays.py --repo PATH --out PATH
The output directory must not exist. Requires the repo's Python and Node dependencies.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone, timedelta


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    repo, out = a.repo.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from parkour import courses as C
    from parkour.env import YAW_STEPS_DEG
    from parkour.live import load_model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    generated = datetime.now(timezone(timedelta(hours=2))).isoformat()
    bundle = {
        "schema": "parkour-replay-v1", "created_at": generated,
        "source_commit": commit, "tick_hz": 20,
        "measurement": "Offline prismarine-physics 1.21.4; no Minecraft server",
        "purpose": "Fixed showcase recordings made after training, not original training rollouts or validation/test benchmarks",
        "selection": "Existing best checkpoints selected on validation only; no model or recipe selection from these showcase outcomes",
        "coordinates": {"axes": ["x", "y", "z"], "up": "y", "units": "blocks", "position": "bot feet", "player_width": 0.6, "player_height": 1.8,
            "yaw": "radians; forward = [-sin(yaw), 0, -cos(yaw)]", "platform": "[x_min,x_max,z_min,z_max,block_y], inclusive block coordinates; surface y=block_y+1",
            "extra": "obstacle cells [x,y,z]", "ice": "indices into the platform array"},
        "frame_columns": ["x", "y", "z", "vx", "vy", "vz", "yaw", "on_ground", "jump_ticks", "platform_reached"],
        "timing": "frames[0] is the initial state; frames[t+1] follows actions[t]; time_seconds = frame_index/20; velocities are blocks per tick",
        "action_columns": ["move", "strafe", "jump", "sprint", "turn_index"],
        "yaw_steps_degrees": list(YAW_STEPS_DEG),
        "runs": [], "episodes": []}
    checks = []
    for run_index, name in enumerate(("g995-adv", "g995-ice")):
        checkpoint = repo / "runs" / name / "checkpoints" / "best.pt"
        original_hash = sha256(checkpoint)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        summary = json.loads((repo / "runs" / name / "summary.json").read_text())
        assert ck["iteration"] == summary["best"]["iteration"], "Wrong selected checkpoint"
        cfg = ck["config"]
        model = load_model(checkpoint, "cpu")
        assert sha256(checkpoint) == original_hash, "Checkpoint changed while loading"
        raw = out / "raw" / name
        raw.mkdir(parents=True)
        model.export(raw / "policy.json", {"run": name, "iteration": ck["iteration"], "checkpoint_sha256": original_hash})
        levels = C.load_levels(advanced=name == "g995-adv")
        courses = []
        for level in range(len(levels)):
            for sample in range(3):
                # A fixed visualization namespace, separate from held-out families.
                seed_components = [2026, 1029, 712, run_index, level, sample]
                c = C.generate(np.random.default_rng(seed_components), levels, level, 10,
                               seed=sample, obstacle_prob=0.3, ice_prob=0.3 if name == "g995-ice" else 0.0,
                               wide_neo=bool(cfg.get("wide_neo", False)))
                c.name = f"{name} showcase L{level} sample {sample + 1}"
                item = c.to_json()
                item["visualization_seed"] = seed_components
                courses.append(item)
        write_json(raw / "courses.json", {"courses": courses})
        command = [str(repo / ".local/runtime/node/node.exe"), str(repo / "bridge/offline_bot.js"),
                   "--policy", str(raw / "policy.json"), "--courses", str(raw / "courses.json"),
                   "--out", str(raw / "episodes.jsonl"), "--settle", "10", "--stall", "200", "--max-ticks", "3000"]
        print(f"Recording {name}: {len(courses)} fixed showcase courses on CPU", flush=True)
        result = subprocess.run(command, cwd=repo / "bridge", text=True, capture_output=True, check=False)
        (raw / "recorder.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (raw / "recorder.stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Recorder failed: {result.stderr}")
        episodes = [json.loads(line) for line in (raw / "episodes.jsonl").read_text().splitlines() if line.strip()]
        assert len(episodes) == len(courses)
        info = {"run": name, "checkpoint_iteration": ck["iteration"], "checkpoint_sha256": original_hash,
                "checkpoint_selection": "validation", "validation_success": summary["best"]["validation_success"],
                "course_count": len(courses), "levels": list(range(len(levels))), "obstacle_prob": 0.3,
                "ice_prob": 0.3 if name == "g995-ice" else 0.0, "showcase_successes": sum(ep["success"] for ep in episodes)}
        bundle["runs"].append(info)
        for ep in episodes:
            course = courses[ep["index"]]
            assert ep["level"] == course["level"]
            assert len(ep["traj"]) == len(ep["actions"]) == ep["ticks"]
            initial = ep["init"]
            yaw = initial["yaw"]
            cp = 0
            frames = [[*initial["pos"], *initial["vel"], yaw, int(initial["onGround"]), initial["jumpTicks"], 0]]
            for row, action in zip(ep["traj"], ep["actions"]):
                yaw = (yaw + math.radians(YAW_STEPS_DEG[action[4]]) + math.pi) % (2 * math.pi) - math.pi
                x, y, z = row[:3]
                if row[6]:
                    for k, (x0, x1, z0, z1, py) in enumerate(course["platforms"]):
                        if k > cp and abs(y - (py + 1)) < 1e-6 and x + 0.3 > x0 and x - 0.3 < x1 + 1 and z + 0.3 > z0 and z - 0.3 < z1 + 1:
                            cp = k
                frames.append([*row[:6], yaw, int(row[6]), row[7], cp])
            assert all(math.isfinite(v) for frame in frames for v in frame)
            assert len(frames) == ep["ticks"] + 1
            assert np.allclose(np.array(frames[1:])[:, :6], np.array(ep["traj"])[:, :6], rtol=0, atol=0)
            if ep["success"]:
                assert cp == len(course["platforms"]) - 1
            bundle["episodes"].append({"id": f"{name}-{ep['index']:03d}", "run": name,
                "checkpoint_iteration": ck["iteration"], "title": course["name"], "course": course,
                "result": ep["result"], "success": ep["success"], "ticks": ep["ticks"],
                "duration_seconds": ep["ticks"] / 20, "platform_reached": ep["cp"],
                "actions": ep["actions"], "frames": frames})
        checks.append({"run": name, "episodes": len(episodes), "frames": sum(ep["ticks"] + 1 for ep in episodes),
                       "raw_alignment_exact": True, "finite_values": True, "selected_checkpoint_matches": True})
        print(json.dumps(info), flush=True)
    write_json(out / "replays.json", bundle)
    manifest = {"created_at": generated, "source_commit": commit, "schema": bundle["schema"],
                "episodes": len(bundle["episodes"]), "frames": sum(len(ep["frames"]) for ep in bundle["episodes"]),
                "checks": checks, "files": {}}
    for p in sorted(out.rglob("*")):
        if p.is_file():
            manifest["files"][p.relative_to(out).as_posix()] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"complete": True, "episodes": manifest["episodes"], "frames": manifest["frames"], "bundle_bytes": (out / "replays.json").stat().st_size}), flush=True)


if __name__ == "__main__":
    main()

"""Cross-evaluation for one network for every skill (docs/combine_policies.md, section 4).

    python -m parkour.skills matrix --candidate final=results/final_policy.pt --candidate adv=runs/g995-adv/checkpoints/best.pt ...
    python -m parkour.skills matrix --part 1/2 ...      # every second candidate, for two processes on two GPU slots
    python -m parkour.skills choose results/skills/validation_matrix.part*.json --out results/skills/validation_matrix.json
    python -m parkour.skills matrix --split test --groups ice --candidate ... --out results/skills/test_ice.json

Simulator, deterministic policy. Every candidate plays the skills-mode held-out sets of the chosen groups
(``parkour.train.skill_sets``: validation families 2/5 and ice 17 by default, 100 courses per level;
``--split test``: 1/4 and ice 16, 200 per level, only for the one report after training). The output has each
candidate's success per group (standard courses, obstacle courses, both) and per level. Finished candidates
are written after each one, and a restarted command skips them.

``choose`` applies the rules fixed before any run, on validation only: each group's teacher is the candidate
with the highest validation success on that group (basic: the mean of basic and ice), and the student starts
from the candidate with the highest mean over basic, ice, advanced and extra. Ties go to the earlier candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from . import train as T
from .live import load_model
from .paths import ROOT

PER_LEVEL = {"validation": 100, "test": 200}  # as TrainConfig's val_per_level and eval_per_level
CHOICE_SCORES = {  # group -> how its teacher is chosen (validation success)
    "basic": ("basic", "ice"),
    "advanced": ("advanced",),
    "extra": ("extra",),
}


def sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def evaluate_candidate(path: Path, sets: dict, device, batch_size: int, stall_ticks: int) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = load_model(path, device)
    out = {"checkpoint": Path(path).as_posix(), "sha1": sha1(path), "iteration": ck.get("iteration"),
           "run": (ck.get("config") or {}).get("run"), "source": ck.get("source"), "groups": {}, "per_level": {}}
    t0 = time.time()
    for g, parts in sets.items():
        res = {}
        for part, courses in parts.items():
            if not courses:
                res[part] = None
                continue
            r = T.evaluate_course_batches(model, courses, device, stall_ticks, batch_size, compact_ice=True)
            res[part] = {"success": r["success"], "n": len(courses)}
            out["per_level"].setdefault(part, {}).update({int(L): v for L, v in r["per_level"].items()})
        done = [v for v in res.values() if v]
        n = sum(v["n"] for v in done)
        out["groups"][g] = {"standard": (res.get("standard") or {}).get("success"),
                            "obstacles": (res.get("obstacles") or {}).get("success"),
                            "all": sum(v["success"] * v["n"] for v in done) / n, "n": n}
    if all(g in out["groups"] for g in T.GROUP_ORDER):
        out["macro"] = T.macro_score({g: v["all"] for g, v in out["groups"].items()})
    out["seconds"] = round(time.time() - t0, 1)
    return out


def choose(candidates: dict) -> dict:
    names = list(candidates)
    scores = {g: {n: sum(candidates[n]["groups"][k]["all"] for k in ks) / len(ks) for n in names}
              for g, ks in CHOICE_SCORES.items()}
    macro = {n: candidates[n]["macro"] for n in names}
    return {"teachers": {g: max(names, key=s.get) for g, s in scores.items()},  # max keeps the first of equals
            "init": max(names, key=macro.get), "teacher_scores": scores, "macro": macro,
            "rules": "teacher of each group: highest validation success on it (basic: mean of basic and ice); "
                     "student start: highest mean over basic, ice, advanced and extra; ties to the earlier candidate"}


def matrix(a) -> None:
    out = Path(a.out or ROOT / "results" / "skills" / f"{a.split}_matrix.json")
    cands = dict(c.split("=", 1) for c in a.candidate)
    if a.part:
        k, n = (int(v) for v in a.part.split("/"))
        cands = dict(list(cands.items())[k - 1::n])
    groups = [g for g in a.groups.split(",") if g]
    data = json.loads(out.read_text()) if out.exists() else {}
    if data and (data["split"], data["groups"]) != (a.split, groups):
        raise ValueError(f"{out} holds another split or other groups")
    data.update({"what": "parkour.skills matrix: success per skill group, simulator, deterministic policy",
                 "split": a.split, "groups": groups, "per_level_courses": PER_LEVEL[a.split],
                 "families": ({"standard": T.VALIDATION_FAMILY, "obstacles": T.OBSTACLE_VALIDATION_FAMILY,
                               "ice": T.ICE_VALIDATION_FAMILY} if a.split == "validation" else
                              {"standard": T.TEST_SEED_FAMILY, "obstacles": T.OBSTACLE_TEST_FAMILY,
                               "ice": T.ICE_TEST_FAMILY}),
                 "level_ids": {g: [gid for gid, gg in T.level_groups().items() if gg == g] for g in groups}})
    data.setdefault("candidates", {})
    sets = T.skill_sets(a.split, PER_LEVEL[a.split], 10, [g for g in groups if g != "ice"], ice="ice" in groups)
    sets = {g: sets[g] for g in groups}
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, path in cands.items():
        old = data["candidates"].get(name)
        if old and old["checkpoint"] == Path(path).as_posix() and old["sha1"] == sha1(Path(path)):
            continue
        data["candidates"][name] = evaluate_candidate(Path(path), sets, a.device, a.batch_size, a.stall_ticks)
        out.write_text(json.dumps(data, indent=1))
        c = data["candidates"][name]
        print(json.dumps({"candidate": name, **{g: round(v["all"], 4) for g, v in c["groups"].items()},
                          "macro": c.get("macro"), "seconds": c["seconds"]}), flush=True)
    out.write_text(json.dumps(data, indent=1))


def choose_cmd(a) -> None:
    parts = [json.loads(Path(p).read_text()) for p in a.parts]
    if {p["split"] for p in parts} != {"validation"}:
        raise ValueError("choices are made on validation only")
    order = [c.split("=", 1)[0] for c in a.order] if a.order else None
    cands = {}
    for p in parts:
        cands.update(p["candidates"])
    if order:
        cands = {n: cands[n] for n in order}
    data = {k: v for k, v in parts[0].items() if k != "candidates"}
    data["parts"] = [Path(p).as_posix() for p in a.parts]
    data["candidates"] = cands
    data["choice"] = choose(cands)
    Path(a.out).write_text(json.dumps(data, indent=1))
    print(json.dumps({k: data["choice"][k] for k in ("teachers", "init")}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("matrix")
    m.add_argument("--candidate", action="append", required=True, help="NAME=CHECKPOINT, in the order ties go")
    m.add_argument("--split", choices=("validation", "test"), default="validation")
    m.add_argument("--groups", default=",".join(T.GROUP_ORDER))
    m.add_argument("--part", default="", help="K/N: only every N-th candidate, starting at the K-th")
    m.add_argument("--out", default="")
    m.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    m.add_argument("--batch-size", type=int, default=600)
    m.add_argument("--stall-ticks", type=int, default=200)
    c = sub.add_parser("choose")
    c.add_argument("parts", nargs="+")
    c.add_argument("--order", action="append", default=[], help="NAME=CHECKPOINT as given to matrix (tie order)")
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    matrix(a) if a.cmd == "matrix" else choose_cmd(a)


if __name__ == "__main__":
    main()

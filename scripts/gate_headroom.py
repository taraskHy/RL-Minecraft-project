"""Headroom of a skill gate (a learned choice of expert per jump), measured before building one. Validation only.

    python scripts/gate_headroom.py [--out results/skills/gate_headroom.json]

Run after skills-hh-ft and neo-combined-r1 have finished, so that every expert is current. Every expert plays
the four groups' validation sets (basic, ice, advanced, extra: families 2, 5 and ice 17, 100 courses per level
and part, the sets of parkour.skills matrix) and its success on every course is kept. Then, per group and as the
mean over the four groups:
  - each expert alone;
  - a perfect group-level gate: the best expert on each group;
  - a perfect per-course gate: a course counts when any expert completes it.
Rule fixed before the measurement (requests of 29 Oct): build the gate only if the per-course ceiling of the
experts is at least 2 points above the best single network on the mean over the four groups; otherwise the
measurement is the reason one distilled network is the design. The best single network is the student with the
highest validation mean over the four groups at its best.pt (summary.json) among combined-distill,
skills-hh-ft and neo-combined-r1, ties to the earlier. The per-course ceiling knows each outcome in advance, so
it overstates what a gate can learn; a gate that switches at each landing could also mix experts within a
course, which this does not measure.

Experts: the final policy, the best ice specialist (g995-ice: the highest validation mean over basic and ice
among the ice-trained runs, ties to the earlier, as the matrix chose the basic teacher), the neo policy,
g995-extra and the three students. g995-ice-s11, tied with g995-ice on that mean, is measured as a sensitivity
check and kept out of the rule. Same evaluation as the matrix: deterministic policy, stall 200, compact ice,
batches of 600 courses.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import parkour.courses as C  # noqa: E402
import parkour.train as T  # noqa: E402
from parkour.env import CoursePool  # noqa: E402
from parkour.live import load_model  # noqa: E402
from parkour.skills import sha1  # noqa: E402

STUDENTS = ("combined-distill", "skills-hh-ft", "neo-combined-r1")
EXPERTS = {"final": "results/final_policy.pt", "g995-ice": "runs/g995-ice/checkpoints/best.pt",
           "neo": "results/neo_policy.pt", "g995-extra": "runs/g995-extra/checkpoints/best.pt",
           **{s: f"runs/{s}/checkpoints/best.pt" for s in STUDENTS}}
SENSITIVITY = {"g995-ice-s11": "runs/g995-ice-s11/checkpoints/best.pt"}
THRESHOLD = 0.02
RULE = ("build the gate only if the per-course ceiling of the experts is at least 2 points above the best single "
        "network (the student with the highest validation mean among " + ", ".join(STUDENTS) + ") on the mean "
        "over the four groups (fixed before the measurement, 29 Oct)")


def per_course(model, courses, device, batch_size, stall_ticks):
    """Success on every course, in order (evaluate_course_batches with per-course results)."""
    out = []
    for start in range(0, len(courses), batch_size):
        pool = CoursePool(courses[start:start + batch_size], device, C.GRID, compact_ice=True)
        out += [bool(x) for x in T.evaluate(model, pool, device, stall_ticks, per_course=True)["per_course"]["success"]]
        del pool
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return out


def mean(xs):
    return sum(xs) / len(xs)


def ceilings(results: dict, names: list, student: str) -> dict:
    groups = {}
    for g in T.GROUP_ORDER:
        rows = {n: results[n][g]["standard"] + results[n][g]["obstacles"] for n in names}
        alone = {n: mean(rows[n]) for n in names}
        any_ = [any(rows[n][i] for n in names) for i in range(len(rows[student]))]
        groups[g] = {"courses": len(any_), "alone": alone, "group_gate": max(alone.values()),
                     "group_gate_expert": max(alone, key=alone.get), "per_course_gate": mean(any_),
                     "student_fails_another_completes": sum(a and not s for a, s in zip(any_, rows[student])),
                     "student_completes_no_other_does": sum(s and not any(rows[n][i] for n in names if n != student)
                                                            for i, s in enumerate(rows[student]))}
    macro = lambda key: T.macro_score({g: groups[g][key] for g in T.GROUP_ORDER})  # noqa: E731
    return {"experts": names, "student": student, "groups": groups,
            "mean": {"student": T.macro_score({g: groups[g]["alone"][student] for g in T.GROUP_ORDER}),
                     "group_gate": macro("group_gate"), "per_course_gate": macro("per_course_gate"),
                     "each_alone": {n: T.macro_score({g: groups[g]["alone"][n] for g in T.GROUP_ORDER}) for n in names}}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/skills/gate_headroom.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=600)
    ap.add_argument("--stall-ticks", type=int, default=200)
    ap.add_argument("--per-level", type=int, default=100, help="validation courses per level and part (100: the matrix's)")
    a = ap.parse_args()
    paths = {s: ROOT / "runs" / s / "summary.json" for s in STUDENTS}
    missing = [s for s, p in paths.items() if not p.exists()]
    if missing:
        raise SystemExit(f"not started yet: {', '.join(missing)} (every expert must be current)")
    summaries = {s: json.loads(p.read_text()) for s, p in paths.items()}
    unfinished = [s for s, v in summaries.items() if not v.get("completed")]
    if unfinished:
        raise SystemExit(f"not finished yet: {', '.join(unfinished)} (every expert must be current)")
    val = {s: summaries[s]["best"]["validation_success"] for s in STUDENTS}
    student = max(STUDENTS, key=val.get)  # max keeps the first of equals
    sets = T.skill_sets("validation", a.per_level, 10, ("basic", "advanced", "extra"), ice=True)
    results, info = {}, {}
    for name, path in {**EXPERTS, **SENSITIVITY}.items():
        t0 = time.time()
        model = load_model(Path(path), a.device)
        results[name] = {g: {p: per_course(model, sets[g][p], a.device, a.batch_size, a.stall_ticks) if sets[g][p] else []
                             for p in ("standard", "obstacles")} for g in T.GROUP_ORDER}
        info[name] = {"checkpoint": path, "sha1": sha1(Path(path)), "seconds": round(time.time() - t0, 1)}
        print(json.dumps({"expert": name, **{g: round(mean(sum(results[name][g].values(), [])), 4) for g in T.GROUP_ORDER},
                          "seconds": info[name]["seconds"]}), flush=True)
    main_c = ceilings(results, list(EXPERTS), student)
    sens = ceilings(results, list(EXPERTS) + list(SENSITIVITY), student)
    gain = main_c["mean"]["per_course_gate"] - main_c["mean"]["student"]
    # References for reading the gain: the three students are near-identical networks (one lineage), and
    # g995-ice and g995-ice-s11 are two seeds of one recipe, so their per-course gains are mostly run-to-run variance.
    students_only = ceilings(results, list(STUDENTS), student)
    seed_pair = ceilings(results, ["g995-ice", "g995-ice-s11"], "g995-ice")["groups"]["basic"]
    # The same courses and settings as the validation matrix and the students' own validation: report both.
    check = {}
    if a.per_level == 100:
        m = json.loads((ROOT / "results/skills/validation_matrix.json").read_text())["candidates"]
        check = {n: {g: {"here": mean(sum(results[n][g].values(), [])), "before": m[n]["groups"][g]["all"]}
                     for g in T.GROUP_ORDER} for n in list(EXPERTS) + list(SENSITIVITY) if n in m}
        for s in STUDENTS:
            check[s] = {g: {"here": mean(sum(results[s][g].values(), [])), "before": summaries[s]["best"]["validation_groups"][g],
                            "note": "before: training-time validation, batches of 256"} for g in T.GROUP_ORDER}
    out = {"what": "skill-gate headroom on the validation sets (simulator, deterministic policies)", "rule": RULE,
           "families": {"standard": T.VALIDATION_FAMILY, "obstacles": T.OBSTACLE_VALIDATION_FAMILY, "ice": T.ICE_VALIDATION_FAMILY},
           "per_level_courses": a.per_level, "students_validation_mean": val, "best_single_network": student,
           "experts": info, "ceilings": main_c, "with_sensitivity_expert": sens,
           "gain_per_course_gate_over_best_single": gain, "threshold": THRESHOLD,
           "decision": "build the gate" if round(gain, 9) >= THRESHOLD else "do not build: one distilled network",
           "references": {"students_only": students_only["mean"],
                          "gain_over_students_only": main_c["mean"]["per_course_gate"] - students_only["mean"]["per_course_gate"],
                          "seed_pair_basic": {k: seed_pair[k] for k in ("alone", "per_course_gate")}},
           "caveat": "the per-course ceiling knows each outcome in advance, so it overstates what a gate can learn; "
                     "part of it is run-to-run variance between similar networks (see references)",
           "consistency_with_earlier_validation": check,
           "per_course_success": {n: {g: {p: "".join("1" if x else "0" for x in v) for p, v in r.items() if v}
                                      for g, r in results[n].items()} for n in results}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"best_single": student, "student": round(main_c["mean"]["student"], 4),
                      "group_gate": round(main_c["mean"]["group_gate"], 4),
                      "per_course_gate": round(main_c["mean"]["per_course_gate"], 4), "gain": round(gain, 4),
                      "with_s11": round(sens["mean"]["per_course_gate"], 4), "decision": out["decision"]}), flush=True)


if __name__ == "__main__":
    main()

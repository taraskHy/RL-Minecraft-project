"""Per-group test comparison of the combined students with every specialist (simulator, same courses).

Existing test rows are reused; pairs no test had covered come from results/skills/test_matrix_*.json
(parkour.skills matrix --split test). One replay: the final policy's standard courses at extra levels 9-11 (600
courses) had been played on 28 Oct (results/extra/course_preview_family1.json); the matrix played them again, and
this script checks that the results are identical. It reads the run logs archived in results/runs/, so it also
runs on a fresh clone. Run from the repository root:
    python scripts/combined_compare.py results/skills/combined_test_comparison.json
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import parkour.courses as C  # noqa: E402
import parkour.train as T  # noqa: E402

GROUPS = T.GROUP_ORDER
STUDENTS = ("combined-distill", "combined-union")
LOCAL = {"basic": (0, 5, 0), "advanced": (6, 8, 6), "extra": (6, 11, 9)}  # specialist level range -> first global id


def csv_row(run, name, it):
    rows = [r for r in csv.DictReader(open(f"results/runs/{run}/{name}")) if r["iteration"] == str(it)]
    assert len(rows) == 1, (run, name, it, len(rows))
    return {int(k[len("success_L"):]): float(v) for k, v in rows[0].items() if k.startswith("success_L") and v != ""}


def from_levels(std: dict, obs: dict, group: str, n=200):
    """Group scores from one specialist's own per-level test results (local level ids)."""
    lo, hi, _ = LOCAL[group]
    s = [std[L] for L in range(lo, hi + 1)] if std else None
    o = [obs[L] for L in range(lo, hi + 1)] if obs else None
    out = {"standard": sum(s) / len(s) if s else None, "obstacles": sum(o) / len(o) if o else None}
    both = (s or []) + (o or [])
    out["all"] = sum(both) / len(both)
    out["n"] = n * len(both)
    return out


def summary_levels(run):
    te = json.load(open(f"results/runs/{run}/summary.json"))["test_eval"]
    std = {int(L): v["success"] for L, v in te["per_level"].items()}
    obs = {int(L): v["success"] for L, v in te["obstacles"]["per_level"].items()}
    return te["iteration"], std, obs


def check_same_courses():
    """Levels 0-5 of the advanced and extra tables give the basic group's test courses."""
    base = C.load_levels()
    for kw in ({"advanced": True}, {"extra": True}):
        other = C.load_levels(**kw)
        for fam, op in ((T.TEST_SEED_FAMILY, 0.0), (T.OBSTACLE_TEST_FAMILY, 0.5)):
            a = [T.course_hash(c) for c in T.held_out_courses(base, 200, 10, obstacle_prob=op, family=fam)]
            b = [T.course_hash(c) for c in T.held_out_courses(other, 200, 10, obstacle_prob=op, family=fam) if c.level <= 5]
            assert a == b, (kw, fam)
    return True


def main(out_path):
    res = {"what": "test success per skill group, simulator, deterministic policy; families 1 (standard), "
                   "4 (obstacles), 16 (ice); 200 courses per level and part",
           "selection": "every checkpoint was chosen on validation (families 2, 5, 17); these test numbers choose nothing",
           "levels_0_5_same_courses_in_every_table": check_same_courses(),
           "candidates": {}, "sources": {}}
    # The combined students: their one test at the validation-chosen checkpoint.
    for run in STUDENTS:
        s = json.load(open(f"results/runs/{run}/summary.json"))
        te = s["test_eval"]
        res["candidates"][run] = {"iteration": te["iteration"], "groups": te["groups"], "macro": te["macro"],
                                  "validation_macro": s["best"]["validation_success"],
                                  "validation_groups": s["best"].get("validation_groups")}
        res["sources"][run] = f"results/runs/{run}/summary.json test_eval (test once, at best.pt)"
    # Specialists: their own test rows at the validation-chosen checkpoint, then the matrix for the rest.
    spec = {}
    std, obs = csv_row("obstacles-g995", "eval_log.csv", 699), csv_row("obstacles-g995", "eval_obstacles_log.csv", 699)
    spec["final"] = {"iteration": 699, "groups": {"basic": from_levels(std, obs, "basic")}}
    res["sources"]["final"] = ("basic: results/runs/obstacles-g995/eval_log.csv and eval_obstacles_log.csv, iteration 699; "
                               "advanced standard: results/neo/one_policy_comparison.json; the rest: test matrix (its standard "
                               "courses at extra levels 9-11 replay results/extra/course_preview_family1.json, 28 Oct, with "
                               "identical results)")
    l68 = json.load(open("results/neo/one_policy_comparison.json"))["final_policies"]["final_policy"]["levels_6_8"]
    adv_std = sum(l68[str(L)] for L in (6, 7, 8)) / 300.0
    spec["final"]["advanced_standard"] = adv_std
    for name, run, groups in (("g995-ice", "g995-ice", ("basic",)), ("g995-adv", "g995-adv", ("basic", "advanced")),
                              ("g995-extra", "g995-extra", ("basic", "extra"))):
        it, std, obs = summary_levels(run)
        spec[name] = {"iteration": it, "groups": {g: from_levels(std, obs, g) for g in groups}}
        res["sources"][name] = f"{', '.join(groups)}: results/runs/{run}/summary.json test_eval, iteration {it}; the rest: test matrix"
    std, obs = csv_row("neo-dagger3", "eval_log.csv", 449), csv_row("neo-dagger3", "eval_obstacles_log.csv", 449)
    spec["neo"] = {"iteration": 449, "groups": {g: from_levels(std, obs, g) for g in ("basic", "advanced")}}
    res["sources"]["neo"] = ("basic, advanced: results/runs/neo-dagger3/eval_log.csv and eval_obstacles_log.csv, iteration 449; "
                             "the rest: test matrix")
    for f in sorted(Path("results/skills").glob("test_matrix_*.json")):
        m = json.loads(f.read_text())
        assert m["split"] == "test"
        for name, c in m["candidates"].items():
            for g, v in c["groups"].items():
                cur = spec[name]["groups"].get(g)
                if cur is None:
                    spec[name]["groups"][g] = dict(v)
                elif v["standard"] is None and cur["standard"] is None:
                    raise ValueError(f"{name} {g}: two obstacle-only parts")
                else:
                    raise ValueError(f"{name} {g}: already covered; a test was rerun ({f})")
        res["sources"].setdefault("matrix_files", []).append(f.as_posix())
    # The one replay: the final policy's standard courses at extra levels 9-11 were first played on 28 Oct.
    first = json.load(open("results/extra/course_preview_family1.json"))
    assert first["checkpoint"] == "results/final_policy.pt" and first["family"] == T.TEST_SEED_FAMILY
    again = json.load(open("results/skills/test_matrix_extra.json"))["candidates"]["final"]["per_level"]["standard"]
    for gid, kind in zip((9, 10, 11), ("head hitter", "hard head hitter", "ice run-up")):
        assert round(100 * again[str(gid)]["success"], 6) == first["policy_alone"][kind], (gid, kind)
        assert round(100 * again[str(gid)]["fell"], 6) == first["policy_alone_fell"][kind], (gid, kind)
    res["replayed"] = ("the final policy's 600 standard courses at extra levels 9-11 (head hitter, hard head hitter, "
                       "ice run-up), first played on 28 Oct (results/extra/course_preview_family1.json), were played "
                       "again by test_matrix_extra.json with identical success and fall rates")
    # The final policy's advanced group: standard from the earlier file, obstacles from the matrix.
    a = spec["final"]["groups"]["advanced"]
    assert a["standard"] is None and a["obstacles"] is not None
    a["standard"] = spec["final"].pop("advanced_standard")
    a["all"] = (a["standard"] * 600 + a["obstacles"] * a["n"]) / (600 + a["n"])
    a["n"] += 600
    for name, c in spec.items():
        missing = [g for g in GROUPS if g not in c["groups"]]
        assert not missing, (name, missing)
        c["macro"] = T.macro_score({g: c["groups"][g]["all"] for g in GROUPS})
        res["candidates"][name] = c
    best = {g: max(spec, key=lambda n: spec[n]["groups"][g]["all"]) for g in GROUPS}
    res["best_specialist_per_group"] = {g: {"name": n, "all": spec[n]["groups"][g]["all"]} for g, n in best.items()}
    for run in STUDENTS:
        res["candidates"][run]["minus_best_specialist"] = {
            g: res["candidates"][run]["groups"][g]["all"] - spec[best[g]]["groups"][g]["all"] for g in GROUPS}
    Path(out_path).write_text(json.dumps(res, indent=1))
    for n, c in res["candidates"].items():
        print(f"{n:17s}", " ".join(f"{g} {100 * c['groups'][g]['all']:.2f}" for g in GROUPS), f"macro {100 * c['macro']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])

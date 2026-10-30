"""Tables and figures from finished runs.

    python -m parkour.report [--out results]

Reads runs/*/eval_log.csv (held-out test set, deterministic policy), copies
each run's small logs into OUT/runs/NAME/ for the record, and writes
OUT/README.md (tables) and OUT/figures/*.png. All numbers are simulator
results unless a table says otherwise. With the default OUT it also refreshes the
generated snapshot in HANDOFF.md (parkour.handoff).

It needs the training PC's runs/ folder: elsewhere (a fresh clone, another PC) it
stops instead of rewriting results/README.md without the sections whose runs are
missing; run `python -m parkour.handoff` there (or pass --force).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

import numpy as np

from . import handoff
from .paths import ROOT, RUNS

ABLATION_ITERATIONS = 1000
ARMS = {
    "full": ("progress + landing + goal + fall, adaptive curriculum", ["main-s0", "full-s1", "full-s2"]),
    "sparse": ("no progress (distance) shaping", ["sparse-s0", "sparse-s1", "sparse-s2"]),
    "nolanding": ("no landing bonus", ["nolanding-s0", "nolanding-s1", "nolanding-s2"]),
    "nofall": ("no fall penalty", ["nofall-s0", "nofall-s1", "nofall-s2"]),
    "uniform": ("no curriculum (levels uniform)", ["uniform-s0", "uniform-s1", "uniform-s2"]),
    "noroute": ("platform positions hidden: block scan + goal only", ["noroute-s0", "noroute-s1", "noroute-s2"]),
}
# Reference categorical palette (dataviz skill), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def read_eval(run: str, max_iteration: int | None = None) -> dict | None:
    p = RUNS / run / "eval_log.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open()))
    if max_iteration is not None:
        rows = [r for r in rows if int(r["iteration"]) <= max_iteration]
    if not rows:
        return None
    cols = rows[0].keys()
    return {c: np.array([float(r[c]) for r in rows]) for c in cols}


def levels_in(ev: dict) -> list[int]:
    return sorted(int(c.split("L")[1]) for c in ev if c.startswith("success_L"))


def _end_labels(ax, ends, gap=4.5, top=100.0):
    """Direct labels at line ends, spread downward from the top so they never overlap."""
    placed = []
    for x, text, y in sorted(ends, key=lambda e: -e[2]):
        y = min(y, top)
        if placed and placed[-1] - y < gap:
            y = placed[-1] - gap
        placed.append(y)
        ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points", color=INK2, fontsize=8, va="center",
                    annotation_clip=False)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_main(ev: dict, path: Path, title: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    _style(ax)
    x = ev["env_steps"] / 1e9
    ends = []
    for i, L in enumerate(levels_in(ev)):
        y = ev[f"success_L{L}"] * 100
        ax.plot(x, y, color=SERIES[i], linewidth=2, label=f"level {L}")
        ends.append((x[-1], f"L{L}", y[-1]))
    _end_labels(ax, ends)
    ax.set_xlabel("game ticks of experience (billions)", color=INK2)
    ax.set_ylabel("held-out courses completed (%)", color=INK2)
    ax.set_ylim(0, 102)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_ablation(curves: dict, path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    _style(ax)
    ends = []
    for i, (arm, runs) in enumerate(curves.items()):
        if not runs:
            continue
        n = min(len(r["success"]) for r in runs)
        x = runs[0]["env_steps"][:n] / 1e9
        ys = np.stack([r["success"][:n] for r in runs]) * 100
        ax.plot(x, ys.mean(0), color=SERIES[i], linewidth=2, label=f"{arm} ({len(runs)} seeds)")
        if len(runs) > 1:
            ax.fill_between(x, ys.min(0), ys.max(0), color=SERIES[i], alpha=0.15, linewidth=0)
        ends.append((x[-1], arm, ys.mean(0)[-1]))
    ax.set_xlabel("game ticks of experience (billions)", color=INK2)
    _end_labels(ax, ends)
    ax.set_ylabel("held-out courses completed, all levels (%)", color=INK2)
    ax.set_ylim(0, 100)
    ax.set_title("Reward and curriculum ablation (mean, band = min to max over seeds)", color=INK, fontsize=11,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_jumps(jump: dict, path: Path):
    """Two series only (possible / impossible for the scripts); levels are x bands."""
    import matplotlib.pyplot as plt
    from .feasibility import LEVEL_EDGES
    rows = jump["specs"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.set_xscale("symlog", linthresh=1)
    edges = [e * 100 for e in LEVEL_EDGES[:-1]]
    for e in edges:
        ax.axvline(e, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    bounds = [100] + edges + [0]
    for L in range(len(bounds) - 1):
        hi, lo = bounds[L], bounds[L + 1]
        mid = (hi * max(lo, 0.3)) ** 0.5 if lo > 0 else 0.5
        ax.text(mid, 107, f"L{L}", color=INK2, fontsize=8, ha="center")
    groups = (("possible for scripted strategies", SERIES[0], [r for r in rows if r["level"] is not None]),
              ("impossible for scripted strategies", SERIES[1], [r for r in rows if r["level"] is None]))
    for label, color, rl in groups:
        ax.scatter([r["forgiveness"] * 100 for r in rl], [r["policy_success"] * 100 for r in rl], s=30,
                   color=color, edgecolor=SURFACE, linewidth=1, label=f"{label} ({len(rl)})", zorder=3)
    ax.set_ylim(-5, 112)
    ax.set_xlabel("scripted strategies that land the jump (%), the jump's forgiveness", color=INK2)
    ax.set_ylabel("policy success over 16 start angles (%)", color=INK2)
    ax.set_title("Every jump type: learned policy vs scripted strategies", color=INK, fontsize=11, loc="left", pad=18)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def read_train(run: str) -> dict | None:
    p = RUNS / run / "train_log.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open()))
    out = {}
    for c in rows[0]:
        out[c] = np.array([float(r[c]) if r[c] not in ("", None) else np.nan for r in rows])
    return out


def _smooth(y: np.ndarray, k: int = 25) -> np.ndarray:
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - k + 1): i + 1]
        w = w[~np.isnan(w)]
        out[i] = w.mean() if len(w) else np.nan
    return out


def fig_training(run: str, path: Path, title: str) -> bool:
    """Ten panels from one run's per-iteration log: what training did, iteration by iteration."""
    import matplotlib.pyplot as plt
    tr = read_train(run)
    if tr is None:
        return False
    ev = read_eval(run)
    x = tr["env_steps"] / 1e9
    panels = [
        ("Success (% of finished episodes)", [("training episodes", tr["success"] * 100, True)]),
        ("Mean episode return (reward)", [("return", tr["ep_return"], True)]),
        ("Falls (% of finished episodes)", [("fell", tr["fell"] * 100, True)]),
        ("Episode length (ticks)", [("length", tr["ep_len"], True)]),
        ("Curriculum frontier (level)", [("frontier", tr["frontier"], False)]),
        ("Policy entropy", [("entropy", tr["entropy"], True)]),
        ("Policy loss (PPO clipped objective)", [("policy loss", tr["pi_loss"], True)]),
        ("Value loss", [("value loss", tr["v_loss"], True)]),
        ("Approximate KL per update", [("KL", tr["kl"], True)]),
        ("Learning rate (KL-adaptive)" if "lr" in tr else "Learning rate (fixed, before the PPO fixes)",
         [("lr", tr.get("lr", np.full_like(x, 3e-4)), "log")]),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(18, 6.4), facecolor=SURFACE)
    for ax, (name, series) in zip(axes.flat, panels):
        _style(ax)
        for label, y, smooth in series:
            if smooth == "log":  # geometric mean of a quantity that jumps by factors of 1.5
                ax.plot(x, y, color=SERIES[0], linewidth=0.6, alpha=0.2)
                ax.plot(x, 10 ** _smooth(np.log10(y)), color=SERIES[0], linewidth=2, label=label)
            elif smooth:
                ax.plot(x, y, color=SERIES[0], linewidth=0.6, alpha=0.25)
                ax.plot(x, _smooth(y), color=SERIES[0], linewidth=2, label=label + " (25-iteration mean)")
            else:
                ax.plot(x, y, color=SERIES[0], linewidth=1.5, label=label)
        if name.startswith("Success") and ev is not None:
            ax.plot(ev["env_steps"] / 1e9, ev["success"] * 100, color=SERIES[1], linewidth=2, marker="o",
                    markersize=3, label="held-out test courses")
            ax.legend(frameon=False, fontsize=7, labelcolor=INK2, loc="lower right")
        if name.startswith("Learning rate"):
            ax.set_yscale("log")
        ax.set_title(name, color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("game ticks (billions)", color=INK2, fontsize=8)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


RUN_FILES = ("config.json", "train_log.csv", "eval_log.csv", "eval_val_log.csv", "eval_obstacles_log.csv",
             "summary.json")
GAME_FILES = ("courses.json", "summary.json", "summary.md")


def archive(out: Path) -> None:
    """Copy every run's records into ``out`` (checkpoints stay in runs/, they are large).

    out/runs/NAME              training runs: config, per-iteration log, evaluation logs
    out/runs/_v1_unstable/NAME the first runs, discarded after PPO collapsed (kept as evidence)
    out/games/NAME             bot runs (live, offline): courses, every tick (episodes.jsonl.gz),
                               summaries, and the server log of live runs
    """
    sources = [(d, out / "runs" / d.name) for d in sorted(RUNS.iterdir())] if RUNS.is_dir() else []
    if (RUNS / "_v1_unstable").is_dir():
        sources += [(d, out / "runs" / "_v1_unstable" / d.name) for d in sorted((RUNS / "_v1_unstable").iterdir())]
    for d, dst in sources:
        if d.is_dir() and (d / "train_log.csv").exists():
            dst.mkdir(parents=True, exist_ok=True)
            for f in RUN_FILES:
                if (d / f).exists():
                    shutil.copy2(d / f, dst / f)
    for d in sorted(RUNS.rglob("episodes.jsonl")):
        run = d.parent
        dst = out / "games" / run.relative_to(RUNS).as_posix().replace("/", "__")
        dst.mkdir(parents=True, exist_ok=True)
        for f in GAME_FILES:
            if (run / f).exists():
                shutil.copy2(run / f, dst / f)
        for f in ("episodes.jsonl", "server.log"):
            if (run / f).exists():
                # mtime=0: the same records give the same bytes, so re-archiving does not touch Git.
                with open(run / f, "rb") as src, gzip.GzipFile(dst / (f + ".gz"), "wb", compresslevel=9, mtime=0) as gz:
                    shutil.copyfileobj(src, gz)


def game_row(out: Path, name: str, label: str | None = None) -> str | None:
    sp = out / "games" / name / "summary.json"
    if not sp.exists():
        return None
    s = json.loads(sp.read_text())
    eps = s["episodes"]
    label = label or name
    if s.get("assist", {}).get("planner_errors"):
        label += f" (degraded assist: {s['assist']['planner_errors']} planner errors)"
    corrections = "-" if s['mode'].startswith('offline') else sum(e.get('server_corrections') or 0 for e in eps)
    return (f"| {label} | {s['mode']} | {sum(e['result'] == 'success' for e in eps)}/{len(eps)} | "
            f"{pct(s['sim_success_same_courses'])} | {pct(s['outcome_agreement'])} | "
            f"{corrections} | {s['replay_max_pos_err']:.1e} |")


GAME_HEADER = ["| Run | Where | Completed | Simulator, same courses | Same outcome | Server corrections | Replay error (blocks) |",
               "| --- | --- | --- | --- | --- | --- | --- |"]


def pct(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}"


L5_RUNS = ("obstacles-g995", "l5-obs05", "l5-obs05-s1", "l5-obs05-s2", "l5-obs05-w3", "l5-obs05-w3-s1", "l5-obs05-w3-s2",
           "l5-w3", "l5-w3-s1")


def level5_recipe(out: Path) -> list:
    """Fine-tunes of the final policy for level 5 with obstacles (30 Oct): reported, not shipped."""
    f = out / "level5_closer_look.json"
    if not f.exists() or not all((RUNS / n / "summary.json").exists() for n in L5_RUNS):
        return []
    d = json.loads(f.read_text())
    sets = {**d["seeds"]["completed_of_2000"], **d["ablation"]["runs"], "obstacles-g995": d["seeds"]["final_policy"]}
    groups, cfgs = {}, {}
    for name in L5_RUNS:
        run = RUNS / name
        cfgs[name] = cfg = json.loads((run / "config.json").read_text())
        it = json.loads((run / "summary.json").read_text())["best"]
        t = {int(r["iteration"]): r for r in csv.DictReader(open(run / "eval_log.csv"))}[it["iteration"]]
        o = {int(r["iteration"]): r for r in csv.DictReader(open(run / "eval_obstacles_log.csv"))}[it["iteration"]]
        groups.setdefault((cfg["obstacle_prob"], cfg.get("level_weights") or "adaptive"), []).append(
            (name, it["iteration"], it["validation_success"], sets[name]["46"], sets[name]["47"],
             *(float(x) for x in (t["success"], t["success_L5"], o["success"], o["success_L5"]))))
    setting = lambda c: (f"{c['num_envs']:,} games, rollout {c['rollout']}, {c['pool_size']:,} courses in the pool, "
                         f"{c['minibatches']} minibatches")
    fp, ft = cfgs["obstacles-g995"], cfgs["l5-obs05"]
    same = all(setting(cfgs[n]) == setting(ft) and cfgs[n]["iterations"] == ft["iterations"] for n in L5_RUNS[1:])
    ticks = {c["num_envs"] * c["rollout"] for c in (fp, ft)}
    lines = ["### Level 5 with obstacles: a fine-tune recipe (reported, not shipped)", "",
             f"The final policy (first row) against fine-tunes of it. The fine-tunes run {ft['iterations']:,} iterations at "
             f"discount {ft['gamma']} with the training queue's settings ({setting(ft)}"
             + ("" if same else "; settings differ between the runs, see each `config.json`")
             + f"; the final policy's own obstacle fine-tune had {setting(fp)}, "
             + (f"the same {ticks.pop():,} game ticks per update" if len(ticks) == 1 else "other ticks per update")
             + "), choose their checkpoint on validation and play each test set once (the final policy's run scored its "
             "test sets at every checkpoint, without choosing on them).",
             "They change the share of jumps with obstacles (`--obstacle-prob`; the final policy trained with 0.3, the",
             "obstacle test sets have 0.5), the mix of levels (`--level-weights 1,1,1,1,1,3`: level 5 weighted 3 against 1",
             "for each other level, so 3 of every 8 training games, where the adaptive curriculum gives each level 1 of 6",
             "once it reaches level 5), or both; none has neither change. Families 46 and 47 are new validation-only sets",
             "of 2,000 level-5 courses each, with obstacles on half the jumps and without",
             "(`results/level5_closer_look.json`, deterministic first episode).", "",
             "| Obstacles | Levels | Run | Iteration | Validation | Family 46 | Family 47 | Test, standard | L5 | "
             "Test, obstacles | Obstacles L5 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    mean = {}
    for (prob, weights), rows in groups.items():
        for name, it, val, f46, f47, *test in rows:
            lines.append(f"| {prob} | {weights} | `{name}` | {it} | {pct(val)} | {f46:,} ({f46 / 20:.2f}) | "
                         f"{f47:,} ({f47 / 20:.2f}) | " + " | ".join(pct(v) for v in test) + " |")
        mean[prob, weights] = m = [sum(c) / len(rows) for c in list(zip(*rows))[3:]]
        if len(rows) > 1:
            lines.append(f"| {prob} | {weights} | mean of {len(rows)} | | | {m[0] / 20:.2f} | {m[1] / 20:.2f} | "
                         + " | ".join(pct(v) for v in m[2:]) + " |")
    base, both, obs, lw = (mean[k] for k in ((0.3, "adaptive"), (0.5, "1,1,1,1,1,3"), (0.5, "adaptive"), (0.3, "1,1,1,1,1,3")))
    gain = lambda m, i: f"{(m[i] - base[i]) / 20:+.2f}"
    above = all(r[3] > base[0] for r in groups[0.5, "1,1,1,1,1,3"])
    lines += ["", "Family 46 was fixed at 06:39:26 (HANDOFF section 3), after the first two runs (`l5-obs05`, `l5-obs05-w3`) were",
              "tested and before any run was played on it; the other six runs came after it. One run's bar was 2 points",
              "above the final policy; the seeds compared the recipes' means (with a family-47 guard), and the level",
              f"weights alone had to be 2 points above on their mean. Mean of the seeds on family 46: both changes {gain(both, 0)}"
              + (" (every seed above the final policy)" if above else "") + f", obstacles alone {gain(obs, 0)}, the level "
              f"weights alone {gain(lw, 0)}; on family 47 {gain(both, 1)}, {gain(obs, 1)} and {gain(lw, 1)}. On the obstacle "
              f"test set's level 5 (200 courses per run, played once) the means are {pct(both[5])}, {pct(obs[5])} and "
              f"{pct(lw[5])} against {pct(base[5])}: the test does not confirm that both changes are needed, and with no "
              "fine-tune that has neither change it cannot separate them from the extra training they share.",
              "Reported, not shipped: the final policy stays `results/final_policy.pt` (decided 30 Oct). The planner, the",
              "neo planner, the router and the final policy's bot runs were all measured with it.", ""]
    return lines


def new_abilities(out: Path) -> list:
    """Tables of the planning assist, the extra jump kinds and the new blocks' physics checks (simulator)."""
    lines = []
    runs = []
    for split in ("validation", "test"):
        f = out / "neo" / f"assist_{split}.jsonl"
        if f.exists():
            runs += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    if runs:
        lines += ["## Planning the hard jumps: the neo assist (simulator)", "",
                  "When the bot first stands on a planned takeoff, a cross-entropy search with the exact physics finds an",
                  "action sequence from its exact state and the bot plays it (`parkour/neo_assist.py`). Jumps planned:",
                  "`neo` the neos; `advanced` also the wide and momentum jumps; `all` every jump. Official: the held-out",
                  "level 6-8 courses with a neo (family 2 validation, family 1 test); random: courses of level 0-5 jumps",
                  "where half the jumps are neos (13 validation, 12 test); mixed: ordinary and special jump kinds",
                  "including the extra levels (15 validation, 14 test). Settings are chosen on validation; test lines",
                  "are the chosen settings.", "",
                  "| Split | Courses | Policy | Setting | Jumps planned | Completed | Neo jumps passed / reached | "
                  "Plans landed / played |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in runs:
            ckpt = Path(r["checkpoint"]).stem
            setting = "policy alone" if not r["assist"] else (r.get("tag") or "assist").split(";")[0]
            if r["assist"] and r["config"].get("batch", 1) != 1:  # searches batched: other random draws, not the bots'
                setting += f" (batch {r['config']['batch']})"
            jumps = r["config"]["jumps"] if r["assist"] else ""
            plans = (f"{r['plans_landed_as_planned']}/{r['plans_played']}" if r.get("plans_played") is not None else "")
            lines.append(f"| {r['split']} | {r['set']} (family {r['family']}) | `{ckpt}` | {setting} | {jumps} | "
                         f"{r['completed']}/{r['courses']} ({pct(r['completed'] / r['courses'])}) | "
                         f"{r['neo_jumps_passed']}/{r.get('neo_jumps_reached', '?')} | {plans} |")
        lines.append("")
    games = sorted((out / "games").glob("neo-assist-*/summary.json"))
    if games:
        rows = []
        for g in games:
            a = json.loads(g.read_text()).get("assist", {}).get("config", {})
            extra = [f"{k} {a[k]}" for k in ("retries", "brake", "approach") if a.get(k)]
            rows.append(game_row(out, g.parent.name, f"{g.parent.name} (jumps {a.get('jumps', '?')}"
                                 + (", " + ", ".join(extra) if extra else "") + ")"))
        lines += ["The assist in the bots, on the neo course set of the neo policy's runs (`configs/neo_live_courses.json`:",
                  "the first 10 held-out test courses with a neo of levels 6, 7 and 8, and the showcase neo course; the",
                  "neo policy alone made 11/31 there, offline and in real Minecraft). The final policy plays; the bot",
                  "pauses for the Python planner service and checks every plan with prismarine-physics:", ""] + GAME_HEADER + rows + [""]
    je = out / "extra" / "jump_eval_final_policy.json"
    previews = [(f, json.loads(f.read_text())) for f in (out / "extra" / f"course_preview_family{F}.json" for F in (2, 1))
                if f.exists()]
    if je.exists() or previews:
        lines += ["## New jump kinds and blocks: head hitters, ice run-ups, shaped blocks, slime, ladders (simulator)", "",
                  "Opt-in levels (`configs/jump_levels_extra.json`, `run.cmd train --extra`) that the final policy was",
                  "never trained on. Every kind alone from rest at 16 angles (`parkour.jumpeval --extra`) and whole",
                  "held-out courses of each level (`scripts/extra_preview.py`: family 2 validation, family 1 test), the",
                  "policy alone and with the planner (jumps `advanced`: the run-up jumps; `all`: every jump) on the first",
                  "courses of each level.", ""]
        if je.exists():
            j = json.loads(je.read_text())
            lines += ["| Level | Kinds | Mean success alone | Kinds never made |", "| --- | --- | --- | --- |"]
            for name, v in j["by_level"].items():
                label = "no scripted landing" if name == "impossible" else name
                lines.append(f"| {label} | {v['specs']} | {pct(v['mean_policy_success'])} | {v['specs_policy_never']} |")
            lines.append("")
        for f, c in previews:
            settings = c.get("assist_configs") or {"": c.get("assist_config", {})}
            planner = c["with_planner"] if "assist_configs" in c else {"": c["with_planner"]}
            heads = [f"planner, jumps {cfg.get('jumps', '?')}" for cfg in settings.values()]
            lines += [f"Family {c['family']} ({'test' if c['family'] == 1 else 'validation'}, `{f.name}`):", "",
                      "| Level | Policy alone (all courses) | Same first courses, policy alone | " + " | ".join(heads) + " |",
                      "| --- | --- | --- | " + " | ".join("---" for _ in heads) + " |"]
            for name, v in c["policy_alone"].items():
                ws = [planner[s].get(name, {}) for s in settings]
                first = next((w for w in ws if w), {})
                lines.append(f"| {name} | {v}% of {c['courses_per_level']} | "
                             f"{first.get('completed_policy_alone', '')}/{first.get('courses', '')} | " +
                             " | ".join(f"{w.get('completed', '')}/{w.get('courses', '')}" for w in ws) + " |")
            lines.append("")
    rows = []
    for name, label in (("physics_parity_shapes.json", "block shapes"), ("physics_parity_slime.json", "slime"),
                        ("physics_parity_climb.json", "ladders and vines")):
        f = out / name
        if f.exists():
            d = json.loads(f.read_text())
            r = d["result"]
            eff = d.get("effect_of_shapes") or d.get("effect_of_slime") or d.get("effect_of_climbing")
            rows.append(f"| {label} | {r['cases']} | {r['ticks']} | {r['max_pos_err']:.1e} | {r['flag_mismatch_ticks']} | "
                        f"{eff['cases_changed']}/{eff['cases']} |")
    if rows:
        lines += ["## The new blocks against prismarine-physics (physics library)", "",
                  "| Blocks | Random fields | Ticks | Largest position error (blocks) | Ticks with a flag mismatch | "
                  "Trajectories the blocks change |", "| --- | --- | --- | --- | --- | --- |"] + rows + [""]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--main", default="main-s0")
    ap.add_argument("--force", action="store_true", help="write even without the training PC's runs/ folder")
    a = ap.parse_args(argv)
    if not (RUNS / a.main / "eval_log.csv").exists() and not a.force:
        raise SystemExit(f"{RUNS / a.main} is missing: this checkout does not have the training PC's runs/, and the report "
                         "would drop the sections built from them. Run `python -m parkour.handoff` instead, or pass --force.")
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    archive(out)

    lines = ["# Results", "", "Generated by `python -m parkour.report`. Unless a section says otherwise, numbers are",
             "**simulator** results on the fixed held-out test set: 200 courses per difficulty level, 10 jumps each,",
             "never seen in training, played by the deterministic (argmax) policy.", ""]

    main_ev = read_eval(a.main)
    if main_ev is not None:
        fig_main(main_ev, out / "figures" / "main_learning_curve.png",
                 f"{a.main}: held-out success by difficulty level")
        best = int(np.argmax(main_ev["success"]))
        lv = levels_in(main_ev)
        lines += [f"## Main run `{a.main}`", "",
                  f"![learning curve](figures/main_learning_curve.png)", "",
                  "| Evaluation | Game ticks | All levels | " + " | ".join(f"L{L}" for L in lv) + " |",
                  "| --- | --- | --- | " + " | ".join("---" for _ in lv) + " |"]
        last = len(main_ev["success"]) - 1
        has_val = (RUNS / a.main / "eval_val_log.csv").exists()
        if has_val:
            v = list(csv.DictReader((RUNS / a.main / "eval_val_log.csv").open()))
            best = int(np.argmax([float(r["success"]) for r in v]))
            best_name = "best on validation"
        else:
            best_name = "highest test score (picked on the test set itself, so optimistic)"
        for name, i in (("last checkpoint (headline)", last), (best_name, best)):
            lines.append(f"| {name} (iteration {int(main_ev['iteration'][i])}) | {main_ev['env_steps'][i] / 1e9:.2f} B | "
                         f"{pct(main_ev['success'][i])} | " + " | ".join(pct(main_ev[f'success_L{L}'][i]) for L in lv) + " |")
        lines.append("")

    curves, table = {}, []
    for arm, (desc, runs) in ARMS.items():
        evs = [e for e in (read_eval(r, ABLATION_ITERATIONS - 1) for r in runs) if e is not None]
        done = [e for e in evs if int(e["iteration"][-1]) >= ABLATION_ITERATIONS - 1]
        curves[arm] = done
        if not done:
            table.append((arm, desc, 0, None, None, None, None))
            continue
        final = np.array([e["success"][-1] for e in done])
        per_level = np.array([[e[f"success_L{L}"][-1] for L in levels_in(e)] for e in done])
        auc = np.array([e["success"].mean() for e in done])
        reach = []
        for e in done:
            hit = np.nonzero(e["success"] >= 0.8)[0]
            reach.append(e["env_steps"][hit[0]] / 1e9 if len(hit) else np.nan)
        table.append((arm, desc, len(done), final, per_level, auc, np.array(reach)))
    if any(t[2] for t in table):
        fig_ablation(curves, out / "figures" / "ablation_curves.png")
        lines += ["## Reward and curriculum ablation", "",
                  f"Each arm trains {ABLATION_ITERATIONS} PPO iterations (0.52 B game ticks) per seed; the `full` arm's",
                  "seed 0 is the first 1000 iterations of the main run (identical settings).", "",
                  "![ablation](figures/ablation_curves.png)", "",
                  "| Arm | What changes | Seeds | Final success, mean (min-max) | L4 | L5, mean (min-max) | Mean over training | Ticks to 80% (B) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for arm, desc, n, final, per_level, auc, reach in table:
            if not n:
                lines.append(f"| {arm} | {desc} | 0 | not finished | | | | |")
                continue
            r = reach[~np.isnan(reach)]
            lines.append(f"| {arm} | {desc} | {n} | {pct(final.mean())} ({pct(final.min())}-{pct(final.max())}) | "
                         f"{pct(per_level[:, 4].mean())} | {pct(per_level[:, 5].mean())} ({pct(per_level[:, 5].min())}-{pct(per_level[:, 5].max())}) | {pct(auc.mean())} | "
                         f"{'%.2f' % r.mean() if len(r) else 'never'}{'' if len(r) == n else f' ({len(r)}/{n} seeds)'} |")
        lines.append("")

    jpath = out / "jump_eval.json"
    if jpath.exists():
        jump = json.loads(jpath.read_text())
        fig_jumps(jump, out / "figures" / "jumps.png")
        lines += ["## Every jump type", "", f"Checkpoint `{jump['checkpoint']}`. Each jump type starts from rest on a 1x1",
                  "block, with 16 start angles from -60 to +60 degrees off the target.", "",
                  "![jumps](figures/jumps.png)", "",
                  "| Level | Jump types | Mean policy success | Types never made |", "| --- | --- | --- | --- |"]
        for k, v in jump["by_level"].items():
            lines.append(f"| {k} | {v['specs']} | {pct(v['mean_policy_success'])} | {v['specs_policy_never']} |")
        lines.append("")

    hp_runs = sorted(d.name for d in RUNS.iterdir() if d.is_dir() and d.name.startswith("hp-"))
    if hp_runs:
        lines += ["## Hyperparameters (one at a time around the defaults, seed 0)", "",
                  "Selection uses the validation courses (seed family 2); the test score is reported alongside. The",
                  "reference is the `full` arm (defaults), mean over its seeds. All runs see 0.52 B game ticks.", "",
                  "| Run | Iteration | Validation success | Test success | L4 | L5 |", "| --- | --- | --- | --- | --- | --- |"]
        ref = curves.get("full") or []
        if ref:
            dv = out / "hparams_defaults_validation.json"
            dval = [v for k, v in json.loads(dv.read_text()).items() if k != "method"] if dv.exists() else []
            vtxt = f"{pct(np.mean(dval))} ({len(dval)} seeds)" if dval else "-"
            lines.append(f"| defaults (`full`, {len(ref)} seeds) | 999 | {vtxt} | {pct(np.mean([e['success'][-1] for e in ref]))} | "
                         f"{pct(np.mean([e['success_L4'][-1] for e in ref]))} | {pct(np.mean([e['success_L5'][-1] for e in ref]))} |")
        for r in hp_runs:
            e = read_eval(r)
            vp = RUNS / r / "eval_val_log.csv"
            v = list(csv.DictReader(vp.open())) if vp.exists() else []
            if e is None:
                continue
            its = json.loads((RUNS / r / "config.json").read_text())["iterations"]
            it = int(e["iteration"][-1])
            state = str(it) if it >= its - 1 else f"{it} of {its} (running)"
            lines.append(f"| `{r}` | {state} | {pct(float(v[-1]['success'])) if v else '-'} | {pct(e['success'][-1])} | "
                         f"{pct(e['success_L4'][-1])} | {pct(e['success_L5'][-1])} |")
        lines.append("")

    spath = out / "stress.json"
    if spath.exists():
        st = json.loads(spath.read_text())
        conds = st["conditions"]
        lv = sorted({int(L) for c in conds.values() for L in c["per_level"]})
        lines += ["## Course styles outside the training distribution", "",
                  f"Checkpoint `{st['checkpoint']}`, {st['per_level']} new courses per level and condition "
                  "(`python -m parkour.stress`).", "",
                  "| Condition | All levels | " + " | ".join(f"L{L}" for L in lv) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in lv) + " |"]
        for name, c in conds.items():
            lines.append(f"| {name} | {pct(c['success'])} | " + " | ".join(pct(c['per_level'].get(str(L), c['per_level'].get(L))) for L in lv) + " |")
        lines.append("")

    obs_ev = read_eval("obstacles-ft")
    obs_o = None
    p = RUNS / "obstacles-ft" / "eval_obstacles_log.csv"
    if p.exists():
        rows = list(csv.DictReader(p.open()))
        obs_o = {c: np.array([float(r[c]) for r in rows]) for c in rows[0]} if rows else None
    if obs_ev is not None and obs_o is not None:
        lv = levels_in(obs_ev)
        lines += ["## Obstacles (head-hitter beams and decoy pillars)", "",
                  "`obstacles-ft` starts from the main policy and trains with an obstacle on 30% of jumps. Both test",
                  "sets are held out; the obstacle set puts an obstacle on 50% of jumps."]
        zs = out / "stress_main.json"
        if zs.exists():
            z = json.loads(zs.read_text())["conditions"].get("obstacles")
            if z:
                lines += [f"Before fine-tuning, the main policy completed {pct(z['success'])}% of new obstacle courses",
                          "(stress test, 100 per level), against 96.7% of standard ones."]
        lines += ["",
                  "| Evaluation | Test set | All levels | " + " | ".join(f"L{L}" for L in lv) + " |",
                  "| --- | --- | --- | " + " | ".join("---" for _ in lv) + " |"]
        for name, i in (("first", 0), ("last", -1)):
            it = int(obs_ev["iteration"][i])
            lines.append(f"| {name} (iteration {it}) | standard | {pct(obs_ev['success'][i])} | "
                         + " | ".join(pct(obs_ev[f'success_L{L}'][i]) for L in lv) + " |")
            lines.append(f"| {name} (iteration {it}) | obstacles | {pct(obs_o['success'][i])} | "
                         + " | ".join(pct(obs_o[f'success_L{L}'][i]) for L in lv) + " |")
        lines.append("")

    plan_std, plan_obs = out / "planner" / "sim_test_standard.json", out / "planner" / "sim_test_obstacles.json"
    if plan_std.exists() and plan_obs.exists():
        lines += ["## Without the course list: greedy route planner", "",
                  "The final policy, unchanged, with its platform slots filled by `parkour.planner` from the blocks",
                  "within 12 blocks of the bot (only the goal position is given; settings chosen on the validation",
                  "courses). \"Remaining\" counts a drop onto any later, lower platform as a landing, not a fall.", "",
                  "| Test set | Fall rule | Course list given | Planner | " + " | ".join(f"L{L}" for L in range(6)) + " |",
                  "| --- | --- | --- | --- | " + " | ".join("---" for _ in range(6)) + " |"]
        for label, path in (("standard", plan_std), ("obstacles", plan_obs)):
            d = json.loads(path.read_text())
            for rule in ("local", "remaining"):
                g, p = d[rule]["given_route"], d[rule]["planner_route"]
                lines.append(f"| {label} ({d['courses']}) | {rule} | {pct(g['success'])} | {pct(p['success'])} | "
                             + " | ".join(pct(p["per_level"][str(L)]["success"]) for L in range(6)) + " |")
        lines.append("")
        rows = [r for r in (game_row(out, n) for n in ("v2-planner-offline", "v2-planner-offline-obstacles", "v2-planner-live",
                                                       "v2-planner-live-obstacles")) if r]
        tuned = [(k, out / "planner" / f"sim_test_{k}_tuned.json") for k in ("standard", "obstacles")]
        if all(f.exists() for _, f in tuned):
            cfg = json.loads(tuned[0][1].read_text())["planner"]
            lines += [f"Retuned for the final policy on the validation courses (`cost={cfg['cost']}`, `align={cfg['align']}`;",
                      "every variant tried is in `results/planner/validation_tuning.jsonl`), the test sets used once:", "",
                      "| Test set | Fall rule | Planner, retuned | " + " | ".join(f"L{L}" for L in range(6)) + " |",
                      "| --- | --- | --- | " + " | ".join("---" for _ in range(6)) + " |"]
            for label, path in tuned:
                d = json.loads(path.read_text())
                for rule in ("local", "remaining"):
                    pr = d[rule]["planner_route"]
                    lines.append(f"| {label} ({d['courses']}) | {rule} | {pct(pr['success'])} | "
                                 + " | ".join(pct(pr["per_level"][str(L)]["success"]) for L in range(6)) + " |")
            lines.append("")
            rows += [r for r in (game_row(out, n, n + " (retuned planner)") for n in (
                "v2-tuned-planner-offline", "v2-tuned-planner-offline-obstacles")) if r]
        rows += [r for r in (game_row(out, n, n + " (first final policy)") for n in (
            "planner-offline", "planner-offline-obstacles", "planner-live", "planner-live-obstacles")) if r]
        v1 = [(out / "planner" / f"sim_test_{k}_v1.json") for k in ("standard", "obstacles")]
        if all(f.exists() for f in v1):
            a_, b_ = (json.loads(f.read_text())["local"]["planner_route"]["success"] for f in v1)
            lines += [f"With the first final policy the planner made {pct(a_)} (standard) and {pct(b_)} (obstacles) on the",
                      "same test sets (`results/planner/sim_test_*_v1.json`).", ""]
        if rows:
            lines += GAME_HEADER + rows + [""]

    nr = RUNS / "noroute-v2" / "summary.json"
    if nr.exists():
        best = json.loads(nr.read_text())["best"]
        it = best["iteration"]
        tl = list(csv.DictReader(open(RUNS / "noroute-v2" / "eval_log.csv")))
        ol = list(csv.DictReader(open(RUNS / "noroute-v2" / "eval_obstacles_log.csv")))
        t = next(r for r in tl if int(r["iteration"]) == it)
        o = next(r for r in ol if int(r["iteration"]) == it)
        lines += ["## Learned no-route model `noroute-v2`", "",
                  "Trained from scratch for 3,000 iterations (1.57 B ticks, obstacles on 30% of jumps) without the",
                  "platform slots or the progress input; instead a wide coarse block scan and its last three landing",
                  "spots (`EnvConfig.extended_obs`). The route-based progress reward is used in training only.",
                  f"Checkpoint chosen on the validation courses: iteration {it} (validation {pct(best['validation_success'])}).", "",
                  "| Test set | All levels | " + " | ".join(f"L{L}" for L in range(6)) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in range(6)) + " |",
                  f"| standard (1200) | {pct(float(t['success']))} | " + " | ".join(pct(float(t[f'success_L{L}'])) for L in range(6)) + " |",
                  f"| obstacles (1200) | {pct(float(o['success']))} | " + " | ".join(pct(float(o[f'success_L{L}'])) for L in range(6)) + " |", ""]
        rows = [r for r in (game_row(out, n) for n in ("noroute-offline", "noroute-offline-obstacles", "noroute-live",
                                                       "noroute-live-obstacles")) if r]
        if rows:
            lines += GAME_HEADER + rows + [""]

    # The final policy since 28 October: discount 0.995 (section "Hyperparameters").
    v2 = RUNS / "obstacles-g995" / "summary.json"
    v1 = RUNS / "obstacles-ft" / "summary.json"
    if v2.exists() and v1.exists():
        lines += ["## Final policy: discount 0.995", "",
                  "The hyperparameter study's only setting that beat the defaults by more than 3 points on validation",
                  "(discount 0.995, all three seeds) was trained with the main run's recipe (`main-g995`, 3,000",
                  "iterations) and the same obstacle fine-tune (`obstacles-g995`, 1,000 iterations). Chosen on validation",
                  "against the first final policy (`obstacles-ft`); since then `results/final_policy.pt` (the first one is",
                  "`results/final_policy_v1.pt`).", "",
                  "| Policy | Iteration | Validation | Test, standard | L5 | Test, obstacles | Obstacles L5 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, path in (("first final policy `obstacles-ft` (discount 0.99)", v1), ("final policy `obstacles-g995` (discount 0.995)", v2)):
            b = json.loads(path.read_text())["best"]
            run = path.parent
            t = {int(r["iteration"]): r for r in csv.DictReader(open(run / "eval_log.csv"))}[b["iteration"]]
            o = {int(r["iteration"]): r for r in csv.DictReader(open(run / "eval_obstacles_log.csv"))}[b["iteration"]]
            lines.append(f"| {name} | {b['iteration']} | {pct(b['validation_success'])} | {pct(float(t['success']))} | "
                         f"{pct(float(t['success_L5']))} | {pct(float(o['success']))} | {pct(float(o['success_L5']))} |")
        lines.append("")
        rows = [r for r in (game_row(out, n) for n in ("v2-offline", "v2-offline-obstacles", "v2-offline-showcase",
                                                       "v2-live", "v2-live-obstacles", "v2-live-showcase")) if r]
        if rows:
            lines += ["The bot with the final policy, on the same 60 + 60 held-out courses as before and the four",
                      "hand-designed courses (the neo course is not in its training):", ""] + GAME_HEADER + rows + [""]
        lines += level5_recipe(out)

    # Neo jumps: learning from demonstrations.
    ne = out / "neo" / "course_eval.json"
    n3 = out / "neo" / "round3_eval.json"
    if ne.exists():
        d = json.loads(ne.read_text())
        lines += ["## Neo jumps: learning from demonstrations", "",
                  f"Held-out test courses of levels 6-8 that contain a neo: {d['courses']} courses, {d['neo_jumps']} neo jumps.",
                  "Neo jumps passed counts the neo jumps the bot reached and got past. Round 3's checkpoint was chosen",
                  "on the validation family's neo courses.", "",
                  "| Neo policy | Courses completed | Neo jumps passed / reached |", "| --- | --- | --- |"]
        for name, r in d["results"].items():
            lines.append(f"| {name} | {r['courses']}/{d['courses']} ({pct(r['courses'] / d['courses'])}) | "
                         f"{r['neo_passed']}/{r['neo_reached']} ({pct(r['neo_passed'] / max(r['neo_reached'], 1))}) |")
        if n3.exists():
            for key, (cs, n, passed, reached) in json.loads(n3.read_text()).items():
                if key.startswith("test") and "neo-dagger3" in key and "best" in key:
                    lines.append(f"| + DAgger round 3 (neo-dagger3, chosen on validation) | {cs}/{n} ({pct(cs / n)}) | "
                                 f"{passed}/{reached} ({pct(passed / max(reached, 1))}) |")
        lines.append("")
        rows = [r for r in (game_row(out, n) for n in ("neo-offline-final", "neo-live-final")) if r]
        if rows:
            lines += ["The neo policy on the neo course set (`configs/neo_live_courses.json`: the first 10 held-out test",
                      "courses with a neo of levels 6, 7 and 8, and the showcase neo course):", ""] + GAME_HEADER + rows + [""]
        op = out / "neo" / "one_policy_comparison.json"
        if op.exists():
            c = json.loads(op.read_text())
            lines += ["One policy for everything? Levels 0-5 of the advanced test set are the standard test courses, so the",
                      "neo policies' scores there compare with the final policies' (`scripts/one_policy_check.py`; neo",
                      "policies at their validation-chosen checkpoint; their neo-course results are in the table above):", "",
                      "| Policy | Standard, L0-5 | Obstacles, L0-5 | L6 | L7 | L8 | Neo courses completed |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for name, f in c["final_policies"].items():
                lines.append(f"| `{name}` | {f['standard_levels_0_5']} | {f['obstacles_levels_0_5']} | "
                             + " | ".join(str(f["levels_6_8"][str(L)]) for L in (6, 7, 8))
                             + f" | {f['neo_courses_completed']}/{f['neo_courses']} |")
            for name, r in c["neo_policies"].items():
                lines.append(f"| `{name}` (iteration {r['iteration']}) | {r['standard_levels_0_5']} | "
                             f"{r['obstacles_levels_0_5']} | " + " | ".join(str(r["per_level"][str(L)]) for L in (6, 7, 8))
                             + " | |")
            lines.append("")

    lines += new_abilities(out)

    for name in ("offline", "live", "offline_v1", "live_v1"):
        p = out / f"{name}_summary.md"
        if p.exists():
            lines += [f"## {name.capitalize()} check", "", p.read_text().replace("# ", "### ", 1), ""]

    (out / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if out.resolve() == (ROOT / "results").resolve():
        try:
            print(f"HANDOFF.md snapshot {'updated' if handoff.update() else 'already current'}.")
        except (OSError, ValueError) as e:
            print(f"HANDOFF.md snapshot not updated: {e}")


if __name__ == "__main__":
    main()

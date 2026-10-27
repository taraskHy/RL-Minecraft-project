"""Tables and figures from finished runs.

    python -m parkour.report [--out results]

Reads runs/*/eval_log.csv (held-out test set, deterministic policy), copies
each run's small logs into OUT/runs/NAME/ for the record, and writes
OUT/README.md (tables) and OUT/figures/*.png. All numbers are simulator
results unless a table says otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

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


def _end_labels(ax, ends, gap=4.5):
    """Direct labels at line ends, nudged apart so they never overlap."""
    ends = sorted(ends, key=lambda e: e[2])
    placed = []
    for x, text, y in ends:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
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


def pct(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--main", default="main-s0")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # Copy small per-run records.
    for d in sorted(RUNS.iterdir()):
        if d.is_dir() and (d / "eval_log.csv").exists():
            dst = out / "runs" / d.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("config.json", "eval_log.csv", "train_log.csv", "summary.json"):
                if (d / f).exists():
                    shutil.copy2(d / f, dst / f)

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
        for name, i in (("best", best), ("last", len(main_ev["success"]) - 1)):
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
                  "| Arm | What changes | Seeds | Final success, mean (min-max) | L4 | L5 | Mean over training | Ticks to 80% (B) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for arm, desc, n, final, per_level, auc, reach in table:
            if not n:
                lines.append(f"| {arm} | {desc} | 0 | not finished | | | | |")
                continue
            r = reach[~np.isnan(reach)]
            lines.append(f"| {arm} | {desc} | {n} | {pct(final.mean())} ({pct(final.min())}-{pct(final.max())}) | "
                         f"{pct(per_level[:, 4].mean())} | {pct(per_level[:, 5].mean())} | {pct(auc.mean())} | "
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

    for name in ("offline", "live"):
        p = out / f"{name}_summary.md"
        if p.exists():
            lines += [f"## {name.capitalize()} check", "", p.read_text().replace("# ", "### ", 1), ""]

    (out / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

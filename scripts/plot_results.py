"""Optional publication-style figure: python scripts/plot_results.py RUN_FOLDER.

Install plotting support separately: python -m pip install -e '.[plots]'.
The RL learner itself has no third-party Python dependencies.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    methods = result["methods"]
    labels = ["Random"] + [m["algorithm"].replace("q_learning", "Q-learning").replace("sarsa", "SARSA")
                               + "\n" + m["reward_mode"] for m in methods]
    means = [100 * result["random_baseline"]["success_rate"]]
    means += [100 * m["test"]["mean_success_rate"] for m in methods]
    fig, ax = plt.subplots(figsize=(10, 5.7), layout="constrained")
    fig.suptitle("Learning to craft a wooden pickaxe", fontsize=18, weight="bold")
    ax.set_title("Simulator only | fixed map | dots show independently trained policies", fontsize=11, pad=20)
    bars = ax.bar(labels, means, color=["#98a2b3", "#259a9b", "#70c1bc", "#4268a8", "#90a8d2"], width=0.65)
    label_heights = list(means)
    for i, method in enumerate(methods, 1):
        scores = [100 * s["success_rate"] for s in method["test_by_training_seed"]]
        offsets = [(j - (len(scores)-1)/2)*0.08 for j in range(len(scores))]
        ax.scatter([i + offset for offset in offsets], scores, s=22, color="#182230", zorder=3)
        label_heights[i] = max(scores + [means[i]])
    for bar, value, label_height in zip(bars, means, label_heights):
        ax.text(bar.get_x() + bar.get_width()/2, label_height+3, f"{value:.2f}%",
                ha="center", fontsize=11, weight="bold")
    ax.set_ylabel("Evaluation success (%)")
    ax.set_ylim(0, 116)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    n = methods[0]["test"]["training_seeds"]
    episodes = methods[0]["test_by_training_seed"][0]["episodes"]
    fig.supxlabel(f"{n} training seeds per learned method; {episodes} evaluation episodes per seed.\n"
                  "Hyperparameters selected on validation. These results do not measure Minecraft performance.",
                  fontsize=10)
    fig.savefig(args.run / "comparison.png", dpi=170)
    fig.savefig(args.run / "comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()

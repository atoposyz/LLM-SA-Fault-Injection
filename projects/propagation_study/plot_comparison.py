"""
Plot accuracy drop vs propagation degree — bit=25/26/27 stuck-at-1 + bit=27/28 stuck-at-0 comparison.

Usage:
  uv run python projects/propagation_study/plot_comparison.py
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")

# (bit, stuck) -> (label, color, linestyle)
DATASETS = {
    (25, 1): ("bit=25, stuck-at-1", "#2ecc71", "--"),
    (26, 1): ("bit=26, stuck-at-1", "#e74c3c", "-"),
    (27, 1): ("bit=27, stuck-at-1", "#3498db", "-"),
    (27, 0): ("bit=27, stuck-at-0", "#f39c12", "--"),
    (28, 0): ("bit=28, stuck-at-0", "#9b59b6", "--"),
}


def load_csv(bit: int, stuck: int) -> tuple[list[int], list[float]]:
    if stuck == 1:
        dirname = f"bit{bit}_n200"
    else:
        dirname = f"bit{bit}_sa{stuck}_n200"
    path = os.path.join(RESULT_DIR, dirname, "accuracy_vs_p.csv")
    ps, drops = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ps.append(int(row["p"]))
            drops.append(float(row["acc_drop"]))
    return ps, drops


def main():
    fig, ax = plt.subplots(figsize=(8, 5))

    for (bit, stuck), (label, color, ls) in DATASETS.items():
        ps, drops = load_csv(bit, stuck)
        ax.plot(ps, drops, color=color, linestyle=ls, linewidth=2,
                marker="o", markersize=6, markerfacecolor="white",
                markeredgewidth=1.5, markeredgecolor=color,
                label=label)

    # Annotate bit=26 stuck-at-1 key points
    ax.annotate("p=160, drop=61.2%", xy=(160, 61.17), xytext=(100, 45),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
                fontsize=9, color="#e74c3c", fontweight="bold")
    ax.annotate("p=256, drop=94.5%", xy=(256, 94.5), xytext=(190, 80),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
                fontsize=9, color="#e74c3c", fontweight="bold")

    ax.set_xlabel("Propagation degree (p)", fontsize=12)
    ax.set_ylabel("Accuracy Drop (%)", fontsize=12)
    ax.set_title("Accuracy Drop vs Propagation Degree\nL0, k&v kernels, WS dataflow",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xticks([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256])
    ax.set_xticklabels([str(t) for t in [0, 16, "", 48, "", 80, "", 112, "", 144, "", 192, "", 256]])
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = os.path.join(RESULT_DIR, "comparison_all.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

"""
Per-bit PE heatmaps: 32x32 acc_drop for each bit separately.
Rows = 8 bits (23-30), cols = 4 views:
  [acc_drop heatmap, per-row mean, per-col mean, row vs col scatter]
"""

import csv, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "config")
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
BITS = list(range(23, 31))

def load_bit_matrix(path, bit):
    """Return [32,32] matrix of acc_drop for a specific bit."""
    mat = np.full((32, 32), np.nan)
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["bit"]) != bit:
                continue
            r, c = int(row["pe_row"]), int(row["pe_col"])
            mat[r, c] = float(row["acc_drop"])
    return mat


for fname, tag in [("pe_accuracy_input_stuck1_single.csv", "sa1"),
                     ("pe_accuracy_input_stuck0_single.csv", "sa0")]:
    path = os.path.join(RESULT_DIR, fname)

    fig, axes = plt.subplots(len(BITS), 4, figsize=(20, 2.8 * len(BITS)))
    fig.suptitle(f"Per-Bit PE Position acc_drop — input {tag}", fontsize=15, fontweight="bold")

    for i, bit in enumerate(BITS):
        mat = load_bit_matrix(path, bit)
        vmin = np.nanmin(mat)
        vmax = np.nanmax(mat)

        # ── col 0: heatmap ──
        ax = axes[i, 0]
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_ylabel(f"bit {bit}")
        if i == 0:
            ax.set_title("acc_drop heatmap")
        plt.colorbar(im, ax=ax)

        # ── col 1: per-row mean ──
        ax = axes[i, 1]
        row_means = np.nanmean(mat, axis=1)
        row_stds = np.nanstd(mat, axis=1)
        ax.barh(range(32), row_means, xerr=row_stds, height=0.8,
                color=plt.cm.YlOrRd((row_means - vmin) / max(vmax - vmin, 1e-9)))
        ax.set_xlim(vmin * 0.95, vmax * 1.05)
        ax.invert_yaxis()
        if i == 0:
            ax.set_title("per-row mean ± std")
        if i == len(BITS) - 1:
            ax.set_xlabel("acc_drop")

        # ── col 2: per-col mean ──
        ax = axes[i, 2]
        col_means = np.nanmean(mat, axis=0)
        col_stds = np.nanstd(mat, axis=0)
        ax.bar(range(32), col_means, yerr=col_stds, width=0.8,
               color=plt.cm.YlOrRd((col_means - vmin) / max(vmax - vmin, 1e-9)))
        ax.set_ylim(vmin * 0.95, vmax * 1.05)
        if i == 0:
            ax.set_title("per-col mean ± std")
        if i == len(BITS) - 1:
            ax.set_xlabel("PE column")

        # ── col 3: row/col range ──
        ax = axes[i, 3]
        ax.text(0.5, 0.5,
                f"range: [{vmin:.3f}, {vmax:.3f}]\n"
                f"row range: [{row_means.min():.3f}, {row_means.max():.3f}]\n"
                f"row max/min: {row_means.max()/row_means.min():.2f}x\n"
                f"col range: [{col_means.min():.3f}, {col_means.max():.3f}]\n"
                f"col max/min: {col_means.max()/col_means.min():.2f}x",
                transform=ax.transAxes, fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title("summary")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"pe_perbit_heatmap_{tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

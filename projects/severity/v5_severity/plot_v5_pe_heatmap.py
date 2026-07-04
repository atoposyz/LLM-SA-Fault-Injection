"""
PE-position heatmaps: v5_lin predicted vs measured acc_drop.
2x2 grid: rows = measured/predicted, cols = sa0/sa1.
"""

import csv
import json
import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "result")

BITS = list(range(23, 31))
COLS = list(range(32))

PE_CSV = {
    ("input", 0): "pe_accuracy_input_stuck0.csv",
    ("input", 1): "pe_accuracy_input_stuck1.csv",
}

ORDER = [("input", 0), ("input", 1)]


def load_actual(et, sv):
    path = os.path.join(RESULT_DIR, PE_CSV[(et, sv)])
    by_key = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["mode"] != et:
                continue
            by_key[(int(row["bit"]), int(row["pe_col"]))].append(float(row["acc_drop"]))
    return {k: float(np.mean(v)) for k, v in by_key.items()}


def load_predicted(et, sv):
    with open(os.path.join(SCRIPT_DIR, f"joint_severity_stuck{sv}_ws_v5_lin.json")) as f:
        data = json.load(f)
    by_key = defaultdict(list)
    for e in data.get("operator_entries", []):
        if e.get("type") != et:
            continue
        js = e["joint_severity"]
        if js == 0:
            continue
        by_key[(e["bit"], e["pe_col"])].append(js)
    return {k: float(np.mean(v)) for k, v in by_key.items()}


def build_matrix(data_dict):
    mat = np.full((len(BITS), len(COLS)), np.nan)
    for i, b in enumerate(BITS):
        for j, c in enumerate(COLS):
            mat[i, j] = data_dict.get((b, c), np.nan)
    return mat


def norm(mat):
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    return (mat - vmin) / (vmax - vmin + 1e-12)


def compute_rho(pred, actual):
    common_a, common_p = [], []
    for key in set(pred) & set(actual):
        common_a.append(actual[key])
        common_p.append(pred[key])
    return spearmanr(common_p, common_a)[0] if len(common_a) > 1 else 0.0


def draw_panel(ax, mat, show_y=False, show_x=False):
    im = ax.imshow(mat, cmap="OrRd", norm=mcolors.Normalize(0, 1),
                   aspect="auto", interpolation="nearest")

    ax.set_yticks(range(len(BITS)))
    ax.set_yticklabels(BITS if show_y else [], fontsize=20)
    ax.set_xticks(range(0, len(COLS), 4))
    ax.set_xticklabels(COLS[::4] if show_x else [], fontsize=20)
    ax.tick_params(length=0, pad=6)

    ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BITS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#d0d0d0")
    return im


def main():
    norm_obj = mcolors.Normalize(0, 1)
    cmap = "OrRd"

    fig = plt.figure(figsize=(20, 9))
    left, right = 0.105, 0.895
    bottom, top = 0.125, 0.89
    col_gap = 0.055
    row_gap = 0.070
    col_w = (right - left - col_gap) / 2
    row_h = (top - bottom - row_gap) / 2

    for col_idx, (et, sv) in enumerate(ORDER):
        actual = load_actual(et, sv)
        pred = load_predicted(et, sv)
        rho = compute_rho(pred, actual)

        a_norm = norm(build_matrix(actual))
        p_norm = norm(build_matrix(pred))

        x = left + col_idx * (col_w + col_gap)
        y_top = bottom + row_h + row_gap
        y_bottom = bottom

        ax_actual = fig.add_axes([x, y_top, col_w, row_h])
        ax_pred = fig.add_axes([x, y_bottom, col_w, row_h])
        draw_panel(ax_actual, a_norm, show_y=(col_idx == 0), show_x=False)
        draw_panel(ax_pred, p_norm, show_y=(col_idx == 0), show_x=True)

        title = f"stuck-at-{sv}    ρ = {rho:.3f}"
        fig.text(x + col_w / 2, top + 0.035, title,
                 ha="center", va="center", fontsize=20, fontweight="bold", color="#222222")

    fig.text(left - 0.055, bottom + row_h + row_gap + row_h / 2, "measured",
             ha="center", va="center", rotation=90, fontsize=20, fontweight="bold", color="#333333")
    fig.text(left - 0.055, bottom + row_h / 2, "predicted",
             ha="center", va="center", rotation=90, fontsize=20, fontweight="bold", color="#333333")
    fig.text(left - 0.078, (bottom + top) / 2, "bit position",
             ha="center", va="center", rotation=90, fontsize=20, color="#333333")
    fig.text((left + right) / 2, 0.055, "PE column",
             ha="center", va="center", fontsize=20, color="#333333")

    for sep_x in [left + col_w + col_gap / 2]:
        fig.add_artist(plt.Line2D([sep_x, sep_x], [bottom - 0.012, top + 0.012],
                                  transform=fig.transFigure, color="#dddddd", linewidth=1.0))

    cbar_ax = fig.add_axes([0.915, bottom, 0.010, top - bottom])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_obj, cmap=cmap),
                       cax=cbar_ax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.ax.tick_params(labelsize=16, length=0, pad=4)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#cccccc")
    cb.set_label("normalized", fontsize=18, labelpad=8)

    for ext in ("png", "pdf"):
        out = os.path.join(SCRIPT_DIR, f"v5_pe_heatmap_input.{ext}")
        fig.savefig(out, dpi=220, facecolor="white", bbox_inches=None)
        print(f"Saved: {os.path.basename(out)}")
    plt.close(fig)


if __name__ == "__main__":
    main()

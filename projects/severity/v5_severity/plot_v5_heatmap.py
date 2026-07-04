"""
Bit-level predicted severity vs measured accuracy drop — single combined heatmap.
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
RESULT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "..", "bert", "result_fp32")

BITS = list(range(0, 31))

ACTUAL_CSV = {
    ("input", 0):  "accuracy_drop_perbit_input_stuck_0.csv",
    ("input", 1):  "accuracy_drop_perbit_input_stuck_1.csv",
    ("weight", 0): "accuracy_drop_perbit_weight_stuck_0.csv",
    ("weight", 1): "accuracy_drop_perbit_weight_stuck_1.csv",
}

ORDER = [("input", 0), ("input", 1), ("weight", 0), ("weight", 1)]

LABELS = {
    ("input", 0, "meas"): "Input SA0, Meas.",
    ("input", 0, "pred"): "Input SA0, Pred.",
    ("input", 1, "meas"): "Input SA1, Meas.",
    ("input", 1, "pred"): "Input SA1, Pred.",
    ("weight", 0, "meas"): "Weight SA0, Meas.",
    ("weight", 0, "pred"): "Weight SA0, Pred.",
    ("weight", 1, "meas"): "Weight SA1, Meas.",
    ("weight", 1, "pred"): "Weight SA1, Pred.",
}

FIELD_BG = {
    "mantissa": dict(x0=-0.5, x1=22.5, clr="#2ecc71"),
    "exponent": dict(x0=22.5, x1=30.5, clr="#e67e22"),
}


def load_actual(et, sv):
    path = os.path.join(RESULT_DIR, ACTUAL_CSV[(et, sv)])
    return {int(r["bit"]): float(r["acc_drop"]) for r in csv.DictReader(open(path))}


def load_predicted(et, sv):
    with open(os.path.join(SCRIPT_DIR, f"joint_severity_stuck{sv}_ws_v5_lin.json")) as f:
        data = json.load(f)
    by_bit = defaultdict(list)
    for e in data.get("operator_entries", []):
        if e.get("type") != et:
            continue
        if e["joint_severity"] == 0:
            continue
        by_bit[e["bit"]].append(e["joint_severity"])
    return {b: float(np.mean(v)) for b, v in by_bit.items()}


def build_row(d):
    return np.array([d.get(b, np.nan) for b in BITS])


def norm(arr):
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    return (arr - vmin) / (vmax - vmin + 1e-12)


def rho_exp(pred, actual):
    keys = [b for b in range(23, 31) if b in pred and b in actual]
    return spearmanr([pred[k] for k in keys], [actual[k] for k in keys])[0] if len(keys) > 1 else 0.0


def draw_strip(ax, arr_norm, show_xticks=False):
    cmap = "OrRd"
    mat = arr_norm.reshape(1, -1)
    ax.imshow(mat, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)

    ax.set_yticks([])
    if show_xticks:
        ax.set_xticks(range(len(BITS)))
        ax.set_xticklabels([str(b) if b % 5 == 0 else "" for b in BITS], fontsize=18)
        ax.tick_params(axis="x", length=0, pad=4)
    else:
        ax.set_xticks([])
        ax.tick_params(axis="x", length=0)

    for field, cfg in FIELD_BG.items():
        ax.axvspan(cfg["x0"], cfg["x1"], alpha=0.06, color=cfg["clr"], zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_color("#cccccc")


def main():
    n = len(ORDER)

    fig = plt.figure(figsize=(20, 8))
    left, right = 0.20, 0.93
    bottom, top = 0.10, 0.97
    pair_gap = 0.018
    group_gap = 0.050
    row_h = (top - bottom - n * pair_gap - (n - 1) * group_gap) / (n * 2)

    y = top
    for k, (et, sv) in enumerate(ORDER):
        actual = load_actual(et, sv)
        pred = load_predicted(et, sv)

        a_norm = norm(build_row(actual))
        p_norm = norm(build_row(pred))

        y -= row_h
        ax_actual = fig.add_axes([left, y, right - left, row_h])
        draw_strip(ax_actual, a_norm, show_xticks=False)
        fig.text(left - 0.015, y + row_h / 2, LABELS[(et, sv, "meas")],
                 ha="right", va="center", fontsize=20, fontweight="bold", color="#333333")

        y -= pair_gap + row_h
        ax_pred = fig.add_axes([left, y, right - left, row_h])
        draw_strip(ax_pred, p_norm, show_xticks=(k == n - 1))
        fig.text(left - 0.015, y + row_h / 2, LABELS[(et, sv, "pred")],
                 ha="right", va="center", fontsize=20, fontweight="bold", color="#333333")
        fig.text(left + 0.012, y + row_h + pair_gap / 2, f"ρ={rho_exp(pred, actual):.3f}",
                 ha="left", va="center", fontsize=20, fontweight="bold", color="#222222",
                 zorder=10)

        if k < n - 1:
            sep_y = y - group_gap / 2
            fig.add_artist(plt.Line2D([left, right], [sep_y, sep_y], transform=fig.transFigure,
                                      color="#aaaaaa", linewidth=1.2))
            y -= group_gap

    fig.text((left + right) / 2, 0.03, "bit position", ha="center", va="center", fontsize=20)

    cbar_ax = fig.add_axes([0.945, bottom, 0.004, top - bottom])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap="OrRd"),
                       cax=cbar_ax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.ax.tick_params(labelsize=16)
    cb.set_label("normalized", fontsize=20)

    for ext in ("png", "pdf"):
        out = os.path.join(SCRIPT_DIR, f"v5_heatmap_all.{ext}")
        fig.savefig(out, dpi=200, facecolor="white", bbox_inches=None)
        print(f"Saved: {os.path.basename(out)}")
    plt.close(fig)


if __name__ == "__main__":
    main()

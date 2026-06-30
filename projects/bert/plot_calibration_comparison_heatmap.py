"""
Paper-quality heatmaps: calibrated vs theoretical joint severity against
PE-level measured accuracy drop.

Style follows v5_severity/plot_v5_pe_heatmap.py layout, extended with a
third row for theoretical (no-calibration) prediction.

Two separate figures — input and weight — each 2 cols (sa0/sa1) × 3 rows
(measured, calibrated, theoretical).

Usage:
  uv run python projects/bert/plot_calibration_comparison_heatmap.py
"""

import csv
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
V5_DIR = os.path.join(SCRIPT_DIR, "v5_severity")
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "comparison_results")

BITS = list(range(23, 31))
COLS = list(range(32))
OPERATOR_GROUPS = ["attention", "intermediate", "output"]

PANEL_ROWS = [
    ("Measured", "measured_mat"),
    ("Theoretical", "th_mat"),
    ("Calibrated", "cal_mat"),
]
PANEL_LETTERS = tuple("abcdef")

CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_orange_red",
    ["#fff7ec", "#fee8c8", "#fdbb84", "#e34a33", "#7f0000"],
)
CMAP.set_bad("#f2f2f2")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.55,
    "axes.edgecolor": "#6f6f6f",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 450,
})

PE_CSV = {
    ("input", 0): "pe_accuracy_input_stuck0.csv",
    ("input", 1): "pe_accuracy_input_stuck1.csv",
    ("weight", 0): "pe_accuracy_weight_stuck0.csv",
    ("weight", 1): "pe_accuracy_weight_stuck1.csv",
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_measured(et: str, sv: int) -> dict[tuple, float]:
    path = os.path.join(RESULT_DIR, PE_CSV[(et, sv)])
    by_key = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["mode"] != et:
                continue
            by_key[(int(row["bit"]), int(row["pe_col"]))].append(float(row["acc_drop"]))
    return {k: float(np.mean(v)) for k, v in by_key.items()}


def load_calibrated_predicted(et: str, sv: int) -> dict[tuple, float]:
    path = os.path.join(V5_DIR, f"joint_severity_stuck{sv}_ws_v5_lin.json")
    with open(path) as f:
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


def theoretical_bit_conditional(bit: int) -> float:
    if bit == 31:
        return 2.0
    elif 23 <= bit <= 30:
        k = 2 ** (bit - 23)
        return 2.0 ** k - 1.0
    elif 0 <= bit <= 22:
        return 2.0 ** (bit - 23)
    return 0.0


def theoretical_bit_unconditional(bit: int) -> float:
    return theoretical_bit_conditional(bit) * 0.5


def load_position_coverage() -> dict:
    with open(os.path.join(CONFIG_DIR, "position_severity_ws.json")) as f:
        return json.load(f)


def build_theoretical_predicted(et: str, sv: int) -> dict[tuple, float]:
    pos_data = load_position_coverage()
    R, C = pos_data["sa_rows"], pos_data["sa_cols"]

    op_mats = {}
    for op in OPERATOR_GROUPS:
        op_mats[op] = np.array(pos_data["severity"][op][et]["raw_matrix"])

    op_reach = {}
    for op, mat in op_mats.items():
        positive = mat[mat > 0]
        C_min = float(positive.min()) if positive.size > 0 else 1.0
        op_reach[op] = np.where(mat > 0, mat / C_min, 0.0)

    by_key: defaultdict[tuple, list] = defaultdict(list)
    for bit in BITS:
        raw = theoretical_bit_conditional(bit) if et == "input" else theoretical_bit_unconditional(bit)
        bit_depth = math.log1p(max(raw, 0))
        if bit_depth == 0:
            continue
        for op in OPERATOR_GROUPS:
            reach_mat = op_reach.get(op)
            if reach_mat is None:
                continue
            for r in range(R):
                for c in range(C):
                    reach = float(reach_mat[r, c])
                    if reach == 0:
                        continue
                    by_key[(bit, c)].append(bit_depth * reach)
    return {k: float(np.mean(v)) for k, v in by_key.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_matrix(data_dict: dict[tuple, float]) -> np.ndarray:
    mat = np.full((len(BITS), len(COLS)), np.nan)
    for i, b in enumerate(BITS):
        for j, c in enumerate(COLS):
            mat[i, j] = data_dict.get((b, c), np.nan)
    return mat


def norm(mat: np.ndarray) -> np.ndarray:
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    if vmax - vmin < 1e-12:
        return np.zeros_like(mat)
    return (mat - vmin) / (vmax - vmin)


def safe_rho(pred: dict, actual: dict) -> float:
    common_a, common_p = [], []
    for key in set(pred) & set(actual):
        common_a.append(actual[key])
        common_p.append(pred[key])
    if len(common_a) < 2:
        return float("nan")
    if np.std(common_a) < 1e-15 or np.std(common_p) < 1e-15:
        return float("nan")
    return float(spearmanr(common_p, common_a)[0])


# ---------------------------------------------------------------------------
# Panel drawing (v5_pe_heatmap style)
# ---------------------------------------------------------------------------


def draw_panel(
    ax,
    mat: np.ndarray,
    panel_letter: str,
    show_y: bool = False,
    show_x: bool = False,
):
    im = ax.imshow(
        np.ma.masked_invalid(mat),
        cmap=CMAP,
        norm=mcolors.Normalize(0, 1),
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )

    ax.set_yticks(range(len(BITS)))
    ax.set_yticklabels(BITS if show_y else [], fontsize=7)

    x_ticks = list(range(0, len(COLS), 4)) + [len(COLS) - 1]
    x_labels = [COLS[i] for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels if show_x else [], fontsize=7)
    ax.tick_params(axis="both", which="major", length=2.2, width=0.5, pad=1.8)

    ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BITS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.28, alpha=0.72)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.text(
        0.012,
        0.93,
        f"({panel_letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color="#1f1f1f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.1},
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color("#777777")
    return im


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def draw_figure(exp_type: str, all_data: dict):
    """Draw one paper-ready 2-column x 3-row comparison figure."""
    fig = plt.figure(figsize=(7.15, 5.15))

    left, right = 0.135, 0.875
    bottom, top = 0.115, 0.850
    col_gap = 0.060
    row_gap = 0.046
    n_rows = len(PANEL_ROWS)

    col_w = (right - left - col_gap) / 2
    row_h = (top - bottom - row_gap * (n_rows - 1)) / n_rows
    norm_obj = mcolors.Normalize(0, 1)

    for col_idx, sv in enumerate([0, 1]):
        x = left + col_idx * (col_w + col_gap)
        d = all_data[(exp_type, sv)]

        for row_idx, (_, key) in enumerate(PANEL_ROWS):
            y = bottom + (n_rows - 1 - row_idx) * (row_h + row_gap)
            ax = fig.add_axes([x, y, col_w, row_h])
            draw_panel(
                ax,
                d[key],
                PANEL_LETTERS[row_idx * 2 + col_idx],
                show_y=(col_idx == 0),
                show_x=(row_idx == n_rows - 1),
            )

        rho_cal = d["rho_cal"]
        rho_th = d["rho_th"]
        rho_cal_str = f"{rho_cal:.3f}" if not np.isnan(rho_cal) else "N/A"
        rho_th_str = f"{rho_th:.3f}" if not np.isnan(rho_th) else "N/A"
        fig.text(
            x + col_w / 2,
            top + 0.025,
            f"Stuck-at-{sv}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#111111",
        )
        fig.text(
            x + col_w / 2,
            top + 0.006,
            f"Calibrated ρ={rho_cal_str}; Theory ρ={rho_th_str}",
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="#4a4a4a",
        )

    for row_idx, (label, _) in enumerate(PANEL_ROWS):
        y_pos = bottom + (n_rows - 1 - row_idx) * (row_h + row_gap) + row_h / 2
        fig.text(
            left - 0.047,
            y_pos,
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
        )

    fig.text(
        left - 0.078,
        (bottom + top) / 2,
        "Bit position",
        ha="center",
        va="center",
        rotation=90,
        fontsize=8.5,
        color="#222222",
    )
    fig.text(
        (left + right) / 2,
        0.055,
        "PE column",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#222222",
    )

    sep_x = left + col_w + col_gap / 2
    fig.add_artist(plt.Line2D(
        [sep_x, sep_x],
        [bottom - 0.010, top + 0.040],
        transform=fig.transFigure,
        color="#d6d6d6",
        linewidth=0.65,
    ))

    cbar_ax = fig.add_axes([0.902, bottom, 0.014, top - bottom])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_obj, cmap=CMAP),
        cax=cbar_ax,
        ticks=[0, 0.25, 0.5, 0.75, 1],
    )
    cb.ax.tick_params(labelsize=7, length=2.0, width=0.45, pad=1.5)
    cb.outline.set_linewidth(0.45)
    cb.outline.set_edgecolor("#777777")
    cb.set_label("Normalized value", fontsize=7.6, labelpad=5)


    for ext in ("png", "pdf"):
        out = os.path.join(OUTPUT_DIR, f"calibration_comparison_{exp_type}.{ext}")
        fig.savefig(out, dpi=450, facecolor="white", bbox_inches="tight", pad_inches=0.015)
        print(f"Saved: {os.path.basename(out)}")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Computing all data...")
    all_data = {}
    for et in ["input", "weight"]:
        for sv in [0, 1]:
            measured = load_measured(et, sv)
            cal_pred = load_calibrated_predicted(et, sv)
            th_pred = build_theoretical_predicted(et, sv)
            all_data[(et, sv)] = {
                "measured_mat": norm(build_matrix(measured)),
                "cal_mat": norm(build_matrix(cal_pred)),
                "th_mat": norm(build_matrix(th_pred)),
                "rho_cal": safe_rho(cal_pred, measured),
                "rho_th": safe_rho(th_pred, measured),
            }
    print("  Done.\n")

    draw_figure("input", all_data)
    print()
    draw_figure("weight", all_data)

    # Console summary
    print("\n" + "=" * 62)
    print("PE-level: calibrated vs theoretical  →  Spearman ρ")
    print("=" * 62)
    print(f"{'config':>16s}  {'ρ_cal':>8s}  {'ρ_th':>8s}  {'Δ':>8s}")
    print("-" * 46)
    for et in ["input", "weight"]:
        for sv in [0, 1]:
            rc = all_data[(et, sv)]["rho_cal"]
            rt = all_data[(et, sv)]["rho_th"]
            print(f"  {et}_sa{sv:<9d}  {rc:8.4f}  {rt:8.4f}  {rc-rt:+8.4f}")


if __name__ == "__main__":
    main()

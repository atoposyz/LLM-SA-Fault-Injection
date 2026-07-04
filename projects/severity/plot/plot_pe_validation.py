"""
Plot PE-level validation: v4 vs v5_lin joint severity vs measured acc_drop.

Outputs saved to tables/:
  - pe_validation_heatmap_input.png   : per-(bit, col) heatmaps (pred vs true)
  - pe_validation_scatter_input.png   : scatter plots (pred vs true)
  - pe_validation_col_gradient.png    : column gradient (reach vs measured drop)
"""

import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "tables")
RESULT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "bert", "result")
OUTPUT_DIR = TABLES_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

BITS = list(range(23, 31))
COLS = list(range(32))
FORMULAS_TO_PLOT = ["v4", "v5_lin", "v5_exp"]

CSV_MAP = {
    ("input", 0): "pe_accuracy_input_stuck0.csv",
    ("input", 1): "pe_accuracy_input_stuck1.csv",
}


def load_severity(path, exp_type):
    """Load joint severity aggregated to per-(bit, col) across operators and rows."""
    with open(path) as f:
        data = json.load(f)
    by_key = defaultdict(list)
    for entry in data.get("operator_entries", []):
        if entry.get("type") != exp_type:
            continue
        js = entry.get("joint_severity", 0)
        if js == 0:
            continue
        by_key[(entry["bit"], entry["pe_col"])].append(js)
    return {k: float(np.mean(v)) for k, v in by_key.items()}


def load_accuracy(exp_type, stuck_value):
    key = (exp_type, stuck_value)
    path = os.path.join(RESULT_DIR, CSV_MAP[key])
    by_key = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["mode"] != exp_type:
                continue
            by_key[(int(row["bit"]), int(row["pe_col"]))].append(float(row["acc_drop"]))
    return {k: float(np.mean(v)) for k, v in by_key.items()}


def build_matrix(data_dict, bits, cols):
    """Build [n_bits, n_cols] matrix from {(bit, col): val} dict."""
    mat = np.full((len(bits), len(cols)), np.nan)
    for i, b in enumerate(bits):
        for j, c in enumerate(cols):
            mat[i, j] = data_dict.get((b, c), np.nan)
    return mat


# ─── Heatmap ────────────────────────────────────────────────────
def plot_heatmaps():
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    for row_idx, (exp_type, stuck_value) in enumerate([("input", 0), ("input", 1)]):
        acc = load_accuracy(exp_type, stuck_value)
        acc_mat = build_matrix(acc, BITS, COLS)

        col_idx = 0
        # Ground truth
        im = axes[row_idx, col_idx].imshow(acc_mat, aspect="auto", cmap="YlOrRd",
                                            interpolation="nearest")
        axes[row_idx, col_idx].set_title(f"Measured acc_drop\n{exp_type} sa{stuck_value}")
        axes[row_idx, col_idx].set_xlabel("PE column")
        axes[row_idx, col_idx].set_ylabel("bit")
        axes[row_idx, col_idx].set_yticks(range(len(BITS)))
        axes[row_idx, col_idx].set_yticklabels(BITS)
        plt.colorbar(im, ax=axes[row_idx, col_idx])

        for fi, formula in enumerate(FORMULAS_TO_PLOT):
            col_idx = fi + 1
            sv_path = os.path.join(TABLES_DIR, f"joint_severity_stuck{stuck_value}_ws"
                                                f"{'_' + formula if formula != 'v3' else ''}.json")
            if not os.path.exists(sv_path):
                continue
            sev = load_severity(sv_path, exp_type)
            sev_mat = build_matrix(sev, BITS, COLS)
            rho, _ = spearmanr(sev_mat.flatten(), acc_mat.flatten(), nan_policy="omit")

            im = axes[row_idx, col_idx].imshow(sev_mat, aspect="auto", cmap="YlOrRd",
                                                interpolation="nearest")
            axes[row_idx, col_idx].set_title(f"{formula}  ρ={rho:.3f}")
            axes[row_idx, col_idx].set_xlabel("PE column")
            axes[row_idx, col_idx].set_yticks(range(len(BITS)))
            axes[row_idx, col_idx].set_yticklabels(BITS)
            plt.colorbar(im, ax=axes[row_idx, col_idx])

    fig.suptitle("Per-(bit, PE column) Severity Heatmaps: Prediction vs Measurement",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pe_validation_heatmap_input.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─── Scatter ────────────────────────────────────────────────────
def plot_scatter():
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for row_idx, (exp_type, stuck_value) in enumerate([("input", 0), ("input", 1)]):
        acc = load_accuracy(exp_type, stuck_value)
        acc_vals_all = []
        for (b, c), v in sorted(acc.items()):
            if b in BITS:
                acc_vals_all.append(v)

        for fi, formula in enumerate(FORMULAS_TO_PLOT):
            ax = axes[row_idx, fi]
            sv_path = os.path.join(TABLES_DIR, f"joint_severity_stuck{stuck_value}_ws"
                                                f"{'_' + formula if formula != 'v3' else ''}.json")
            if not os.path.exists(sv_path):
                continue
            sev = load_severity(sv_path, exp_type)

            common = sorted(set(sev.keys()) & set(acc.keys()))
            sev_vals = [sev[k] for k in common]
            acc_vals = [acc[k] for k in common]
            cols_arr = [k[1] for k in common]  # PE column for color

            rho, _ = spearmanr(sev_vals, acc_vals)

            sc = ax.scatter(sev_vals, acc_vals, c=cols_arr, cmap="viridis",
                           alpha=0.7, s=30, edgecolors="grey", linewidth=0.3)
            ax.set_xlabel("Predicted joint_severity")
            ax.set_ylabel("Measured acc_drop")
            ax.set_title(f"{formula}  ρ={rho:.3f}\n{exp_type} sa{stuck_value}")
            plt.colorbar(sc, ax=ax, label="PE column")

    fig.suptitle("Predicted Severity vs Measured Accuracy Drop", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pe_validation_scatter_input.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─── Column Gradient ────────────────────────────────────────────
def plot_col_gradient():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, (exp_type, stuck_value) in enumerate([("input", 0), ("input", 1)]):
        ax = axes[ax_idx]
        acc = load_accuracy(exp_type, stuck_value)
        # Average acc_drop across bits per column
        col_acc = {}
        for (b, c), v in acc.items():
            if b not in BITS:
                continue
            col_acc.setdefault(c, []).append(v)
        cols_sorted = sorted(col_acc.keys())
        acc_by_col = np.array([np.mean(col_acc[c]) for c in cols_sorted])
        # Normalise to [0,1]
        acc_norm = (acc_by_col - acc_by_col.min()) / (acc_by_col.max() - acc_by_col.min())

        ax.plot(cols_sorted, acc_norm, "ko-", linewidth=2.5, markersize=6, label="Measured acc_drop (norm)")

        # Predicted reach (v5_lin and v5_exp)
        sv_lin = os.path.join(TABLES_DIR, "joint_severity_stuck{}_ws_v5_lin.json".format(stuck_value))
        sv_exp = os.path.join(TABLES_DIR, "joint_severity_stuck{}_ws_v5_exp.json".format(stuck_value))

        for sv_path, label, style in [
            (sv_lin, "v5_lin reach (C/C_min)", "s--"),
            (sv_exp, "v5_exp reach ((1+λ)^P)", "^--"),
        ]:
            if not os.path.exists(sv_path):
                continue
            sev = load_severity(sv_path, exp_type)
            col_reach = {}
            for (b, c), v in sev.items():
                if b not in BITS:
                    continue
                # Get reach from operator_entries
                pass
            # Reconstruct reach: since joint_severity = bit_depth * reach,
            # for a given column, reach is constant across bits.
            # Average (joint_severity / mean_joint_per_bit) to extract reach
            with open(sv_path) as f:
                data = json.load(f)
            col_js = defaultdict(list)
            for entry in data.get("operator_entries", []):
                if entry.get("type") != exp_type or entry["bit"] not in BITS:
                    continue
                c = entry["pe_col"]
                col_js[c].append(entry["joint_severity"] / max(entry.get("bit_depth", 1), 1e-8)
                                 if entry.get("bit_depth", 0) > 0 else 1.0)

            # Actually, simpler: just read `reach` field directly
            col_reach_direct = defaultdict(list)
            for entry in data.get("operator_entries", []):
                if entry.get("type") != exp_type or entry["bit"] not in BITS:
                    continue
                col_reach_direct[entry["pe_col"]].append(entry.get("reach", 1.0))

            cols_r = sorted(col_reach_direct.keys())
            reach_by_col = np.array([np.mean(col_reach_direct[c]) for c in cols_r])
            reach_norm = (reach_by_col - reach_by_col.min()) / (reach_by_col.max() - reach_by_col.min()) \
                if reach_by_col.max() > reach_by_col.min() else reach_by_col

            ax.plot(cols_r, reach_norm, style, linewidth=2, markersize=7,
                   markerfacecolor="white", label=label)

        ax.set_xlabel("PE column")
        ax.set_ylabel("Normalised value")
        ax.set_title(f"{exp_type} sa{stuck_value}: Column gradient")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("PE Column Gradient: Predicted Reach vs Measured acc_drop",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pe_validation_col_gradient.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_heatmaps()
    plot_scatter()
    plot_col_gradient()
    print("Done.")

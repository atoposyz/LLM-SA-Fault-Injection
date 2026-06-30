"""
Side-by-side comparison: predicted severity vs actual accuracy drop.

Both predicted and actual are normalized the same way (divide by max),
so the bar heights are directly comparable.

Algorithm (shared for BERT and Qwen, implemented in tool/bit_severity.py):
  1. Build per-bit conditional severity from calibration tensors
  2. Apply IEEE 754 theoretical floor for exponent bits with insufficient samples
  3. Normalize: log1p(conditional) / max(log1p(conditional))  — preserves proportionality

Usage:
  uv run python projects/qwen3-8b/plot_calibrated_severity.py
"""
import csv
import json
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config")
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")

OPERATOR_GROUPS = ["attention", "intermediate", "output"]

COLORS = {"input": "#FF5722", "weight": "#2196F3"}


# ---------------------------------------------------------------------------
# IEEE 754 theoretical severity (shared with tool/bit_severity.py)
# ---------------------------------------------------------------------------

def _theoretical_exponent_severity(bit, precision="fp32"):
    if precision == "fp32":
        if not (23 <= bit <= 30):
            return 0.0
        offset = 23
    elif precision == "bf16":
        if not (7 <= bit <= 14):
            return 0.0
        offset = 7
    else:
        return 0.0
    return math.log(2) * (2 ** (bit - offset))


# ---------------------------------------------------------------------------
# Load predicted severity (same algorithm as tool/bit_severity.py)
# ---------------------------------------------------------------------------

def load_predicted(precision="fp32"):
    """Load conditional severity, apply theoretical floor, normalize with log1p_max."""
    raw = {}
    for typ, src in [("input", "activation_input"), ("weight", "weight")]:
        for op in OPERATOR_GROUPS:
            path = os.path.join(CONFIG_DIR, f"severity_table_{src}_{precision}_{op}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                tbl = json.load(f)
            for e in tbl["table"]:
                b = e["bit"]
                raw.setdefault((typ, b), {})
                for sv in [0, 1]:
                    raw[(typ, b)].setdefault(f"sa{sv}_cond", []).append(
                        e.get(f"sa{sv}_conditional", 0))
                    raw[(typ, b)].setdefault(f"sa{sv}_eff", []).append(
                        e.get(f"sa{sv}_effective_rate", 0))

    for (typ, b), d in raw.items():
        for k in d:
            d[k] = float(np.mean(d[k]))

    # Step 2: IEEE 754 theoretical floor
    for (typ, b), d in raw.items():
        theory = _theoretical_exponent_severity(b, precision)
        if theory > 0:
            for sv in [0, 1]:
                ck = f"sa{sv}_cond"
                ek = f"sa{sv}_eff"
                if d[ck] < theory * 0.01:
                    proxy_eff = max(d[ek], 0.001)
                    d[ck] = max(d[ck], theory * proxy_eff)

    # Step 3: log1p → divide by max (preserves proportionality)
    result = {}
    for typ in ["input", "weight"]:
        bits = sorted(set(b for (t, b) in raw if t == typ))
        for sv in [0, 1]:
            vals = np.array([math.log1p(max(raw[(typ, b)][f"sa{sv}_cond"], 0))
                             for b in bits])
            vmax = vals.max()
            norms = vals / vmax if vmax > 1e-12 else vals
            for i, b in enumerate(bits):
                result[(typ, sv, b)] = float(norms[i])
    return result


# ---------------------------------------------------------------------------
# Load actual accuracy drop (normalized same way: divide by max)
# ---------------------------------------------------------------------------

def load_actual():
    actual = {}
    csv_map = {
        ("input", 0): "accuracy_drop_plot_form_input_stuck_0.csv",
        ("input", 1): "accuracy_drop_plot_form_input_stuck_1.csv",
        ("weight", 0): "accuracy_drop_plot_form_weight_stuck_0.csv",
        ("weight", 1): "accuracy_drop_plot_form_weight_stuck_1.csv",
    }
    for (typ, sv), fname in csv_map.items():
        path = os.path.join(RESULT_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                actual[(typ, sv, int(r["bit"]))] = max(float(r["accuracy_drop"]), 0)

    # Normalize per (typ, sv) using divide-by-max (same as predicted)
    norm = {}
    for typ in ["input", "weight"]:
        for sv in [0, 1]:
            vals = np.array([actual.get((typ, sv, b), 0) for b in range(32)])
            vmax = vals.max()
            for b in range(32):
                norm[(typ, sv, b)] = float(vals[b] / vmax) if vmax > 1e-12 else 0.0
    return norm


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def add_field_regions(ax):
    ax.axvspan(-0.5, 22.5, alpha=0.06, color='green', zorder=0)
    ax.axvspan(22.5, 30.5, alpha=0.08, color='orange', zorder=0)
    ax.axvspan(30.5, 31.5, alpha=0.12, color='red', zorder=0)


def plot():
    pred = load_predicted("fp32")
    actual = load_actual()

    fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True)
    fig.subplots_adjust(hspace=0.30)

    bits = np.arange(32)
    bar_width = 0.35

    for row, (typ, sv) in enumerate([("input", 1), ("input", 0),
                                       ("weight", 1), ("weight", 0)]):
        ax = axes[row]

        pv = np.array([pred.get((typ, sv, b), 0) for b in bits])
        av = np.array([actual.get((typ, sv, b), 0) for b in bits])

        # Both already normalized with divide-by-max — directly comparable
        pr, _ = pearsonr(pv, av)
        sr, _ = spearmanr(pv, av)

        x_pred = bits - bar_width / 2
        x_act = bits + bar_width / 2

        ax.bar(x_pred, pv, bar_width,
               color=COLORS[typ], alpha=0.75, label="Predicted severity")
        ax.bar(x_act, av, bar_width,
               color="none", edgecolor="black", linewidth=1.5,
               alpha=0.9, label="Actual accuracy drop")

        add_field_regions(ax)
        ax.set_ylabel("Normalized (÷ max)", fontsize=10)
        stuck_label = "Stuck-at-1" if sv == 1 else "Stuck-at-0"
        ax.set_title(f"{typ} — {stuck_label}  |  Pearson r = {pr:.4f}  |  Spearman ρ = {sr:.4f}",
                     fontsize=12, fontweight='bold', loc='left', color=COLORS[typ])
        ax.set_ylim(-0.02, 1.15)
        ax.grid(True, alpha=0.25, linestyle=':', axis='y')
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

        # Highlight top-5 actual bits
        top5 = set(np.argsort(-av)[:5])
        for i in top5:
            if av[i] > 0.01:
                ax.annotate(str(i), (bits[i], max(pv[i], av[i]) + 0.03),
                           fontsize=7, ha='center', fontweight='bold',
                           color='red' if av[i] > 0.5 else 'black')

    axes[-1].set_xlabel("Bit Position (IEEE 754 FP32)", fontsize=12)
    axes[-1].set_xticks(range(0, 32))
    axes[-1].set_xticklabels([str(i) for i in range(32)], fontsize=7)
    axes[-1].set_xlim(-0.8, 31.8)

    fig.suptitle(
        "Predicted Severity vs Actual Accuracy Drop\n"
        "Algorithm: conditional + IEEE 754 floor + log1p → ÷max  |  Qwen3-8B  |  FP32",
        fontsize=14, fontweight='bold', y=1.005
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path = os.path.join(CONFIG_DIR, "severity_pred_vs_actual.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    plot()

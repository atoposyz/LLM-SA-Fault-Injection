"""
Final comparison plot: v3 predicted severity vs measured accuracy drop.

v3 formula (physically motivated):
  Input:  min(sa1_conditional, 10), sign_bit × 0.08
  Weight: sa1_unconditional
  Both:   bit-level only (position coverage is secondary due to saturation / uniformity)

Generates a clean 3-panel figure:
  Left:   Input — predicted vs measured per-bit
  Center: Weight — predicted vs measured per-bit
  Right:  Scatter — predicted severity vs accuracy drop (all data points)
"""

import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "config")

SAT_POINT = 10.0
SIGN_RESILIENCE = 0.08

# ---- data loading ---------------------------------------------------------

def load_data():
    """Load severity tables, position coverage, and ground truth."""
    sev_cache = {}
    for typ, src in [("input", "activation_input"), ("weight", "weight")]:
        for op in ["attention", "intermediate", "output"]:
            path = os.path.join(
                SCRIPT_DIR, "config", f"severity_table_{src}_fp32_{op}.json"
            )
            with open(path) as f:
                t = json.load(f)
            for e in t["table"]:
                sev_cache[(typ, op, e["bit"])] = {
                    "sa1_uncond": e["sa1_unconditional"],
                    "sa1_cond": e["sa1_conditional"],
                    "p0": e["p0"],
                }

    with open(os.path.join(SCRIPT_DIR, "result", "combined_top1_position_long.csv")) as f:
        rows = list(csv.DictReader(f))
    baseline = max(float(r["top1"]) for r in rows)

    return sev_cache, rows, baseline


def compute_v3_severity(typ, bit, sev_cache):
    """Compute v3 severity for a given (type, bit). Averages across operators."""
    vals = []
    for op in ["attention", "intermediate", "output"]:
        sk = (typ, op, bit)
        if sk not in sev_cache:
            continue
        s = sev_cache[sk]
        if typ == "input":
            sev = min(s["sa1_cond"], SAT_POINT)
        else:
            sev = s["sa1_uncond"]
        if bit == 31:
            sev *= SIGN_RESILIENCE
        vals.append(sev)
    return float(np.mean(vals)) if vals else 0.0


# ---- plot -----------------------------------------------------------------

def main():
    sev_cache, rows, baseline = load_data()
    print(f"Baseline top1: {baseline:.4f}")

    # Aggregate: per-bit mean severity and mean drop
    bit_data = defaultdict(lambda: {"sev": [], "drop": []})
    all_sevs, all_drops, all_types = [], [], []

    for r in rows:
        typ = r["type"]
        bit = int(r["bit"])
        drop = baseline - float(r["top1"])
        sev = compute_v3_severity(typ, bit, sev_cache)

        bit_data[(typ, bit)]["sev"].append(sev)
        bit_data[(typ, bit)]["drop"].append(drop)
        all_sevs.append(sev)
        all_drops.append(drop)
        all_types.append(typ)

    # ---- Figure ----
    fig = plt.figure(figsize=(20, 6))

    # === Panel 1: Input per-bit ===
    ax1 = fig.add_subplot(1, 3, 1)
    _plot_per_bit(ax1, bit_data, "input", "#FF5722", "Input Faults (stuck-at-1)")

    # === Panel 2: Weight per-bit ===
    ax2 = fig.add_subplot(1, 3, 2)
    _plot_per_bit(ax2, bit_data, "weight", "#2196F3", "Weight Faults (stuck-at-1)")

    # === Panel 3: Scatter (all) ===
    ax3 = fig.add_subplot(1, 3, 3)
    _plot_scatter(ax3, all_sevs, all_drops, all_types, baseline)

    fig.suptitle(
        "Predicted Severity (v3) vs Measured Accuracy Drop\n"
        "Input: min(conditional_severity, 10), sign×0.08  |  "
        "Weight: unconditional_severity  |  "
        "boltuix/bert-emotion, 32×32 WS systolic array",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = os.path.join(OUTPUT_DIR, "validation_v3_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close()


def _plot_per_bit(ax, bit_data, exp_type, color, title):
    """Bar chart: per-bit predicted severity vs measured accuracy drop."""
    bits = sorted(b for (t, b) in bit_data if t == exp_type)
    if not bits:
        ax.set_title(f"{title} (no data)")
        return

    x = np.arange(len(bits))
    width = 0.35

    sev_means = [np.mean(bit_data[(exp_type, b)]["sev"]) for b in bits]
    sev_stds = [np.std(bit_data[(exp_type, b)]["sev"]) for b in bits]
    drop_means = [np.mean(bit_data[(exp_type, b)]["drop"]) for b in bits]
    drop_stds = [np.std(bit_data[(exp_type, b)]["drop"]) for b in bits]

    # Normalize severity to [0, max_drop] for visual overlay
    sev_arr = np.array(sev_means, dtype=float)
    if sev_arr.max() > 0:
        sev_display = sev_arr / sev_arr.max() * max(drop_means)
    else:
        sev_display = np.zeros_like(sev_arr)

    ax.bar(x - width / 2, sev_display, width, color=color, alpha=0.35,
           edgecolor=color, linewidth=0.8, label="Predicted severity (norm.)")
    bars = ax.bar(x + width / 2, drop_means, width, color=color, alpha=0.85,
                  edgecolor="white", linewidth=0.5, label="Measured acc. drop")
    ax.errorbar(x + width / 2, drop_means, yerr=drop_stds, fmt="none",
                ecolor="#333333", capsize=3, linewidth=0.8, alpha=0.6)

    # Annotate raw severity values
    for i, (b, s) in enumerate(zip(bits, sev_means)):
        label = f"{s:.2f}"
        if b == 31 and exp_type == "input":
            label += "×"
        ax.annotate(label, (x[i] - width / 2, sev_display[i]),
                    textcoords="offset points", xytext=(0, 5),
                    fontsize=7, ha="center", color="#555555", fontweight="bold")

    # Compute Spearman
    if len(bits) > 2:
        rho, p = spearmanr(sev_means, drop_means)
    else:
        rho, p = 0, 1

    ax.set_title(f"{title}\nSpearman ρ = {rho:.3f}", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bits], fontsize=9)
    ax.set_xlabel("Bit position", fontsize=10)
    ax.set_ylabel("Accuracy drop / norm. severity", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_ylim(-0.02, max(drop_means) * 1.25 if drop_means else 0.7)

    # Field shading
    _add_field_regions(ax, bits)


def _add_field_regions(ax, bits):
    """Add IEEE 754 field background shading."""
    x_min, x_max = -0.5, len(bits) - 0.5
    sign_bit = None
    exp_start, exp_end = None, None
    for i, b in enumerate(bits):
        if b == 31:
            sign_bit = i
        if 23 <= b <= 30:
            if exp_start is None:
                exp_start = i
            exp_end = i

    if exp_start is not None:
        ax.axvspan(exp_start - 0.5, exp_end + 0.5, alpha=0.06, color="orange", zorder=0)
    if sign_bit is not None:
        ax.axvspan(sign_bit - 0.5, sign_bit + 0.5, alpha=0.10, color="red", zorder=0)
    # Mantissa is the rest
    mantissa_bits = [i for i, b in enumerate(bits) if b <= 22]
    if mantissa_bits:
        ax.axvspan(min(mantissa_bits) - 0.5, max(mantissa_bits) + 0.5,
                   alpha=0.04, color="green", zorder=0)


def _plot_scatter(ax, sevs, drops, types, baseline):
    """Scatter: predicted severity vs measured accuracy drop."""
    sevs = np.array(sevs)
    drops = np.array(drops)
    types = np.array(types)

    colors = {"input": "#FF5722", "weight": "#2196F3"}
    for typ in ["input", "weight"]:
        mask = types == typ
        if mask.sum() == 0:
            continue
        ax.scatter(sevs[mask], drops[mask], c=colors[typ], alpha=0.25, s=6,
                   label=typ, edgecolors="none", rasterized=True)

    # Spearman
    rho, p = spearmanr(sevs, drops)
    ax.text(0.95, 0.08, f"Spearman ρ = {rho:.3f}\np = {p:.2e}",
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_xlabel("Predicted severity (v3)", fontsize=10)
    ax.set_ylabel("Measured accuracy drop", fontsize=10)
    ax.set_title(f"All Data Points (n={len(sevs):,})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, markerscale=3, loc="upper right")
    ax.grid(alpha=0.2, linestyle=":")

    # saturation reference line
    ax.axhline(y=baseline - 0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.annotate("accuracy floor ≈ 0.05–0.10", xy=(ax.get_xlim()[1] * 0.6, baseline - 0.08),
                fontsize=7, color="gray", alpha=0.7)


if __name__ == "__main__":
    main()

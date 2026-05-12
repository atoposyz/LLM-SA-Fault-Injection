"""
V3 vs V4 comparison: INPUT / WEIGHT stuck-at-1, bit-level data.
Dual y-axis: left = severity, right = accuracy drop.
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "config")
BITS = list(range(23, 32))


def load_cache():
    c = {}
    for typ, src in [("input", "activation_input"), ("weight", "weight")]:
        for op in ["attention", "intermediate", "output"]:
            path = os.path.join(SCRIPT_DIR, "config", f"severity_table_{src}_fp32_{op}.json")
            with open(path) as f:
                t = json.load(f)
            for e in t["table"]:
                c[(typ, op, e["bit"])] = {
                    "sa0_u": e["sa0_unconditional"], "sa0_c": e["sa0_conditional"],
                    "sa1_u": e["sa1_unconditional"], "sa1_c": e["sa1_conditional"],
                }
    return c


def sev(typ, bit, sv, cache, formula):
    vals = []
    for op in ["attention", "intermediate", "output"]:
        sk = (typ, op, bit)
        if sk not in cache: continue
        s = cache[sk]
        raw = s["sa1_c"] if sv == 1 else s["sa0_c"] if typ == "input" else s["sa1_u"] if sv == 1 else s["sa0_u"]
        if formula == "v3":
            if bit == 31: raw *= 0.08
            v = min(raw, 10.0)
        else:  # v4
            v = np.log1p(raw)
        vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


def load_bit_csv(path, col):
    with open(path) as f:
        return {int(r["bit"]): float(r[col]) for r in csv.DictReader(f)}


def draw_figure(typ, cache, old_data, output_path):
    color_v3 = "#FF5722" if typ == "input" else "#2196F3"
    color_v4 = "#4CAF50"
    label = "Input" if typ == "input" else "Weight"

    old_drops = np.array([max(0, old_data[(typ, 1)].get(b, 0)) for b in BITS])

    sevs_v3 = np.array([sev(typ, b, 1, cache, "v3") for b in BITS])
    sevs_v4 = np.array([sev(typ, b, 1, cache, "v4") for b in BITS])

    sev_all = np.concatenate([sevs_v3, sevs_v4])
    drop_all = old_drops
    sev_ymax = max(sev_all) * 1.2
    drop_ymax = max(drop_all) * 1.3 if max(drop_all) > 0 else 0.2

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(BITS))

    # ---- V3 line ----
    ax.plot(x, sevs_v3, "o-", color=color_v3, linewidth=2.5, markersize=9,
            markerfacecolor="white", markeredgewidth=2, zorder=3, label="V3 severity")
    for i, s in enumerate(sevs_v3):
        ax.annotate(f"{s:.2f}", (x[i], sevs_v3[i]),
                    textcoords="offset points", xytext=(-12, 8),
                    fontsize=7.5, ha="center", color=color_v3, fontweight="bold")

    # ---- V4 line ----
    ax.plot(x, sevs_v4, "s--", color=color_v4, linewidth=2.5, markersize=9,
            markerfacecolor="white", markeredgewidth=2, zorder=3, label="V4 severity (log1p)")
    for i, s in enumerate(sevs_v4):
        ax.annotate(f"{s:.2f}", (x[i], sevs_v4[i]),
                    textcoords="offset points", xytext=(12, -2),
                    fontsize=7.5, ha="center", color=color_v4, fontweight="bold")

    # ---- accuracy drop (right axis) ----
    ax2 = ax.twinx()
    ax2.bar(x, old_drops, 0.35, color="#333333", alpha=0.35, edgecolor="#333333",
            linewidth=0.5, label="Measured acc. drop")
    for i, d in enumerate(old_drops):
        if d > 0.001:
            ax2.annotate(f"{d:.3f}", (x[i], old_drops[i]),
                        textcoords="offset points", xytext=(0, 6),
                        fontsize=7, ha="center", color="#333333")

    # ---- thresholds ----
    ax.axhline(y=10, color=color_v3, linestyle=":", linewidth=1, alpha=0.4)
    ax.annotate("V3 sat=10", (len(BITS)-1.3, 10.5), fontsize=8, color=color_v3, alpha=0.5, ha="right")
    ax.axhline(y=np.log1p(5), color=color_v4, linestyle=":", linewidth=1, alpha=0.4)
    ax.annotate(f"V4 ref={np.log1p(5):.2f}", (len(BITS)-1.3, np.log1p(5)+0.15),
                fontsize=8, color=color_v4, alpha=0.5, ha="right")

    # ---- Spearman ----
    mask3 = (sevs_v3 > 0.001) | (old_drops > 0.001)
    mask4 = (sevs_v4 > 0.001) | (old_drops > 0.001)
    rho3 = spearmanr(sevs_v3[mask3], old_drops[mask3])[0] if mask3.sum() > 2 and np.std(old_drops[mask3])>1e-8 else float("nan")
    rho4 = spearmanr(sevs_v4[mask4], old_drops[mask4])[0] if mask4.sum() > 2 and np.std(old_drops[mask4])>1e-8 else float("nan")

    # ---- styling ----
    ax.set_title(f"{label} stuck-at-1 — V3 vs V4", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BITS], fontsize=10)
    ax.set_xlabel("Bit position", fontsize=12)
    ax.set_ylabel("Predicted severity", fontsize=12)
    ax.set_ylim(-sev_ymax * 0.03, sev_ymax)
    ax.grid(axis="y", alpha=0.2, linestyle=":")
    ax2.set_ylabel("Measured accuracy drop", fontsize=12)
    ax2.set_ylim(-drop_ymax * 0.03, drop_ymax)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left", framealpha=0.9)

    ax.text(0.98, 0.92, f"V3 ρ = {rho3:.3f}\nV4 ρ = {rho4:.3f}",
            transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9))

    for i, b in enumerate(BITS):
        if b == 31: ax.axvspan(i-0.5, i+0.5, alpha=0.06, color="red", zorder=0)
        else:       ax.axvspan(i-0.5, i+0.5, alpha=0.04, color="orange", zorder=0)

    fig.suptitle(f"{label} Faults — V3 vs V4 Severity Formulas  |  boltuix/bert-emotion",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    cache = load_cache()
    old_data = {
        ("input",  1): load_bit_csv(os.path.join(SCRIPT_DIR, "result", "accuracy_drop_plot_form_input_stuck_1.csv"),  "stuck_1_input_acc_drop"),
        ("weight", 1): load_bit_csv(os.path.join(SCRIPT_DIR, "result", "accuracy_drop_plot_form_weight_stuck_1.csv"), "stuck_1_weight_acc_drop"),
    }

    draw_figure("input",  cache, old_data, os.path.join(OUTPUT_DIR, "validation_v3v4_input.png"))
    draw_figure("weight", cache, old_data, os.path.join(OUTPUT_DIR, "validation_v3v4_weight.png"))


if __name__ == "__main__":
    main()

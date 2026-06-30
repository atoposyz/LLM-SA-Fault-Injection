"""
V4 heatmaps: one strip per fault type, all 32 bits (0-31).
Light-green to dark-green, linear normalization, separate range per type.
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "config")
ALL_BITS = list(range(32))


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


def v4(typ, bit, sv, cache):
    vals = []
    for op in ["attention", "intermediate", "output"]:
        sk = (typ, op, bit)
        if sk not in cache: continue
        s = cache[sk]
        raw = s["sa1_c"] if sv == 1 else s["sa0_c"] if typ == "input" else s["sa1_u"] if sv == 1 else s["sa0_u"]
        vals.append(np.log1p(raw))
    return float(np.mean(vals)) if vals else 0.0


def load_bit_csv(path, col):
    with open(path) as f:
        return {int(r["bit"]): float(r[col]) for r in csv.DictReader(f)}


def draw_strip(ax, values, title, cmap, norm):
    """Draw 1-row heatmap with all 32 bits."""
    data = np.array(values).reshape(1, -1)
    ax.imshow(data, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_yticks([])
    ax.set_xticks(ALL_BITS)
    ax.set_xticklabels([str(b) if b % 4 == 0 else "" for b in ALL_BITS], fontsize=7)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")

    # Value annotations for exponent+sign bits and non-zero mantissa
    vmax = norm.vmax
    for i, b in enumerate(ALL_BITS):
        v = values[i]
        if b >= 23 or v > vmax * 0.01:
            color = "white" if v > vmax * 0.6 else "black"
            ax.text(i, 0, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color=color)

    # IEEE 754 field labels
    ax.annotate("mantissa (0–22)", (-0.5, -1.2), fontsize=7, color="green", alpha=0.7)
    ax.annotate("exponent (23–30)", (22.5, -1.2), fontsize=7, color="orange", alpha=0.7)
    ax.annotate("sign (31)", (30.5, -1.2), fontsize=7, color="red", alpha=0.7)

    # Field background
    ax.axvspan(-0.5, 22.5, alpha=0.04, color="green", zorder=0)
    ax.axvspan(22.5, 30.5, alpha=0.06, color="orange", zorder=0)
    ax.axvspan(30.5, 31.5, alpha=0.10, color="red", zorder=0)


def main():
    cache = load_cache()
    old_data = {
        ("input",  1): load_bit_csv(os.path.join(SCRIPT_DIR, "result", "accuracy_drop_plot_form_input_stuck_1.csv"),  "stuck_1_input_acc_drop"),
        ("input",  0): load_bit_csv(os.path.join(SCRIPT_DIR, "result", "accuracy_drop_plot_form_input_stuck_0.csv"),  "stuck_0_input_acc_drop"),
        ("weight", 1): load_bit_csv(os.path.join(SCRIPT_DIR, "result", "accuracy_drop_plot_form_weight_stuck_1.csv"), "stuck_1_weight_acc_drop"),
        ("weight", 0): load_bit_csv(os.path.join(SCRIPT_DIR, "result", "accuracy_drop_plot_form_weight_stuck_0.csv"), "stuck_0_weight_acc_drop"),
    }

    cmap = plt.cm.Greens

    # ---- compute values ----
    def get_row(typ, sv, kind):
        if kind == "sev":
            return [v4(typ, b, sv, cache) for b in ALL_BITS]
        else:
            return [max(0, old_data[(typ, sv)].get(b, 0)) for b in ALL_BITS]

    # ---- color ranges (linear, uniform) ----
    input_sev_all  = get_row("input",  1, "sev") + get_row("input",  0, "sev")
    weight_sev_all = get_row("weight", 1, "sev") + get_row("weight", 0, "sev")
    input_drop_all  = get_row("input",  1, "drop") + get_row("input",  0, "drop")
    weight_drop_all = get_row("weight", 1, "drop") + get_row("weight", 0, "drop")

    norm_in_sev  = mcolors.Normalize(vmin=min(input_sev_all),  vmax=max(input_sev_all))
    norm_wt_sev  = mcolors.Normalize(vmin=min(weight_sev_all), vmax=max(weight_sev_all))
    norm_in_drop = mcolors.Normalize(vmin=0,                    vmax=max(input_drop_all))
    norm_wt_drop = mcolors.Normalize(vmin=0,                    vmax=max(weight_drop_all))

    # ================================================================
    # Figure 1: INPUT (4 strips)
    # ================================================================
    fig, axes = plt.subplots(4, 1, figsize=(20, 6))
    fig.subplots_adjust(hspace=0.5)

    draw_strip(axes[0], get_row("input", 1, "sev"),
               "Input stuck-at-1 — V4 predicted severity", cmap, norm_in_sev)
    draw_strip(axes[1], get_row("input", 1, "drop"),
               "Input stuck-at-1 — measured accuracy drop", cmap, norm_in_drop)
    draw_strip(axes[2], get_row("input", 0, "sev"),
               "Input stuck-at-0 — V4 predicted severity", cmap, norm_in_sev)
    draw_strip(axes[3], get_row("input", 0, "drop"),
               "Input stuck-at-0 — measured accuracy drop", cmap, norm_in_drop)

    cax1 = fig.add_axes([0.92, 0.55, 0.012, 0.35])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_in_sev, cmap=cmap), cax=cax1,
                 label="V4 severity")
    cax2 = fig.add_axes([0.96, 0.55, 0.012, 0.35])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_in_drop, cmap=cmap), cax=cax2,
                 label="Acc. drop")

    fig.suptitle("INPUT Faults — V4 Heatmaps (all 32 bits)  |  boltuix/bert-emotion",
                 fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(OUTPUT_DIR, "validation_v4_heatmap_input.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved: validation_v4_heatmap_input.png")
    plt.close()

    # ================================================================
    # Figure 2: WEIGHT (4 strips)
    # ================================================================
    fig, axes = plt.subplots(4, 1, figsize=(20, 6))
    fig.subplots_adjust(hspace=0.5)

    draw_strip(axes[0], get_row("weight", 1, "sev"),
               "Weight stuck-at-1 — V4 predicted severity", cmap, norm_wt_sev)
    draw_strip(axes[1], get_row("weight", 1, "drop"),
               "Weight stuck-at-1 — measured accuracy drop", cmap, norm_wt_drop)
    draw_strip(axes[2], get_row("weight", 0, "sev"),
               "Weight stuck-at-0 — V4 predicted severity", cmap, norm_wt_sev)
    draw_strip(axes[3], get_row("weight", 0, "drop"),
               "Weight stuck-at-0 — measured accuracy drop", cmap, norm_wt_drop)

    cax1 = fig.add_axes([0.92, 0.55, 0.012, 0.35])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_wt_sev, cmap=cmap), cax=cax1,
                 label="V4 severity")
    cax2 = fig.add_axes([0.96, 0.55, 0.012, 0.35])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_wt_drop, cmap=cmap), cax=cax2,
                 label="Acc. drop")

    fig.suptitle("WEIGHT Faults — V4 Heatmaps (all 32 bits)  |  boltuix/bert-emotion",
                 fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(OUTPUT_DIR, "validation_v4_heatmap_weight.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved: validation_v4_heatmap_weight.png")
    plt.close()


if __name__ == "__main__":
    main()

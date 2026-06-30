"""Sample 1: 浅/中/深层 stable_rank 对比 (inject vs clean)，同组多层取均值。"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

RESULT_DIR = "/workplace/home/mayongzhe/faultinject/projects/runtimemetrics/result"
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")

INJ_FILE = "runtime_metrics_record_sid56_kernel-v_layer-35_cfg-input_bitflip_13_reg-input_df-WS_pe86_82_intv50_detL0_3_6_9_12_15_18_21_24_27_30_33_affect0_inject.jsonl"
CLEAN_FILE = "runtime_metrics_record_sid56_kernel-v_layer-0_cfg-weight_bitflip_10_reg-weight_df-WS_pe_random_intv50_detL0_3_6_9_12_15_18_21_24_27_30_33_affect1_noinject.jsonl"

SID = "56"
METRIC = "stable_rank"

GROUPS = {
    "Shallow": [0],
    "Middle": [18],
    "Deep": [33],
}

COLOR_FAULTY = "#d64531"
COLOR_BASELINE = "#2b7bba"


def load_sample(filepath, sid):
    with open(filepath) as f:
        for line in f:
            rec = json.loads(line)
            if str(rec["sample_id"]) == str(sid):
                return rec["runtime_metrics_values"]
    return []


def group_metrics(metrics, groups):
    grouped = {name: {} for name in groups}
    for m in metrics:
        lid = m["layer"]
        te = m["token_end"]
        for gname, g_layers in groups.items():
            if lid in g_layers:
                if te not in grouped[gname]:
                    grouped[gname][te] = []
                grouped[gname][te].append(m[METRIC])

    result = {}
    for gname in groups:
        pts = sorted([(te, np.mean(vals)) for te, vals in grouped[gname].items()])
        result[gname] = {"x": [p[0] for p in pts], "y": [p[1] for p in pts]}
    return result


def main():
    inj_metrics = load_sample(os.path.join(RESULT_DIR, INJ_FILE), SID)
    clean_metrics = load_sample(os.path.join(RESULT_DIR, CLEAN_FILE), SID)

    inj_grouped = group_metrics(inj_metrics, GROUPS)
    clean_grouped = group_metrics(clean_metrics, GROUPS)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, (gname, g_layers) in zip(axes, GROUPS.items()):
        x_f = inj_grouped[gname]["x"]
        y_f = inj_grouped[gname]["y"]
        x_b = clean_grouped[gname]["x"]
        y_b = clean_grouped[gname]["y"]

        ax.plot(x_f, y_f, color=COLOR_FAULTY, linewidth=2.2, label="Faulty", zorder=3)
        ax.plot(x_b, y_b, color=COLOR_BASELINE, linewidth=2.2, label="Baseline", zorder=2)

        step = max(1, len(x_f) // 20)
        ax.scatter(x_f[::step], y_f[::step], color=COLOR_FAULTY, s=20, zorder=4, alpha=0.7, edgecolors="none")
        ax.scatter(x_b[::step], y_b[::step], color=COLOR_BASELINE, s=20, zorder=4, alpha=0.7, edgecolors="none")

        ax.set_title(gname, fontweight="bold", pad=14)
        ax.set_xlabel("Token")
        ax.legend(frameon=True, fancybox=True, framealpha=0.95, edgecolor="#ccc",
                  loc="upper right", handlelength=1.8, borderpad=0.5)
        ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        ax.set_xlim(left=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(width=0.6)

    axes[0].set_ylabel(METRIC)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"sample_{SID}_{METRIC}_grouped.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    pdf_path = os.path.join(PLOTS_DIR, f"sample_{SID}_{METRIC}_grouped.pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {out_path}")
    print(f"[OK] {pdf_path}")


if __name__ == "__main__":
    main()

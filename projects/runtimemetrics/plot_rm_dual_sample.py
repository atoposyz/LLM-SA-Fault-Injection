"""2样本对比：浅/中/深层 stable_rank 对比 (inject vs clean)，竖向排列，3行×2列。"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
matplotlib.use("Agg")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "legend.fontsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

RESULT_DIR = "/workplace/home/mayongzhe/faultinject/projects/runtimemetrics/result"
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")

# ---- 样本1 ----
S1_INJ_FILE = "runtime_metrics_record_sid56_kernel-v_layer-35_cfg-input_bitflip_13_reg-input_df-WS_pe86_82_intv50_detL0_3_6_9_12_15_18_21_24_27_30_33_affect0_inject.jsonl"
S1_CLEAN_FILE = "runtime_metrics_record_sid56_kernel-v_layer-0_cfg-weight_bitflip_10_reg-weight_df-WS_pe_random_intv50_detL0_3_6_9_12_15_18_21_24_27_30_33_affect1_noinject.jsonl"
S1_SID = "56"
# ---- 样本2 ----
S2_INJ_FILE = "runtime_metrics_record_sid932_kernel-v_layer-35_cfg-input_bitflip_13_reg-input_df-WS_pe193_71_intv50_detL0_3_6_9_12_15_18_21_24_27_30_33_affect0_inject.jsonl"
S2_CLEAN_FILE = "runtime_metrics_record_sidall_kernel-v_layer-35_cfg-input_bitflip_13_reg-input_df-WS_pe_random_intv50_detL0_3_6_9_12_15_18_21_24_27_30_33_affect0_noinject.jsonl"
S2_CLEAN_INDEX = 931  # 0-indexed, 第932个
S2_SID = "932"

METRIC = "stable_rank"

GROUPS = {
    "Shallow": [0],
    "Middle": [18],
    "Deep": [33],
}

COLOR_FAULTY = "#e0554a"
COLOR_BASELINE = "#3a7db8"
BG_LEFT = "#fef5f4"
BG_RIGHT = "#f3f6fb"


def load_sample(filepath, sid):
    with open(filepath) as f:
        for line in f:
            rec = json.loads(line)
            if str(rec["sample_id"]) == str(sid):
                return rec["runtime_metrics_values"]
    return []


def load_sample_by_index(filepath, idx):
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)["runtime_metrics_values"]
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


def plot_one_column(ax, inj_grouped, clean_grouped, gname, is_left, facecolor):
    x_f = inj_grouped[gname]["x"]
    y_f = inj_grouped[gname]["y"]
    x_b = clean_grouped[gname]["x"]
    y_b = clean_grouped[gname]["y"]

    ax.set_facecolor(facecolor)

    ax.plot(x_f, y_f, color=COLOR_FAULTY, linewidth=1.8, label="Faulty", zorder=3)
    ax.plot(x_b, y_b, color=COLOR_BASELINE, linewidth=1.8, label="Fault-free", zorder=2)

    step = max(1, len(x_f) // 25)
    ax.scatter(x_f[::step], y_f[::step], color=COLOR_FAULTY, s=10, zorder=4, alpha=0.6, edgecolors="none", linewidth=0)
    ax.scatter(x_b[::step], y_b[::step], color=COLOR_BASELINE, s=10, zorder=4, alpha=0.6, edgecolors="none", linewidth=0)

    if is_left:
        ax.text(-0.22, 0.5, gname, transform=ax.transAxes,
                fontsize=plt.rcParams["axes.titlesize"], fontweight="bold",
                rotation=90, va="center", ha="center", color="#333333")
    ax.set_ylabel("Stable Rank", labelpad=8)

    leg = ax.legend(frameon=True, fancybox=False, framealpha=0.9,
                    edgecolor="#dddddd", loc="upper right",
                    handlelength=1.4, borderpad=0.4, borderaxespad=0.5,
                    ncol=1, columnspacing=0.8)
    leg.get_frame().set_linewidth(0.5)

    ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.4)
    ax.set_xlim(left=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["left"].set_color("#aaaaaa")
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.tick_params(width=0.4, colors="#666666")


def main():
    s1_inj = group_metrics(load_sample(os.path.join(RESULT_DIR, S1_INJ_FILE), S1_SID), GROUPS)
    s1_clean = group_metrics(load_sample(os.path.join(RESULT_DIR, S1_CLEAN_FILE), S1_SID), GROUPS)
    s2_inj = group_metrics(load_sample(os.path.join(RESULT_DIR, S2_INJ_FILE), S2_SID), GROUPS)
    s2_clean = group_metrics(load_sample_by_index(os.path.join(RESULT_DIR, S2_CLEAN_FILE), S2_CLEAN_INDEX), GROUPS)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    group_names = list(GROUPS.keys())

    for row, gname in enumerate(group_names):
        plot_one_column(axes[row, 0], s1_inj, s1_clean, gname, is_left=True,
                        facecolor=BG_LEFT)
        plot_one_column(axes[row, 1], s2_inj, s2_clean, gname, is_left=False,
                        facecolor=BG_RIGHT)

    axes[-1, 0].set_xlabel("Token", labelpad=6)
    axes[-1, 1].set_xlabel("Token", labelpad=6)

    axes[0, 0].set_title("nonsensical-generation", fontweight="bold", pad=12, color="#333333")
    axes[0, 1].set_title("repetitive-generation", fontweight="bold", pad=12, color="#333333")

    fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.12, wspace=0.22, hspace=0.28)

    fig.canvas.draw()
    pad = 0.05
    pad_left = 0.07
    pad_bottom = 0.06
    mid_x = (axes[0, 0].get_position().x1 + axes[0, 1].get_position().x0) / 2
    for col, bg in enumerate([BG_LEFT, BG_RIGHT]):
        col_axes = axes[:, col]
        y0 = min(ax.get_position().y0 for ax in col_axes) - pad_bottom
        y1 = max(ax.get_position().y1 for ax in col_axes) + pad
        if col == 0:
            x0 = min(ax.get_position().x0 for ax in col_axes) - pad_left + 0.01
            x1 = mid_x
        else:
            x0 = mid_x - 0.02
            x1 = max(ax.get_position().x1 for ax in col_axes) + pad
        rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  facecolor=bg, edgecolor="#e8e8e8",
                                  linewidth=0.5, transform=fig.transFigure,
                                  zorder=0)
        fig.add_artist(rect)

    out_path = os.path.join(PLOTS_DIR, f"dual_sample_{S1_SID}_{S2_SID}_{METRIC}_grouped.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    pdf_path = os.path.join(PLOTS_DIR, f"dual_sample_{S1_SID}_{S2_SID}_{METRIC}_grouped.pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] {out_path}")
    print(f"[OK] {pdf_path}")


if __name__ == "__main__":
    main()

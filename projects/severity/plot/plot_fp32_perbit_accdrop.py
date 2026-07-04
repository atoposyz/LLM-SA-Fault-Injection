"""
Plot per-bit accuracy drop for FP32 stuck-at faults.

Two subplots (SA0 / SA1), each with input and weight as separate lines.
Data: result_fp32/accuracy_drop_perbit_{type}_stuck_{value}.csv

Usage:
  uv run python projects/severity/plot_fp32_perbit_accdrop.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "comparison_results")

MODEL_SPECS = {
    "bert": {
        "label": "BERT",
        "color": "#d55e00",
        "data_dir": os.path.join(SCRIPT_DIR, "..", "..", "bert", "result_fp32"),
        "template": "accuracy_drop_perbit_{typ}_stuck_{sv}.csv",
        "drop_col": "acc_drop",
    },
    "qwen3-8b": {
        "label": "Qwen3-8B",
        "color": "#0072b2",
        "data_dir": os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "qwen3-8b", "result"),
        "template": "accuracy_drop_plot_form_{typ}_stuck_{sv}.csv",
        "drop_col": "accuracy_drop",
    },
}
TYPE_LABELS = {"input": "Input", "weight": "Weight"}
LINE_STYLES = {"input": "-", "weight": "--"}
MARKERS = {"input": "o", "weight": "s"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.55,
    "axes.edgecolor": "#666666",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 450,
})


def load_csv(model: str, typ: str, sv: int) -> dict[int, float]:
    spec = MODEL_SPECS[model]
    path = os.path.join(spec["data_dir"], spec["template"].format(typ=typ, sv=sv))
    result = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            result[int(row["bit"])] = float(row[spec["drop_col"]])
    return result


def add_field_regions(ax, show_labels: bool = False):
    """Shade IEEE 754 FP32 field regions without dominating the data."""
    regions = [
        (-0.5, 22.5, "Mantissa", "#5aae61", 0.055),
        (22.5, 30.5, "Exponent", "#f1a340", 0.085),
        (30.5, 31.5, "Sign", "#b2182b", 0.105),
    ]
    for start, end, _, color, alpha in regions:
        ax.axvspan(start, end, color=color, alpha=alpha, linewidth=0, zorder=0)
    for boundary in (22.5, 30.5):
        ax.axvline(boundary, color="#7a7a7a", linewidth=0.65, linestyle="--", dashes=(3, 2), zorder=1)

    if not show_labels:
        return

    label_specs = [
        (11.0, "Mantissa", "#2f7d32"),
        (26.5, "Exponent", "#9a5b00"),
        (31.0, "Sign", "#8b1a1a"),
    ]
    for x, label, color in label_specs:
        ax.text(
            x,
            1.015,
            label,
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color=color,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )


def annotate_peak(ax, bits, values, model: str, typ: str):
    peak_idx = int(np.argmax(values))
    bit = bits[peak_idx]
    value = values[peak_idx]
    color = MODEL_SPECS[model]["color"]
    model_label = MODEL_SPECS[model]["label"]
    yoff = 8 if typ == "input" else -13
    va = "bottom" if typ == "input" else "top"
    ax.annotate(
        f"{model_label} b{bit}",
        xy=(bit, value),
        xytext=(0, yoff),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=5.9,
        color=color,
        arrowprops={
            "arrowstyle": "-",
            "linewidth": 0.42,
            "color": color,
            "shrinkA": 1,
            "shrinkB": 2,
        },
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bits = list(range(32))
    loaded = {
        (model, typ, sv): load_csv(model, typ, sv)
        for model in MODEL_SPECS
        for typ in ("input", "weight")
        for sv in (0, 1)
    }
    ymax = max(max(data.values()) for data in loaded.values())
    ymin = min(min(data.values()) for data in loaded.values())
    y_bottom = max(0.0, ymin - 0.045)
    y_top = min(0.70, ymax + 0.065)

    fig, axes = plt.subplots(2, 1, figsize=(7.15, 4.35), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.125, top=0.865, hspace=0.175)

    for panel_idx, (ax, sv, title) in enumerate([
        (axes[0], 0, "Stuck-at-0"),
        (axes[1], 1, "Stuck-at-1"),
    ]):
        add_field_regions(ax, show_labels=(panel_idx == 0))

        for model, spec in MODEL_SPECS.items():
            for typ in ("input", "weight"):
                data = loaded[(model, typ, sv)]
                y = np.array([data[b] for b in bits], dtype=float)
                ax.plot(
                    bits,
                    y,
                    color=spec["color"],
                    linestyle=LINE_STYLES[typ],
                    linewidth=1.45,
                    marker=MARKERS[typ],
                    markersize=3.3,
                    markerfacecolor="white",
                    markeredgewidth=0.78,
                    label=f"{spec["label"]} {TYPE_LABELS[typ]}",
                    alpha=0.95 if model == "bert" else 0.9,
                    zorder=3 if model == "bert" else 2,
                )
                annotate_peak(ax, bits, y, model, typ)

        ax.text(
            0.012,
            0.91,
            f"({chr(97 + panel_idx)}) {title}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            fontweight="bold",
            color="#111111",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
        )
        ax.set_ylabel("Accuracy drop", fontsize=8.5)
        ax.set_ylim(y_bottom, y_top)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.grid(axis="y", alpha=0.32, linestyle=":", linewidth=0.55)
        ax.grid(axis="x", alpha=0.12, linestyle=":", linewidth=0.45)
        ax.tick_params(axis="both", which="major", labelsize=7.2, length=2.5, width=0.5, pad=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(
        fontsize=7.0,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=4,
        frameon=True,
        framealpha=0.92,
        edgecolor="#d0d0d0",
        handlelength=1.8,
        columnspacing=0.75,
        handletextpad=0.45,
        borderpad=0.35,
    )

    axes[1].set_xlabel("Bit position in IEEE 754 FP32", fontsize=8.5)
    axes[1].set_xticks(range(0, 32, 2))
    axes[1].set_xticklabels([str(i) for i in range(0, 32, 2)])
    axes[1].set_xlim(-0.6, 31.6)

    for ext in ("png", "pdf"):
        out = os.path.join(OUTPUT_DIR, f"fp32_perbit_accdrop.{ext}")
        fig.savefig(out, dpi=450, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        print(f"Saved: {os.path.basename(out)}")
    plt.close(fig)


if __name__ == "__main__":
    main()

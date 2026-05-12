"""
Bit-level severity comparison: Key vs Value projections.
3 subplot columns (weight / activation-input / psum-output) × 2 rows (SA0 / SA1).
"""
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SOURCES = ["weight", "activation_input", "psum_output"]
SRC_LABELS = {"weight": "Weight", "activation_input": "Activation (input)", "psum_output": "Psum (output)"}

COLORS = {"key": "#2196F3", "value": "#FF5722"}


def load_kv(source: str):
    data = {}
    for module in ("key", "value"):
        path = f"projects/bert/config/severity_table_{source}_fp32_{module}.json"
        with open(path) as f:
            t = json.load(f)
        entries = t["table"]
        xs, sa0, sa1 = [], [], []
        for e in entries:
            xs.append(e["bit"])
            sa0.append(e.get("sa0_unconditional_norm", 0))
            sa1.append(e.get("sa1_unconditional_norm", 0))
        order = np.argsort(xs)
        data[module] = {
            "x": np.array(xs)[order],
            "sa0": np.array(sa0)[order],
            "sa1": np.array(sa1)[order],
        }
    return data


def add_field_regions(ax):
    for lo, hi, color, alpha in [
        (-0.5, 22.5, "green", 0.06),
        (22.5, 30.5, "orange", 0.08),
        (30.5, 31.5, "red", 0.12),
    ]:
        ax.axvspan(lo, hi, alpha=alpha, color=color, zorder=0)
    ax.text(11, 1.07, "Mantissa\n(23 bits)", ha="center", va="bottom",
            fontsize=7, color="green", fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(26.5, 1.07, "Exponent\n(8 bits)", ha="center", va="bottom",
            fontsize=7, color="orange", fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(31, 1.07, "Sign\n(1 bit)", ha="center", va="bottom",
            fontsize=7, color="red", fontweight="bold", transform=ax.get_xaxis_transform())


def plot():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    fig.subplots_adjust(hspace=0.18, wspace=0.18)

    for col, src in enumerate(SOURCES):
        data = load_kv(src)
        for row, (direction, title) in enumerate([
            ("sa0", "Stuck-at-0"),
            ("sa1", "Stuck-at-1"),
        ]):
            ax = axes[row][col]
            for mod in ("key", "value"):
                d = data[mod]
                ax.plot(d["x"], d[direction],
                        linewidth=2.0, marker="o", markersize=3, markevery=4,
                        color=COLORS[mod], label=mod.capitalize(), alpha=0.85)

            add_field_regions(ax)
            ax.set_ylim(-0.03, 1.10)
            ax.set_yticks(np.arange(0, 1.1, 0.25))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            ax.grid(True, alpha=0.3, linestyle=":")
            if col == 0:
                ax.set_ylabel("Normalised Severity", fontsize=10)
            if row == 0:
                ax.set_title(SRC_LABELS[src], fontsize=11, fontweight="bold")
            if row == 0 and col == len(SOURCES) - 1:
                ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
            ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=10,
                    fontweight="bold", va="top", color="#333333")

    for col in range(3):
        axes[1][col].set_xlabel("Bit Position (IEEE 754 FP32)", fontsize=9)
        axes[1][col].set_xticks(range(0, 32))
        axes[1][col].set_xticklabels([str(i) for i in range(32)], fontsize=6)
        axes[1][col].set_xlim(-0.8, 31.8)

    fig.suptitle(
        "Key vs Value — Bit-Level Stuck-at Fault Severity  (FP32, log1p, 2M elements)\n"
        "boltuix/bert-emotion  —  Key/Value projection layers only",
        fontsize=13, fontweight="bold", y=1.01,
    )
    out_path = "projects/bert/config/severity_kv_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    plot()

"""
Plot stuck-at normalized severity weights — FP32 only, 3 sources:
weight, activation (input), psum (output).
"""
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

TABLES = {
    "weight":      "projects/qwen3-0.6b/config/severity_table_weight_fp32.json",
    "activation":  "projects/qwen3-0.6b/config/severity_table_activation_input_fp32.json",
    "psum":        "projects/qwen3-0.6b/config/severity_table_psum_output_fp32.json",
}

COLORS = {
    "weight":      "#2196F3",
    "activation":  "#FF5722",
    "psum":        "#4CAF50",
}

STYLES = {
    "weight":      "--",
    "activation":  "-",
    "psum":        "-.",
}

LABELS = {
    "weight":      "Weight",
    "activation":  "Activation (input)",
    "psum":        "Psum (output)",
}


def load_data():
    data = {}
    for name, path in TABLES.items():
        with open(path) as f:
            t = json.load(f)
        entries = t["table"]
        xs, sa0, sa1 = [], [], []
        for e in entries:
            xs.append(e["bit"])
            sa0.append(e.get("sa0_unconditional_norm", 0))
            sa1.append(e.get("sa1_unconditional_norm", 0))
        order = np.argsort(xs)
        data[name] = {
            "x": np.array(xs)[order],
            "sa0": np.array(sa0)[order],
            "sa1": np.array(sa1)[order],
        }
    return data


def add_field_regions(ax):
    """Shade IEEE 754 field regions."""
    ax.axvspan(-0.5, 22.5, alpha=0.06, color='green', zorder=0)
    ax.axvspan(22.5, 30.5, alpha=0.08, color='orange', zorder=0)
    ax.axvspan(30.5, 31.5, alpha=0.12, color='red', zorder=0)

    # Labels at top
    ax.text(11, 1.07, "Mantissa\n(23 bits)", ha='center', va='bottom',
            fontsize=9, color='green', fontweight='bold',
            transform=ax.get_xaxis_transform())
    ax.text(26.5, 1.07, "Exponent\n(8 bits)", ha='center', va='bottom',
            fontsize=9, color='orange', fontweight='bold',
            transform=ax.get_xaxis_transform())
    ax.text(31, 1.07, "Sign\n(1 bit)", ha='center', va='bottom',
            fontsize=9, color='red', fontweight='bold',
            transform=ax.get_xaxis_transform())


def plot():
    data = load_data()

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    for ax, direction, title in [
        (ax0, "sa0", "Stuck-at-0  (force bit → 0)"),
        (ax1, "sa1", "Stuck-at-1  (force bit → 1)"),
    ]:
        for name in ["weight", "activation", "psum"]:
            d = data[name]
            ax.plot(d["x"], d[direction],
                    STYLES[name], color=COLORS[name],
                    label=LABELS[name], linewidth=2.0, marker='o',
                    markersize=4, markevery=4,
                    alpha=0.85)

        add_field_regions(ax)
        ax.set_ylabel("Normalised Severity", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left', color='#333333')
        ax.set_ylim(-0.03, 1.10)
        ax.set_yticks(np.arange(0, 1.1, 0.25))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(fontsize=9, loc='upper left', framealpha=0.92,
                  ncol=3, columnspacing=0.6)

    ax1.set_xlabel("Bit Position (IEEE 754 FP32)", fontsize=11)
    ax1.set_xticks(range(0, 32))
    ax1.set_xticklabels([str(i) for i in range(32)], fontsize=7)
    ax1.set_xlim(-0.8, 31.8)

    fig.suptitle(
        "Bit-Level Stuck-at Fault Severity — FP32 | Weight vs Activation vs Psum\n"
        "Qwen3-0.6B Attention Layers  |  transform=log1p  |  2M elements each",
        fontsize=13, fontweight='bold', y=0.985
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = "projects/qwen3-0.6b/config/severity_stuckat_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    plot()

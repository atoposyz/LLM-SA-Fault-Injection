"""
Plot stuck-at normalized severity curves — FP32, 3 sources:
weight, activation (input), psum (output).

Uses CONDITIONAL severity (normalized per-bit across operator groups)
which better captures the per-element impact of rare-but-catastrophic faults.

Qwen/Qwen3-8B  |  Dataset: cais/mmlu

Usage:
  uv run python projects/qwen3-8b/plot_severity.py [--precision fp32|bf16]
"""
import argparse
import json
import math
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config")

OPERATOR_GROUPS = ["attention", "intermediate", "output"]

SOURCES = {
    "weight":      "weight",
    "activation":  "activation_input",
    "psum":        "activation_output",
}

SOURCE_LABELS = {
    "weight":      "Weight",
    "activation":  "Activation (input)",
    "psum":        "Psum (output)",
}

COLORS = {
    "weight":      "#2196F3",
    "activation":  "#FF5722",
    "psum":        "#4CAF50",
}

STYLES = {
    "weight":      "dashed",
    "activation":  "solid",
    "psum":        "dashdot",
}

MARKERS = {
    "weight":      "o",
    "activation":  "s",
    "psum":        "^",
}


# ---------------------------------------------------------------------------
# Data loading with conditional severity
# ---------------------------------------------------------------------------

def _load_raw(source, precision):
    """Load raw severity values (conditional + unconditional) per bit,
    aggregated across operator groups.

    Returns dict[bit] -> {sa0_cond, sa0_uncond, sa1_cond, sa1_uncond}
    """
    per_bit = {}
    for op in OPERATOR_GROUPS:
        path = os.path.join(CONFIG_DIR, f"severity_table_{source}_{precision}_{op}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            tbl = json.load(f)
        for e in tbl["table"]:
            b = e["bit"]
            per_bit.setdefault(b, {})
            for k in ["sa0_conditional", "sa0_unconditional",
                      "sa1_conditional", "sa1_unconditional"]:
                per_bit[b].setdefault(k, []).append(e.get(k, 0))

    result = {}
    for b, d in per_bit.items():
        result[b] = {k: float(np.mean(v)) for k, v in d.items()}

    # Apply IEEE 754 theoretical floor for exponent bits with insufficient samples
    _apply_theoretical_floor(result, precision)
    return result


def _theoretical_exponent_severity(bit, precision):
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


def _apply_theoretical_floor(raw_data, precision):
    """Patch conditional severity for exponent bits that lack calibration samples."""
    for b, d in raw_data.items():
        theory = _theoretical_exponent_severity(b, precision)
        if theory <= 0:
            continue
        for prefix in ["sa0", "sa1"]:
            cond_key = f"{prefix}_conditional"
            if cond_key not in d:
                continue
            cal_val = d[cond_key]
            if cal_val < theory * 0.01:
                # effective_rate from the same fault type
                eff = d.get(f"{prefix}_effective_rate", d.get("sa1_effective_rate", 0.001))
                proxy_eff = max(eff, 0.001)
                d[cond_key] = max(cal_val, theory * proxy_eff)


def _normalize_across_bits(raw_data, field):
    """Min-max normalize a given field across all bits in raw_data."""
    vals = np.array([raw_data[b].get(field, 0) for b in sorted(raw_data)])
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-12:
        return {b: 0.0 for b in raw_data}
    return {b: float((raw_data[b].get(field, 0) - vmin) / (vmax - vmin))
            for b in raw_data}


def load_data(precision, use_conditional=True):
    """Load and normalize severity data for all 3 sources."""
    data = {}
    for name, table_source in SOURCES.items():
        raw = _load_raw(table_source, precision)
        if not raw:
            print(f"[WARNING] No tables found for {name} ({table_source}), skipping")
            continue

        bits = sorted(raw.keys())
        if use_conditional:
            sa0_norm = _normalize_across_bits(raw, "sa0_conditional")
            sa1_norm = _normalize_across_bits(raw, "sa1_conditional")
        else:
            sa0_norm = _normalize_across_bits(raw, "sa0_unconditional")
            sa1_norm = _normalize_across_bits(raw, "sa1_unconditional")

        data[name] = {
            "x": np.array(bits),
            "sa0": np.array([sa0_norm[b] for b in bits]),
            "sa1": np.array([sa1_norm[b] for b in bits]),
        }
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def add_field_regions_fp32(ax):
    ax.axvspan(-0.5, 22.5, alpha=0.06, color='green', zorder=0)
    ax.axvspan(22.5, 30.5, alpha=0.08, color='orange', zorder=0)
    ax.axvspan(30.5, 31.5, alpha=0.12, color='red', zorder=0)
    ax.text(11, 1.07, "Mantissa\n(23 bits)", ha='center', va='bottom',
            fontsize=9, color='green', fontweight='bold',
            transform=ax.get_xaxis_transform())
    ax.text(26.5, 1.07, "Exponent\n(8 bits)", ha='center', va='bottom',
            fontsize=9, color='orange', fontweight='bold',
            transform=ax.get_xaxis_transform())
    ax.text(31, 1.07, "Sign\n(1 bit)", ha='center', va='bottom',
            fontsize=9, color='red', fontweight='bold',
            transform=ax.get_xaxis_transform())


def add_field_regions_bf16(ax):
    ax.axvspan(-0.5, 6.5, alpha=0.06, color='green', zorder=0)
    ax.axvspan(6.5, 14.5, alpha=0.08, color='orange', zorder=0)
    ax.axvspan(14.5, 15.5, alpha=0.12, color='red', zorder=0)
    ax.text(3, 1.07, "Mantissa\n(7 bits)", ha='center', va='bottom',
            fontsize=9, color='green', fontweight='bold',
            transform=ax.get_xaxis_transform())
    ax.text(10.5, 1.07, "Exponent\n(8 bits)", ha='center', va='bottom',
            fontsize=9, color='orange', fontweight='bold',
            transform=ax.get_xaxis_transform())
    ax.text(15, 1.07, "Sign\n(1 bit)", ha='center', va='bottom',
            fontsize=9, color='red', fontweight='bold',
            transform=ax.get_xaxis_transform())


def plot(precision="fp32", use_conditional=True):
    data = load_data(precision, use_conditional)

    num_bits = 32 if precision == "fp32" else 16
    add_regions = add_field_regions_fp32 if precision == "fp32" else add_field_regions_bf16
    precision_label = precision.upper()
    mode_label = "Conditional" if use_conditional else "Unconditional"

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    for ax, direction, title in [
        (ax0, "sa0", "Stuck-at-0  (force bit → 0)"),
        (ax1, "sa1", "Stuck-at-1  (force bit → 1)"),
    ]:
        for name in ["weight", "activation", "psum"]:
            if name not in data:
                continue
            d = data[name]
            ax.plot(d["x"], d[direction],
                    linestyle=STYLES[name], color=COLORS[name],
                    marker=MARKERS[name], markersize=4, markevery=4,
                    label=SOURCE_LABELS[name], linewidth=2.0,
                    alpha=0.85)

        add_regions(ax)
        ax.set_ylabel("Normalised Severity", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left', color='#333333')
        ax.set_ylim(-0.03, 1.10)
        ax.set_yticks(np.arange(0, 1.1, 0.25))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(fontsize=9, loc='upper left', framealpha=0.92,
                  ncol=3, columnspacing=0.6)

    ax1.set_xlabel(f"Bit Position (IEEE 754 {precision_label})", fontsize=11)
    ax1.set_xticks(range(0, num_bits))
    ax1.set_xticklabels([str(i) for i in range(num_bits)], fontsize=7)
    ax1.set_xlim(-0.8, num_bits - 0.2)

    fig.suptitle(
        f"Bit-Level Stuck-at Fault Severity — {precision_label} | {mode_label} Severity\n"
        f"Qwen/Qwen3-8B  |  transform=log1p  |  Dataset: cais/mmlu",
        fontsize=13, fontweight='bold', y=0.985
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    suffix = "cond" if use_conditional else "uncond"
    out_path = os.path.join(CONFIG_DIR, f"severity_stuckat_comparison_{precision}_{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot stuck-at severity comparison for Qwen3-8B"
    )
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32",
                        help="Precision (default: fp32)")
    parser.add_argument("--conditional", action="store_true", default=True,
                        help="Use conditional severity (default)")
    parser.add_argument("--unconditional", action="store_false", dest="conditional",
                        help="Use unconditional severity (legacy)")
    args = parser.parse_args()

    plot(args.precision, args.conditional)


if __name__ == "__main__":
    main()

"""
Find the best severity formula transform by comparing predicted severity
against measured accuracy drop.

Tests multiple candidate transforms on raw unconditional severity to find
which best predicts actual accuracy impact, WITHOUT direct per-bit fitting.

The goal is to find a transform that:
  1. Has high rank correlation (Spearman) with actual accuracy drop
  2. Generalizes — uses the same formula parameters for all types/bits
  3. Is monotonic and interpretable

Usage:
  uv run python projects/qwen3-8b/calibrate_formula.py
"""
import csv
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config")
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
PRECISION = "fp32"
NUM_BITS = 32

OPERATOR_GROUPS = ["attention", "intermediate", "output"]

SOURCE_KEY = {"input": "activation_input", "weight": "weight"}

CSV_MAP = {
    ("input",  0): "accuracy_drop_plot_form_input_stuck_0.csv",
    ("input",  1): "accuracy_drop_plot_form_input_stuck_1.csv",
    ("weight", 0): "accuracy_drop_plot_form_weight_stuck_0.csv",
    ("weight", 1): "accuracy_drop_plot_form_weight_stuck_1.csv",
}

COLORS = {"input": "#FF5722", "weight": "#2196F3"}


# ---------------------------------------------------------------------------
# Load raw severity + actual drop
# ---------------------------------------------------------------------------

def load_raw_data():
    """Return (raw_severity, actual_drop, bit_fields) per (type, stuck_value, bit)."""
    raw = {}   # (typ, sv, bit) -> raw unconditional severity
    actual = {}  # (typ, sv, bit) -> accuracy_drop
    fields = {}  # (typ, bit) -> field name

    for typ, src in SOURCE_KEY.items():
        for op in OPERATOR_GROUPS:
            path = os.path.join(CONFIG_DIR, f"severity_table_{src}_{PRECISION}_{op}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                tbl = json.load(f)
            for e in tbl["table"]:
                b = e["bit"]
                fields[(typ, b)] = e.get("field", "?")
                for sv in [0, 1]:
                    val = e.get(f"sa{sv}_unconditional", 0)
                    raw.setdefault((typ, sv, b), []).append(val)

    # Average across operator groups
    raw_mean = {k: float(np.mean(v)) for k, v in raw.items()}

    # Load actual
    for (typ, sv), fname in CSV_MAP.items():
        path = os.path.join(RESULT_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                b = int(r["bit"])
                actual[(typ, sv, b)] = max(float(r["accuracy_drop"]), 0)

    return raw_mean, actual, fields


# ---------------------------------------------------------------------------
# Candidate transforms
# ---------------------------------------------------------------------------

def make_transforms():
    """Return dict of {name: (fn, param_range)} for candidate transforms.

    Each fn(raw, **params) -> transformed value.
    We wrap with min-max normalization since downstream uses normalized values.
    """
    def log1p(raw, **_kw):
        return np.log1p(raw)

    def power(raw, alpha=0.3):
        return np.power(np.maximum(raw, 1e-12), alpha)

    def sqrt(raw, **_kw):
        return np.sqrt(np.maximum(raw, 0))

    def log1p_exp(raw, beta=0.5):
        return np.log1p(np.expm1(raw) * beta)

    def sigmoid_like(raw, scale=1.0):
        return raw / (scale + raw)

    transforms = {
        "log1p":          (log1p,          {}),
        "sqrt":           (sqrt,           {}),
        "power_0.2":       (power,          {"alpha": 0.2}),
        "power_0.3":       (power,          {"alpha": 0.3}),
        "power_0.4":       (power,          {"alpha": 0.4}),
        "power_0.5":       (power,          {"alpha": 0.5}),
        "sigmoid_s1":     (sigmoid_like,   {"scale": 1.0}),
        "sigmoid_s5":     (sigmoid_like,   {"scale": 5.0}),
        "sigmoid_s10":    (sigmoid_like,   {"scale": 10.0}),
        "log1p_exp_0.3":  (log1p_exp,     {"beta": 0.3}),
        "log1p_exp_0.5":  (log1p_exp,     {"beta": 0.5}),
    }
    return transforms


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_transforms(raw, actual):
    """For each (typ, sv) and each transform, compute Spearman rho."""
    transforms = make_transforms()
    results = {}

    for (typ, sv) in [("input", 0), ("input", 1), ("weight", 0), ("weight", 1)]:
        bits = sorted(set(b for (t, s, b) in raw if t == typ and s == sv))
        if not bits:
            continue

        raw_vals = np.array([raw.get((typ, sv, b), 0) for b in bits])
        act_vals = np.array([actual.get((typ, sv, b), 0) for b in bits])

        for tname, (tfn, tparams) in transforms.items():
            try:
                tf_vals = tfn(raw_vals, **tparams)
                # Handle inf/nan
                tf_vals = np.nan_to_num(tf_vals, nan=0.0, posinf=1e6, neginf=0.0)
                if np.allclose(tf_vals, tf_vals[0]):
                    rho = 0.0
                else:
                    rho, _ = spearmanr(tf_vals, act_vals)
            except Exception:
                rho = 0.0
            results.setdefault((typ, sv), {})[tname] = round(float(rho), 4)

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def add_field_regions(ax):
    ax.axvspan(-0.5, 22.5, alpha=0.06, color='green', zorder=0)
    ax.axvspan(22.5, 30.5, alpha=0.08, color='orange', zorder=0)
    ax.axvspan(30.5, 31.5, alpha=0.12, color='red', zorder=0)
    ax.text(11, 1.07, "Mantissa", ha='center', va='bottom',
            fontsize=8, color='green', fontweight='bold', transform=ax.get_xaxis_transform())
    ax.text(26.5, 1.07, "Exponent", ha='center', va='bottom',
            fontsize=8, color='orange', fontweight='bold', transform=ax.get_xaxis_transform())
    ax.text(31, 1.07, "Sign", ha='center', va='bottom',
            fontsize=8, color='red', fontweight='bold', transform=ax.get_xaxis_transform())


def plot_best_vs_original(raw, actual, fields, results):
    """Plot: original (log1p + norm) vs best-transform prediction vs actual."""
    transforms = make_transforms()

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.subplots_adjust(hspace=0.25)

    bits_arr = np.arange(NUM_BITS)

    for row, (typ, sv) in enumerate([("input", 0), ("input", 1),
                                       ("weight", 0), ("weight", 1)]):
        ax = axes[row]

        raw_vals = np.array([raw.get((typ, sv, b), 0) for b in bits_arr])
        act_vals = np.array([actual.get((typ, sv, b), 0) for b in bits_arr])

        # Best transform
        scores = results.get((typ, sv), {})
        if scores:
            best_name = max(scores, key=scores.get)
            best_fn, best_params = transforms[best_name]
            best_vals = best_fn(raw_vals, **best_params)
            best_vals = np.nan_to_num(best_vals, nan=0.0, posinf=1e6, neginf=0.0)
            # Normalize to [0, 1] for plotting
            vmin, vmax = best_vals.min(), best_vals.max()
            if vmax - vmin > 1e-12:
                best_norm = (best_vals - vmin) / (vmax - vmin)
            else:
                best_norm = np.zeros_like(best_vals)
        else:
            best_name = "N/A"
            best_norm = np.zeros(NUM_BITS)

        # Original (log1p + minmax norm)
        orig_log1p = np.log1p(raw_vals)
        vmin_o, vmax_o = orig_log1p.min(), orig_log1p.max()
        if vmax_o - vmin_o > 1e-12:
            orig_norm = (orig_log1p - vmin_o) / (vmax_o - vmin_o)
        else:
            orig_norm = np.zeros_like(orig_log1p)

        # Normalize actual for visual comparison
        act_max = act_vals.max()
        act_norm = act_vals / act_max if act_max > 0 else act_vals

        rho_orig = spearmanr(orig_norm, act_norm)[0]
        rho_best = spearmanr(best_norm, act_norm)[0]

        ax.plot(bits_arr, orig_norm,
                linestyle="dotted", color="gray", linewidth=1.5,
                marker='x', markersize=4, markevery=4,
                alpha=0.55, label=f"Original (log1p+norm), ρ={rho_orig:.3f}")
        ax.plot(bits_arr, best_norm,
                linestyle="dashed", color=COLORS[typ], linewidth=2.0,
                marker='s', markersize=5, markevery=2,
                alpha=0.8, label=f"Best: {best_name}, ρ={rho_best:.3f}")
        ax.plot(bits_arr, act_norm,
                linestyle="solid", color="black", linewidth=2.5,
                marker='o', markersize=6, markevery=2,
                alpha=0.9, label="Actual (normalized)")

        add_field_regions(ax)
        ax.set_ylabel("Normalized Severity", fontsize=10)
        ax.set_title(f"{typ} — stuck-at-{sv}  |  best transform: {best_name}",
                     fontsize=11, fontweight='bold', loc='left', color=COLORS[typ])
        ax.set_ylim(-0.04, 1.08)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9)

    axes[-1].set_xlabel("Bit Position (IEEE 754 FP32)", fontsize=11)
    axes[-1].set_xticks(range(0, NUM_BITS))
    axes[-1].set_xticklabels([str(i) for i in range(NUM_BITS)], fontsize=7)
    axes[-1].set_xlim(-0.8, 31.8)

    fig.suptitle(
        "Severity Prediction Formula Comparison — Original vs Best Transform vs Actual\n"
        "Qwen/Qwen3-8B  |  FP32  |  cais/mmlu",
        fontsize=14, fontweight='bold', y=1.005
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(CONFIG_DIR, "severity_formula_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    raw, actual, fields = load_raw_data()

    print("Evaluating transform candidates...")
    results = evaluate_transforms(raw, actual)

    # Print results
    print("\n" + "=" * 85)
    print("SPEARMAN RHO: Transform → Accuracy Drop correlation")
    print("=" * 85)
    header = f"{'Transform':<20s}"
    for typ, sv in [("input", 0), ("input", 1), ("weight", 0), ("weight", 1)]:
        header += f" {'i/sa'+str(sv):>10s}" if typ == "input" else f" {'w/sa'+str(sv):>10s}"
    print(header)
    print("-" * 65)

    transforms = make_transforms()
    for tname in transforms:
        row = f"{tname:<20s}"
        for typ, sv in [("input", 0), ("input", 1), ("weight", 0), ("weight", 1)]:
            rho = results.get((typ, sv), {}).get(tname, 0)
            row += f" {rho:10.4f}"
        print(row)

    # Find best overall
    all_rhos = {}
    for tname in transforms:
        vals = []
        for typ, sv in [("input", 0), ("input", 1), ("weight", 0), ("weight", 1)]:
            vals.append(results.get((typ, sv), {}).get(tname, 0))
        all_rhos[tname] = float(np.mean(vals))

    best_overall = max(all_rhos, key=all_rhos.get)
    print(f"\nBest overall transform: {best_overall} (mean ρ = {all_rhos[best_overall]:.4f})")

    # Print per-type best
    print("\nBest per-type:")
    for typ, sv in [("input", 0), ("input", 1), ("weight", 0), ("weight", 1)]:
        scores = results.get((typ, sv), {})
        if scores:
            best = max(scores, key=scores.get)
            print(f"  {typ} stuck-at-{sv}: {best} (ρ = {scores[best]:.4f})")

    # Plot
    os.makedirs(CONFIG_DIR, exist_ok=True)
    plot_best_vs_original(raw, actual, fields, results)
    print("\nDone.")


if __name__ == "__main__":
    main()

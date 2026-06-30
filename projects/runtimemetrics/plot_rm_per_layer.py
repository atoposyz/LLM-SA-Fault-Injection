"""
Plot runtime metric traces per layer — all metrics on one figure per layer.

Each figure contains one subplot per metric (3x3 grid), overlaying
fault-injected vs baseline values over generated token positions.

Usage:
  uv run python projects/runtimemetrics/plot_rm_per_layer.py
  uv run python projects/runtimemetrics/plot_rm_per_layer.py --samples 56
"""

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULT_DIR = os.path.join(SCRIPT_DIR, "result")

INJECT_FILES = []
BASELINE_FILES = [
    "runtime_metrics_record_sidall_kernel-v_layer-35_cfg-input_bitflip_13_reg-input_df-WS_pe_random_intv50_detL0_3_6_9_12_15_18_21_24_27_30_33_affect0_noinject.jsonl",
]

METRICS = [
    "stable_rank",
    "svd_entropy",
    "participation_ratio",
    "top5_energy_ratio",
    "frobenius_norm_sq",
    "spectral_norm_sq",
    "numerical_rank",
    "nuclear_norm",
    "normalized_nuclear_rank",
]

METRIC_LABELS = {
    "stable_rank": "Stable Rank",
    "svd_entropy": "SVD Entropy",
    "participation_ratio": "Participation Ratio",
    "top5_energy_ratio": "Top-5 Energy Ratio",
    "frobenius_norm_sq": "Frobenius Norm Squared",
    "spectral_norm_sq": "Spectral Norm Squared",
    "numerical_rank": "Numerical Rank",
    "nuclear_norm": "Nuclear Norm",
    "normalized_nuclear_rank": "Normalized Nuclear Rank",
}

COLORS = {
    "inject": "#E84D3D",
    "baseline": "#3D7DB5",
    "grid": "#E8E4E0",
    "text": "#2D2D2D",
    "bg": "#FAFAFA",
}


def configure_matplotlib():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": COLORS["bg"],
    })


def numeric_sample_key(sample_id):
    try:
        return int(sample_id)
    except ValueError:
        return sample_id


def load_metric_records(result_dir, inject_files, baseline_files):
    data = {}
    seen_files = set()

    def parse_file(filename, tag):
        path = os.path.join(result_dir, filename)
        if path in seen_files:
            return 0
        seen_files.add(path)

        if not os.path.exists(path):
            print(f"[WARN] Missing {tag} file: {path}")
            return 0

        count = 0
        with open(path) as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[WARN] Skip malformed JSON in {path}:{line_no}: {exc}")
                    continue

                sample_id = str(record.get("sample_id"))
                if sample_id == "None":
                    print(f"[WARN] Skip record without sample_id in {path}:{line_no}")
                    continue

                sample = data.setdefault(sample_id, {"inject": [], "baseline": []})
                run = {}
                for metric_row in record.get("runtime_metrics_values", []):
                    if "layer" not in metric_row or "token_end" not in metric_row:
                        continue
                    layer = int(metric_row["layer"])
                    run.setdefault(layer, []).append((metric_row["token_end"], metric_row))
                if run:
                    sample[tag].append(run)
                count += 1
        return count

    for filename in baseline_files:
        count = parse_file(filename, "baseline")
        print(f"[INFO] Loaded {count} baseline records from {filename}")
    for filename in inject_files:
        count = parse_file(filename, "inject")
        print(f"[INFO] Loaded {count} injected records from {filename}")

    for sample in data.values():
        for tag in ("inject", "baseline"):
            for run in sample[tag]:
                for layer in run:
                    run[layer].sort(key=lambda item: item[0])

    return data


def metric_series(rows, metric):
    points = [(token, values[metric]) for token, values in rows if metric in values]
    if not points:
        return [], []
    x, y = zip(*points)
    return list(x), list(y)


def sample_layers(sample_data):
    layers = set()
    for tag in ("inject", "baseline"):
        for run in sample_data.get(tag, []):
            layers.update(run.keys())
    return sorted(layers)


def plot_sample_per_layer(sample_id, sample_data, output_dir, metrics, layers_filter=None):
    sample_dir = os.path.join(output_dir, f"sample_{sample_id}")
    os.makedirs(sample_dir, exist_ok=True)

    layers = sample_layers(sample_data)
    if layers_filter is not None:
        layers = [l for l in layers if l in layers_filter]
    if not layers:
        print(f"[WARN] Sample {sample_id} has no layer data, skipping")
        return []

    cols = min(3, len(metrics))
    rows = math.ceil(len(metrics) / cols)
    saved = []

    for layer in layers:
        fig, axes = plt.subplots(rows, cols, figsize=(12.0, 4.8), squeeze=False)
        axes = axes.ravel()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            baseline_plotted = False
            for run in sample_data.get("baseline", []):
                x, y = metric_series(run.get(layer, []), metric)
                if not x:
                    continue
                ax.plot(x, y, color=COLORS["baseline"], linewidth=1.5, alpha=0.95,
                        label="Fault-free" if not baseline_plotted else None)
                baseline_plotted = True

            injected_plotted = False
            for run in sample_data.get("inject", []):
                x, y = metric_series(run.get(layer, []), metric)
                if not x:
                    continue
                ax.plot(x, y, color=COLORS["inject"], linewidth=1.5, alpha=0.95,
                        label="Faulty" if not injected_plotted else None)
                injected_plotted = True
                injected_plotted = True

            ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=12, fontweight="bold",
                         color=COLORS["text"], pad=6)
            ax.grid(color=COLORS["grid"], linestyle="-", linewidth=0.5, alpha=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=9, pad=2, width=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#AAAAAA")
            ax.spines["left"].set_linewidth(0.8)
            ax.spines["bottom"].set_color("#AAAAAA")
            ax.spines["bottom"].set_linewidth(0.8)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("medium")

        for idx in range(len(metrics), len(axes)):
            axes[idx].set_visible(False)

        for idx in range(len(metrics)):
            handles, labels = axes[idx].get_legend_handles_labels()
            if handles:
                axes[idx].legend(handles, labels, loc="best",
                                 frameon=True, framealpha=0.85, fontsize=8,
                                 edgecolor="#DDDDDD", borderpad=0.3, prop={"weight": "medium"})

        fig.tight_layout(pad=1.2, w_pad=1.5, h_pad=1.5)

        output_path = os.path.join(sample_dir, f"layer_{layer}.pdf")
        fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        saved.append(output_path)
        print(f"[OK] {output_path}")

    return saved


def parse_sample_ids(value):
    if not value:
        return []
    result = []
    for item in value:
        for part in item.split(","):
            part = part.strip()
            if part:
                result.append(part)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-layer runtime metric comparison (all metrics on one figure)."
    )
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--inject-file", nargs="*", default=None)
    parser.add_argument("--baseline-file", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--metrics", nargs="*", default=METRICS)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Only plot specific layers")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_matplotlib()

    output_dir = args.output_dir or os.path.join(args.result_dir, "plots_per_layer")
    data = load_metric_records(
        args.result_dir,
        args.inject_file if args.inject_file is not None else INJECT_FILES,
        args.baseline_file if args.baseline_file is not None else BASELINE_FILES,
    )

    all_samples = sorted(data.keys(), key=numeric_sample_key)
    if args.all_samples:
        selected_samples = all_samples
    else:
        requested_samples = parse_sample_ids(args.samples)
        selected_samples = requested_samples or all_samples[:args.limit]

    print(f"[INFO] Found {len(all_samples)} samples")
    print(f"[INFO] Plotting samples: {selected_samples}")

    for sample_id in selected_samples:
        if sample_id not in data:
            print(f"[WARN] Sample {sample_id} not found, skipping")
            continue
        plot_sample_per_layer(sample_id, data[sample_id], output_dir, args.metrics, args.layers)


if __name__ == "__main__":
    main()

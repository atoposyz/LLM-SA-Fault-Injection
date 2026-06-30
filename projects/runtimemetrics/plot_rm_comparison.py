"""
Plot runtime metric traces for fault-injected and baseline runs.

For each selected sample, the script writes one figure per metric. Each figure
contains one subplot per recorded layer and overlays fault-injected vs baseline
metric values over generated token positions.

Usage:
  uv run python projects/runtimemetrics/plot_rm_comparison.py
  uv run python projects/runtimemetrics/plot_rm_comparison.py --samples 0 1 2
  uv run python projects/runtimemetrics/plot_rm_comparison.py --all-samples
"""
import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULT_DIR = os.path.join(SCRIPT_DIR, "result", "rm_test")

INJECT_FILES = [
    "runtime_metrics_record_kernel-v_layer-35_reg-input_df-WS_cfg-input_bitflip_13_pe_random_intv50_detL0_3_6_9_12_15_18_21_24_27_30_33_affect0_inject.jsonl",
]
BASELINE_FILES = [
    "runtime_metrics_record_kernel-v_layer-35_reg-input_df-WS_cfg-input_bitflip_13_pe_random_intv50_detL0_3_6_9_12_15_18_21_24_27_30_33_affect0_noinject.jsonl",
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
    "inject": "#D95F02",
    "baseline": "#1B9E77",
    "grid": "#D8D2C8",
    "text": "#222222",
}


def configure_matplotlib():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 9,
        "axes.labelsize": 10,
        "axes.edgecolor": "#7A746D",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
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


def plot_sample(sample_id, sample_data, output_dir, metrics):
    sample_dir = os.path.join(output_dir, f"sample_{sample_id}")
    os.makedirs(sample_dir, exist_ok=True)

    layers = sample_layers(sample_data)
    if not layers:
        print(f"[WARN] Sample {sample_id} has no layer data, skipping")
        return []

    cols = 4
    rows = math.ceil(len(layers) / cols)
    saved = []

    for metric in metrics:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.0), squeeze=False)
        axes = axes.ravel()

        for idx, layer in enumerate(layers):
            ax = axes[idx]

            baseline_plotted = False
            for run in sample_data.get("baseline", []):
                x, y = metric_series(run.get(layer, []), metric)
                if not x:
                    continue
                ax.plot(
                    x,
                    y,
                    color=COLORS["baseline"],
                    linewidth=1.05,
                    alpha=0.9,
                    label="Baseline" if not baseline_plotted else None,
                )
                baseline_plotted = True

            injected_plotted = False
            for run in sample_data.get("inject", []):
                x, y = metric_series(run.get(layer, []), metric)
                if not x:
                    continue
                ax.plot(
                    x,
                    y,
                    color=COLORS["inject"],
                    linewidth=1.05,
                    alpha=0.9,
                    label="Fault-injected" if not injected_plotted else None,
                )
                injected_plotted = True

            ax.set_title(f"Layer {layer}", fontweight="bold", color=COLORS["text"])
            ax.grid(axis="y", color=COLORS["grid"], linestyle="--", linewidth=0.6, alpha=0.7)
            ax.tick_params(labelsize=7)
            if idx == 0:
                ax.legend(loc="best", frameon=True, framealpha=0.92)

        for idx in range(len(layers), len(axes)):
            axes[idx].set_visible(False)

        metric_label = METRIC_LABELS.get(metric, metric)
        fig.suptitle(f"Sample {sample_id}: {metric_label}", fontsize=13, fontweight="bold", y=0.985)
        fig.supxlabel("Generated Token Position", fontsize=10)
        fig.supylabel(metric_label, fontsize=10)
        fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])

        output_path = os.path.join(sample_dir, f"{metric}.png")
        fig.savefig(output_path, dpi=180)
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
        description="Plot runtime metric comparison traces for baseline vs fault-injected runs."
    )
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR, help="Directory containing JSONL records")
    parser.add_argument("--inject-file", nargs="*", default=None, help="Inject JSONL filename(s), overrides built-in INJECT_FILES")
    parser.add_argument("--baseline-file", nargs="*", default=None, help="Baseline JSONL filename(s), overrides built-in BASELINE_FILES")
    parser.add_argument("--output-dir", default=None, help="Directory for plots; defaults to RESULT_DIR/plots")
    parser.add_argument("--samples", nargs="*", default=None, help="Sample ids to plot, e.g. --samples 0 1 2")
    parser.add_argument("--all-samples", action="store_true", help="Plot every sample found")
    parser.add_argument("--limit", type=int, default=3, help="Number of samples to plot when --samples is omitted")
    parser.add_argument("--metrics", nargs="*", default=METRICS, help="Metrics to plot")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_matplotlib()

    output_dir = args.output_dir or os.path.join(args.result_dir, "plots")
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
        plot_sample(sample_id, data[sample_id], output_dir, args.metrics)


if __name__ == "__main__":
    main()

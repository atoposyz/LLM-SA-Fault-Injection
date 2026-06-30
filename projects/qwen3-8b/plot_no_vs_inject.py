"""
Compare no-injection and injection runs by token length and inference time.

The default plot is paper-oriented: bars show sample means and error bars show
bootstrap 95% confidence intervals. Individual samples can be overlaid with
--show-points for debugging or appendix figures.

Usage:
  uv run python projects/qwen3-8b/plot_no_vs_inject.py
  uv run python projects/qwen3-8b/plot_no_vs_inject.py --n 200 --output /tmp/fault_injection_generation_latency.png
  uv run python projects/qwen3-8b/plot_no_vs_inject.py --show-points
"""
import argparse
import json
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")

DEFAULT_NO_INJECT = os.path.join(
    RESULT_DIR,
    "no_simple_kernel-v_layer-35_reg-input_df-WS_cfg-input_bitflip_13_pe_random_affect0.jsonl",
)
DEFAULT_INJECT = os.path.join(
    RESULT_DIR,
    "simple_kernel-v_layer-35_reg-input_df-WS_cfg-input_bitflip_13_pe_random_affect0.jsonl",
)
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "fault_injection_generation_latency.png")

COLORS = {
    "token": "#4477AA",
    "time": "#CC6677",
    "delta": "#882255",
    "grid": "#E0E0E0",
    "text": "#2D2D2D",
    "axis": "#8C8C8C",
    "background": "#FFFFFF",
}


@dataclass(frozen=True)
class RunData:
    label: str
    token_lengths: np.ndarray
    inference_times: np.ndarray


@dataclass(frozen=True)
class MeanCI:
    mean: float
    low: float
    high: float

    @property
    def lower_err(self):
        return self.mean - self.low

    @property
    def upper_err(self):
        return self.high - self.mean


def configure_matplotlib():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.edgecolor": COLORS["axis"],
        "axes.linewidth": 0.9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def load_first_n(path, label, n=100):
    token_lengths = []
    inference_times = []

    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            token_lengths.append(row["token_length"])
            inference_times.append(row["inference_time_s"])

    if not token_lengths:
        raise ValueError(f"No records loaded from {path}")

    return RunData(
        label=label,
        token_lengths=np.asarray(token_lengths, dtype=float),
        inference_times=np.asarray(inference_times, dtype=float),
    )


def bootstrap_mean_ci(values, rng, confidence=95.0, resamples=10000):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) == 1:
        return MeanCI(mean, mean, mean)

    samples = rng.choice(values, size=(resamples, len(values)), replace=True)
    means = np.mean(samples, axis=1)
    alpha = (100.0 - confidence) / 2.0
    low, high = np.percentile(means, [alpha, 100.0 - alpha])
    return MeanCI(mean, float(low), float(high))


def mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def percent_delta(baseline, current):
    if abs(baseline) < 1e-12:
        return 0.0
    return current / baseline * 100.0 - 100.0


def add_sample_points(ax, x_pos, values, color, rng, jitter=0.055):
    x_jitter = rng.uniform(-jitter, jitter, size=len(values))
    ax.scatter(
        np.full(len(values), x_pos) + x_jitter,
        values,
        s=13,
        facecolors="white",
        edgecolors=color,
        linewidth=0.5,
        alpha=0.32,
        zorder=5,
    )


def add_value_labels(ax, bars, stats, formatter, color):
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.025
    for bar, stat in zip(bars, stats):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            stat.high + offset,
            formatter(stat.mean),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=color,
            zorder=8,
        )


def add_delta_label(ax, x_pos, base_mean, current_stat, color):
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.075
    delta = percent_delta(base_mean, current_stat.mean)
    sign = "+" if delta >= 0 else ""
    ax.text(
        x_pos,
        current_stat.high + offset,
        f"{sign}{delta:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#C97A7A",
            "linewidth": 0.8,
            "alpha": 0.96,
        },
        zorder=9,
    )


def axis_upper(stats, pad_ratio=0.25):
    high = max(stat.high for stat in stats)
    return high * (1.0 + pad_ratio) if high > 0 else 1.0


def ci_yerr(stats):
    return np.array([
        [stat.lower_err for stat in stats],
        [stat.upper_err for stat in stats],
    ])


def output_paths(output_path):
    base, ext = os.path.splitext(output_path)
    if ext.lower() == ".pdf":
        return f"{base}.png", output_path
    return output_path, f"{base}.pdf"


def plot_comparison(no_data, inject_data, output_path, confidence=95.0, resamples=10000, show_points=False):
    configure_matplotlib()

    labels = [no_data.label, inject_data.label]
    x = np.arange(len(labels), dtype=float)
    bar_width = 0.30
    point_rng = np.random.default_rng(42)
    ci_rng = np.random.default_rng(2026)

    token_stats = [
        bootstrap_mean_ci(no_data.token_lengths, ci_rng, confidence, resamples),
        bootstrap_mean_ci(inject_data.token_lengths, ci_rng, confidence, resamples),
    ]
    time_stats = [
        bootstrap_mean_ci(no_data.inference_times, ci_rng, confidence, resamples),
        bootstrap_mean_ci(inject_data.inference_times, ci_rng, confidence, resamples),
    ]
    token_means = [stat.mean for stat in token_stats]
    time_means = [stat.mean for stat in time_stats]

    fig, ax_token = plt.subplots(figsize=(9.2, 5.6))
    ax_time = ax_token.twinx()

    fig.patch.set_facecolor("white")
    ax_token.set_facecolor(COLORS["background"])

    token_x = x - bar_width / 2
    time_x = x + bar_width / 2
    error_style = {"linewidth": 1.4, "ecolor": "#4B4742", "capthick": 1.4}

    token_bars = ax_token.bar(
        token_x,
        token_means,
        bar_width,
        yerr=ci_yerr(token_stats),
        capsize=5,
        error_kw=error_style,
        color=COLORS["token"],
        edgecolor="white",
        linewidth=1.0,
        alpha=0.90,
        label="Generated tokens",
        zorder=3,
    )
    time_bars = ax_time.bar(
        time_x,
        time_means,
        bar_width,
        yerr=ci_yerr(time_stats),
        capsize=5,
        error_kw=error_style,
        color=COLORS["time"],
        edgecolor="white",
        linewidth=1.0,
        alpha=0.88,
        label="Inference latency",
        zorder=3,
    )

    if show_points:
        for x_pos, values in zip(token_x, [no_data.token_lengths, inject_data.token_lengths]):
            add_sample_points(ax_token, x_pos, values, COLORS["token"], point_rng)
        for x_pos, values in zip(time_x, [no_data.inference_times, inject_data.inference_times]):
            add_sample_points(ax_time, x_pos, values, COLORS["time"], point_rng)

    ax_token.set_ylim(0, axis_upper(token_stats))
    ax_time.set_ylim(0, axis_upper(time_stats))

    add_value_labels(ax_token, token_bars, token_stats, lambda v: f"{v:,.0f}", COLORS["token"])
    add_value_labels(ax_time, time_bars, time_stats, lambda v: f"{v:.1f}", COLORS["time"])
    add_delta_label(ax_token, token_x[1], token_stats[0].mean, token_stats[1], COLORS["delta"])
    add_delta_label(ax_time, time_x[1], time_stats[0].mean, time_stats[1], COLORS["delta"])

    ax_token.set_ylabel("Generated Tokens", color=COLORS["token"])
    ax_time.set_ylabel("Inference Latency (s)", color=COLORS["time"])
    ax_token.tick_params(axis="y", colors=COLORS["token"])
    ax_time.tick_params(axis="y", colors=COLORS["time"])
    ax_token.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}")
    )

    ax_token.set_xticks(x)
    ax_token.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax_token.set_xlim(-0.55, 1.55)
    ax_token.grid(axis="y", color=COLORS["grid"], linewidth=0.8, linestyle="--", alpha=0.65)
    ax_token.set_axisbelow(True)

    for spine in ["top"]:
        ax_token.spines[spine].set_visible(False)
        ax_time.spines[spine].set_visible(False)

    handles1, labels1 = ax_token.get_legend_handles_labels()
    handles2, labels2 = ax_time.get_legend_handles_labels()
    ax_token.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#E7E5E4",
        framealpha=0.96,
    )

    fig.suptitle(
        "Impact of Fault Injection on Generation Length and Latency",
        fontsize=15,
        fontweight="bold",
        color=COLORS["text"],
        y=0.965,
    )
    if show_points:
        ax_token.set_title(
            "Hollow markers show individual samples",
            fontsize=9,
            fontweight="normal",
            color="#57534E",
            loc="left",
            pad=4,
        )

    png_path, pdf_path = output_paths(output_path)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def print_summary(no_data, inject_data, confidence=95.0, resamples=10000):
    rng = np.random.default_rng(2026)
    rows = [
        (
            "Avg Token Length",
            bootstrap_mean_ci(no_data.token_lengths, rng, confidence, resamples),
            bootstrap_mean_ci(inject_data.token_lengths, rng, confidence, resamples),
            "{:.1f}",
        ),
        (
            "Avg Time (s)",
            bootstrap_mean_ci(no_data.inference_times, rng, confidence, resamples),
            bootstrap_mean_ci(inject_data.inference_times, rng, confidence, resamples),
            "{:.2f}",
        ),
    ]

    print(f"{'Metric':>20} {no_data.label:>12} {inject_data.label:>12} {'Delta':>12} {'Delta %':>10}")
    print("-" * 70)
    for name, base, current, fmt in rows:
        delta = current.mean - base.mean
        delta_pct = percent_delta(base.mean, current.mean)
        print(
            f"{name:>20} "
            f"{fmt.format(base.mean):>12} "
            f"{fmt.format(current.mean):>12} "
            f"{fmt.format(delta):>12} "
            f"{delta_pct:+9.1f}%"
        )
        print(
            f"{'95% CI':>20} "
            f"[{fmt.format(base.low)}, {fmt.format(base.high)}]".rjust(12) + " "
            f"[{fmt.format(current.low)}, {fmt.format(current.high)}]".rjust(12)
        )

    for name, base_values, current_values, fmt in [
        ("Std Token Length", no_data.token_lengths, inject_data.token_lengths, "{:.1f}"),
        ("Std Time (s)", no_data.inference_times, inject_data.inference_times, "{:.2f}"),
    ]:
        base_std, current_std = mean_std(base_values)[1], mean_std(current_values)[1]
        print(
            f"{name:>20} "
            f"{fmt.format(base_std):>12} "
            f"{fmt.format(current_std):>12} "
            f"{fmt.format(current_std - base_std):>12}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot token length and inference time for no-injection vs injection runs."
    )
    parser.add_argument("--no-inject", default=DEFAULT_NO_INJECT, help="No-injection JSONL path")
    parser.add_argument("--inject", default=DEFAULT_INJECT, help="Injection JSONL path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image path")
    parser.add_argument("-n", "--n", type=int, default=100, help="Number of records to load")
    parser.add_argument("--ci", type=float, default=95.0, help="Bootstrap confidence level")
    parser.add_argument("--resamples", type=int, default=10000, help="Bootstrap resample count")
    parser.add_argument("--show-points", action="store_true", help="Overlay individual samples")
    return parser.parse_args()


def main():
    args = parse_args()
    no_data = load_first_n(args.no_inject, "Baseline", args.n)
    inject_data = load_first_n(args.inject, "Fault-injected", args.n)

    print_summary(no_data, inject_data, args.ci, args.resamples)
    saved_paths = plot_comparison(
        no_data,
        inject_data,
        args.output,
        confidence=args.ci,
        resamples=args.resamples,
        show_points=args.show_points,
    )
    print("\nSaved:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

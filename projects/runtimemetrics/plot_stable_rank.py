"""
Plot stable rank vs token position from SAStableRankRecord.py output.
Each line = one detect layer.
Default: average across all samples.  --per-sample: one plot per sample.
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CMAP = plt.colormaps["tab10"]


def plot_one(ax, layers, token_positions, sr_by_layer, sample_id=""):
    for i, layer in enumerate(layers):
        vals = [sr_by_layer[layer].get(tp, np.nan) for tp in token_positions]
        # Drop NaN to prevent line breaks
        tps = [tp for tp, v in zip(token_positions, vals) if not np.isnan(v)]
        vals = [v for v in vals if not np.isnan(v)]
        ax.plot(tps, vals, color=CMAP(i % 10), linewidth=1.2,
                label=f"Layer {layer}")
    ax.set_xlabel("Token position", fontsize=13)
    ax.set_ylabel("Stable Rank", fontsize=13)
    title = f"Stable Rank vs Token Position{(' — sample ' + sample_id) if sample_id else ''}"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="JSONL file from SAStableRankRecord.py")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to read (0=all)")
    parser.add_argument("--output", "-o", type=str, default="",
                        help="Output image path (aggregate mode)")
    parser.add_argument("--per-sample", action="store_true",
                        help="Save one plot per sample in a subdirectory")
    parser.add_argument("--out-dir", type=str, default="",
                        help="Per-sample output directory (default: input_dir/per_sample/)")
    args = parser.parse_args()

    if args.per_sample:
        # Read all data into memory (need per-sample grouping)
        samples = []
        with open(args.input) as f:
            for i, line in enumerate(f):
                if args.max_samples and i >= args.max_samples:
                    break
                rec = json.loads(line)
                # Build per-layer dict for this sample
                sr_by_layer = defaultdict(dict)
                for pt in rec["stable_rank_values"]:
                    sr_by_layer[pt["layer"]][pt["token_end"]] = pt["stable_rank"]
                samples.append((rec["sample_id"], sr_by_layer))

        layers = sorted(set(k for s in samples for k in s[1]))
        token_positions = sorted(set(tp for s in samples for d in s[1].values() for tp in d))

        out_dir = args.out_dir or os.path.join(os.path.dirname(args.input), "per_sample")
        os.makedirs(out_dir, exist_ok=True)

        for sample_id, sr_by_layer in samples:
            # token positions for THIS sample only — avoids NaN gaps from shorter samples
            tp_sample = sorted(set(tp for d in sr_by_layer.values() for tp in d))
            fig, ax = plt.subplots(figsize=(14, 6))
            plot_one(ax, layers, tp_sample, sr_by_layer, sample_id=sample_id)
            fig.tight_layout()
            out_path = os.path.join(out_dir, f"sr_sample_{sample_id}.png")
            fig.savefig(out_path, dpi=150, facecolor="white")
            plt.close(fig)

        print(f"Saved {len(samples)} plots to {out_dir}/")

    else:
        # Aggregate mode: mean across samples
        accum = defaultdict(list)
        n_samples = 0
        with open(args.input) as f:
            for line in f:
                if args.max_samples and n_samples >= args.max_samples:
                    break
                rec = json.loads(line)
                for pt in rec["stable_rank_values"]:
                    accum[(pt["layer"], pt["token_end"])].append(pt["stable_rank"])
                n_samples += 1
                if n_samples % 10 == 0:
                    print(f"\rRead {n_samples} samples...", end="", flush=True)
        print(f"\rRead {n_samples} samples. Aggregating...")

        layers = sorted(set(k[0] for k in accum))
        token_positions = sorted(set(k[1] for k in accum))

        sr_by_layer = defaultdict(dict)
        for layer in layers:
            for tp in token_positions:
                vals = accum.get((layer, tp), [])
                sr_by_layer[layer][tp] = np.mean(vals) if vals else np.nan

        fig, ax = plt.subplots(figsize=(14, 6))
        plot_one(ax, layers, token_positions, sr_by_layer)
        fig.tight_layout()

        out_path = args.output or os.path.splitext(args.input)[0] + "_sr_plot.png"
        fig.savefig(out_path, dpi=150, facecolor="white")
        print(f"Saved: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()

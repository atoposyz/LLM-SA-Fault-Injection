import json
import matplotlib.pyplot as plt
import numpy as np

INJECT_PATH = "projects/qwen3-8b/result/runtime_metrics_inject.jsonl"
NOINJECT_PATH = "projects/qwen3-8b/result/runtime_metrics_noinject.jsonl"
OUTPUT_DIR = "result/qwen3-8b-metrics/"

METRIC_KEYS = [
    "stable_rank", "svd_entropy", "participation_ratio",
    "top5_energy_ratio", "frobenius_norm_sq", "spectral_norm_sq",
    "numerical_rank", "nuclear_norm", "normalized_nuclear_rank",
    "nuclear_rank",
]

LAYER_ORDER = ["layers.0", "layers.1", "layers.2", "layers.33", "layers.34", "layers.35"]
COLORS = {"inject": "#e74c3c", "noinject": "#3498db"}


def load_data(path, sample_id=0):
    data = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["sample_id"] != sample_id:
                continue
            layer = rec["layer_name"]
            if layer not in data:
                data[layer] = {"module_step": [], "metrics": {k: [] for k in METRIC_KEYS}}
            data[layer]["module_step"].append(rec["module_step"])
            for k in METRIC_KEYS:
                if k in rec:
                    data[layer]["metrics"][k].append(rec[k])
    return data


def plot_metric(metric_name, inject_data, noinject_data):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Qwen3-8B — {metric_name}  (sample 0, inject vs noinject)",
                 fontsize=14, fontweight="bold")

    for idx, layer in enumerate(LAYER_ORDER):
        ax = axes[idx // 3][idx % 3]

        if layer in inject_data:
            d = inject_data[layer]
            ax.plot(d["module_step"], d["metrics"][metric_name],
                    color=COLORS["inject"], linewidth=1.2, label="inject", alpha=0.85)

        if layer in noinject_data:
            d = noinject_data[layer]
            ax.plot(d["module_step"], d["metrics"][metric_name],
                    color=COLORS["noinject"], linewidth=1.2, label="noinject", alpha=0.85)

        ax.set_title(layer, fontsize=11)
        ax.set_xlabel("module_step")
        ax.set_ylabel(metric_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}{metric_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading inject data...")
    inject = load_data(INJECT_PATH, sample_id=0)
    print(f"  Layers: {list(inject.keys())}, steps per layer: {[len(inject[l]['module_step']) for l in LAYER_ORDER]}")

    print("Loading noinject data...")
    noinject = load_data(NOINJECT_PATH, sample_id=0)
    print(f"  Layers: {list(noinject.keys())}, steps per layer: {[len(noinject[l]['module_step']) for l in LAYER_ORDER]}")

    print(f"\nPlotting {len(METRIC_KEYS)} metrics...")
    for mk in METRIC_KEYS:
        plot_metric(mk, inject, noinject)

    print("\nDone.")


if __name__ == "__main__":
    main()

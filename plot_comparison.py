import json
import os
import matplotlib.pyplot as plt
import numpy as np

NOINJECT = "/tmp/test_v29_noinject/runtime_metrics_noinject.jsonl"
SAINJECT = "/tmp/test_v29_sainject/runtime_metrics_inject.jsonl"
OUTDIR = "/workplace/home/mayongzhe/faultinject/result/"

METRICS = [
    ("stable_rank", "Stable Rank"),
    ("svd_entropy", "SVD Entropy"),
    ("participation_ratio", "Participation Ratio"),
    ("top5_energy_ratio", "Top-5 Energy Ratio"),
    ("frobenius_norm_sq", "Frobenius Norm²"),
    ("spectral_norm_sq", "Spectral Norm²"),
    ("numerical_rank", "Numerical Rank"),
    ("normalized_nuclear_rank", "Normalized Nuclear Rank"),
]

def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def group_by_layer(records):
    by_layer = {}
    for r in records:
        layer = r["layer_name"]
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append((r["module_step"], r["stage"], r))
    return by_layer

noinject_all = load_records(NOINJECT)
sainject_all = load_records(SAINJECT)

noinject_by_layer = group_by_layer(noinject_all)
sainject_by_layer = group_by_layer(sainject_all)

layer_order = sorted(noinject_by_layer.keys(), key=lambda x: int(x.split(".")[1]))

# --- Per-layer plots (2 rows x 4 cols per layer) ---
for li, layer_name in enumerate(layer_order):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle(f"{layer_name} — NoInject (blue) vs SAInject v-proj L35 bit29 (red)", fontsize=11, fontweight="bold")

    ni_records = noinject_by_layer[layer_name]
    si_records = sainject_by_layer[layer_name]

    ni_data = [(s, r) for s, st, r in ni_records if st == "output"]
    si_data = [(s, r) for s, st, r in si_records if st == "output"]

    ni_steps = np.array([s for s, _ in ni_data])
    si_steps = np.array([s for s, _ in si_data])

    for mi, (key, label) in enumerate(METRICS):
        ax = axes[mi // 4][mi % 4]
        ni_vals = np.array([r[key] for _, r in ni_data], dtype=float)
        si_vals = np.array([r[key] for _, r in si_data], dtype=float)

        ax.plot(ni_steps, ni_vals, 'b-', alpha=0.6, linewidth=0.6, label='NoInject')
        ax.plot(si_steps, si_vals, 'r-', alpha=0.6, linewidth=0.6, label='SAInject')
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if mi == 0:
            ax.legend(fontsize=7)

        if key in ("frobenius_norm_sq", "spectral_norm_sq"):
            ax.set_yscale("log")

    plt.tight_layout()
    out_path = os.path.join(OUTDIR, f"comparison_v29_{layer_name.replace('.', '_')}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

# --- Numerical summary ---
print("\n" + "="*80)
print("NUMERICAL SUMMARY: Mean relative difference |SAInject - NoInject| / |NoInject|")
print("="*80)

for li, layer_name in enumerate(layer_order):
    ni_records = noinject_by_layer[layer_name]
    si_records = sainject_by_layer[layer_name]

    ni_data = [(s, r) for s, st, r in ni_records if st == "output"]
    si_data = [(s, r) for s, st, r in si_records if st == "output"]

    ni_map = {s: r for s, r in ni_data}
    si_map = {s: r for s, r in si_data}
    common_steps = sorted(set(ni_map.keys()) & set(si_map.keys()))

    print(f"\n--- {layer_name} ({len(common_steps)} shared steps) ---")
    for key, label in METRICS:
        diffs = []
        for s in common_steps:
            ni_val = ni_map[s][key]
            si_val = si_map[s][key]
            if abs(ni_val) > 1e-9:
                diffs.append(abs(si_val - ni_val) / abs(ni_val))
        if diffs:
            mean_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            print(f"  {label:25s}: mean_diff={mean_diff:.4f}, max_diff={max_diff:.4f}")

print(f"\nPlots saved to: {OUTDIR}")

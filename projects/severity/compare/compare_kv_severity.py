"""
Quick comparison: K vs V projection weight severity (stuck-at-0/1, FP32).
"""
import json, os, sys
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tool/src"))
from tool.bit_severity import build_severity_lookup_table, normalize_table_scores

hf_logging.set_verbosity_error()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL = "boltuix/bert-emotion"
OUT = "projects/bert/config/severity_kv_comparison.png"

def build_table(tensors, label):
    t = build_severity_lookup_table(tensors, source_name=label, precision="fp32",
                                      transform="log1p", max_elements=2_000_000)
    t = normalize_table_scores(t, pre_log1p=True)
    return t

print("Loading model...")
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32)
model.eval()

# Collect K and V weights
k_weights, v_weights = [], []
for name, mod in model.named_modules():
    if not isinstance(mod, torch.nn.Linear):
        continue
    if ".key" in name:
        k_weights.append(mod.weight.detach().cpu())
    elif ".value" in name:
        v_weights.append(mod.weight.detach().cpu())

print(f"K layers: {len(k_weights)} ({sum(w.numel() for w in k_weights):,} elems)")
print(f"V layers: {len(v_weights)} ({sum(w.numel() for w in v_weights):,} elems)")

t_k = build_table(k_weights, "K_proj")
t_v = build_table(v_weights, "V_proj")

# ---- Plot ----
def extract(tbl, key):
    xs, ys = [], []
    for e in tbl["table"]:
        xs.append(e["bit"])
        ys.append(e.get(key, 0))
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.subplots_adjust(hspace=0.15)

for ax, key, title in [
    (ax0, "sa0_unconditional_norm", "Stuck-at-0"),
    (ax1, "sa1_unconditional_norm", "Stuck-at-1"),
]:
    for tbl, label, color, ls in [
        (t_k, "K projection", "#2196F3", "-"),
        (t_v, "V projection", "#FF5722", "--"),
    ]:
        x, y = extract(tbl, key)
        ax.plot(x, y, linestyle=ls, color=color, label=label, linewidth=2.0,
                marker='o', markersize=4, markevery=4, alpha=0.85)

    # IEEE 754 regions
    ax.axvspan(-0.5, 22.5, alpha=0.06, color='green', zorder=0)
    ax.axvspan(22.5, 30.5, alpha=0.08, color='orange', zorder=0)
    ax.axvspan(30.5, 31.5, alpha=0.12, color='red', zorder=0)
    ax.text(11, 1.07, "Mantissa (23b)", ha='center', va='bottom', fontsize=9, color='green', fontweight='bold', transform=ax.get_xaxis_transform())
    ax.text(26.5, 1.07, "Exponent (8b)", ha='center', va='bottom', fontsize=9, color='orange', fontweight='bold', transform=ax.get_xaxis_transform())
    ax.text(31, 1.07, "Sign (1b)", ha='center', va='bottom', fontsize=9, color='red', fontweight='bold', transform=ax.get_xaxis_transform())

    ax.set_ylabel("Normalised Severity", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left', color='#333333')
    ax.set_ylim(-0.03, 1.10)
    ax.set_yticks(np.arange(0, 1.1, 0.25))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.92)

ax1.set_xlabel("Bit Position (IEEE 754 FP32)", fontsize=11)
ax1.set_xticks(range(0, 32))
ax1.set_xticklabels([str(i) for i in range(32)], fontsize=7)
ax1.set_xlim(-0.8, 31.8)

fig.suptitle("K vs V Projection Weight — Stuck-at Fault Severity (FP32)\nboltuix/bert-emotion  |  transform=log1p",
             fontsize=13, fontweight='bold', y=0.985)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT}")
plt.close()

# Also print key differences
print("\n--- K vs V: largest absolute differences in sa0_unconditional_norm ---")
k_data = {e["bit"]: e for e in t_k["table"]}
v_data = {e["bit"]: e for e in t_v["table"]}
diffs = []
for b in range(32):
    d = abs(k_data[b]["sa0_unconditional_norm"] - v_data[b]["sa0_unconditional_norm"])
    diffs.append((b, d, k_data[b]["sa0_unconditional_norm"], v_data[b]["sa0_unconditional_norm"]))
diffs.sort(key=lambda x: -x[1])
print(f"{'bit':>4s} {'field':>9s} {'K_sa0_norm':>12s} {'V_sa0_norm':>12s} {'|diff|':>10s}")
for b, d, kv, vv in diffs[:8]:
    print(f"{b:4d} {k_data[b]['field']:>9s} {kv:12.6f} {vv:12.6f} {d:10.6f}")

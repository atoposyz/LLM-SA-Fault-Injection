"""
K vs V severity: separate plots for each operator, each showing weight / input / psum.
"""
import json, os, sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tool/src"))
from tool.bit_severity import build_severity_lookup_table, normalize_table_scores

hf_logging.set_verbosity_error()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL = "boltuix/bert-emotion"
MAX_ELEMS = 2_000_000

# ---------------------------------------------------------------------------
# Calibration texts
# ---------------------------------------------------------------------------
_CAL_TEXTS = []
try:
    import datasets
    ds = datasets.load_dataset("boltuix/emotions-dataset", split="train", trust_remote_code=True)
    from random import Random
    rng = Random(42)
    for s in rng.sample(list(ds), min(16, len(ds))):
        _CAL_TEXTS.append(s["Sentence"])
    print(f"[INFO] Loaded {len(_CAL_TEXTS)} calibration samples from emotions-dataset")
except Exception:
    _CAL_TEXTS = ["I feel happy and joyful today! " * 20] * 16
    print("[INFO] Using fallback calibration texts")


# ---------------------------------------------------------------------------
# Collect weights for a specific operator
# ---------------------------------------------------------------------------
def collect_weights(model, op_suffix):
    """op_suffix: e.g. '.key' or '.value'"""
    tensors = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and op_suffix in name:
            tensors.append(mod.weight.detach().cpu())
    return tensors


# ---------------------------------------------------------------------------
# Collect activations for a specific operator
# ---------------------------------------------------------------------------
def collect_activations(model, tokenizer, texts, op_suffix, kind, device):
    """
    kind: 'input' or 'output'
    Returns list of flattened activation tensors.
    """
    acts = []
    total_elems = 0
    hooks = []

    target = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and op_suffix in name:
            target.append((name, mod))

    n = max(len(target), 1)
    limit = MAX_ELEMS
    budget = max(int(limit / n), 10000)
    collect_input = (kind == "input")

    def _hook_factory():
        def hook_fn(module, input_tup, output_tup):
            nonlocal total_elems
            if total_elems >= limit:
                return
            if collect_input:
                t = input_tup[0].detach().cpu().float().flatten()
            else:
                t = output_tup if isinstance(output_tup, torch.Tensor) else output_tup[0]
                t = t.detach().cpu().float().flatten()
            if t.numel() > budget:
                idx = torch.randperm(t.numel())[:budget]
                t = t[idx]
            acts.append(t)
            total_elems += t.numel()
        return hook_fn

    for _, mod in target:
        hooks.append(mod.register_forward_hook(_hook_factory()))

    model.eval()
    with torch.no_grad():
        for text in texts:
            if total_elems >= limit:
                break
            try:
                inp = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
                if inp.input_ids.numel() == 0:
                    continue
                _ = model(**inp)
            except Exception as e:
                # ignore
                continue

    for h in hooks:
        h.remove()
    return acts


# ---------------------------------------------------------------------------
# Build table from tensors
# ---------------------------------------------------------------------------
def build_table(tensors, label):
    if not tensors or all(t.numel() == 0 for t in tensors):
        return None
    t = build_severity_lookup_table(tensors, source_name=label, precision="fp32",
                                     transform="log1p", max_elements=MAX_ELEMS)
    t = normalize_table_scores(t, pre_log1p=True)
    return t


# ---------------------------------------------------------------------------
# Plot one operator (weight + input + psum)
# ---------------------------------------------------------------------------
COLORS = {"weight": "#2196F3", "activation_input": "#FF5722", "psum_output": "#4CAF50"}
STYLES = {"weight": "--", "activation_input": "-", "psum_output": "-."}
LABELS = {"weight": "Weight", "activation_input": "Activation (input)", "psum_output": "Psum (output)"}

def plot_one_operator(tables, op_name, out_path):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    for ax, key, title in [
        (ax0, "sa0_unconditional_norm", "Stuck-at-0"),
        (ax1, "sa1_unconditional_norm", "Stuck-at-1"),
    ]:
        for src, tbl in tables.items():
            if tbl is None:
                continue
            xs, ys = [], []
            for e in tbl["table"]:
                xs.append(e["bit"])
                ys.append(e.get(key, 0))
            order = np.argsort(xs)
            xs, ys = np.array(xs)[order], np.array(ys)[order]
            ax.plot(xs, ys, linestyle=STYLES[src], color=COLORS[src],
                    label=LABELS[src], linewidth=2.0, marker='o',
                    markersize=4, markevery=4, alpha=0.85)

        # IEEE 754 field regions
        ax.axvspan(-0.5, 22.5, alpha=0.06, color='green', zorder=0)
        ax.axvspan(22.5, 30.5, alpha=0.08, color='orange', zorder=0)
        ax.axvspan(30.5, 31.5, alpha=0.12, color='red', zorder=0)
        ax.text(11, 1.07, "Mantissa (23b)", ha='center', va='bottom', fontsize=9,
                color='green', fontweight='bold', transform=ax.get_xaxis_transform())
        ax.text(26.5, 1.07, "Exponent (8b)", ha='center', va='bottom', fontsize=9,
                color='orange', fontweight='bold', transform=ax.get_xaxis_transform())
        ax.text(31, 1.07, "Sign (1b)", ha='center', va='bottom', fontsize=9,
                color='red', fontweight='bold', transform=ax.get_xaxis_transform())

        ax.set_ylabel("Normalised Severity", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left', color='#333333')
        ax.set_ylim(-0.03, 1.10)
        ax.set_yticks(np.arange(0, 1.1, 0.25))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(fontsize=9, loc='upper left', framealpha=0.92, ncol=3, columnspacing=0.6)

    ax1.set_xlabel("Bit Position (IEEE 754 FP32)", fontsize=11)
    ax1.set_xticks(range(0, 32))
    ax1.set_xticklabels([str(i) for i in range(32)], fontsize=7)
    ax1.set_xlim(-0.8, 31.8)

    fig.suptitle(f"{op_name} — Bit-Level Stuck-at Fault Severity (FP32)\n"
                 "boltuix/bert-emotion  |  transform=log1p  |  weight / input / psum",
                 fontsize=13, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


# ===================================================================
# Main
# ===================================================================
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32, device_map="auto")
model.eval()

for op_suffix, op_label in [(".key", "K Projection"), (".value", "V Projection")]:
    print(f"\n{'='*60}")
    print(f"Building tables for: {op_label}")

    # Weight
    w = collect_weights(model, op_suffix)
    print(f"  Weight tensors: {len(w)} ({sum(t.numel() for t in w):,} elements)")
    tw = build_table(w, f"{op_label}_weight")

    # Input activation
    a_in = collect_activations(model, tokenizer, _CAL_TEXTS, op_suffix, "input", device)
    print(f"  Input activation tensors: {len(a_in)} ({sum(t.numel() for t in a_in):,} elements)")
    ta_in = build_table(a_in, f"{op_label}_input")

    # Output activation (psum)
    a_out = collect_activations(model, tokenizer, _CAL_TEXTS, op_suffix, "output", device)
    print(f"  Output activation tensors: {len(a_out)} ({sum(t.numel() for t in a_out):,} elements)")
    ta_out = build_table(a_out, f"{op_label}_psum")

    tables = {"weight": tw, "activation_input": ta_in, "psum_output": ta_out}
    out_file = f"projects/bert/config/severity_{op_suffix.strip('.')}_fp32.png"
    plot_one_operator(tables, op_label, out_file)

"""
Offline calibration script to build bit severity lookup tables for Qwen3-8B.

Collects weight tensors (and optionally activation tensors) from Qwen/Qwen3-8B,
then builds per-bit severity tables for FP32/BF16.

When --modules all (default), tables are emitted per operator group:
  attention    — Q/K/V/O dense (q_proj/k_proj/v_proj/o_proj)
  intermediate — gate_proj/up_proj dense
  output       — down_proj dense

Usage:
  # Weight-only table, all operator groups
  uv run python projects/qwen3-8b/build_severity_table.py --source weight

  # Activation-only table (input activations)
  uv run python projects/qwen3-8b/build_severity_table.py --source activation --num-samples 8

  # Both weight + activation, BF16 precision
  uv run python projects/qwen3-8b/build_severity_table.py --source both --precision bf16
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tool/src"))
from tool.bit_severity import (
    apply_theoretical_floor,
    build_severity_lookup_table,
    normalize_table_scores,
    save_lookup_table,
    print_lookup_table,
)

hf_logging.set_verbosity_error()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------------------
# Operator groups (Qwen3-8B naming conventions)
# ---------------------------------------------------------------------------

OPERATOR_GROUPS = {
    "attention": {
        "patterns": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ],
        "desc": "Q/K/V/O dense (K=4096, N=4096 / k,v N=1024)",
    },
    "intermediate": {
        "patterns": [
            "mlp.gate_proj",
            "mlp.up_proj",
        ],
        "desc": "gate_proj/up_proj dense (K=4096, N=12288)",
    },
    "output": {
        "patterns": ["mlp.down_proj"],
        "desc": "down_proj dense (K=12288, N=4096)",
    },
}

# ---------------------------------------------------------------------------
# Calibration prompts (fallback when MMLU dataset unavailable)
# ---------------------------------------------------------------------------

_CALIBRATION_TEXTS = [
    "The quick brown fox jumps over the lazy dog. " * 10,
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data. " * 8,
    "Neural networks are composed of layers of interconnected nodes that process information hierarchically. " * 8,
    "Transformer architectures use self-attention mechanisms to capture dependencies between tokens. " * 8,
    "Large language models are trained on vast corpora of text data to predict the next token. " * 8,
    "The Industrial Revolution began in the late 18th century and transformed manufacturing processes. " * 8,
    "Quantum computing leverages principles of superposition and entanglement to solve certain problems. " * 8,
    "Natural language processing combines linguistics and computer science to understand human language. " * 8,
    "Climate change refers to long-term shifts in temperatures and weather patterns across the globe. " * 8,
    "Deep reinforcement learning combines neural networks with reward-driven decision making. " * 8,
    "Graph neural networks operate on graph-structured data by passing messages between nodes. " * 8,
    "Edge computing brings computation and data storage closer to the sources of data generation. " * 8,
    "Convolutional neural networks are specialized for processing grid-like topology data such as images. " * 8,
    "The scientific method involves systematic observation, measurement, experiment, and formulation. " * 8,
    "Blockchain technology provides a decentralized ledger that records transactions across many computers. " * 8,
    "Generative adversarial networks consist of two neural networks contesting with each other in a game. " * 8,
]


def _format_mmlu_sample(sample: dict) -> str:
    """Format an MMLU sample into a natural-language prompt."""
    letters = ["A", "B", "C", "D"]
    choices_text = "\n".join(
        f"{letters[i]}. {c}" for i, c in enumerate(sample["choices"])
    )
    return f"Question: {sample['question']}\n{choices_text}\nAnswer:"


def _get_calibration_texts(dataset_name, num_samples, max_length, tokenizer):
    """Load calibration texts from cais/mmlu or use built-in prompts."""
    prompts = []
    try:
        import datasets
        ds = datasets.load_dataset(dataset_name, "all", split="test", trust_remote_code=True)
        from random import Random
        rng = Random(42)
        samples = rng.sample(list(ds), min(num_samples, len(ds)))
        for s in samples:
            text = _format_mmlu_sample(s)
            prompts.append(text)
        print(f"[INFO] Loaded {len(prompts)} MMLU samples for calibration")
    except Exception as e:
        print(f"[WARNING] Could not load {dataset_name}: {e}")
        prompts = _CALIBRATION_TEXTS[:num_samples]
        print(f"[INFO] Using {len(prompts)} built-in calibration prompts (dataset unavailable)")

    texts = []
    for p in prompts:
        input_ids = tokenizer(p, truncation=True, max_length=max_length, return_tensors="pt").input_ids
        decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if decoded:
            texts.append(decoded)
    return texts


# ---------------------------------------------------------------------------
# Module name matching (Qwen3 naming conventions)
# ---------------------------------------------------------------------------

def _get_operator_group(name: str) -> str | None:
    """Map a module name to its operator group. Returns None for excluded modules."""
    for group_name, info in OPERATOR_GROUPS.items():
        patterns = info.get("patterns", [])
        exclude = info.get("exclude_patterns", [])
        if any(p in name for p in patterns) and not any(e in name for e in exclude):
            return group_name
    return None


def _module_matches(name: str, module_filter: str) -> bool:
    """Check if a named module matches the filter (Qwen3 naming)."""
    if module_filter in ("all", "linear"):
        return True

    if module_filter == "attention":
        return any(p in name for p in OPERATOR_GROUPS["attention"]["patterns"])
    if module_filter == "mlp":
        return any(p in name for p in OPERATOR_GROUPS["intermediate"]["patterns"]
                             + OPERATOR_GROUPS["output"]["patterns"])

    return True


# ---------------------------------------------------------------------------
# 1. Weight collection
# ---------------------------------------------------------------------------

def collect_weight_tensors(model, module_filter: str = "linear",
                            grouped: bool = False) -> dict[str, list[torch.Tensor]] | list[torch.Tensor]:
    """Collect weight tensors from target Linear modules.

    When grouped=True and module_filter='all', returns a dict keyed by operator group.
    Otherwise returns a flat list (backward compatible).
    """
    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    group_module_counts: dict[str, int] = defaultdict(int)
    group_elements: dict[str, int] = defaultdict(int)
    flat_tensors: list[torch.Tensor] = []
    total_elements = 0
    module_count = 0

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not _module_matches(name, module_filter):
            continue
        w = module.weight.detach().cpu()
        flat_tensors.append(w)
        total_elements += w.numel()
        module_count += 1

        if grouped:
            op_group = _get_operator_group(name)
            if op_group is None:
                continue
            groups[op_group].append(w)
            group_module_counts[op_group] += 1
            group_elements[op_group] += w.numel()

    print(f"[INFO] Weight collection: {module_count} modules, {total_elements:,} elements")
    if grouped:
        for g in OPERATOR_GROUPS:
            cnt = group_module_counts.get(g, 0)
            elems = group_elements.get(g, 0)
            print(f"       {g:>12s}: {cnt:3d} modules, {elems:>12,} elements")
        return dict(groups)

    return flat_tensors


# ---------------------------------------------------------------------------
# 2. Activation collection
# ---------------------------------------------------------------------------

def collect_activation_tensors(
    model,
    tokenizer,
    calibration_texts: list[str],
    module_filter: str = "linear",
    activation_kind: str = "input",
    max_elements: int | None = None,
    device: str = "cuda",
    grouped: bool = False,
) -> tuple:
    """
    Collect input and/or output activations from target Linear modules
    by running the model on calibration texts.

    When grouped=True and module_filter='all', returns dicts keyed by operator group.
    Otherwise returns flat lists (backward compatible).

    Returns:
        When grouped=False: (input_acts: list, output_acts: list)
        When grouped=True:  (input_acts: dict[str, list], output_acts: dict[str, list])
    """
    import torch.nn as nn

    collect_input = activation_kind in ("input", "both")
    collect_output = activation_kind in ("output", "both")

    target_modules: list[tuple[str, nn.Module, str | None]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _module_matches(name, module_filter):
            op_group = _get_operator_group(name) if grouped else None
            if grouped and op_group is None:
                continue
            target_modules.append((name, module, op_group))

    n_targets = max(len(target_modules), 1)
    limit = max_elements or float("inf")

    # Per-group budgets so attention doesn't consume the entire limit
    if grouped:
        group_counts: dict[str, int] = defaultdict(int)
        for _, _, g in target_modules:
            if g:
                group_counts[g] += 1
        n_groups = len(group_counts)
        per_group_limit = max(int(limit / max(n_groups, 1)), 100000)
    else:
        per_group_limit = limit

    per_tensor_budget = max(int(limit / n_targets), 10000)

    group_elems: dict[str, int] = defaultdict(int)
    total_input_elems = 0
    total_output_elems = 0

    input_acts_flat: list[torch.Tensor] = []
    output_acts_flat: list[torch.Tensor] = []
    input_acts_grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    output_acts_grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    hooks = []

    print(f"[INFO] Activation collection: {n_targets} target modules, "
          f"per-tensor budget: {per_tensor_budget:,} elements")
    if grouped:
        for g, cnt in sorted(group_counts.items()):
            print(f"       {g:>12s}: {cnt:3d} modules, per-group limit: {per_group_limit:,}")

    def _make_hook(name, op_group):
        def hook_fn(module, input_tup, output_tup):
            nonlocal total_input_elems, total_output_elems

            if collect_input:
                if grouped and group_elems[op_group] >= per_group_limit:
                    pass
                elif not grouped and total_input_elems >= limit:
                    pass
                else:
                    inp = input_tup[0].detach().cpu().float().flatten()
                    if inp.numel() > per_tensor_budget:
                        idx = torch.randperm(inp.numel())[:per_tensor_budget]
                        inp = inp[idx]
                    if grouped:
                        input_acts_grouped[op_group].append(inp)
                        group_elems[op_group] += inp.numel()
                    else:
                        input_acts_flat.append(inp)
                        total_input_elems += inp.numel()

            if collect_output:
                if grouped and group_elems[op_group] >= per_group_limit:
                    pass
                elif not grouped and total_output_elems >= limit:
                    pass
                else:
                    out = output_tup.detach().cpu() if isinstance(output_tup, torch.Tensor) else output_tup[0].detach().cpu()
                    if isinstance(out, torch.Tensor) and out.numel() > 0:
                        out = out.float().flatten()
                        if out.numel() > per_tensor_budget:
                            idx = torch.randperm(out.numel())[:per_tensor_budget]
                            out = out[idx]
                        if grouped:
                            output_acts_grouped[op_group].append(out)
                            group_elems[op_group] += out.numel()
                        else:
                            output_acts_flat.append(out)
                            total_output_elems += out.numel()
        return hook_fn

    for name, module, op_group in target_modules:
        hooks.append(module.register_forward_hook(_make_hook(name, op_group)))

    model.eval()
    collected_texts = 0
    with torch.no_grad():
        for text in calibration_texts:
            if max_elements:
                if grouped:
                    all_saturated = all(
                        group_elems[g] >= per_group_limit for g in group_counts
                    )
                    if all_saturated:
                        break
                else:
                    sat = (total_input_elems >= limit if collect_input else True) and \
                          (total_output_elems >= limit if collect_output else True)
                    if sat:
                        break
            try:
                inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
                if inputs.input_ids.numel() == 0:
                    continue
                _ = model(**inputs)
                collected_texts += 1
            except Exception as e:
                print(f"       [WARNING] Error on calibration text: {e}")
                continue

    for h in hooks:
        h.remove()

    print(f"[INFO] Processed {collected_texts} calibration samples")
    if grouped:
        for g in sorted(group_counts):
            in_count = len(input_acts_grouped.get(g, []))
            out_count = len(output_acts_grouped.get(g, []))
            in_elems = sum(t.numel() for t in input_acts_grouped.get(g, []))
            out_elems = sum(t.numel() for t in output_acts_grouped.get(g, []))
            print(f"       {g:>12s}: {in_count:3d} input tensors, {in_elems:>12,} elements"
                  f"  |  {out_count:3d} output tensors, {out_elems:>12,} elements")
        return dict(input_acts_grouped), dict(output_acts_grouped)
    else:
        in_elems = sum(t.numel() for t in input_acts_flat)
        out_elems = sum(t.numel() for t in output_acts_flat)
        print(f"       Input activations:  {len(input_acts_flat)} tensors, {in_elems:,} elements")
        print(f"       Output activations: {len(output_acts_flat)} tensors, {out_elems:,} elements")
        return input_acts_flat, output_acts_flat


# ---------------------------------------------------------------------------
# 3. Build & save
# ---------------------------------------------------------------------------

def build_and_save(
    tensors: list[torch.Tensor],
    source_name: str,
    precision: str,
    output_dir: str,
    transform: str,
    clip_value: float | None,
    max_elements: int | None,
    metadata: dict,
    module_filter: str = "all",
) -> str:
    """Build severity table from tensors, normalise, and save. Returns output path."""
    if not tensors or all(t.numel() == 0 for t in tensors):
        print(f"[WARNING] No tensors for {source_name}, skipping")
        return ""

    table = build_severity_lookup_table(
        tensors,
        source_name=source_name,
        precision=precision,
        transform=transform,
        clip_value=clip_value,
        max_elements=max_elements,
    )
    table = apply_theoretical_floor(table)
    table = normalize_table_scores(table, method="log1p_max")
    table["metadata"] = metadata

    os.makedirs(output_dir, exist_ok=True)
    if module_filter and module_filter != "all":
        filename = f"severity_table_{source_name}_{precision}_{module_filter}.json"
    else:
        filename = f"severity_table_{source_name}_{precision}.json"
    path = os.path.join(output_dir, filename)
    save_lookup_table(table, path)
    print(f"[INFO] Saved: {path}")

    print(f"[SUMMARY] {source_name} ({precision}): {table['num_elements']:,} elements")
    print_lookup_table(table, sort_by="sa0_unconditional_norm")

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build bit severity lookup tables for Qwen3-8B")

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Model name or path")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32",
                        help="Precision for table bit indexing")
    parser.add_argument("--source", type=str, choices=["weight", "activation", "both", "psum"], default="both",
                        help="Which tensor source to use")
    parser.add_argument("--dataset", type=str, default="cais/mmlu",
                        help="Calibration dataset name")
    parser.add_argument("--num-samples", type=int, default=32,
                        help="Number of calibration samples for activation")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length for calibration texts")
    parser.add_argument("--max-elements", type=int, default=2_000_000,
                        help="Max elements per table (subsampled if exceeded)")
    parser.add_argument("--modules", type=str, choices=["linear", "attention", "mlp", "all"], default="all",
                        help="Module filter for collecting tensors")
    parser.add_argument("--activation-kind", type=str, choices=["input", "output", "both"], default="input",
                        help="Which activations to collect")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: projects/qwen3-8b/config/)")
    parser.add_argument("--transform", type=str, choices=["log1p", "identity"], default="log1p",
                        help="Severity stabilising transform")
    parser.add_argument("--clip-value", type=float, default=None,
                        help="Clip raw delta before transform")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for computation")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[WARNING] CUDA not available, using CPU")

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "config")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    print("=" * 60)
    print(f"Bit Severity Table Builder — Qwen3-8B")
    print(f"  Model:       {args.model}")
    print(f"  Precision:   {args.precision}")
    print(f"  Torch dtype: {torch_dtype}")
    print(f"  Source:      {args.source}")
    print(f"  Modules:     {args.modules}")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Output:      {output_dir}")
    print(f"  Transform:   {args.transform}")
    print(f"  Max elements:{args.max_elements:,}")
    print("=" * 60)

    print("[INFO] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.eval()
    print("[INFO] Model loaded.")

    metadata = {
        "model": args.model,
        "modules": args.modules,
        "activation_kind": args.activation_kind,
        "num_calibration_samples": args.num_samples,
        "max_length": args.max_length,
        "precision": args.precision,
        "torch_dtype": str(torch_dtype),
        "transform": args.transform,
        "clip_value": args.clip_value,
        "timestamp": timestamp,
    }

    use_grouped = args.modules == "all"

    # -----------------------------------------------------------------------
    # Weight table
    # -----------------------------------------------------------------------
    if args.source in ("weight", "both"):
        print("-" * 40)
        print("[STEP] Building weight severity table...")
        result = collect_weight_tensors(model, args.modules, grouped=use_grouped)
        if use_grouped:
            for op_group in OPERATOR_GROUPS:
                tensors = result.get(op_group, [])
                if not tensors:
                    print(f"[WARNING] No weight tensors for {op_group}, skipping")
                    continue
                build_and_save(
                    tensors,
                    source_name="weight",
                    precision=args.precision,
                    output_dir=output_dir,
                    transform=args.transform,
                    clip_value=args.clip_value,
                    max_elements=args.max_elements,
                    metadata={**metadata, "operator_group": op_group},
                    module_filter=op_group,
                )
        else:
            if result:
                build_and_save(
                    result,
                    source_name="weight",
                    precision=args.precision,
                    output_dir=output_dir,
                    transform=args.transform,
                    clip_value=args.clip_value,
                    max_elements=args.max_elements,
                    metadata=metadata,
                    module_filter=args.modules,
                )
            else:
                print("[WARNING] No weight tensors collected")

    # -----------------------------------------------------------------------
    # Activation / PSUM table
    # -----------------------------------------------------------------------
    if args.source in ("activation", "both", "psum"):
        if args.source == "psum":
            table_label = "psum"
            collect_kind = "output"
        elif args.source == "activation":
            table_label = "activation"
            collect_kind = args.activation_kind
        else:
            table_label = "activation"
            collect_kind = args.activation_kind

        print("-" * 40)
        print(f"[STEP] Building {table_label} severity table (kind={collect_kind})...")
        cal_texts = _get_calibration_texts(
            args.dataset, args.num_samples, args.max_length, tokenizer
        )
        in_acts, out_acts = collect_activation_tensors(
            model, tokenizer, cal_texts,
            module_filter=args.modules,
            activation_kind=collect_kind,
            max_elements=args.max_elements,
            device=device,
            grouped=use_grouped,
        )

        def _save_activation(source_suffix, acts, op_group=None):
            """Save one activation severity table."""
            if not acts:
                return
            source_name = f"{table_label}_{source_suffix}"
            mf = op_group if op_group else args.modules
            build_and_save(
                acts,
                source_name=source_name,
                precision=args.precision,
                output_dir=output_dir,
                transform=args.transform,
                clip_value=args.clip_value,
                max_elements=args.max_elements,
                metadata={**metadata, "operator_group": op_group} if op_group else metadata,
                module_filter=mf,
            )

        if use_grouped:
            for op_group in OPERATOR_GROUPS:
                if collect_kind in ("input", "both"):
                    _save_activation("input", in_acts.get(op_group, []), op_group)
                if collect_kind in ("output", "both"):
                    _save_activation("output", out_acts.get(op_group, []), op_group)
        else:
            if collect_kind in ("input", "both") and in_acts:
                _save_activation("input", in_acts)
            if collect_kind in ("output", "both") and out_acts:
                _save_activation("output", out_acts)

    print()
    print("=" * 60)
    print("Done. Tables saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Offline calibration script to build bit severity lookup tables.

Collects weight tensors (and optionally activation tensors) from a model,
then builds per-bit severity tables for FP32/BF16.

Usage:
  # Weight-only FP32 table
  python build_severity_table.py --source weight --precision fp32

  # Activation-only BF16 table
  python build_severity_table.py --source activation --precision bf16 --num-samples 8

  # Both sources, all modules
  python build_severity_table.py --source both --modules all
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tool/src"))
from tool.bit_severity import (
    build_severity_lookup_table,
    normalize_table_scores,
    save_lookup_table,
    print_lookup_table,
)

hf_logging.set_verbosity_error()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------------------
# Calibration prompts (fallback when external dataset unavailable)
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


def _get_calibration_texts(dataset_name, num_samples, max_length, tokenizer):
    """Load calibration texts from a dataset or use built-in prompts."""
    prompts = []
    try:
        import datasets
        ds = datasets.load_dataset("rajpurkar/squad", split="validation", trust_remote_code=True)
        from random import Random
        rng = Random(42)
        samples = rng.sample(list(ds), min(num_samples, len(ds)))
        for s in samples:
            text = f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer: {s['answers']['text'][0] if s['answers']['text'] else ''}"
            prompts.append(text)
        print(f"[INFO] Loaded {len(prompts)} SQuAD samples for calibration")
    except Exception:
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
# Module name matching
# ---------------------------------------------------------------------------

def _module_matches(name: str, module_filter: str) -> bool:
    """Check if a named module matches the filter."""
    if module_filter == "all":
        return True
    if module_filter == "linear":
        return True  # already filtered by isinstance(nn.Linear)

    attention_names = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "in_proj_qkv"}
    mlp_names = {"gate_proj", "up_proj", "down_proj", "fc1", "fc2"}

    if module_filter == "attention":
        return any(attn_name in name for attn_name in attention_names)
    if module_filter == "mlp":
        return any(mlp_name in name for mlp_name in mlp_names)

    return True


# ---------------------------------------------------------------------------
# 1. Weight collection
# ---------------------------------------------------------------------------

def collect_weight_tensors(model, module_filter: str = "linear") -> list[torch.Tensor]:
    """Collect weight tensors from target Linear modules."""
    tensors = []
    module_names = []
    total_elements = 0

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not _module_matches(name, module_filter):
            continue
        w = module.weight.detach().cpu()
        tensors.append(w)
        module_names.append(name)
        total_elements += w.numel()

    print(f"[INFO] Weight collection: {len(tensors)} modules, {total_elements:,} elements")
    if module_names:
        print(f"       Examples: {module_names[:5]}..." if len(module_names) > 5 else f"       Modules: {module_names}")
    return tensors


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
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Collect input and/or output activations from target Linear modules
    by running the model on calibration texts.

    Returns (input_activations, output_activations).
    """
    import torch.nn as nn

    import torch.nn as nn

    input_acts: list[torch.Tensor] = []
    output_acts: list[torch.Tensor] = []
    total_input_elems = 0
    total_output_elems = 0
    hooks = []

    collect_input = activation_kind in ("input", "both")
    collect_output = activation_kind in ("output", "both")

    # Find target modules first to compute per-tensor sampling budget
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _module_matches(name, module_filter):
            target_modules.append((name, module))

    n_targets = max(len(target_modules), 1)
    limit = max_elements or float("inf")
    # Reserve some capacity for each target module so we sample across layers
    per_tensor_budget = max(int(limit / n_targets), 10000)

    print(f"[INFO] Activation collection: {n_targets} target modules, "
          f"per-tensor budget: {per_tensor_budget:,} elements")

    def _make_hook(name):
        def hook_fn(module, input_tup, output_tup):
            nonlocal total_input_elems, total_output_elems
            if collect_input and total_input_elems < limit:
                inp = input_tup[0].detach().cpu().float().flatten()
                if inp.numel() > per_tensor_budget:
                    idx = torch.randperm(inp.numel())[:per_tensor_budget]
                    inp = inp[idx]
                input_acts.append(inp)
                total_input_elems += inp.numel()
            if collect_output and total_output_elems < limit:
                out = output_tup.detach().cpu() if isinstance(output_tup, torch.Tensor) else output_tup[0].detach().cpu()
                if isinstance(out, torch.Tensor) and out.numel() > 0:
                    out = out.float().flatten()
                    if out.numel() > per_tensor_budget:
                        idx = torch.randperm(out.numel())[:per_tensor_budget]
                        out = out[idx]
                    output_acts.append(out)
                    total_output_elems += out.numel()
        return hook_fn

    for name, module in target_modules:
        hooks.append(module.register_forward_hook(_make_hook(name)))

    # Run calibration forward passes
    model.eval()
    collected_texts = 0
    with torch.no_grad():
        for text in calibration_texts:
            if (max_elements and total_input_elems >= max_elements
                    and total_output_elems >= max_elements):
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

    # Cleanup hooks
    for h in hooks:
        h.remove()

    print(f"[INFO] Processed {collected_texts} calibration samples")
    print(f"       Input activations:  {len(input_acts)} tensors, {total_input_elems:,} elements")
    print(f"       Output activations: {len(output_acts)} tensors, {total_output_elems:,} elements")

    return input_acts, output_acts


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
    table = normalize_table_scores(table)
    table["metadata"] = metadata

    os.makedirs(output_dir, exist_ok=True)
    filename = f"severity_table_{source_name}_{precision}.json"
    path = os.path.join(output_dir, filename)
    save_lookup_table(table, path)
    print(f"[INFO] Saved: {path}")

    # Brief summary
    print(f"[SUMMARY] {source_name} ({precision}): {table['num_elements']:,} elements")
    print_lookup_table(table, sort_by="sa0_unconditional_norm")

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build bit severity lookup tables")

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model name or path")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32",
                        help="Precision for table bit indexing")
    parser.add_argument("--source", type=str, choices=["weight", "activation", "both", "psum"], default="both",
                        help="Which tensor source to use")
    parser.add_argument("--dataset", type=str, default="squad",
                        help="Calibration dataset name (fallback to built-in if unavailable)")
    parser.add_argument("--num-samples", type=int, default=32,
                        help="Number of calibration samples for activation")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length for calibration texts")
    parser.add_argument("--max-elements", type=int, default=2_000_000,
                        help="Max elements per table (subsampled if exceeded)")
    parser.add_argument("--modules", type=str, choices=["linear", "attention", "mlp", "all"], default="attention",
                        help="Module filter for collecting tensors")
    parser.add_argument("--activation-kind", type=str, choices=["input", "output", "both"], default="input",
                        help="Which activations to collect")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: projects/<model>/config/)")
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

    # Determine output directory
    if args.output_dir is None:
        model_slug = args.model.replace("/", "--").replace("\\", "--")
        args.output_dir = f"projects/qwen3-0.6b/config"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")

    print("=" * 60)
    print(f"Bit Severity Table Builder")
    print(f"  Model:       {args.model}")
    print(f"  Precision:   {args.precision}")
    print(f"  Source:      {args.source}")
    print(f"  Modules:     {args.modules}")
    print(f"  Output:      {output_dir}")
    print(f"  Transform:   {args.transform}")
    print(f"  Max elements:{args.max_elements:,}")
    print("=" * 60)

    # Load model (weight collection can be cheap; activation needs full model)
    print("[INFO] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32,
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
        "transform": args.transform,
        "clip_value": args.clip_value,
        "timestamp": timestamp,
    }

    # -----------------------------------------------------------------------
    # Weight table
    # -----------------------------------------------------------------------
    if args.source in ("weight", "both"):
        print("-" * 40)
        print("[STEP] Building weight severity table...")
        weight_tensors = collect_weight_tensors(model, args.modules)
        if weight_tensors:
            build_and_save(
                weight_tensors,
                source_name="weight",
                precision=args.precision,
                output_dir=output_dir,
                transform=args.transform,
                clip_value=args.clip_value,
                max_elements=args.max_elements,
                metadata=metadata,
            )
        else:
            print("[WARNING] No weight tensors collected")

    # -----------------------------------------------------------------------
    # PSUM table (output activation = final accumulated psum)
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
        )

        if in_acts and collect_kind in ("input", "both"):
            build_and_save(
                in_acts,
                source_name=f"{table_label}_input",
                precision=args.precision,
                output_dir=output_dir,
                transform=args.transform,
                clip_value=args.clip_value,
                max_elements=args.max_elements,
                metadata=metadata,
            )

        if out_acts and collect_kind in ("output", "both"):
            build_and_save(
                out_acts,
                source_name=f"{table_label}_output",
                precision=args.precision,
                output_dir=output_dir,
                transform=args.transform,
                clip_value=args.clip_value,
                max_elements=args.max_elements,
                metadata=metadata,
            )

    print()
    print("=" * 60)
    print("Done. Tables saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()

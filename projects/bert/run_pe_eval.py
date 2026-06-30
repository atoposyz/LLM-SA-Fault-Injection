"""
Per-PE-position fault injection for bits 23–30, all 1024 PEs.

Input mode: injects per-column (32 rows per column) since coverage only varies by column.
Weight mode: injects per-PE (all 1024) — fast because _simulate_ws for weight is single matmul.

Runs 4 independent processes (one per mode+stuck) for parallelism.
Each process loads the model once and evaluates all PEs × bits internally.

Usage:
  uv run python projects/bert/run_pe_eval.py
  uv run python projects/bert/run_pe_eval.py --mode input --stuck 1
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time

import torch

# ── per-process imports happen inside worker to avoid fork issues ──


MODEL_NAME = "boltuix/bert-emotion"
DATASET_NAME = "boltuix/emotions-dataset"
NUM_SAMPLES = 500
BATCH_SIZE = 16
SA_ROWS = 32
SA_COLS = 32
BITS = list(range(23, 31))  # 23–30


def worker(args_dict: dict):
    """Single-process worker: load model, run all PEs × bits, save CSV."""
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
    from datasets import load_dataset

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tool/src"))
    from tool.fault_injector_next import Fast_SA_FaultInjector

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    hf_logging.set_verbosity_error()

    mode = args_dict["mode"]
    stuck = args_dict["stuck"]
    device = args_dict["device"]
    pe_positions = args_dict["pe_positions"]  # list of (r, c) tuples, or list of lists
    output_dir = args_dict["output_dir"]
    output_tag = args_dict["output_tag"]
    bits = args_dict["bits"]

    pid = os.getpid()
    print(f"[{pid}] Loading model for mode={mode} stuck={stuck} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    import random
    ds = load_dataset(DATASET_NAME, split="train")
    random.seed(42)
    samples = random.sample(list(ds), min(NUM_SAMPLES, len(ds)))

    label2id = model.config.label2id

    # Target patterns
    TARGET_PATTERNS = [
        "attention.self.query", "attention.self.key", "attention.self.value",
        "attention.output.dense", "intermediate.dense", "output.dense",
    ]

    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(p in name for p in TARGET_PATTERNS):
            target_modules.append((name, module))

    print(f"[{pid}] Target modules: {len(target_modules)}")

    def evaluate():
        correct = 0
        total = 0
        for i in range(0, len(samples), BATCH_SIZE):
            batch = samples[i:i + BATCH_SIZE]
            texts = [s["Sentence"] for s in batch]
            labels = torch.tensor([label2id[s["Label"]] for s in batch], device=device)
            inputs = tokenizer(texts, truncation=True, max_length=256, padding=True,
                              return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(batch)
        return correct / total if total > 0 else 0.0

    # Baseline
    print(f"[{pid}] Measuring baseline...")
    baseline = evaluate()
    print(f"[{pid}] Baseline accuracy: {baseline:.4f}")

    all_results = []

    total_experiments = len(bits) * len(pe_positions)
    done = 0
    t_start = time.time()

    for bit in bits:
        for pe_group in pe_positions:
            # pe_group is either [(r,c)] (single PE) for weight, or [(r,c) for r in range(32)] for input column
            fault_type = f"{mode}_stuck_{stuck}_{bit}"
            injector = Fast_SA_FaultInjector(
                sa_rows=SA_ROWS, sa_cols=SA_COLS,
                fault_type=fault_type, dataflow="WS",
            )

            if isinstance(pe_group[0], (list, tuple)):
                injector.set_multi_fault_positions(pe_group)
                # Representative PE for reporting: first in group
                rep_row, rep_col = pe_group[0]
            else:
                injector.set_multi_fault_positions([pe_group])
                rep_row, rep_col = pe_group

            handles = []
            for _name, module in target_modules:
                handles.append(module.register_forward_hook(injector.hook_fn))

            acc = evaluate()

            for h in handles:
                h.remove()

            drop = baseline - acc
            all_results.append({
                "mode": mode,
                "stuck": stuck,
                "bit": bit,
                "pe_row": rep_row,
                "pe_col": rep_col,
                "baseline": round(baseline, 6),
                "accuracy": round(acc, 6),
                "acc_drop": round(drop, 6),
            })

            done += 1
            elapsed = time.time() - t_start
            eta = (elapsed / done) * (total_experiments - done) if done > 0 else 0
            print(f"[{pid}] bit={bit:2d} pe=({rep_row:2d},{rep_col:2d})  "
                  f"acc={acc:.4f}  drop={drop:+.4f}  "
                  f"[{done}/{total_experiments}  ETA {eta:.0f}s]")

    # Save CSV
    csv_path = os.path.join(output_dir, f"pe_accuracy_{output_tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "stuck", "bit", "pe_row", "pe_col",
            "baseline", "accuracy", "acc_drop",
        ])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"[{pid}] Saved: {csv_path}  ({len(all_results)} rows)")
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Per-PE fault injection evaluation for bits 23-30"
    )
    parser.add_argument("--mode", type=str, choices=["input", "weight", "all"],
                        default="all")
    parser.add_argument("--stuck", type=str, choices=["0", "1", "all"],
                        default="all")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(args.output_dir, exist_ok=True)

    modes = ["input", "weight"] if args.mode == "all" else [args.mode]
    stuck_vals = [0, 1] if args.stuck == "all" else [int(args.stuck)]

    # Determine PE position strategy
    # input: per-column (32 positions) since coverage only varies by column
    # weight: per-PE (1024 positions) but per-column also valid (uniform)
    # For speed, weight also uses per-column
    input_pe_positions = [[(r, c) for r in range(SA_ROWS)] for c in range(SA_COLS)]
    weight_pe_positions = [[(r, c) for r in range(SA_ROWS)] for c in range(SA_COLS)]

    # Detect available GPUs
    ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Available GPUs: {ngpus}")

    # Build work items
    work_items = []
    for mode in modes:
        for sv in stuck_vals:
            pe_pos = input_pe_positions if mode == "input" else weight_pe_positions
            tag = f"{mode}_stuck{sv}"
            device_idx = len(work_items) % ngpus
            work_items.append({
                "mode": mode,
                "stuck": sv,
                "device": f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu",
                "pe_positions": pe_pos,
                "output_dir": args.output_dir,
                "output_tag": tag,
                "bits": BITS,
            })

    print(f"Launching {len(work_items)} worker processes...")
    print(f"  Bits: {BITS}")
    print(f"  Input:  32 columns per bit  → {32 * len(BITS)} experiments per worker")
    print(f"  Weight: 32 columns per bit  → {32 * len(BITS)} experiments per worker")
    print(f"  Total experiments: {len(work_items) * 32 * len(BITS)}")

    # Use spawn for CUDA compatibility
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(len(work_items), ngpus * 2)) as pool:
        results = pool.map(worker, work_items)

    print("\nAll workers done. Output files:")
    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    main()

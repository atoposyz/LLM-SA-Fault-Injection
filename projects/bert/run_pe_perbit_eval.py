"""
Single-PE per-position fault injection for input mode, bits 23-30.

Each PE is injected INDIVIDUALLY (not column-grouped), so row-to-row
variation within the same column can be observed.

Runs 2 workers (sa0, sa1) in parallel, each handling 8 bits × 1024 PEs.

Usage:
  uv run python projects/bert/run_pe_perbit_eval.py
"""

import multiprocessing as mp
import os
import sys
import time

import torch

MODEL_NAME = "boltuix/bert-emotion"
DATASET_NAME = "boltuix/emotions-dataset"
NUM_SAMPLES = 500
BATCH_SIZE = 16
SA_ROWS = 32
SA_COLS = 32
BITS = list(range(23, 31))

ALL_PE_POSITIONS = [(r, c) for r in range(SA_ROWS) for c in range(SA_COLS)]


def worker(args_dict: dict):
    """Single worker: load model, test all PEs × bits with single-PE injection."""
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
    from datasets import load_dataset

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tool/src"))
    from tool.fault_injector_next import Fast_SA_FaultInjector

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    hf_logging.set_verbosity_error()

    stuck = args_dict["stuck"]
    device = args_dict["device"]
    output_dir = args_dict["output_dir"]
    bits = args_dict["bits"]
    pe_positions = args_dict["pe_positions"]

    pid = os.getpid()
    print(f"[{pid}] Loading model for input sa{stuck} on {device}...")

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

    print(f"[{pid}] Baseline...")
    baseline = evaluate()
    print(f"[{pid}] Baseline accuracy: {baseline:.4f}")

    all_results = []
    total_experiments = len(bits) * len(pe_positions)
    done = 0
    t_start = time.time()

    for bit in bits:
        for pe_row, pe_col in pe_positions:
            fault_type = f"input_stuck_{stuck}_{bit}"
            injector = Fast_SA_FaultInjector(
                sa_rows=SA_ROWS, sa_cols=SA_COLS,
                fault_type=fault_type, dataflow="WS",
            )
            injector.set_fault_position(pe_row, pe_col)

            handles = []
            for _name, module in target_modules:
                handles.append(module.register_forward_hook(injector.hook_fn))

            acc = evaluate()

            for h in handles:
                h.remove()

            drop = baseline - acc
            all_results.append({
                "mode": "input",
                "stuck": stuck,
                "bit": bit,
                "pe_row": pe_row,
                "pe_col": pe_col,
                "baseline": round(baseline, 6),
                "accuracy": round(acc, 6),
                "acc_drop": round(drop, 6),
            })

            done += 1
            if done % 128 == 0:
                elapsed = time.time() - t_start
                eta = (elapsed / done) * (total_experiments - done) if done > 0 else 0
                print(f"[{pid}] {done}/{total_experiments} "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  last_drop={drop:+.4f}")

    # Save CSV
    csv_path = os.path.join(output_dir, f"pe_accuracy_input_stuck{stuck}_single.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "stuck", "bit", "pe_row", "pe_col",
            "baseline", "accuracy", "acc_drop",
        ])
        writer.writeheader()
        writer.writerows(all_results)
    elapsed_total = time.time() - t_start
    print(f"[{pid}] Saved: {csv_path}  ({len(all_results)} rows, {elapsed_total:.0f}s)")
    return csv_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(args.output_dir, exist_ok=True)

    ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Available GPUs: {ngpus}")
    print(f"Total experiments per worker: {len(BITS)} bits × {len(ALL_PE_POSITIONS)} PEs = {len(BITS)*len(ALL_PE_POSITIONS)}")

    work_items = []
    for sv in [0, 1]:
        device_idx = len(work_items) % ngpus
        work_items.append({
            "stuck": sv,
            "device": f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu",
            "output_dir": args.output_dir,
            "bits": BITS,
            "pe_positions": ALL_PE_POSITIONS,
        })

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(work_items)) as pool:
        results = pool.map(worker, work_items)

    print("\nAll done:")
    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    main()

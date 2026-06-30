"""
Per-bit fault injection evaluation for BERT emotion classification.

Uses Fast_SA_FaultInjector with ALL 1024 PEs faulted per bit to maximise signal.
Outputs per-bit accuracy_drop CSVs for downstream ranking validation.

Usage:
  uv run python projects/bert/run_perbit_eval.py --mode input --stuck 1
  uv run python projects/bert/run_perbit_eval.py --mode weight --stuck 0
  uv run python projects/bert/run_perbit_eval.py --mode all --stuck all
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tool/src"))
from tool.fault_injector_next import Fast_SA_FaultInjector

hf_logging.set_verbosity_error()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_NAME = "boltuix/bert-emotion"
DATASET_NAME = "boltuix/emotions-dataset"
NUM_SAMPLES = 500
BATCH_SIZE = 16
SA_ROWS = 32
SA_COLS = 32

ALL_PE_POSITIONS = [(r, c) for r in range(SA_ROWS) for c in range(SA_COLS)]
# Input mode: only column 0 (leftmost), which propagates to ALL output columns.
# Full 1024 PEs is 32× slower due to per-column matmul in _simulate_ws.
INPUT_PE_POSITIONS = [(r, 0) for r in range(SA_ROWS)]

# Target Linear layers matching operator groups
TARGET_PATTERNS = [
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense",
]


def load_model_and_data(device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    ds = load_dataset(DATASET_NAME, split="train")
    import random
    random.seed(42)
    samples = random.sample(list(ds), min(NUM_SAMPLES, len(ds)))

    return model, tokenizer, samples


def evaluate(model, tokenizer, samples, device="cuda"):
    correct = 0
    total = 0
    model.eval()

    label2id = model.config.label2id

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


def build_fault_type(mode: str, stuck_value: int, bit: int) -> str:
    """e.g. 'input_stuck_1_30' or 'weight_stuck_0_15'"""
    return f"{mode}_stuck_{stuck_value}_{bit}"


def find_target_modules(model):
    """Return list of (name, module) for all target Linear layers."""
    result = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(p in name for p in TARGET_PATTERNS):
            result.append((name, module))
    return result


def run_fault_eval(model, tokenizer, samples, mode, stuck_value, bit, device="cuda", precision="bf16"):
    """Evaluate accuracy with fault injected at ALL PEs on all target layers."""
    fault_type = build_fault_type(mode, stuck_value, bit)
    injector = Fast_SA_FaultInjector(
        sa_rows=SA_ROWS, sa_cols=SA_COLS,
        fault_type=fault_type, dataflow="WS",
        precision=precision,
    )
    injector.set_multi_fault_positions(
        ALL_PE_POSITIONS if mode == "weight" else INPUT_PE_POSITIONS
    )

    target_modules = find_target_modules(model)

    # Register hooks
    handles = []
    for _name, module in target_modules:
        handles.append(module.register_forward_hook(injector.hook_fn))

    acc = evaluate(model, tokenizer, samples, device)

    for h in handles:
        h.remove()

    return acc


def main():
    parser = argparse.ArgumentParser(
        description="Per-bit fault injection evaluation for BERT"
    )
    parser.add_argument("--mode", type=str, choices=["input", "weight", "all"],
                        default="all")
    parser.add_argument("--stuck", type=str, choices=["0", "1", "all"],
                        default="all")
    parser.add_argument("--bits", type=int, nargs="+", default=list(range(32)),
                        help="Bit positions to test (default: 0-31)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp32"], default="bf16",
                        help="Precision for bit position mapping (bf16 shifts bits 0-15 to 16-31)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.output_dir is None:
        suffix = f"_{args.precision}" if args.precision != "bf16" else ""
        args.output_dir = os.path.join(os.path.dirname(__file__), f"result{suffix}")
    os.makedirs(args.output_dir, exist_ok=True)

    modes = ["input", "weight"] if args.mode == "all" else [args.mode]
    stuck_vals = [0, 1] if args.stuck == "all" else [int(args.stuck)]

    print("=" * 60)
    print("Per-bit SA Fault Injection Evaluation")
    print(f"  SA: {SA_ROWS}x{SA_COLS} (ALL PEs)  |  Dataflow: WS")
    print(f"  Precision: {args.precision}")
    print(f"  Modes: {modes}  |  Stuck values: {stuck_vals}")
    print(f"  Bits: {min(args.bits)}-{max(args.bits)} ({len(args.bits)} bits)")
    print(f"  Samples: {NUM_SAMPLES}  |  Batch: {BATCH_SIZE}  |  Device: {device}")
    print("=" * 60)

    model, tokenizer, samples = load_model_and_data(device)

    print("\n[INFO] Measuring baseline accuracy...")
    baseline = evaluate(model, tokenizer, samples, device)
    print(f"[INFO] Baseline accuracy: {baseline:.4f}")

    for mode in modes:
        for sv in stuck_vals:
            print(f"\n{'='*40}")
            print(f"[RUN] mode={mode}  stuck={sv}")
            print(f"{'='*40}")

            results = []
            for bit in tqdm(args.bits, desc=f"{mode}_sa{sv}"):
                t0 = time.time()
                acc = run_fault_eval(model, tokenizer, samples, mode, sv, bit, device, args.precision)
                elapsed = time.time() - t0
                drop = baseline - acc
                results.append({
                    "bit": bit,
                    "baseline_accuracy": round(baseline, 6),
                    "avg_accuracy": round(acc, 6),
                    "acc_drop": round(drop, 6),
                    "time_s": round(elapsed, 2),
                })
                tqdm.write(f"  bit={bit:2d}  acc={acc:.4f}  drop={drop:+.4f}  ({elapsed:.1f}s)")

            csv_path = os.path.join(
                args.output_dir,
                f"accuracy_drop_perbit_{mode}_stuck_{sv}.csv"
            )
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "bit", "baseline_accuracy", "avg_accuracy", "acc_drop", "time_s"
                ])
                writer.writeheader()
                writer.writerows(results)
            print(f"[INFO] Saved: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Quick single-PE test: check row-to-row variation within same column.
Theory says all rows in column 0 have same coverage for input mode WS.
Injects ONE PE at a time and measures per-PE accuracy drop.
"""

import os, sys, time
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tool/src"))
from tool.fault_injector_next import Fast_SA_FaultInjector

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
hf_logging.set_verbosity_error()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 500
BATCH_SIZE = 16
SA_ROWS = 32
SA_COLS = 32

TARGET_PATTERNS = [
    "attention.self.query", "attention.self.key", "attention.self.value",
    "attention.output.dense", "intermediate.dense", "output.dense",
]

# Test: column 0, 8 rows spread across the 32 rows
TEST_PES = [(r, 0) for r in [0, 4, 8, 12, 16, 20, 24, 28]]
TEST_BITS = [30]  # highest exponent bit, strongest signal

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("boltuix/bert-emotion", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    "boltuix/bert-emotion", trust_remote_code=True, torch_dtype=torch.float32,
).to(DEVICE)
model.eval()

import random
ds = load_dataset("boltuix/emotions-dataset", split="train")
random.seed(42)
samples = random.sample(list(ds), min(NUM_SAMPLES, len(ds)))
label2id = model.config.label2id

target_modules = []
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and any(p in name for p in TARGET_PATTERNS):
        target_modules.append((name, module))
print(f"Target modules: {len(target_modules)}")

def evaluate():
    correct = 0
    total = 0
    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i + BATCH_SIZE]
        texts = [s["Sentence"] for s in batch]
        labels = torch.tensor([label2id[s["Label"]] for s in batch], device=DEVICE)
        inputs = tokenizer(texts, truncation=True, max_length=256, padding=True,
                          return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(batch)
    return correct / total if total > 0 else 0.0

print("Baseline...")
baseline = evaluate()
print(f"Baseline accuracy: {baseline:.4f}")

print(f"\n{'='*70}")
print(f"Single-PE test: input sa1, bit=30, column=0, rows = {[r for r,_ in TEST_PES]}")
print(f"{'='*70}")
print(f"{'row':>4s}  {'col':>4s}  {'accuracy':>10s}  {'drop':>10s}  {'time_s':>7s}")
print("-" * 45)

for pe_row, pe_col in TEST_PES:
    t0 = time.time()
    injector = Fast_SA_FaultInjector(
        sa_rows=SA_ROWS, sa_cols=SA_COLS,
        fault_type="input_stuck_1_30", dataflow="WS",
    )
    injector.set_fault_position(pe_row, pe_col)

    handles = []
    for _name, module in target_modules:
        handles.append(module.register_forward_hook(injector.hook_fn))

    acc = evaluate()

    for h in handles:
        h.remove()

    drop = baseline - acc
    elapsed = time.time() - t0
    print(f"{pe_row:4d}  {pe_col:4d}  {acc:10.4f}  {drop:+10.4f}  {elapsed:6.1f}s")

# Also test the per-column approach for comparison
print(f"\n{'='*70}")
print("Per-column comparison: all 32 rows in column 0 injected together")
print(f"{'='*70}")
t0 = time.time()
injector = Fast_SA_FaultInjector(
    sa_rows=SA_ROWS, sa_cols=SA_COLS,
    fault_type="input_stuck_1_30", dataflow="WS",
)
injector.set_multi_fault_positions([(r, 0) for r in range(SA_ROWS)])
handles = []
for _name, module in target_modules:
    handles.append(module.register_forward_hook(injector.hook_fn))
acc_col = evaluate()
for h in handles:
    h.remove()
print(f"  all-32-rows col=0:  acc={acc_col:.4f}  drop={baseline-acc_col:+.4f}")

print("\nDone.")

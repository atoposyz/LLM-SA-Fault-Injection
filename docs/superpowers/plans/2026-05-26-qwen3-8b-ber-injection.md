# Qwen3-8B BER Injection Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `run_ber_inject.py` for systematic BER fault injection on Qwen3-8B (60 combinations: dataflow × reg × BER), plus a dispatch script for 7-machine/14-GPU distribution.

**Architecture:** `run_ber_inject.py` follows the existing qwen3-8b pattern — load model once, iterate experiment combinations, for each combination reset seeds and iterate samples, hook all linear layers with `BER_Fast_SA_FaultInjector`, generate, save JSONL. `dispatch_ber_inject.sh` SSH-scans GPU availability across machines and assigns combination chunks to free GPUs.

**Tech Stack:** Python 3.12, PyTorch, Transformers, uv, bash

**Spec:** `docs/superpowers/specs/2026-05-26-qwen3-8b-ber-injection-design.md`

---

## File Map

| Action | Path | Role |
|---|---|---|
| Create | `projects/qwen3-8b/run_ber_inject.py` | Main BER injection script |
| Create | `scripts/dispatch_ber_inject.sh` | Multi-machine GPU-aware dispatch |

---

### Task 1: Create `run_ber_inject.py` — imports, constants, CLI

**Files:**
- Create: `projects/qwen3-8b/run_ber_inject.py`

- [ ] **Step 1: Write the script skeleton with imports, constants, and CLI**

```python
"""
BER 多点故障注入脚本 — Qwen3-8B。
遍历 dataflow × reg × BER 组合，每组合输出独立 JSONL。
支持 --filter-* 参数拆分工作到多 GPU/多机器。
"""

import argparse
import json
import os
import random
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tool", "src"))
from tool.ber_injector import BER_Fast_SA_FaultInjector

logging.set_verbosity_error()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_ID = "Qwen/Qwen3-8B"

pos_mapping = {
    "q":   "self_attn.q_proj",
    "k":   "self_attn.k_proj",
    "v":   "self_attn.v_proj",
    "o":   "self_attn.o_proj",
    "mlp-gate":  "mlp.gate_proj",
    "mlp-up":    "mlp.up_proj",
    "mlp-down":  "mlp.down_proj",
}

BER_VALUES = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

# 10 mode groups: (dataflow, reg)
MODE_GROUPS = [
    ("random", "mixed"),
    ("IS", "input"), ("IS", "weight"), ("IS", "psum"),
    ("OS", "input"), ("OS", "weight"), ("OS", "psum"),
    ("WS", "input"), ("WS", "weight"), ("WS", "psum"),
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_module_by_path(module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module


def ensure_dir(path_value: str) -> str:
    if not path_value.endswith("/"):
        path_value += "/"
    os.makedirs(path_value, exist_ok=True)
    return path_value


def load_model_and_tokenizer(model_id: str, use_cache_first: bool = True):
    if use_cache_first:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", trust_remote_code=True, local_files_only=True)
            print(f"[cache] using local cache for {model_id}")
            return tokenizer, model
        except (OSError, EnvironmentError):
            print("[cache] local cache miss, downloading...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", trust_remote_code=True)
    return tokenizer, model


def load_dataset_with_cache(*args, use_cache_first=True, **kwargs):
    if use_cache_first:
        try:
            return load_dataset(*args, download_mode="reuse_dataset_if_exists", **kwargs)
        except Exception:
            pass
    return load_dataset(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-8B BER Fault Injection")
    parser.add_argument("--outputfile", "-o", type=str, default="./result")
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--filter-dataflow", type=str, nargs="*",
                        choices=["random", "IS", "OS", "WS"],
                        default=["random", "IS", "OS", "WS"])
    parser.add_argument("--filter-reg", type=str, nargs="*",
                        choices=["mixed", "input", "weight", "psum"],
                        default=["mixed", "input", "weight", "psum"])
    parser.add_argument("--filter-ber", type=float, nargs="*",
                        default=BER_VALUES)
    parser.add_argument("--num-samples", "-n", type=int, default=200)
    parser.add_argument("--no-cache-priority", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_path = ensure_dir(args.outputfile)
    use_cache = not args.no_cache_priority
    # ... continues in next step

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the skeleton parses correctly**

```bash
uv run python projects/qwen3-8b/run_ber_inject.py --help
```

Expected: argparse help output showing all options.

- [ ] **Step 3: Commit skeleton**

```bash
git add projects/qwen3-8b/run_ber_inject.py
git commit -m "feat: add BER injection script skeleton for Qwen3-8B"
```

---

### Task 2: Add model loading, dataset, and combo iteration

**Files:**
- Modify: `projects/qwen3-8b/run_ber_inject.py`

- [ ] **Step 1: Add model/dataset loading and combo building after arg parsing**

Replace `# ... continues in next step` in `main()` with:

```python
    # --- model & dataset ---
    tokenizer, model = load_model_and_tokenizer(MODEL_ID, use_cache_first=use_cache)
    model.eval()
    print(f"[INFO] model loaded: {MODEL_ID}, layers={model.config.num_hidden_layers}")

    ds = load_dataset_with_cache("openai/gsm8k", "main", split="train", use_cache_first=use_cache)
    random.seed(37)
    samples = random.sample(list(ds), args.num_samples)
    sample_index = [str(i) for i in range(len(samples))]
    print(f"[INFO] dataset loaded: {len(samples)} samples")

    # --- build combo list (filtered) ---
    combos = []
    for df, reg in MODE_GROUPS:
        if df not in args.filter_dataflow:
            continue
        if reg not in args.filter_reg:
            continue
        for ber in args.filter_ber:
            if ber not in BER_VALUES:
                continue
            combos.append((df, reg, ber))
    print(f"[INFO] {len(combos)} combinations to run")

    # --- per-combo loop ---
    reg_map = {"input": 0, "weight": 1, "psum": 2}

    for df, reg, ber in combos:
        _run_combo(df, reg, ber, out_path, samples, sample_index,
                   tokenizer, model, reg_map)
```

- [ ] **Step 2: Verify it loads the model without errors (dry-run with 1 sample)**

```bash
uv run python projects/qwen3-8b/run_ber_inject.py --num-samples 1 --filter-ber 1e-4
```

Expected: model loads, prints combo count, then errors because `_run_combo` is not yet defined (acceptable at this step).

- [ ] **Step 3: Commit**

```bash
git add projects/qwen3-8b/run_ber_inject.py
git commit -m "feat: add model loading and combo iteration to BER script"
```

---

### Task 3: Implement `_run_combo` — seed reset, injector init, sample loop

**Files:**
- Modify: `projects/qwen3-8b/run_ber_inject.py`

- [ ] **Step 1: Add `_run_combo` function that orchestrates a single combination**

```python
def _run_combo(df: str, reg: str, ber: float, out_path: str,
               samples, sample_index, tokenizer, model, reg_map):
    # --- output file & resume ---
    ber_str = f"{ber:.0e}".replace("e-0", "e-").replace("e+0", "e")
    fname = f"ber_{ber_str}_df{df}_reg{reg}.jsonl"
    filepath = out_path + fname

    if os.path.exists(filepath):
        print(f"[SKIP] {fname} exists")
        return

    # --- seed reset per combo ---
    random.seed(42)
    torch.manual_seed(42)
    rng = random.Random(42)  # for per-sample decisions

    # --- injector ---
    injector = BER_Fast_SA_FaultInjector(
        sa_rows=256, sa_cols=256, dataflow=df,
        fault_type="random_stuck_0_mixed", precision="bf16")
    injector.enabled = True

    num_regs = 3 if reg == "mixed" else 1
    total_bits = 256 * 256 * num_regs * 32
    expected_faults = total_bits * ber

    print(f"[RUN] {fname}  expected_faults={expected_faults:.3f}  num_regs={num_regs}")

    # --- sample loop ---
    with open(filepath, "a", encoding="utf-8") as f:
        for idx, sample in enumerate(tqdm(samples, desc=fname)):
            _run_sample(df, reg, ber, expected_faults, num_regs, reg_map,
                        idx, sample, sample_index, injector,
                        tokenizer, model, f)
    print(f"[DONE] {fname}")
```

- [ ] **Step 2: Verify combo function works with `_run_sample` as a stub**

- [ ] **Step 3: Commit**

```bash
git add projects/qwen3-8b/run_ber_inject.py
git commit -m "feat: add _run_combo with seed reset and resume logic"
```

---

### Task 4: Implement `_run_sample` — fault config, BER injection, hooks, generation

**Files:**
- Modify: `projects/qwen3-8b/run_ber_inject.py`

- [ ] **Step 1: Add `_run_sample` with full injection logic**

```python
def _run_sample(df: str, reg: str, ber: float, expected_faults: float,
                num_regs: int, reg_map: dict, idx: int, sample,
                sample_index, injector, tokenizer, model, file_handle):
    handles = []

    # --- fault_type: stuck_0 or stuck_1 only ---
    f_type = random.choice(["stuck_0", "stuck_1"])
    injector.fault_type_str = f"random_{f_type}_mixed"
    injector.parse_fault_type()

    # --- dataflow: random per sample in random mode ---
    if df == "random":
        injector.dataflow = random.choice(["WS", "OS", "IS"])

    # --- BER injection ---
    if expected_faults >= 1.0:
        injector.init_faults_by_ber(ber, num_regs=num_regs, num_bits=32)
    else:
        # Always consume 1 random position (keep sequence aligned)
        injector.init_multi_fault_positions(1, num_regs=num_regs, num_bits=32)
        if not (random.random() < expected_faults):
            injector.reset_fault_pe()

    # --- fix reg for single-reg modes ---
    if reg != "mixed":
        reg_id = reg_map[reg]
        injector.fault_reg = [reg_id for _ in injector.fault_reg]

    # --- mount hooks on all layers ---
    num_layers = model.config.num_hidden_layers
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for kernel_name, kernel_path in pos_mapping.items():
            try:
                mod = get_module_by_path(layer, kernel_path)
                handles.append(mod.register_forward_hook(injector.hook_fn))
            except AttributeError:
                continue

    # --- generate ---
    messages = [{
        "role": "user",
        "content": ("Solve the following math problem step by step.\n"
                     + sample["question"] + "\nAnswer:")
    }]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    num_tokens = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=2000, do_sample=False,
            top_p=0.95, temperature=0, eos_token_id=tokenizer.eos_token_id)
    token_length = outputs[0].shape[0] - num_tokens

    decoded = tokenizer.decode(outputs[0][num_tokens:], skip_special_tokens=True)
    response = decoded.strip().replace("\n\n", "\n")

    # --- record ---
    result = {
        "sample_id": sample_index[idx],
        "token_length": token_length,
        "reference_answer": sample["answer"],
        "generated_answer": (
            response.split("</think>", 1)[1].strip()
            if "</think>" in response else response),
        "fault_type": injector.fault_type_str,
        "dataflow": injector.dataflow,
        "ber": ber,
        "fix_reg": reg,
        "inject_layer": "all",
        "inject_kernel": "all",
    }
    file_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    file_handle.flush()

    for h in handles:
        h.remove()
```

- [ ] **Step 2: Dry-run with 1 sample to verify the full pipeline works**

```bash
uv run python projects/qwen3-8b/run_ber_inject.py -n 1 --filter-ber 1e-4 --filter-dataflow WS --filter-reg weight
```

Expected: model loads, runs 1 sample, writes `result/ber_1e-4_dfWS_regweight.jsonl`.

- [ ] **Step 3: Test with 3 samples, 2 BER values**

```bash
uv run python projects/qwen3-8b/run_ber_inject.py -n 3 --filter-ber 1e-3 1e-4 --filter-dataflow WS --filter-reg weight
```

Expected: 2 files created, each with 3 records. Verify `fault_type` is consistent across files for same `sample_id`.

- [ ] **Step 4: Verify seed reproducibility — same sample_id gets same fault_type across files**

```bash
python3 -c "
import json
for f in ['result/ber_1e-3_dfWS_regweight.jsonl', 'result/ber_1e-4_dfWS_regweight.jsonl']:
    with open(f) as fp:
        for line in fp:
            r = json.loads(line)
            print(f, r['sample_id'], r['fault_type'])
"
```

Expected: same `sample_id` → same `fault_type` across both files.

- [ ] **Step 5: Test small BER probabilistic injection (1e-9)**

```bash
uv run python projects/qwen3-8b/run_ber_inject.py -n 10 --filter-ber 1e-9 --filter-dataflow WS --filter-reg weight
```

Expected: runs without error. Most samples have 0 faults injected (but process still generates normally).

- [ ] **Step 6: Commit**

```bash
git add projects/qwen3-8b/run_ber_inject.py
git commit -m "feat: implement _run_sample with BER injection and hook mounting"
```

---

### Task 5: Implement `dispatch_ber_inject.sh` — multi-machine GPU-aware dispatch

**Files:**
- Create: `scripts/dispatch_ber_inject.sh`

- [ ] **Step 1: Write the dispatch script**

```bash
#!/bin/bash
set -euo pipefail

# ============================================================
# Multi-machine BER injection dispatcher
# Machines: 10.4.1.112 - 10.4.1.118  (local: 10.4.1.115)
# Each: 2× A30 GPU
# ============================================================

MACHINES=(10.4.1.112 10.4.1.113 10.4.1.114 10.4.1.115 10.4.1.116 10.4.1.117 10.4.1.118)
PROJECT_DIR="/workplace/home/mayongzhe/faultinject"
RESULT_DIR="${PROJECT_DIR}/projects/qwen3-8b/result"

# 10 mode groups as discrete jobs
JOBS=(
  "random mixed"
  "IS input"   "IS weight"   "IS psum"
  "OS input"   "OS weight"   "OS psum"
  "WS input"   "WS weight"   "WS psum"
)

# ----------------------------------------------------------
# GPU util check: returns 0 if GPU $2 on machine $1 is free
# "free" = utilization < 10% AND no compute processes
# ----------------------------------------------------------
gpu_free() {
  local host="$1" gpu_idx="$2"
  local info
  info=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$host" \
    "nvidia-smi -i $gpu_idx --query-gpu=utilization.gpu,compute_mode --format=csv,noheader,nounits" 2>/dev/null || echo "ERR,ERR")
  local util=$(echo "$info" | cut -d',' -f1 | tr -d ' ')
  if [[ "$util" == "ERR" ]] || [[ -z "$util" ]]; then
    return 1
  fi
  # check for running compute processes
  local procs
  procs=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$host" \
    "nvidia-smi -i $gpu_idx --query-compute-apps=pid --format=csv,noheader" 2>/dev/null || echo "ERR")
  if [[ "$procs" != "" ]] && [[ "$procs" != "ERR" ]]; then
    return 1
  fi
  if [[ "$util" -lt 10 ]]; then
    return 0
  fi
  return 1
}

# ----------------------------------------------------------
# Main dispatch loop
# ----------------------------------------------------------
job_idx=0
total_jobs=${#JOBS[@]}
declare -A PIDS

echo "=== BER Injection Dispatcher ==="
echo "Total jobs: $total_jobs  (10 mode groups × 6 BER values = 60 combinations per job)"
echo "Scanning GPU availability..."
echo ""

while [[ $job_idx -lt $total_jobs ]]; do
  dispatched_this_round=0

  for host in "${MACHINES[@]}"; do
    for gpu in 0 1; do
      if [[ $job_idx -ge $total_jobs ]]; then
        break 2
      fi

      if gpu_free "$host" "$gpu"; then
        job=(${JOBS[$job_idx]})
        df="${job[0]}"
        reg="${job[1]}"

        echo "[$(date +%H:%M:%S)] Dispatching df=$df reg=$reg to ${host}:${gpu}"

        ssh -o StrictHostKeyChecking=no "$host" \
          "cd ${PROJECT_DIR} && nohup uv run python projects/qwen3-8b/run_ber_inject.py \
           --gpu $gpu --filter-dataflow $df --filter-reg $reg \
           > /tmp/ber_${df}_${reg}.log 2>&1 &" &

        PIDS["${host}:${gpu}"]=$!
        job_idx=$((job_idx + 1))
        dispatched_this_round=1
        sleep 1  # stagger launches
      fi
    done
  done

  if [[ $dispatched_this_round -eq 0 ]]; then
    echo "[$(date +%H:%M:%S)] No free GPUs found, waiting 30s..."
    sleep 30
  fi
done

echo ""
echo "=== All $total_jobs dispatched ==="
echo "Check logs at /tmp/ber_*.log on each machine"
echo "Results at: ${RESULT_DIR}/"
```

- [ ] **Step 2: Verify the script is syntactically valid**

```bash
bash -n scripts/dispatch_ber_inject.sh
```

Expected: no output (no syntax errors).

- [ ] **Step 3: Make executable**

```bash
chmod +x scripts/dispatch_ber_inject.sh
```

- [ ] **Step 4: Commit**

```bash
git add scripts/dispatch_ber_inject.sh
git commit -m "feat: add multi-machine GPU-aware BER injection dispatcher"
```

---

### Task 6: End-to-end integration test (small scale)

**Files:**
- Test: `projects/qwen3-8b/run_ber_inject.py`
- Test: `scripts/dispatch_ber_inject.sh`

- [ ] **Step 1: Clean previous test results and run a 3-sample, 2-combo test**

```bash
rm -f projects/qwen3-8b/result/ber_*_df*_reg*.jsonl
uv run python projects/qwen3-8b/run_ber_inject.py \
  -n 3 --filter-ber 1e-4 1e-6 --filter-dataflow WS --filter-reg weight input
```

- [ ] **Step 2: Verify all 4 files were created**

```bash
ls -l projects/qwen3-8b/result/ber_*_dfWS_reg*.jsonl
```

Expected: 4 files (2 BER × 2 reg).

- [ ] **Step 3: Verify record counts**

```bash
for f in projects/qwen3-8b/result/ber_*_dfWS_reg*.jsonl; do
  echo "$f: $(wc -l < $f) records"
done
```

Expected: 3 records each.

- [ ] **Step 4: Verify seed consistency across files**

```bash
python3 -c "
import json
files = [
  'projects/qwen3-8b/result/ber_1e-4_dfWS_regweight.jsonl',
  'projects/qwen3-8b/result/ber_1e-6_dfWS_regweight.jsonl',
  'projects/qwen3-8b/result/ber_1e-4_dfWS_reginput.jsonl',
  'projects/qwen3-8b/result/ber_1e-6_dfWS_reginput.jsonl',
]
for f in files:
    recs = [json.loads(l) for l in open(f)]
    print(f'{f}: {[(r[\"sample_id\"], r[\"fault_type\"]) for r in recs]}')
"
```

Expected: same `(sample_id, fault_type)` pairs across all files.

- [ ] **Step 5: Test resume — re-run, verify SKIP messages**

```bash
uv run python projects/qwen3-8b/run_ber_inject.py \
  -n 3 --filter-ber 1e-4 --filter-dataflow WS --filter-reg weight 2>&1 | grep -E "SKIP|DONE"
```

Expected: `[SKIP] ber_1e-4_dfWS_regweight.jsonl exists`

- [ ] **Step 6: Clean test files**

```bash
rm -f projects/qwen3-8b/result/ber_*_dfWS_reg*.jsonl
```

- [ ] **Step 7: Commit any fixes**

```bash
git add -A && git diff --cached --stat
# If changes: git commit -m "fix: integration test fixes for BER injection script"
```

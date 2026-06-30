# Propagation Degree Study — Design Spec

**Date**: 2026-05-30
**Status**: draft

## Motivation

In the systolic array fault injection simulation, when a fault hits a PE at position (r, c), the corrupted partial sum currently propagates to ALL downstream PEs (e.g., WS input mode: all columns j where j_mod >= c). In real hardware, a corrupted partial sum may be overwritten or diluted after a limited number of hops. The "propagation degree" p models this by limiting fault effects to the next p downstream SA columns.

This study investigates how different propagation degrees (p) affect the observable accuracy impact, and whether the relationship between bit-severity and p reveals anything about fault masking in systolic arrays.

## Experiment Design

### Fixed Parameters

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen3-8B |
| Precision | FP32 |
| Dataflow | WS (Weight Stationary) |
| Fault mode | input |
| PE position | (0, 0) — single PE, leftmost column |
| SA size | 256×256 |
| Target layers | All linear layers: q/k/v/o/mlp-gate/mlp-up/mlp-down, all 32 transformer blocks |
| Dataset | GSM8K (openai/gsm8k, main split) |
| Num samples | 200, seed=37 |
| Metric | Exact-match accuracy |

### Varied Parameters

| Parameter | Values | Count |
|---|---|---|
| Bit position | 24, 25, 26 (exponent bits) | 3 |
| Stuck direction | stuck_0, stuck_1 | 2 |
| Propagation p | 0 (no propagation), 1, 4, 16, 64, 256 (full) | 6 |
| **Total combinations** | | **36** |

### Evaluation Flow

For each (bit, stuck, p) combination:
1. Register fault injector hooks on all target layers
2. Run generation on all 200 GSM8K samples
3. Compute exact-match accuracy vs ground truth
4. Compare against baseline accuracy (no injection)

## Implementation

### 1. Injector Subclass (`tool/src/tool/propagation_injector.py`)

**Do NOT copy `fault_injector_next.py`.** Instead, subclass `Fast_SA_FaultInjector` and override only the WS input-mode propagation logic:

```python
from tool.fault_injector_next import Fast_SA_FaultInjector

class Propagation_SA_FaultInjector(Fast_SA_FaultInjector):
    def __init__(self, propagation_degree=256, **kwargs):
        super().__init__(**kwargs)
        self.propagation_degree = propagation_degree

    def _simulate_ws(self, X, W, M, K, N, r_f, c_f, device):
        # For input mode, modify in_mask to limit propagation
        # p=0: fault affects only SA column c (j_mod == c)
        # p=1: fault affects SA columns c and c+1 (j_mod in {c, c+1})
        # p=256: affects all SA columns (j_mod in 0..255), i.e. full propagation
        # Note: for N > 256, j_mod wraps, so p=1 affects ALL output columns
        #       where j % 256 is c or c+1 — this is the correct SA model behavior.
        # ... (override input mode block only, delegate others to parent)
```

The override changes only the `in_mask` line in the input mode block:

```python
# Original (from fault_injector_next.py line 239):
in_mask = (j_mod >= c).squeeze(0)

# New:
# For p=0: c <= j_mod < c+1  → only column c
# For p=1: c <= j_mod < c+2  → columns c and c+1
# For p=256: c <= j_mod < c+257 → all columns (full propagation)
# The +1 is because propagation_degree counts *additional* columns beyond c.
in_mask = (j_mod >= c) & (j_mod < c + self.propagation_degree + 1)
in_mask = in_mask.squeeze(0)
```

**Rationale for subclassing:** The change is ~5 lines. Copying 376 lines creates a maintenance burden where any bugfix to the base injector must be applied in two places.

### 2. Experiment Script (`projects/qwen3-8b/run_propagation_study.py`)

Follows the pattern of `run_direct_inject.py`:
- Load model via `device_map="auto"`, load GSM8K samples
- Define bit list `[24, 25, 26]`, stuck list `["stuck_0", "stuck_1"]`, p list `[0, 1, 4, 16, 64, 256]`
- Outer loops: (bit, stuck) → inner loop: p
- For each combo: create injector, register hooks, run generation, save results
- Skip existing output files (resumability)
- Write per-sample JSONL + aggregate `accuracy_vs_p.csv`

**Baseline acquisition:** Before the combo loop, run a baseline pass with no injection and compute exact-match accuracy. Store this value and include it in every CSV row.

**Hook registration strategy:** Register hooks **once per combo** (not per sample), since fault config is constant within a combo. Remove hooks after the combo completes.

### 3. Output Files

```
projects/qwen3-8b/result_propagation/
├── accuracy_vs_p.csv          # 36 rows: bit, stuck, p, baseline_acc, faulty_acc, acc_drop
└── samples/                    # per-sample generation results (JSONL)
    ├── bit24_stuck_0_p0.jsonl
    ├── bit24_stuck_0_p1.jsonl
    ├── bit24_stuck_0_p4.jsonl
    └── ...
```

CSV columns: `bit, stuck, p, baseline_accuracy, faulty_accuracy, acc_drop`

## Validation

Before running the full 36-combo sweep, run a quick validation:
- **p=0**: Expect zero accuracy drop (fault is self-contained at PE(0,0), no propagation)
- **p=256**: Expect accuracy drop matching the existing full-propagation injector behavior

## Edge Cases

- FP32: bit positions map directly to IEEE 754 single-precision (no offset needed)
- p = 0: no propagation — fault only affects the PE's own output column; `in_mask` limited to `j_mod == c`
- p = 256: full propagation — column 0 fault reaches all 256 SA columns (and therefore all output columns when N is a multiple of 256)
- Resumability: check output file existence before running each combo
- NaN/Inf in faulty outputs: already handled by `torch.nan_to_num` in injector
- **N > SA_size wrapping:** When the output dimension N exceeds 256, j_mod wraps. p=1 affects 2 SA columns, which may map to many output columns. This is the correct systolic array model behavior.

## Accuracy Computation

Exact-match accuracy for GSM8K:
1. Extract the final numeric answer from `generated_answer` (the last number in the chain-of-thought)
2. Compare against `sample["answer"]` (ground truth number string)
3. A sample is "correct" if the extracted number matches exactly
4. Accuracy = correct_count / total_samples

This follows the same evaluation logic as `result/evaluate.py`.


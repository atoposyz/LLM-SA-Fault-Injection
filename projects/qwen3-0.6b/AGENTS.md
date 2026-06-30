# QWEN3-0.6B PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-30

## OVERVIEW

Most developed project with numbered workflow scripts, HW/SW comparison experiments, and the codebase's only unit tests.

## STRUCTURE

```
projects/qwen3-0.6b/
├── 1_run_hw_injection.py         # Hardware-mode SA injection
├── 2_run_sw_injection.py         # Software-mode injection
├── 3_plot_results.py             # Result visualization
├── run_ber_inject.py             # BER sweep
├── run_single_bit_inject.py      # Single-bit experiments
├── SAInjectProRandomex2.py       # Random PE injection
├── SAInjectRuntime.py            # Runtime metrics collection
├── NoInjectRuntime.py            # No-fault baseline
├── build_severity_table.py       # Severity table builder
├── compare_*.py                  # K-matrix / SW-diff / layer-output comparisons
├── test_bit_severity.py          # ONLY unit test in repo
├── test_ws_input_2x2.py          # WS dataflow validation
├── test_fault_effect_heatmap.py  # Fault-effect heatmap experiment
└── config/                       # Model layer configs
```

## WHERE TO LOOK

| Task | Script | Notes |
|---|---|---|
| Full workflow | `1_*.py`, `2_*.py`, `3_*.py` | Numbered HW → SW → plot pipeline |
| Run unit tests | `test_bit_severity.py` | 10 assert-based tests; run via `python test_bit_severity.py` |
| WS validation | `test_ws_input_2x2.py` | Hand-crafted 2×2 matrix, compares expected vs actual Y |
| K-matrix analysis | `compare_k_matrix_variations.py` | K-projection matrix comparison |
| SW vs HW diff | `compare_software_fault_diff.py` | Compares software fault with hardware baseline |

## CONVENTIONS

- **Numbered workflow:** The only project with `1_`, `2_`, `3_` prefixed scripts indicating a pipeline (HW injection → SW injection → plotting).
- **Test runner:** `test_bit_severity.py` uses a custom manual runner (`if __name__ == "__main__":` calling each `test_*()` function). No pytest, no fixtures.
- **Model loading:** All scripts load `Qwen/Qwen3-0.6B` with `trust_remote_code=True` and `device_map="auto"`.

## ANTI-PATTERNS

- **Do not assume `test_*.py` files are all unit tests.** `test.py`, `test2.py`, and `test_fault_effect_heatmap.py` are standalone plotting/experiment scripts with no assertions.
- **Do not remove `sys.path` hacks without testing.** This project has the most scripts relying on `sys.path.insert(0, "../../tool/src")`.

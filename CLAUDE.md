# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Fault injection framework for studying how hardware faults (bit-flips, stuck-at) in systolic array accelerators affect LLM inference. Uses PyTorch forward hooks to intercept linear layer matmuls and simulate PE-level faults under different dataflows (WS/OS/IS) and fault modes (input/weight/psum).

## Build & run

- Package manager: `uv` (workspace with members `tool` and `projects/*`)
- Install: `uv sync`
- Run scripts: `uv run python <script.py>`, not bare `python`
- Python version: `>=3.12` (pinned in `.python-version`)
- HuggingFace mirror: `HF_ENDPOINT=https://hf-mirror.com`

## Architecture

**`tool/src/tool/`** — Reusable library (the `tool` uv workspace member):

| Module | Purpose |
|---|---|
| `fault_injector_next.py` | `Fast_SA_FaultInjector` — general-purpose SA fault injector with hook-based matmul override |
| `single_bit_injector.py` | `SingleBit_Fast_SA_FaultInjector` — optimized for single-point injection experiments |
| `ber_injector.py` | `BER_Fast_SA_FaultInjector` — large-scale multi-bit injection via BER (bit error rate) |
| `fault_injector_runtimemetrics.py` | `Fast_RuntimeMetrics_SA_FaultInjector` — same as above + records stable rank, SVD entropy, participation ratio, nuclear rank, etc. |
| `runtime_metrics.py` | `RuntimeMetricsWriter` and `compute_runtime_metrics()` — JSONL-buffered metric logging |
| `bit_severity.py` | Builds IEEE 754 bit-severity lookup tables (per-bit flip/stuck-at delta stats) for calibrated fault-space reduction |
| `printlayer.py` | Inspects model layer structure, exports to `config/<model>-config/` |

**`projects/<model>/`** — Per-model experiment scripts. Each follows a similar pattern: load HF model via `trust_remote_code=True`, register fault-injector hooks on specific linear layers, run generation on SQuAD2 samples, save results as JSONL.

**Injector interface**: All injectors expose `hook_fn(module, input_tuple, output_tuple)` suitable for `register_forward_hook`. They simulate the matmul `X @ W` by intercepting the hook, computing a faulty result based on dataflow and fault PE positions, and returning the modified output tensor.

**Fault type string format**: `{mode}_{op}[_{pos}]` — e.g. `weight_bitflip_28`, `input_stuck_1_15`, `psum_stuck_0_7`. Mode: `input`/`weight`/`psum`. Op: `bitflip`/`stuck_0`/`stuck_1`. Pos: bit position.

**Bit-severity table**: JSON files in `config/` and `projects/*/config/` that rank each IEEE 754 bit by expected numerical perturbation. Used to guide sampling toward high-impact faults. Typically built once per precision/tensor-type from calibration data.

## Key patterns

- Scripts in `projects/` manually add `tool/src` to `sys.path` rather than installing the package
- Fault injection modifies tensors via int32/uint32 reinterpretation, then converts back to float
- PE masking uses a 256x256 grid (the systolic array), cached per-device for reuse across forward passes
- For BF16 precision, bit positions 0-15 are offset by +16 to target the upper 16 bits of the FP32 representation
- Models use `device_map="auto"`; weights are restored after destructive weight-injection runs via `.copy_()` from a backup clone

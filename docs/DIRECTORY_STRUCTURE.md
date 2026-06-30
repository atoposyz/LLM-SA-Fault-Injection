# Directory Structure

This repository is organized around shared fault-injection tooling, model-specific
experiments, generated outputs, and experiment notes.

## Top-level layout

| Path | Purpose |
| --- | --- |
| `config/` | Shared model layer-name and layer-structure metadata. |
| `docs/` | Experiment plans, design notes, and project documentation. |
| `docs/notes/` | Short research notes and formula explanations. |
| `projects/` | Model- or task-specific experiment entry points. |
| `result/` | Root-level generated metrics, plots, and JSONL experiment outputs. |
| `results/` | Generated artifacts from the Qwen3-0.6B comparison scripts. |
| `scripts/` | Repository-level dispatch and orchestration scripts. |
| `tool/` | Reusable Python package for fault injection and runtime metrics. |

## Project folders

| Path | Purpose |
| --- | --- |
| `projects/bert/` | BERT severity, position, PE, and validation experiments. |
| `projects/gpt2/` | GPT-2 single-bit and BER injection experiments. |
| `projects/gpt-oss/` | GPT-OSS model utilities and configuration. |
| `projects/qwen/` | Legacy Qwen experiment scripts. |
| `projects/qwen3-0.6b/` | Qwen3-0.6B hardware/software fault comparison scripts. |
| `projects/runtimemetrics/` | Runtime metric rerun experiments. |
| `projects/qwen3-8b/` | Qwen3-8B experiments. This directory was intentionally not touched while `scripts/dispatch_ber_inject.sh` was running. |

## Output directories

`result/` and `results/` are both currently in use:

- `result/` is used by root-level plotting scripts such as `plot_comparison.py`
  and `plot_qwen3_8b_metrics.py`.
- `results/` is used by several `projects/qwen3-0.6b/` comparison scripts.

These directories were not merged because the existing scripts contain hard-coded
paths and changing them would alter current workflows.

## Notes from this cleanup

- `instruction.md` moved to `docs/notes/bit-severity.md`.
- `instruction2.md` moved to `docs/notes/position-severity.md`.
- Existing root plotting scripts were left in place because they encode their
  input and output locations directly.
- Existing modified and untracked files under `projects/qwen3-8b/` were left
  untouched.

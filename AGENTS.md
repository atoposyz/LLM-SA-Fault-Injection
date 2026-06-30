# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-30
**Commit:** 280a4d0
**Branch:** master

## OVERVIEW

Fault injection framework for studying hardware faults (bit-flips, stuck-at) in systolic array accelerators during LLM inference. Python 3.12, uv workspace monorepo. PyTorch forward hooks intercept linear layer matmuls; injectors simulate PE-level faults under WS/OS/IS dataflows.

## STRUCTURE

```
./
├── tool/               # Reusable library (uv workspace member)
│   └── src/tool/       # Injector modules + metrics + severity tables
├── projects/           # Per-model experiment scripts (7 members)
│   ├── bert/           # 32×32 PE array, ProPlus32 injector
│   ├── qwen3-0.6b/     # Most developed; numbered workflow scripts
│   ├── qwen3-8b/       # BER sweep + runtime metrics
│   ├── qwen/           # Legacy Qwen scripts
│   ├── gpt2/           # GPT-2 injection scripts
│   ├── gpt-oss/        # GPT-OSS scripts
│   └── runtimemetrics/ # StableRank-triggered reruns
├── config/             # Shared model layer configs (Qwen variants only)
├── result/             # Experiment outputs + embedded scripts
├── results/            # Plots, comparisons (separate from result/)
└── scripts/            # Multi-machine GPU dispatch
```

## WHERE TO LOOK

| Task | Location | Notes |
|---|---|---|
| Add new injector | `tool/src/tool/` | Inherit `hook_fn(module, input, output)` pattern |
| Run BER sweep | `projects/<model>/run_ber_inject.py` | `--filter-ber`, `--filter-dataflow`, `--filter-reg` |
| Run runtime metrics | `projects/<model>/SAInjectRuntime.py` | StableRank / SVD entropy collection |
| Build severity table | `projects/<model>/build_severity_table.py` | IEEE 754 per-bit delta stats |
| Multi-machine dispatch | `scripts/dispatch_ber_inject.sh` | SSH to 10.4.1.117/118, GPU cooldown logic |
| Evaluate results | `result/evaluate.py` | Scans `ber_*.jsonl`, computes accuracy |
| Plot comparisons | `result/plot_ber_comparison.py` | BER vs accuracy curves |
| Inspect layer structure | `tool/src/tool/printlayer.py` | Exports to `config/<model>-config/` |

## CONVENTIONS

- **Package manager:** `uv` only. Never use `pip install` directly. Run with `uv run python <script.py>`.
- **Workspace members:** Root `pyproject.toml` declares `projects/*` and `tool` as members. Projects declare `dependencies = []` and inherit deps from root.
- **Import path hack:** Scripts in `projects/` manually do `sys.path.insert(0, "../../tool/src")` instead of relying on the uv workspace editable install. Do not remove without verifying all scripts still resolve `tool`.
- **Fault type strings:** `{mode}_{op}[_{pos}]` — mode ∈ {input, weight, psum}, op ∈ {bitflip, stuck_0, stuck_1}, pos = IEEE 754 bit position. BF16 pos 0-15 offset by +16 to hit upper FP32 bits.
- **PE grid:** 256×256 systolic array mask, cached per-device. BERT uses 32×32 explicitly.
- **Weight restoration:** After destructive weight injection, restore via `.copy_()` from a backup clone.
- **HF mirror:** `HF_ENDPOINT=https://hf-mirror.com` for model downloads.
- **No linter / formatter / type checker configured.** No `ruff`, `mypy`, `black`, `pytest`. Code style is ad-hoc.

## ANTI-PATTERNS (THIS PROJECT)

- **Do not trust `main.py`.** It is a 6-line `uv init` stub (`print("Hello from faultinject!")`). No real entry point.
- **Do not add transitive deps to root `pyproject.toml`.** The root already incorrectly lists 60+ transitive deps (nvidia-*, six, attrs, etc.). Only direct deps belong there; `uv.lock` handles the rest.
- **Do not merge `result/` and `results/`.** Scripts have hard-coded paths to both. Consolidation requires updating every reference.
- **Do not run bare `python`.** Always `uv run python` so the workspace venv resolves.
- **Do not delete `sys.path` hacks blindly.** They coexist with the uv editable install; removing them may break scripts that rely on source-tree imports.

## UNIQUE STYLES

- **Experiment scripts are standalone executables**, not imported modules. Each has its own `argparse` with no shared CLI framework. Argument names vary across projects (`--injectConfig` vs `--filter-reg`).
- **Copy-paste inheritance:** `SAInjectProRandomex2.py`, `run_ber_inject.py`, `run_single_bit_inject.py` are duplicated across 3-4 projects with near-identical code.
- **Root-level ad-hoc scripts:** `plot_comparison.py` and `plot_qwen3_8b_metrics.py` live at root with hardcoded `/tmp/` paths, outside any workspace member.
- **Data and code mixed:** `result/evaluate.py` and `result/plot_ber_comparison.py` live inside the output directory alongside JSONL data.

## COMMANDS

```bash
# Install dependencies
uv sync

# Run an experiment
uv run python projects/qwen3-8b/run_ber_inject.py --gpu 0 -n 200

# Run the only unit test
uv run python projects/qwen3-0.6b/test_bit_severity.py

# Dispatch across machines
bash scripts/dispatch_ber_inject.sh
```

## NOTES

- `projects/gpt2/README.md` and `projects/qwen/README.md` are verbatim copies of `projects/qwen3-0.6b/README.md` and describe the wrong model.
- `config/` at root only contains Qwen model configs. BERT/Llama/GPT configs live inside their `projects/*/config/` directories.
- `projects/runtimemetrics/` breaks the naming convention: it is named after a concept, not a model.
- Only `tool/` has a `[build-system]` section. Project members are virtual workspace members and cannot be independently published without adding a build backend.

# PROPAGATION STUDY PROJECT

**Created:** 2026-06-25

## OVERVIEW

Studies how far a single PE input stuck-at fault propagates through the systolic array in WS dataflow, by varying `propagation_degree` (p) and measuring GSM8K accuracy on Qwen3-8B.

Key finding: bit position 26 is uniquely sensitive — acc drops from 94% to 0% as p increases from 0 to 256. Bits 25 and 27 show no degradation, confirming a sharp bit-level specificity.

## STRUCTURE

```
projects/propagation_study/
├── run_propagation_study.py     # Main script: sweep p-values, measure acc
├── run_bit_pair.sh              # Slurm script: run 2 bits on 2 GPUs
├── result/
│   ├── bit25_n200/              # bit=25, n=200, p∈{0,16,...,256}
│   ├── bit26_n200/              # bit=26, n=200, p∈{0,16,...,256} ← key result
│   ├── bit27_n200/              # bit=27, n=200, p∈{0,16,...,256}
│   └── legacy/                  # Earlier exploratory runs
└── AGENTS.md
```

## USAGE

```bash
# Single bit
uv run python projects/propagation_study/run_propagation_study.py \
    -o result/bit26_n200 -g 0 -n 200 \
    --filter-layer 0 --filter-bit 26 --filter-stuck 1 \
    --filter-p 0 16 32 48 64 80 96 112 128 144 160 192 256 \
    --kernel-groups k,v

# Bit pair on Slurm (2 GPUs)
sbatch projects/propagation_study/run_bit_pair.sh <bit_a> <bit_b>
```

## KEY RESULTS

| Bit | p=0 | p=128 | p=256 | Trend |
|-----|-----|-------|-------|-------|
| 25 | 95.5% | 94.5% | 94.0% | No degradation |
| 26 | 96.0% | 78.2% | 0.0% | Sharp drop above p≈100 |
| 27 | 95.0% | 94.5% | 95.5% | No degradation |

## DEPENDENCIES

- `tool/src/tool/propagation_injector.py` — `Propagation_SA_FaultInjector`
- Model: `Qwen/Qwen3-8B` (via `trust_remote_code=True`)
- Dataset: `openai/gsm8k` (main, train split)

## CONVENTIONS

- Fault always at PE(0,0), input stuck-at-1, WS dataflow, FP32 precision
- `--kernel-groups` uses commas: `k,v` not `k_v`
- GPU assignment via `-g <idx>` flag (script sets `CUDA_VISIBLE_DEVICES` internally)
- Reusing baseline.jsonl across runs is safe (same model + same seed=37 samples)

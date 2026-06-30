# BERT PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-30

## OVERVIEW

BERT fault-injection experiments on a 32×32 PE systolic array using the `SAInjectProPlus32` injector.

## STRUCTURE

```
projects/bert/
├── SAInjectProPlus32.py          # Main injector (32×32 PE grid)
├── run_pe_eval.py                # PE sensitivity evaluation
├── run_pe_perbit_eval.py         # Per-bit PE evaluation
├── run_perbit_eval.py            # General per-bit evaluation
├── validate_severity.py          # Severity table validation
├── build_severity_table.py       # 610-line severity constructor
├── build_position_severity.py    # Position-based severity
├── build_joint_severity.py       # Joint severity tables
├── compare_*.py                  # Formula / PE / ranking comparisons
├── plot_*.py                     # 6+ plotting scripts
├── allpe.sh                      # Parallel 32×32 PE sweep runner
└── config/                       # Model-specific layer configs
```

## WHERE TO LOOK

| Task | Script | Notes |
|---|---|---|
| Main injection | `SAInjectProPlus32.py` | `--layerType`, `--pos`, `--pe`, `--injectConfig`, `--faultin`, `--sampleid` |
| Parallel PE sweep | `allpe.sh` | `MAX_PARALLEL_JOBS` via `jobs -r -p` |
| Severity validation | `validate_severity.py` | Compares table predictions against ground truth |
| Build tables | `build_severity_table.py` | Longest script; handles IEEE 754 per-bit stats |
| Compare rankings | `compare_pe_ranking.py` | PE sensitivity ranking analysis |

## CONVENTIONS

- **PE grid override:** BERT explicitly uses 32×32 (not the default 256×256). The mask is constructed accordingly in `SAInjectProPlus32.py`.
- **Parallel execution:** `allpe.sh` is the only project with a local parallel sweep script. It iterates all 1024 PE positions.
- **Rich plotting suite:** 6+ dedicated plotting scripts (heatmap, severity, validation, KV-specific, position-specific, fault comparison) — more than any other project.

## ANTI-PATTERNS

- **Do not copy `SAInjectProPlus32.py` to other projects.** It hardcodes 32×32 grid assumptions. Other models should use the generic `Fast_SA_FaultInjector` from `tool/`.

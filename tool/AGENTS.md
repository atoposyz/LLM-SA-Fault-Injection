# TOOL LIBRARY KNOWLEDGE BASE

**Generated:** 2026-05-30

## OVERVIEW

Reusable fault-injection library (`tool` workspace member). All injector classes expose `hook_fn(module, input, output)` for PyTorch `register_forward_hook`.

## STRUCTURE

```
tool/
├── pyproject.toml          # name="tool", uv_build backend
└── src/tool/
    ├── __init__.py         # hello() stub only
    ├── fault_injector_next.py      # Fast_SA_FaultInjector — general-purpose
    ├── single_bit_injector.py      # SingleBit_Fast_SA_FaultInjector
    ├── ber_injector.py             # BER_Fast_SA_FaultInjector
    ├── fault_injector_runtimemetrics.py  # + StableRank, SVD entropy
    ├── runtime_metrics.py          # RuntimeMetricsWriter + compute_runtime_metrics()
    ├── bit_severity.py             # IEEE 754 per-bit severity tables
    ├── printlayer.py               # Model layer inspection → config/
    └── direct_injector.py          # Non-SA (software) baseline
```

## WHERE TO LOOK

| Task | Module | Notes |
|---|---|---|
| General SA injection | `fault_injector_next.py` | `Fast_SA_FaultInjector`, WS/OS/IS dataflows |
| Single-bit experiments | `single_bit_injector.py` | Optimized for one fault at a time |
| BER sweep | `ber_injector.py` | Multi-bit via bit-error rate |
| Runtime metrics | `fault_injector_runtimemetrics.py` | Hooks `RuntimeMetricsWriter` |
| Metric collection | `runtime_metrics.py` | JSONL-buffered logging |
| Severity tables | `bit_severity.py` | `build_severity_lookup_table()`, `normalize_table_scores()` |
| Layer structure export | `printlayer.py` | Writes `config/<model>-config/` |

## CONVENTIONS

- **Hook interface:** All injectors implement `hook_fn(self, module, input_tuple, output_tuple) -> Tensor`. The hook intercepts the linear matmul, simulates the faulty PE array, and returns the modified output.
- **Tensor reinterpretation:** Faults are applied via `.view(torch.int32)` (or uint32), bit manipulation, then `.view(torch.float32)`. This is the core bit-flip mechanism.
- **PE grid caching:** The 256×256 mask is cached per `device` to avoid recomputation across forward passes.
- **BF16 offset:** When precision is BF16, bit positions 0-15 are shifted by +16 to target the upper half of the FP32 storage.

## ANTI-PATTERNS

- **Do not import `tool` as a package in library code.** The `__init__.py` only exports `hello()`. Import submodules directly (`from tool.ber_injector import ...`).
- **Do not assume PE grid size.** BERT scripts override to 32×32; default is 256×256. Check the injector instance attribute.

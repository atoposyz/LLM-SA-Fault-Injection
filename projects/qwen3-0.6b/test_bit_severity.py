"""
Minimal tests for bit_severity.py — verify table semantics,
especially stuck-at effective_rate and unconditional_severity.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tool/src"))

import torch
from tool.bit_severity import (
    get_bit_field,
    apply_forced_bit_transition,
    apply_stuck_at,
    compute_bit_flip_severity,
    compute_stuck_at_severity,
    estimate_bit_value_distribution,
    build_severity_lookup_table,
    normalize_table_scores,
)

def test_sign_bit():
    """FP32 sign bit flip on +1.0: severity should be ~log1p(2.0) ≈ 1.099."""
    x = torch.tensor([1.0])
    stats = compute_bit_flip_severity(31, "0->1", x)
    # +1.0 sign bit is 0 → flips to -1.0, delta = |(-1)-1| / (1+eps) ≈ 2.0
    # log1p(2.0) ≈ 1.0986
    assert 1.0 < stats["unconditional_severity"] < 1.2, f"sign bit severity wrong: {stats['unconditional_severity']:.4f}"
    assert stats["effective_rate"] > 0.99, f"sign bit 0->1 should affect +1.0: {stats['effective_rate']}"
    print(f"  PASS sign bit: severity={stats['unconditional_severity']:.4f}, eff_rate={stats['effective_rate']}")

def test_mantissa_lsb():
    """FP32 mantissa LSB flip: severity should be tiny."""
    x = torch.tensor([1.0])
    stats = compute_bit_flip_severity(0, "0->1", x)
    # mantissa LSB flip changes ~2^-23 ≈ 1.19e-7
    # severity should be very small
    assert stats["unconditional_severity"] < 1e-5, f"LSB severity too large: {stats['unconditional_severity']:.2e}"
    assert stats["field"] == "mantissa"
    print(f"  PASS mantissa LSB: severity={stats['unconditional_severity']:.2e}")

def test_stuck_at_no_change():
    """stuck-at-0 on sign bit of +1.0: original bit is 0, so NO change."""
    x = torch.tensor([1.0])
    _new, effective = apply_stuck_at(x, 31, 0, "fp32")
    assert effective.sum().item() == 0, f"stuck-at-0 should not change +1.0 sign bit"

    stats_sa0 = compute_stuck_at_severity(31, 0, x)
    assert stats_sa0["effective_rate"] == 0.0, f"sa0 effective_rate should be 0: got {stats_sa0['effective_rate']}"
    assert stats_sa0["unconditional_severity"] == 0.0, f"sa0 unconditional should be 0"
    print(f"  PASS stuck-at no-change: sa0_rate={stats_sa0['effective_rate']}, uncond={stats_sa0['unconditional_severity']}")

    # stuck-at-1 on sign bit of +1.0: should be effective (original=0, forced to 1 → sign becomes 1 → -1.0)
    stats_sa1 = compute_stuck_at_severity(31, 1, x)
    assert stats_sa1["effective_rate"] == 1.0, f"sa1 effective_rate should be 1: got {stats_sa1['effective_rate']}"
    assert stats_sa1["unconditional_severity"] > 0.5, f"sa1 unconditional should be positive"
    print(f"  PASS stuck-at effective: sa1_rate={stats_sa1['effective_rate']}, uncond={stats_sa1['unconditional_severity']:.4f}")

def test_stuck_at_effective_rate_equals_bit_distribution():
    """stuck-at-0 effective_rate should equal p1; stuck-at-1 should equal p0."""
    # Mixed signs: +1.0 (sign=0), -1.0 (sign=1), +2.0 (sign=0)
    x = torch.tensor([1.0, -1.0, 2.0])

    dist = estimate_bit_value_distribution(x, 31, "fp32")
    # +1.0 sign=0, -1.0 sign=1, +2.0 sign=0 → p0=2/3, p1=1/3
    assert abs(dist["p0"] - 2/3) < 0.01, f"p0={dist['p0']}"
    assert abs(dist["p1"] - 1/3) < 0.01, f"p1={dist['p1']}"

    stats_sa0 = compute_stuck_at_severity(31, 0, x)
    stats_sa1 = compute_stuck_at_severity(31, 1, x)

    assert abs(stats_sa0["effective_rate"] - dist["p1"]) < 0.01, \
        f"sa0 eff_rate={stats_sa0['effective_rate']} != p1={dist['p1']}"
    assert abs(stats_sa1["effective_rate"] - dist["p0"]) < 0.01, \
        f"sa1 eff_rate={stats_sa1['effective_rate']} != p0={dist['p0']}"

    # sa1 conditional > 0 (changes sign of + values to negative)
    assert stats_sa1["conditional_severity"] > 0
    # sa1 unconditional = conditional * effective_rate
    assert abs(stats_sa1["unconditional_severity"] - stats_sa1["conditional_severity"] * stats_sa1["effective_rate"]) < 0.01

    print(f"  PASS stuck-at bit distribution: p0={dist['p0']:.4f} p1={dist['p1']:.4f}")
    print(f"    sa0: eff_rate={stats_sa0['effective_rate']:.4f} cond={stats_sa0['conditional_severity']:.4f} uncond={stats_sa0['unconditional_severity']:.4f}")
    print(f"    sa1: eff_rate={stats_sa1['effective_rate']:.4f} cond={stats_sa1['conditional_severity']:.4f} uncond={stats_sa1['unconditional_severity']:.4f}")

def test_all_bit_fields():
    """Verify bit field classification for all 32 FP32 bits."""
    fields = {"sign": 1, "exponent": 8, "mantissa": 23}
    counts = {"sign": 0, "exponent": 0, "mantissa": 0}
    for b in range(32):
        f = get_bit_field(b, "fp32")
        counts[f] += 1
    assert counts == fields, f"FP32 field counts wrong: {counts}"

    bf16_fields = {"sign": 1, "exponent": 8, "mantissa": 7}
    bf16_counts = {"sign": 0, "exponent": 0, "mantissa": 0}
    for b in range(16):
        f = get_bit_field(b, "bf16")
        bf16_counts[f] += 1
    assert bf16_counts == bf16_fields, f"BF16 field counts wrong: {bf16_counts}"
    print(f"  PASS bit fields: FP32={counts}, BF16={bf16_counts}")

def test_table_construction():
    """Build a table from sample tensor, check entry count and keys."""
    x = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, 100.0])

    table_fp32 = build_severity_lookup_table([x], "test", precision="fp32")
    assert len(table_fp32["table"]) == 32, f"FP32 table length={len(table_fp32['table'])}"

    table_bf16 = build_severity_lookup_table([x], "test", precision="bf16")
    assert len(table_bf16["table"]) == 16, f"BF16 table length={len(table_bf16['table'])}"

    # Check required keys exist
    required_keys = [
        "bit", "field", "p0", "p1",
        "flip_0to1_conditional", "flip_0to1_unconditional", "flip_0to1_effective_rate",
        "flip_1to0_conditional", "flip_1to0_unconditional", "flip_1to0_effective_rate",
        "sa0_conditional", "sa0_unconditional", "sa0_effective_rate",
        "sa1_conditional", "sa1_unconditional", "sa1_effective_rate",
    ]
    entry = table_fp32["table"][0]
    for k in required_keys:
        assert k in entry, f"Missing key: {k}"

    print(f"  PASS table construction: FP32 has {len(table_fp32['table'])} entries, BF16 has {len(table_bf16['table'])} entries")

    # Sign bit severity >> LSB severity (sign bit is bit 31)
    sign_entry = table_fp32["table"][31]
    lsb_entry = table_fp32["table"][0]
    assert sign_entry["field"] == "sign"
    assert lsb_entry["field"] == "mantissa"
    assert sign_entry["flip_0to1_unconditional"] > lsb_entry["flip_0to1_unconditional"] * 1000, \
        f"sign severity should >> LSB: sign={sign_entry['flip_0to1_unconditional']:.6f} vs lsb={lsb_entry['flip_0to1_unconditional']:.2e}"
    print(f"  PASS severity ordering: sign={sign_entry['flip_0to1_unconditional']:.4f} >> LSB={lsb_entry['flip_0to1_unconditional']:.2e}")

def test_normalization():
    """Test minmax normalization adds _norm keys."""
    x = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, 100.0])
    table = build_severity_lookup_table([x], "test", precision="fp32")
    table = normalize_table_scores(table)

    entry = table["table"][0]
    assert "flip_0to1_unconditional_norm" in entry
    assert "sa0_unconditional_norm" in entry
    # Check values are in [0,1]
    for e in table["table"]:
        assert 0.0 <= e["flip_0to1_unconditional_norm"] <= 1.0
        assert 0.0 <= e["sa0_unconditional_norm"] <= 1.0

    # Max should be 1.0
    max_flip = max(e["flip_0to1_unconditional_norm"] for e in table["table"])
    assert abs(max_flip - 1.0) < 0.01, f"max norm should be 1.0, got {max_flip}"
    print(f"  PASS normalization: _norm keys added, values in [0,1], max={max_flip:.4f}")

def test_direction_asymmetry():
    """0->1 and 1->0 should have different scores for exponent bits."""
    x = torch.randn(10000)  # mixed values
    table = build_severity_lookup_table([x], "test", precision="fp32")

    # Check that at least some exponent bits have asymmetric severity
    asym_count = 0
    for e in table["table"]:
        if e["field"] == "exponent":
            if abs(e["flip_0to1_unconditional"] - e["flip_1to0_unconditional"]) > 0.01:
                asym_count += 1
    assert asym_count > 0, "Expected direction asymmetry for exponent bits"
    print(f"  PASS direction asymmetry: {asym_count} exponent bits show direction asymmetry")

def test_no_table_crash_on_special_values():
    """NaN/Inf in calibration should not crash table building."""
    x = torch.tensor([0.0, -0.0, 1e38, -1e38, 1e-38, -1e-38])
    table = build_severity_lookup_table([x], "test", precision="fp32")
    assert len(table["table"]) == 32

    # Exponent bit flip may create Inf/NaN — check that nan_rate is recorded
    has_nan = any(e.get("nan_rate_flip_0to1", 0) > 0 or e.get("inf_rate_flip_0to1", 0) > 0 for e in table["table"])
    print(f"  PASS special values: any NaN/Inf recorded={has_nan}")

    # Verify JSON serializable
    import json
    s = json.dumps(table)
    assert len(s) > 0
    print(f"  PASS JSON serializable: {len(s)} chars")

def test_bf16_logical_indexing():
    """BF16 uses logical bit indices 0..15, not FP32 16..31."""
    x = torch.tensor([1.0])
    # BF16 bit 7 should be exponent (LSB of exponent in BF16)
    assert get_bit_field(7, "bf16") == "exponent"
    # BF16 bit 15 should be sign
    assert get_bit_field(15, "bf16") == "sign"
    # BF16 bit 0 should be mantissa
    assert get_bit_field(0, "bf16") == "mantissa"
    print(f"  PASS BF16 logical indexing")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing bit_severity semantics")
    print("=" * 60)

    test_sign_bit()
    test_mantissa_lsb()
    test_stuck_at_no_change()
    test_stuck_at_effective_rate_equals_bit_distribution()
    test_all_bit_fields()
    test_table_construction()
    test_normalization()
    test_direction_asymmetry()
    test_no_table_crash_on_special_values()
    test_bf16_logical_indexing()

    print()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)

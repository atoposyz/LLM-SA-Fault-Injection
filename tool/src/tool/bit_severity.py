"""
Bit Severity Lookup Table for Hierarchical Fault-Space Reduction.

Computes per-bit numerical severity scores for IEEE 754 FP32 and BF16 values.
Supports bit-flip and stuck-at fault types, recording both conditional and
unconditional severity, effective rates, and NaN/Inf rates.

The severity table is a ranking/sampling tool — it does NOT modify injected values.
"""

import json
import math

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1. Bit-field classification
# ---------------------------------------------------------------------------

def get_bit_field(bit_pos: int, precision: str = "fp32") -> str:
    """Return the IEEE 754 field name for a logical bit position."""
    precision = precision.lower()
    if precision == "fp32":
        if not (0 <= bit_pos <= 31):
            raise ValueError(f"FP32 bit_pos must be in [0, 31], got {bit_pos}")
        if bit_pos == 31:
            return "sign"
        if 23 <= bit_pos <= 30:
            return "exponent"
        return "mantissa"
    elif precision == "bf16":
        if not (0 <= bit_pos <= 15):
            raise ValueError(f"BF16 bit_pos must be in [0, 15], got {bit_pos}")
        if bit_pos == 15:
            return "sign"
        if 7 <= bit_pos <= 14:
            return "exponent"
        return "mantissa"
    else:
        raise ValueError(f"Unknown precision: {precision}")


# ---------------------------------------------------------------------------
# 2. Tensor normalisation
# ---------------------------------------------------------------------------

def normalize_tensor_to_fp32(tensor: torch.Tensor) -> torch.Tensor:
    """Detach, move to CPU, convert to float32, flatten, drop non-finite."""
    t = tensor.detach().cpu().float().flatten()
    t = t[torch.isfinite(t)]
    return t.contiguous()


# ---------------------------------------------------------------------------
# 3–4. FP32 ↔ uint32 reinterpret helpers (via numpy)
# ---------------------------------------------------------------------------

def _float32_to_uint32_tensor(x: torch.Tensor) -> torch.Tensor:
    """Reinterpret a CPU FP32 tensor as uint32 bit-patterns (returned as int64)."""
    arr = x.numpy().astype(np.float32, copy=False)
    bits = arr.view(np.uint32)
    return torch.from_numpy(bits.astype(np.int64, copy=False))


def _uint32_to_float32_tensor(bits: torch.Tensor) -> torch.Tensor:
    """Reinterpret a uint32 bit-pattern tensor (int64) back to FP32."""
    arr = bits.numpy().astype(np.uint32, copy=False)
    vals = arr.view(np.float32)
    return torch.from_numpy(vals.astype(np.float32, copy=False))


# ---------------------------------------------------------------------------
# 5. BF16 ↔ uint16 helper
# ---------------------------------------------------------------------------

def _bf16_to_uint16_tensor(x: torch.Tensor) -> torch.Tensor:
    """Convert a BF16 tensor to uint16 bit-patterns via the FP32 upper-16-bits path."""
    x_fp32 = x.float().contiguous()
    bits32 = _float32_to_uint32_tensor(x_fp32)
    bits16 = bits32 >> 16
    return bits16


# ---------------------------------------------------------------------------
# Internal: build mask / extract bit
# ---------------------------------------------------------------------------

def _get_bitmask_and_extract(bits: torch.Tensor, bit_pos: int) -> tuple[int, torch.Tensor]:
    """Return (mask as int, bool tensor of current bit values)."""
    mask = 1 << bit_pos
    current = (bits & mask) != 0
    return mask, current


# ---------------------------------------------------------------------------
# 6. apply_forced_bit_transition
# ---------------------------------------------------------------------------

def apply_forced_bit_transition(
    values: torch.Tensor,
    bit_pos: int,
    direction: str,
    precision: str = "fp32",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Force a bit transition on *eligible* elements.

    Returns (new_values, effective_mask).
    - direction "0->1": only elements whose bit is currently 0 are changed.
    - direction "1->0": only elements whose bit is currently 1 are changed.
    """
    if direction not in ("0->1", "1->0"):
        raise ValueError(f"direction must be '0->1' or '1->0', got {direction!r}")

    precision = precision.lower()
    if precision == "fp32":
        bits = _float32_to_uint32_tensor(values)
        mask, current = _get_bitmask_and_extract(bits, bit_pos)
        target_true = direction == "0->1"
        effective = current != target_true  # True where bit currently differs from target
        new_bits = bits.clone()
        if target_true:
            new_bits[effective] = new_bits[effective] | mask
        else:
            new_bits[effective] = new_bits[effective] & ~mask
        new_values = _uint32_to_float32_tensor(new_bits)
        return new_values, effective

    elif precision == "bf16":
        bits16 = _bf16_to_uint16_tensor(values)
        bits32 = _float32_to_uint32_tensor(values)
        mask16 = 1 << bit_pos
        mask32 = mask16 << 16
        current = (bits16 & mask16) != 0
        target_true = direction == "0->1"
        effective = current != target_true
        new_bits32 = bits32.clone()
        if target_true:
            new_bits32[effective] = new_bits32[effective] | mask32
        else:
            new_bits32[effective] = new_bits32[effective] & ~mask32
        new_values = _uint32_to_float32_tensor(new_bits32)
        return new_values, effective

    else:
        raise ValueError(f"Unknown precision: {precision}")


# ---------------------------------------------------------------------------
# 7. apply_stuck_at
# ---------------------------------------------------------------------------

def apply_stuck_at(
    values: torch.Tensor,
    bit_pos: int,
    stuck_value: int,
    precision: str = "fp32",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Force a bit to a stuck value. Only elements where the bit differs are changed.

    Returns (new_values, effective_mask).
    """
    if stuck_value not in (0, 1):
        raise ValueError(f"stuck_value must be 0 or 1, got {stuck_value}")

    direction = "1->0" if stuck_value == 0 else "0->1"
    return apply_forced_bit_transition(values, bit_pos, direction, precision)


# ---------------------------------------------------------------------------
# 8. compute_delta_stats
# ---------------------------------------------------------------------------

def compute_delta_stats(
    original_values: torch.Tensor,
    new_values: torch.Tensor,
    effective_mask: torch.Tensor,
    transform: str = "log1p",
    eps: float = 1e-8,
    clip_value: float | None = None,
) -> dict:
    """
    Compute severity statistics from original and faulted values.

    raw_delta = |new - original| / (|original| + eps)
    severity = transform(raw_delta)  [default: log(1+z)]
    """
    total_count = int(original_values.numel())
    effective_count = int(effective_mask.sum().item())

    abs_orig = original_values.abs() + eps
    raw_delta = (new_values - original_values).abs() / abs_orig

    if clip_value is not None:
        raw_delta = torch.clamp(raw_delta, max=clip_value)

    nan_mask = torch.isnan(new_values)
    inf_mask = torch.isinf(new_values)

    # For NaN new_values, we can't compute raw_delta meaningfully
    # But we still record their count
    raw_delta = torch.nan_to_num(raw_delta, nan=0.0, posinf=0.0, neginf=0.0)

    nan_count = int(nan_mask.sum().item())
    inf_count = int(inf_mask.sum().item())

    if transform == "identity":
        severity_delta = raw_delta
    elif transform == "log1p":
        severity_delta = torch.log1p(raw_delta)
    else:
        raise ValueError(f"Unknown transform: {transform}")

    # Conditional: mean over effective AND finite pairs
    finite_pair = torch.isfinite(original_values) & torch.isfinite(new_values)
    cond_mask = effective_mask & finite_pair
    cond_count = int(cond_mask.sum().item())
    if cond_count > 0:
        conditional_severity = float(severity_delta[cond_mask].mean().item())
        finite_mean_raw_delta_cond = float(raw_delta[cond_mask].mean().item())
    else:
        conditional_severity = 0.0
        finite_mean_raw_delta_cond = 0.0

    # Unconditional: mean over ALL original samples (treat non-effective as 0)
    # Only compute over finite pairs for both
    all_finite = torch.isfinite(original_values) & torch.isfinite(new_values)
    uncond_sev = severity_delta.clone()
    uncond_sev[~effective_mask] = 0.0
    uncond_sev[~all_finite] = 0.0
    uncond_denom = total_count  # use total count for true unconditional
    unconditional_severity = float(uncond_sev.sum().item()) / uncond_denom

    # Finite-mean raw delta unconditional (over finite pairs only)
    raw_uncond = raw_delta.clone()
    raw_uncond[~effective_mask] = 0.0
    raw_uncond[~all_finite] = 0.0
    finite_mean_raw_delta_uncond = float(raw_uncond.sum().item()) / uncond_denom

    # effective_rate among all samples
    effective_rate = effective_count / total_count if total_count > 0 else 0.0

    # max and p99 of raw_delta among effective+finite
    eff_finite_raw = raw_delta[cond_mask] if cond_count > 0 else raw_delta[:0]
    if eff_finite_raw.numel() > 0:
        max_raw_delta = float(eff_finite_raw.max().item())
        p99_raw_delta = float(eff_finite_raw.kthvalue(
            max(1, int(eff_finite_raw.numel() * 0.99))
        ).values.item())
    else:
        max_raw_delta = 0.0
        p99_raw_delta = 0.0

    return {
        "total_count": total_count,
        "effective_count": effective_count,
        "effective_rate": effective_rate,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nan_rate": nan_count / total_count if total_count > 0 else 0.0,
        "inf_rate": inf_count / total_count if total_count > 0 else 0.0,
        "conditional_severity": conditional_severity,
        "unconditional_severity": unconditional_severity,
        "finite_mean_raw_delta_cond": finite_mean_raw_delta_cond,
        "finite_mean_raw_delta_uncond": finite_mean_raw_delta_uncond,
        "max_raw_delta": max_raw_delta,
        "p99_raw_delta": p99_raw_delta,
    }


# ---------------------------------------------------------------------------
# 9. compute_bit_flip_severity
# ---------------------------------------------------------------------------

def compute_bit_flip_severity(
    bit_pos: int,
    direction: str,
    tensor_sample: torch.Tensor,
    precision: str = "fp32",
    transform: str = "log1p",
    eps: float = 1e-8,
    clip_value: float | None = None,
) -> dict:
    """Compute severity statistics for a single-bit flip."""
    values = normalize_tensor_to_fp32(tensor_sample)
    new_values, effective_mask = apply_forced_bit_transition(
        values, bit_pos, direction, precision
    )
    stats = compute_delta_stats(
        values, new_values, effective_mask,
        transform=transform, eps=eps, clip_value=clip_value,
    )
    stats["bit"] = bit_pos
    stats["field"] = get_bit_field(bit_pos, precision)
    stats["precision"] = precision
    stats["fault_type"] = "bitflip"
    stats["direction"] = direction
    return stats


# ---------------------------------------------------------------------------
# 10. compute_stuck_at_severity
# ---------------------------------------------------------------------------

def compute_stuck_at_severity(
    bit_pos: int,
    stuck_value: int,
    tensor_sample: torch.Tensor,
    precision: str = "fp32",
    transform: str = "log1p",
    eps: float = 1e-8,
    clip_value: float | None = None,
) -> dict:
    """Compute severity statistics for a single-bit stuck-at fault."""
    values = normalize_tensor_to_fp32(tensor_sample)
    new_values, effective_mask = apply_stuck_at(
        values, bit_pos, stuck_value, precision
    )
    stats = compute_delta_stats(
        values, new_values, effective_mask,
        transform=transform, eps=eps, clip_value=clip_value,
    )
    stats["bit"] = bit_pos
    stats["field"] = get_bit_field(bit_pos, precision)
    stats["precision"] = precision
    stats["fault_type"] = "stuck-at"
    stats["stuck_value"] = stuck_value
    return stats


# ---------------------------------------------------------------------------
# 11. estimate_bit_value_distribution
# ---------------------------------------------------------------------------

def estimate_bit_value_distribution(
    tensor_sample: torch.Tensor,
    bit_pos: int,
    precision: str = "fp32",
) -> dict:
    """Estimate p0/p1 for a specific bit position."""
    values = normalize_tensor_to_fp32(tensor_sample)
    precision = precision.lower()
    if precision == "fp32":
        bits = _float32_to_uint32_tensor(values)
    elif precision == "bf16":
        bits = _bf16_to_uint16_tensor(values)
    else:
        raise ValueError(f"Unknown precision: {precision}")

    mask = 1 << bit_pos
    ones = (bits & mask) != 0
    count1 = int(ones.sum().item())
    total = int(bits.numel())
    count0 = total - count1
    return {
        "p0": count0 / total if total > 0 else 0.0,
        "p1": count1 / total if total > 0 else 0.0,
        "count0": count0,
        "count1": count1,
        "total_count": total,
    }


# ---------------------------------------------------------------------------
# 12. build_severity_lookup_table
# ---------------------------------------------------------------------------

def build_severity_lookup_table(
    tensors: list[torch.Tensor],
    source_name: str,
    precision: str = "fp32",
    transform: str = "log1p",
    eps: float = 1e-8,
    clip_value: float | None = None,
    max_elements: int | None = None,
) -> dict:
    """
    Build the full bit-severity lookup table from a list of calibration tensors.

    Returns a dictionary with per-bit severity entries for flip and stuck-at faults.
    """
    precision = precision.lower()
    num_bits = 32 if precision == "fp32" else 16

    # Concatenate normalised tensors
    flat_parts = []
    total_elems = 0
    for t in tensors:
        part = normalize_tensor_to_fp32(t)
        if part.numel() == 0:
            continue
        flat_parts.append(part)
        total_elems += part.numel()

    if not flat_parts:
        raise ValueError("No finite elements in provided tensors")

    merged = torch.cat(flat_parts)

    # Subsample if needed
    if max_elements is not None and merged.numel() > max_elements:
        idx = torch.randperm(merged.numel())[:max_elements]
        merged = merged[idx]

    num_elements = int(merged.numel())

    # Bit layout metadata
    if precision == "fp32":
        bit_layout = {"sign": [31], "exponent": [23, 30], "mantissa": [0, 22]}
    else:
        bit_layout = {"sign": [15], "exponent": [7, 14], "mantissa": [0, 6]}

    table_entries = []
    for b in range(num_bits):
        dist = estimate_bit_value_distribution(merged, b, precision)

        # bitflip both directions
        f01 = compute_bit_flip_severity(
            b, "0->1", merged, precision, transform, eps, clip_value
        )
        f10 = compute_bit_flip_severity(
            b, "1->0", merged, precision, transform, eps, clip_value
        )

        # stuck-at both values
        sa0 = compute_stuck_at_severity(
            b, 0, merged, precision, transform, eps, clip_value
        )
        sa1 = compute_stuck_at_severity(
            b, 1, merged, precision, transform, eps, clip_value
        )

        entry = {
            "bit": b,
            "field": get_bit_field(b, precision),
            "p0": dist["p0"],
            "p1": dist["p1"],

            "flip_0to1_conditional": f01["conditional_severity"],
            "flip_0to1_unconditional": f01["unconditional_severity"],
            "flip_0to1_effective_rate": f01["effective_rate"],

            "flip_1to0_conditional": f10["conditional_severity"],
            "flip_1to0_unconditional": f10["unconditional_severity"],
            "flip_1to0_effective_rate": f10["effective_rate"],

            "sa0_conditional": sa0["conditional_severity"],
            "sa0_unconditional": sa0["unconditional_severity"],
            "sa0_effective_rate": sa0["effective_rate"],

            "sa1_conditional": sa1["conditional_severity"],
            "sa1_unconditional": sa1["unconditional_severity"],
            "sa1_effective_rate": sa1["effective_rate"],

            "nan_rate_flip_0to1": f01["nan_rate"],
            "inf_rate_flip_0to1": f01["inf_rate"],
            "nan_rate_flip_1to0": f10["nan_rate"],
            "inf_rate_flip_1to0": f10["inf_rate"],
            "nan_rate_sa0": sa0["nan_rate"],
            "inf_rate_sa0": sa0["inf_rate"],
            "nan_rate_sa1": sa1["nan_rate"],
            "inf_rate_sa1": sa1["inf_rate"],
        }
        table_entries.append(entry)

    return {
        "version": 1,
        "precision": precision,
        "source": source_name,
        "transform": transform,
        "eps": eps,
        "clip_value": clip_value,
        "num_tensors": len(tensors),
        "num_elements": num_elements,
        "bit_indexing": "logical_lsb0",
        "bit_layout": bit_layout,
        "table": table_entries,
    }


# ---------------------------------------------------------------------------
# 13. normalize_table_scores
# ---------------------------------------------------------------------------

def normalize_table_scores(
    table: dict,
    score_keys: list[str] | None = None,
    method: str = "minmax",
) -> dict:
    """
    Add normalized [0,1] scores for each specified key via min-max normalisation.
    Modifies the table in place and also returns it.
    """
    if score_keys is None:
        score_keys = [
            "flip_0to1_unconditional",
            "flip_1to0_unconditional",
            "sa0_unconditional",
            "sa1_unconditional",
        ]

    for key in score_keys:
        values = [entry[key] for entry in table["table"]]
        vmin = min(values)
        vmax = max(values)
        denom = vmax - vmin
        norm_key = f"{key}_norm"
        for entry in table["table"]:
            if denom > 0:
                entry[norm_key] = (entry[key] - vmin) / denom
            else:
                entry[norm_key] = 0.0

    return table


# ---------------------------------------------------------------------------
# 14. save_lookup_table
# ---------------------------------------------------------------------------

def save_lookup_table(table: dict, path: str) -> None:
    """Save the lookup table as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 15. load_lookup_table
# ---------------------------------------------------------------------------

def load_lookup_table(path: str) -> dict:
    """Load a lookup table from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 16. print_lookup_table
# ---------------------------------------------------------------------------

def print_lookup_table(table: dict, sort_by: str | None = None) -> None:
    """Pretty-print the lookup table."""
    entries = table["table"]
    if sort_by:
        entries = sorted(entries, key=lambda e: e.get(sort_by, 0))

    header = (
        f"{'bit':>4s} {'field':>9s} "
        f"{'p0':>7s} {'p1':>7s} "
        f"{'sa0_uncond':>11s} {'sa1_uncond':>11s} "
        f"{'f01_uncond':>11s} {'f10_uncond':>11s}"
    )
    print(header)
    print("-" * len(header))

    for e in entries:
        row = (
            f"{e['bit']:4d} {e['field']:>9s} "
            f"{e['p0']:7.4f} {e['p1']:7.4f} "
            f"{e['sa0_unconditional']:11.6f} {e['sa1_unconditional']:11.6f} "
            f"{e['flip_0to1_unconditional']:11.6f} {e['flip_1to0_unconditional']:11.6f}"
        )
        print(row)

    # Show normalised keys if present
    norm_keys = [k for k in entries[0] if k.endswith("_norm")] if entries else []
    if norm_keys:
        print()
        print("Normalised scores present:", ", ".join(norm_keys))


# ---------------------------------------------------------------------------
# Sampling-weight helper (optional integration point)
# ---------------------------------------------------------------------------

def get_bit_sampling_weights(
    severity_table_path: str,
    fault_type: str,
    stuck_value: int | None = None,
    direction: str | None = None,
    temperature: float = 1.0,
) -> dict[int, float]:
    """
    Load a severity table and return per-bit sampling weights (summing to 1).

    fault_type: 'bitflip' or 'stuck-at'
    For bitflip, uses flip_0to1_unconditional_norm or flip_1to0_unconditional_norm.
    For stuck-at, uses sa0_unconditional_norm or sa1_unconditional_norm.
    """
    table = load_lookup_table(severity_table_path)
    norm_keys = [k for k in table["table"][0] if k.endswith("_norm")]
    if not norm_keys:
        table = normalize_table_scores(table)

    if fault_type == "bitflip":
        if direction == "0->1":
            key = "flip_0to1_unconditional_norm"
        elif direction == "1->0":
            key = "flip_1to0_unconditional_norm"
        else:
            # Average both directions
            scores = {}
            for e in table["table"]:
                scores[e["bit"]] = (
                    e.get("flip_0to1_unconditional_norm", 0)
                    + e.get("flip_1to0_unconditional_norm", 0)
                ) / 2
            return _temp_and_normalize(scores, temperature)
        scores = {e["bit"]: e.get(key, 0) for e in table["table"]}

    elif fault_type == "stuck-at":
        if stuck_value == 0:
            key = "sa0_unconditional_norm"
        elif stuck_value == 1:
            key = "sa1_unconditional_norm"
        else:
            scores = {}
            for e in table["table"]:
                scores[e["bit"]] = (
                    e.get("sa0_unconditional_norm", 0)
                    + e.get("sa1_unconditional_norm", 0)
                ) / 2
            return _temp_and_normalize(scores, temperature)
        scores = {e["bit"]: e.get(key, 0) for e in table["table"]}

    else:
        raise ValueError(f"Unknown fault_type: {fault_type}")

    return _temp_and_normalize(scores, temperature)


def _temp_and_normalize(
    scores: dict[int, float], temperature: float
) -> dict[int, float]:
    eps_val = 1e-12
    weights = {}
    for bit, s in scores.items():
        w = max(s, eps_val) ** (1.0 / max(temperature, 1e-6))
        weights[bit] = w
    total = sum(weights.values())
    return {b: w / total for b, w in weights.items()}

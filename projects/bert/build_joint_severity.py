"""
Build joint bit-position severity ranking for BERT fault injection.

Combines per-operator bit severity tables (Step 1) with dataflow-aware
PE position coverage (Step 2) into a unified joint severity:

    S_joint(type, stuck_value, bit, operator, pe_row, pe_col) =
        S_bit(type, stuck_value, bit, operator)  ×  C_pos(type, operator, pe_row, pe_col)

Outputs:
  - machine-readable JSON with operator-specific and all-mode entries
  - sorted by joint severity for downstream representative sampling

Usage:
  uv run python projects/bert/build_joint_severity.py \
    --fault-type stuck-at --stuck-value 1

The script expects grouped severity tables and position coverage to exist.
Run build_severity_table.py and build_position_severity.py first.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config")

# Per-type severity field mapping
TYPE_MAPPING = {
    "input":  ("activation_input", "sa1_conditional"),
    "weight": ("weight",            "sa1_unconditional"),
}

# ---- formula parameters ----
# Set by --formula flag

FORMULA_CONFIG = {}  # populated in main()

import math


def _apply_formula(raw_sev: float, bit: int, config: dict) -> float:
    """Apply formula-specific transform to raw severity value."""
    name = config.get("name", "v3")
    sign_factor = config.get("sign_factor", 0.08)
    sat_point = config.get("sat_point", 10.0)
    use_log = config.get("use_log", False)

    # Sign bit resilience
    sev = raw_sev
    if bit == 31:
        sev *= sign_factor

    if name == "v3":
        # V3: cap at saturation point (hard ceiling)
        sev = min(sev, sat_point)
    elif name == "v4":
        # V4: log1p to smooth growth, threshold as reference only
        sev = math.log1p(sev)

    return sev

OPERATOR_GROUPS = ["attention", "intermediate", "output"]


def _table_filename(source: str, operator: str) -> str:
    return os.path.join(CONFIG_DIR, f"severity_table_{source}_fp32_{operator}.json")


def _position_filename() -> str:
    return os.path.join(CONFIG_DIR, "position_severity_ws.json")


def load_bit_severity(exp_type: str, operator: str, bit: int) -> float:
    """Return raw unconditional bit severity for a given (type, operator, bit)."""
    source, field = TYPE_MAPPING[exp_type]
    path = _table_filename(source, operator)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing severity table: {path}\n"
            f"Run: uv run python projects/bert/build_severity_table.py "
            f"--source both --modules all"
        )
    with open(path) as f:
        table = json.load(f)
    for entry in table["table"]:
        if entry["bit"] == bit:
            return float(entry[field])
    return 0.0


def load_position_coverage(exp_type: str, operator: str) -> np.ndarray:
    """Return the raw coverage matrix [R, C] for a given (type, operator)."""
    path = _position_filename()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing position coverage: {path}\n"
            f"Run: uv run python projects/bert/build_position_severity.py --sa-rows 32 --sa-cols 32"
        )
    with open(path) as f:
        data = json.load(f)
    return np.array(data["severity"][operator][exp_type]["raw_matrix"])


def compute_joint(
    exp_type: str, stuck_value: int, bit: int, operator: str, pe_row: int, pe_col: int,
    bit_sev: float, coverage: float,
) -> dict:
    # v3: joint_severity = bit_severity (position coverage noted but not multiplied,
    # as saturation/cascade effects make position secondary for ranking)
    return {
        "type": exp_type,
        "stuck_value": stuck_value,
        "bit": bit,
        "operator": operator,
        "pe_row": pe_row,
        "pe_col": pe_col,
        "bit_severity_raw": round(bit_sev, 10),
        "position_coverage": round(coverage, 4),
        "joint_severity": round(bit_sev, 10),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build joint bit-position severity ranking for BERT"
    )
    parser.add_argument("--fault-type", type=str, choices=["stuck-at", "bitflip"],
                        default="stuck-at", help="Fault type")
    parser.add_argument("--stuck-value", type=int, choices=[0, 1],
                        default=1, help="Stuck-at value (ignored for bitflip)")
    parser.add_argument("--types", type=str, nargs="+",
                        choices=["input", "weight", "psum"],
                        default=["input", "weight"],
                        help="Experiment types to include")
    parser.add_argument("--formula", type=str, choices=["v3", "v4"], default="v3",
                        help="Severity formula: v3=cap+saturation+signFactor, v4=log1p+threshold")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path")
    args = parser.parse_args()

    # Formula configuration
    global FORMULA_CONFIG
    if args.formula == "v3":
        FORMULA_CONFIG = {"name": "v3", "sign_factor": 0.08, "sat_point": 10.0, "use_log": False}
    else:
        FORMULA_CONFIG = {"name": "v4", "sign_factor": 1.0,  "sat_point": 5.0,  "use_log": True}

    # Determine output path
    if args.output is None:
        stub = f"stuck{args.stuck_value}" if args.fault_type == "stuck-at" else "flip"
        suffix = f"_{args.formula}" if args.formula != "v3" else ""
        args.output = os.path.join(CONFIG_DIR, f"joint_severity_{stub}_ws{suffix}.json")

    # Load SA dimensions from position file
    pos_data = json.load(open(_position_filename()))
    R, C = pos_data["sa_rows"], pos_data["sa_cols"]

    operator_entries: list[dict] = []
    all_entries: list[dict] = []

    # Pre-load all bit severities to avoid repeated file I/O
    bit_sev_cache: dict[tuple, float] = {}
    for exp_type in args.types:
        for op in OPERATOR_GROUPS:
            for bit in range(32):
                try:
                    sev = load_bit_severity(exp_type, op, bit)
                except FileNotFoundError:
                    print(f"[WARNING] Skipping {exp_type}/{op} — table missing")
                    sev = 0.0
                bit_sev_cache[(exp_type, op, bit)] = sev

    # Pre-load all coverage matrices
    cov_cache: dict[tuple, np.ndarray] = {}
    for exp_type in args.types:
        for op in OPERATOR_GROUPS:
            cov_cache[(exp_type, op)] = load_position_coverage(exp_type, op)

    print(f"Building joint severity ({args.formula}) for {args.fault_type} stuck_value={args.stuck_value}")
    print(f"  Formula: {FORMULA_CONFIG}")
    print(f"  SA: {R}x{C}  |  Types: {args.types}  |  Operators: {OPERATOR_GROUPS}")
    print(f"  Bits: 0–31  |  Total entries: {len(args.types) * 32 * len(OPERATOR_GROUPS) * R * C:,}")

    # Compute operator-specific entries
    for exp_type in args.types:
        for op in OPERATOR_GROUPS:
            cov_mat = cov_cache.get((exp_type, op))
            if cov_mat is None:
                continue
            for bit in range(32):
                bit_sev = bit_sev_cache.get((exp_type, op, bit), 0.0)
                # Apply formula transform
                bit_sev = _apply_formula(bit_sev, bit, FORMULA_CONFIG)
                if bit_sev == 0.0:
                    # Zero bit severity => all joint entries are zero; emit one summary row
                    operator_entries.append({
                        "type": exp_type,
                        "stuck_value": args.stuck_value,
                        "bit": bit,
                        "operator": op,
                        "pe_row": 0, "pe_col": 0,
                        "bit_severity_raw": 0.0,
                        "position_coverage": float(cov_mat[0, 0]),
                        "joint_severity": 0.0,
                        "all_zero": True,
                    })
                    continue
                for r in range(R):
                    for c in range(C):
                        cov = float(cov_mat[r, c])
                        if cov == 0:
                            continue
                        entry = compute_joint(
                            exp_type, args.stuck_value, bit, op, r, c,
                            bit_sev, cov,
                        )
                        operator_entries.append(entry)

    # Compute all-mode (aggregated) entries
    # Group by (type, bit, pe_row, pe_col) and aggregate across operators
    all_groups: dict[tuple, list[float]] = defaultdict(list)
    # Also track operator weights (tensor element counts) for weighted mean
    op_weights = {}
    for op in OPERATOR_GROUPS:
        layers = pos_data.get("layers", {}).get(op, {})
        K = layers.get("K", 256)
        N = layers.get("N", 256)
        # For attention: 12 layers * 4 modules = 48; others: 12 layers * 1 module = 12
        num_modules = 48 if op == "attention" else 12
        op_weights[op] = num_modules * K * N

    for entry in operator_entries:
        if entry.get("all_zero"):
            # Expand zero entries into all PE positions for aggregation
            continue
        key = (entry["type"], entry["bit"], entry["pe_row"], entry["pe_col"])
        all_groups[key].append((entry["operator"], entry["joint_severity"]))

    for (exp_type, bit, pe_row, pe_col), op_sevs in all_groups.items():
        sev_values = [s for _, s in op_sevs]
        # Weighted mean
        total_weight = sum(op_weights.get(op, 1) for op, _ in op_sevs)
        weighted_sum = sum(
            s * op_weights.get(op, 1) for op, s in op_sevs
        )
        all_entries.append({
            "type": exp_type,
            "stuck_value": args.stuck_value,
            "bit": bit,
            "pe_row": pe_row,
            "pe_col": pe_col,
            "joint_mean": round(np.mean(sev_values), 10),
            "joint_max": round(max(sev_values), 10),
            "joint_weighted_mean": round(weighted_sum / total_weight, 10) if total_weight > 0 else 0.0,
        })

    # Sort by severity descending
    operator_entries.sort(key=lambda e: e["joint_severity"], reverse=True)
    all_entries.sort(key=lambda e: e["joint_max"], reverse=True)

    # Output
    output = {
        "fault_type": args.fault_type,
        "stuck_value": args.stuck_value,
        "dataflow": "WS",
        "sa_rows": R,
        "sa_cols": C,
        "operator_entries": operator_entries,
        "all_entries": all_entries,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {args.output}")
    print(f"  operator_entries: {len(operator_entries):,}")
    print(f"  all_entries:      {len(all_entries):,}")

    # Top-10 preview
    print("\nTop-10 (all_entries, by joint_max):")
    print(f"{'rank':>4s} {'type':>6s} {'bit':>4s} {'pe':>8s} {'mean':>12s} {'max':>12s}")
    print("-" * 56)
    for i, entry in enumerate(all_entries[:10]):
        pe = f"({entry['pe_row']},{entry['pe_col']})"
        print(f"{i+1:4d} {entry['type']:>6s} {entry['bit']:4d} {pe:>8s} "
              f"{entry['joint_mean']:12.4f} {entry['joint_max']:12.4f}")


if __name__ == "__main__":
    main()

"""
Build joint bit-position severity ranking for Qwen3-8B fault injection.

Combines per-operator bit severity tables (Step 1) with dataflow-aware
PE position coverage (computed inline) into a unified joint severity:

    S_joint(type, stuck_value, bit, operator, pe_row, pe_col) =
        S_bit(type, stuck_value, bit, operator)  ×  C_pos(type, operator, pe_row, pe_col)

Outputs:
  - machine-readable JSON with operator-specific and all-mode entries
  - sorted by joint severity for downstream representative sampling

Usage:
  uv run python projects/qwen3-8b/build_joint_severity.py \
    --fault-type stuck-at --stuck-value 1

The script expects per-operator severity tables to exist in config/.
Run a per-operator build_severity_table.py first.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config")

# Per-type severity field mapping: (source_name, stuck_at_0_field, stuck_at_1_field)
TYPE_MAPPING = {
    "input":  ("activation_input", "sa0_conditional", "sa1_conditional"),
    "weight": ("weight",            "sa0_conditional", "sa1_conditional"),
}

# ---- Qwen3-8B operator groups and dimensions ----
# K = in_features (reduction dim), N = out_features (output dim)
# Qwen3-8B: hidden=4096, intermediate=12288, 36 layers

OPERATOR_GROUPS = ["attention", "intermediate", "output"]

LAYER_DIMS = {
    "attention":    {"K": 4096,  "N": 4096,  "desc": "Q/K/V/O dense (q_proj/k_proj/v_proj/o_proj)"},
    "intermediate": {"K": 4096,  "N": 12288, "desc": "gate_proj/up_proj dense"},
    "output":       {"K": 12288, "N": 4096,  "desc": "down_proj dense"},
}

# Modules per layer: attention = 4 (q/k/v/o), intermediate = 2 (gate/up), output = 1 (down)
MODULES_PER_LAYER = {
    "attention":    4,
    "intermediate": 2,
    "output":       1,
}

NUM_HIDDEN_LAYERS = 36
DEFAULT_SA_ROWS = 256
DEFAULT_SA_COLS = 256

# ---- formula parameters ----
FORMULA_CONFIG = {}  # populated in main()

import math


def _apply_formula(raw_sev: float, bit: int, config: dict) -> float:
    """Apply formula-specific transform to raw severity value."""
    name = config.get("name", "v3")
    sign_factor = config.get("sign_factor", 0.08)
    sat_point = config.get("sat_point", 10.0)

    sev = raw_sev
    if bit == 31:
        sev *= sign_factor

    if name == "v3":
        sev = min(sev, sat_point)
    elif name == "v4":
        sev = math.log1p(sev)

    return sev


# ---------------------------------------------------------------------------
# Position coverage computation (WS dataflow)
# ---------------------------------------------------------------------------

def _mapped_count(index: int, total: int, dim: int) -> int:
    """Number of elements mapped to PE row/col `index` under round-robin."""
    return total // dim + (1 if index < total % dim else 0)


def _compute_weight_coverage(K: int, N: int, R: int, C: int) -> np.ndarray:
    """WS weight: W[k,j] -> PE(k%R, j%C). Number of W elements per PE."""
    row_counts = np.array([_mapped_count(r, K, R) for r in range(R)], dtype=np.float64)
    col_counts = np.array([_mapped_count(c, N, C) for c in range(C)], dtype=np.float64)
    return np.outer(row_counts, col_counts)


def _compute_input_coverage(K: int, N: int, R: int, C: int) -> np.ndarray:
    """WS input: corrupts X[i,k] at PE(r,c) -> propagates rightward through cols c..C-1."""
    row_counts = np.array([_mapped_count(r, K, R) for r in range(R)], dtype=np.float64)
    col_counts = np.array([_mapped_count(c, N, C) for c in range(C)], dtype=np.float64)
    affected_cols = np.array([
        sum(col_counts[j] for j in range(c, C)) for c in range(C)
    ], dtype=np.float64)
    return np.outer(row_counts, affected_cols)


def _compute_psum_coverage(K: int, N: int, R: int, C: int) -> np.ndarray:
    """WS psum: same coverage pattern as weight."""
    return _compute_weight_coverage(K, N, R, C)


def _normalize_matrix(mat: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    vmin, vmax = mat.min(), mat.max()
    if vmax - vmin < 1e-12:
        return np.ones_like(mat) if vmax > 1e-12 else np.zeros_like(mat)
    return (mat - vmin) / (vmax - vmin)


def build_all_coverage(R: int, C: int) -> dict:
    """Build coverage matrices for all Qwen3-8B operators under WS dataflow."""
    result = {
        "dataflow": "WS",
        "sa_rows": R,
        "sa_cols": C,
        "layers": {},
        "severity": {},
    }

    for op_name, dims in LAYER_DIMS.items():
        K, N = dims["K"], dims["N"]
        result["layers"][op_name] = {"K": K, "N": N, "desc": dims["desc"]}

        weight_mat = _compute_weight_coverage(K, N, R, C)
        input_mat = _compute_input_coverage(K, N, R, C)
        psum_mat = _compute_psum_coverage(K, N, R, C)

        weight_norm = _normalize_matrix(weight_mat)
        input_norm = _normalize_matrix(input_mat)
        psum_norm = _normalize_matrix(psum_mat)

        is_weight_uniform = bool(np.allclose(weight_mat, weight_mat[0, 0]))
        is_psum_uniform = bool(np.allclose(psum_mat, psum_mat[0, 0]))

        result["severity"][op_name] = {
            "weight": {
                "raw_min": float(weight_mat.min()),
                "raw_max": float(weight_mat.max()),
                "uniform": is_weight_uniform,
                "raw_matrix": weight_mat.tolist(),
                "normalized_matrix": weight_norm.tolist(),
            },
            "input": {
                "raw_min": float(input_mat.min()),
                "raw_max": float(input_mat.max()),
                "uniform": False,
                "raw_matrix": input_mat.tolist(),
                "normalized_matrix": input_norm.tolist(),
            },
            "psum": {
                "raw_min": float(psum_mat.min()),
                "raw_max": float(psum_mat.max()),
                "uniform": is_psum_uniform,
                "raw_matrix": psum_mat.tolist(),
                "normalized_matrix": psum_norm.tolist(),
            },
        }

    return result


# ---------------------------------------------------------------------------
# Bit severity table loading
# ---------------------------------------------------------------------------

def _table_filename(source: str, operator: str, precision: str = "fp32") -> str:
    return os.path.join(CONFIG_DIR, f"severity_table_{source}_{precision}_{operator}.json")


def load_bit_severity(exp_type: str, operator: str, bit: int, stuck_value: int = 1, precision: str = "fp32") -> float:
    """Return raw conditional bit severity with theoretical floor for exponent bits."""
    source, field_sa0, field_sa1 = TYPE_MAPPING[exp_type]
    field = field_sa1 if stuck_value == 1 else field_sa0
    path = _table_filename(source, operator, precision)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing severity table: {path}\n"
            f"Run a per-operator build_severity_table.py for {operator} first."
        )
    with open(path) as f:
        table = json.load(f)

    cal_val = 0.0
    eff_rate = 0.0
    for entry in table["table"]:
        if entry["bit"] == bit:
            cal_val = float(entry[field])
            eff_key = field.replace("_conditional", "_effective_rate")
            eff_rate = float(entry.get(eff_key, 0.0))
            break

    # Theoretical floor for exponent bits with insufficient calibration samples
    theory = _theoretical_exponent_severity(bit, precision)
    if theory > 0 and cal_val < theory * 0.01:
        proxy_eff = max(eff_rate, 0.001)
        cal_val = max(cal_val, theory * proxy_eff)

    return cal_val


def _theoretical_exponent_severity(bit: int, precision: str = "fp32") -> float:
    """IEEE 754 theoretical conditional severity for flipping an exponent bit."""
    if precision == "fp32":
        if not (23 <= bit <= 30):
            return 0.0
        offset = 23
    elif precision == "bf16":
        if not (7 <= bit <= 14):
            return 0.0
        offset = 7
    else:
        return 0.0
    return math.log(2) * (2 ** (bit - offset))


def load_position_coverage(pos_data: dict, exp_type: str, operator: str) -> np.ndarray:
    """Return the raw coverage matrix [R, C] from pre-computed position data."""
    return np.array(pos_data["severity"][operator][exp_type]["raw_matrix"])


# ---------------------------------------------------------------------------
# Joint computation
# ---------------------------------------------------------------------------

def compute_joint(
    exp_type: str, stuck_value: int, bit: int, operator: str, pe_row: int, pe_col: int,
    bit_sev: float, coverage: float,
) -> dict:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build joint bit-position severity ranking for Qwen3-8B"
    )
    parser.add_argument("--fault-type", type=str, choices=["stuck-at", "bitflip"],
                        default="stuck-at", help="Fault type")
    parser.add_argument("--stuck-value", type=int, choices=[0, 1],
                        default=1, help="Stuck-at value (ignored for bitflip)")
    parser.add_argument("--types", type=str, nargs="+",
                        choices=["input", "weight", "psum"],
                        default=["input", "weight"],
                        help="Experiment types to include")
    parser.add_argument("--formula", type=str, choices=["v3", "v4"], default="v4",
                        help="Severity formula: v3=cap+saturation+signFactor, v4=log1p (better proportionality)")
    parser.add_argument("--sa-rows", type=int, default=DEFAULT_SA_ROWS,
                        help=f"SA rows (default: {DEFAULT_SA_ROWS})")
    parser.add_argument("--sa-cols", type=int, default=DEFAULT_SA_COLS,
                        help=f"SA cols (default: {DEFAULT_SA_COLS})")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32",
                        help="Precision of the severity tables to load (default: fp32)")
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

    R, C = args.sa_rows, args.sa_cols

    # Compute position coverage inline (no external file needed)
    print(f"Computing position coverage: {R}x{C} WS dataflow for {len(LAYER_DIMS)} operators")
    pos_data = build_all_coverage(R, C)

    operator_entries: list[dict] = []
    all_entries: list[dict] = []

    # Pre-load all bit severities to avoid repeated file I/O
    bit_sev_cache: dict[tuple, float] = {}
    for exp_type in args.types:
        for op in OPERATOR_GROUPS:
            try:
                # Probe the table once to check existence
                load_bit_severity(exp_type, op, 0, args.stuck_value, args.precision)
                for bit in range(32):
                    bit_sev_cache[(exp_type, op, bit)] = load_bit_severity(
                        exp_type, op, bit, args.stuck_value, args.precision
                    )
            except FileNotFoundError:
                print(f"[WARNING] Skipping {exp_type}/{op} — table missing")
                for bit in range(32):
                    bit_sev_cache[(exp_type, op, bit)] = 0.0

    # Pre-load all coverage matrices
    cov_cache: dict[tuple, np.ndarray] = {}
    for exp_type in args.types:
        for op in OPERATOR_GROUPS:
            cov_cache[(exp_type, op)] = load_position_coverage(pos_data, exp_type, op)

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
                bit_sev = _apply_formula(bit_sev, bit, FORMULA_CONFIG)
                if bit_sev == 0.0:
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
    op_weights = {}
    for op in OPERATOR_GROUPS:
        dims = LAYER_DIMS[op]
        K, N = dims["K"], dims["N"]
        num_modules = MODULES_PER_LAYER[op] * NUM_HIDDEN_LAYERS
        op_weights[op] = num_modules * K * N

    for entry in operator_entries:
        if entry.get("all_zero"):
            continue
        key = (entry["type"], entry["bit"], entry["pe_row"], entry["pe_col"])
        all_groups[key].append((entry["operator"], entry["joint_severity"]))

    for (exp_type, bit, pe_row, pe_col), op_sevs in all_groups.items():
        sev_values = [s for _, s in op_sevs]
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
        "model": "Qwen/Qwen3-8B",
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
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

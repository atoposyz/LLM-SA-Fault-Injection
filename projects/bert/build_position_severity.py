"""
PE Position Coverage Prediction for WS Dataflow.

Computes per-PE coverage matrices for the SA fault injector:
  - weight: mapped_count(r,K,R) * mapped_count(c,N,C)   [uniform per PE]
  - input:  sum_{j=c}^{C-1} mapped_count(j,N,C)          [column-dependent, rightward]
  - psum:   mapped_count(r,K,R) * mapped_count(c,N,C)    [accum chain * output cols]

Model: boltuix/bert-emotion | SA: 32x32 (default, matches SAInjectProPlus32.py)
"""

import argparse
import json
import os

import numpy as np

DEFAULT_SA_ROWS = 32
DEFAULT_SA_COLS = 32

LAYER_DIMS = {
    "attention":    {"K": 256,  "N": 256,  "desc": "Q/K/V/attention-output dense"},
    "intermediate": {"K": 256,  "N": 1024, "desc": "intermediate.dense"},
    "output":       {"K": 1024, "N": 256,  "desc": "output.dense"},
}


def mapped_count(index: int, total: int, dim: int) -> int:
    """Number of elements mapped to PE row/col `index` when `total` items
    are distributed over `dim` PEs in round-robin fashion.

    Equivalent to ceil(total/dim) for most positions, but exact for
    remainder (non-divisible) boundaries.
    """
    return total // dim + (1 if index < total % dim else 0)


def compute_weight_coverage(K: int, N: int, R: int, C: int) -> np.ndarray:
    """WS weight: W[k,j] -> PE(k%R, j%C).

    Number of weight elements mapped to PE(r,c).
    """
    row_counts = np.array([mapped_count(r, K, R) for r in range(R)], dtype=np.float64)
    col_counts = np.array([mapped_count(c, N, C) for c in range(C)], dtype=np.float64)
    return np.outer(row_counts, col_counts)  # [R, C]


def compute_input_coverage(K: int, N: int, R: int, C: int) -> np.ndarray:
    """WS input: fault at PE(r,c) corrupts X[i,k] where k%R==r.

    Corrupted input propagates rightward through PE columns c..C-1,
    contributing to output Y[i,j] where j%C >= c.

    Coverage = sum of mapped_count(j,N,C) for j = c..C-1.
    Row-independent (but included for consistency with the SA grid).
    """
    row_counts = np.array([mapped_count(r, K, R) for r in range(R)], dtype=np.float64)
    col_counts = np.array([mapped_count(c, N, C) for c in range(C)], dtype=np.float64)

    # Rightward propagation: affected output columns = sum over c..C-1
    affected_cols = np.array([
        sum(col_counts[j] for j in range(c, C)) for c in range(C)
    ], dtype=np.float64)

    return np.outer(row_counts, affected_cols)  # [R, C]


def compute_psum_coverage(K: int, N: int, R: int, C: int) -> np.ndarray:
    """WS psum: fault at PE(r,c) corrupts accumulated partial sum.

    Coverage = accumulation_chain_length(r) * mapped_output_columns(c).
    accumulation_chain_length = mapped_count(r, K, R) — number of times
    this PE participates in reduction along the K dimension.
    """
    return compute_weight_coverage(K, N, R, C)  # Same formula for WS


def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    vmin, vmax = mat.min(), mat.max()
    if vmax - vmin < 1e-12:
        return np.ones_like(mat) if vmax > 1e-12 else np.zeros_like(mat)
    return (mat - vmin) / (vmax - vmin)


def build_all_coverage(R: int, C: int) -> dict:
    """Build coverage tables for all layer types and modes."""
    result = {
        "dataflow": "WS",
        "sa_rows": R,
        "sa_cols": C,
        "layers": {},
        "severity": {},
    }

    for layer_name, dims in LAYER_DIMS.items():
        K, N = dims["K"], dims["N"]
        result["layers"][layer_name] = {"K": K, "N": N, "desc": dims["desc"]}

        weight_mat = compute_weight_coverage(K, N, R, C)
        input_mat = compute_input_coverage(K, N, R, C)
        psum_mat = compute_psum_coverage(K, N, R, C)

        weight_norm = normalize_matrix(weight_mat)
        input_norm = normalize_matrix(input_mat)
        psum_norm = normalize_matrix(psum_mat)

        is_weight_uniform = np.allclose(weight_mat, weight_mat[0, 0])
        is_psum_uniform = np.allclose(psum_mat, psum_mat[0, 0])

        result["severity"][layer_name] = {
            "weight": {
                "raw_min": float(weight_mat.min()),
                "raw_max": float(weight_mat.max()),
                "uniform": bool(is_weight_uniform),
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
                "uniform": bool(is_psum_uniform),
                "raw_matrix": psum_mat.tolist(),
                "normalized_matrix": psum_norm.tolist(),
            },
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build PE position coverage tables for WS dataflow"
    )
    parser.add_argument("--sa-rows", type=int, default=DEFAULT_SA_ROWS,
                        help=f"SA rows (default: {DEFAULT_SA_ROWS})")
    parser.add_argument("--sa-cols", type=int, default=DEFAULT_SA_COLS,
                        help=f"SA cols (default: {DEFAULT_SA_COLS})")
    args = parser.parse_args()

    R, C = args.sa_rows, args.sa_cols
    print(f"Building position coverage: {R}x{C} WS dataflow")

    data = build_all_coverage(R, C)

    output_dir = os.path.join(os.path.dirname(__file__), "config")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "position_severity_ws.json")

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {out_path}")

    # Print summary
    print("\nCoverage summary:")
    for layer_name, sev in data["severity"].items():
        K = data["layers"][layer_name]["K"]
        N = data["layers"][layer_name]["N"]
        print(f"\n--- {layer_name} (K={K}, N={N}) ---")
        for mode in ["weight", "input", "psum"]:
            info = sev[mode]
            uniform_tag = " [UNIFORM]" if info["uniform"] else ""
            print(f"  {mode:8s}: raw=[{info['raw_min']:.0f}, {info['raw_max']:.0f}]{uniform_tag}")

    # Verification: print input coverage gradient for a sample layer
    print("\nInput coverage gradient (attention, first row, first 8 cols):")
    attn_input = np.array(data["severity"]["attention"]["input"]["raw_matrix"])
    if attn_input.ndim == 2:
        vals = attn_input[0, :8].tolist()
        print(f"  cols 0..7: {[f'{v:.0f}' for v in vals]}")


if __name__ == "__main__":
    main()

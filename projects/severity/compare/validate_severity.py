"""
Validate joint severity ranking against real PE-position measurements.

Loads ground truth from combined_top1_position_long.csv, joins with
joint severity predictions by (type, bit, pe_row, pe_col), and computes
ranking metrics (Spearman rho, Kendall tau, Top-K overlap, high-risk recall).

Usage:
  uv run python projects/bert/validate_severity.py \
    --ground-truth projects/bert/result/combined_top1_position_long.csv \
    --joint-severity projects/bert/config/joint_severity_stuck1_ws.json

The script works without PyTorch — only numpy and scipy are needed for metrics.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from collections import defaultdict


def load_ground_truth(path: str) -> list[dict]:
    """Load and parse the combined position CSV."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            fpe = r["FPE"].split(",")
            if len(fpe) != 2:
                continue
            pe_row, pe_col = int(fpe[0].strip()), int(fpe[1].strip())
            rows.append({
                "type": r["type"],
                "bit": int(r["bit"]),
                "pe_row": pe_row,
                "pe_col": pe_col,
                "top1": float(r["top1"]),
                "samples": int(r["samples"]),
            })
    print(f"[INFO] Loaded {len(rows)} ground-truth rows")
    return rows


def load_joint_severity(path: str) -> dict[tuple, dict]:
    """Load joint severity JSON and build a lookup by (type, bit, pe_row, pe_col)."""
    with open(path) as f:
        data = json.load(f)
    lookup = {}
    for entry in data.get("all_entries", []):
        key = (entry["type"], entry["bit"], entry["pe_row"], entry["pe_col"])
        lookup[key] = entry
    print(f"[INFO] Loaded {len(lookup)} joint severity all_entries")
    return lookup


def load_position_coverage_aggregated(
    pos_path: str, operators: list[str]
) -> dict[tuple, float]:
    """Load position_severity_ws.json and compute aggregated coverage.

    Returns a dict keyed by (type, pe_row, pe_col) -> total coverage
    (summed across all operators for the 'all' layer type).
    """
    with open(pos_path) as f:
        data = json.load(f)

    agg = defaultdict(float)
    for op in operators:
        if op not in data["severity"]:
            continue
        for exp_type in ["input", "weight"]:
            if exp_type not in data["severity"][op]:
                continue
            raw = np.array(data["severity"][op][exp_type]["raw_matrix"])
            for r in range(raw.shape[0]):
                for c in range(raw.shape[1]):
                    agg[(exp_type, r, c)] += float(raw[r, c])
    print(f"[INFO] Loaded aggregated position coverage: {len(agg)} entries")
    return dict(agg)


def compute_metrics(
    severity_scores: np.ndarray,
    measured_drops: np.ndarray,
    top_k: int = 10,
    high_drop_threshold: float = 0.05,
) -> dict:
    """Compute ranking metrics between predicted severity and measured accuracy drop.

    Args:
        severity_scores: predicted joint severity (higher = worse)
        measured_drops: observed accuracy drop (higher = worse)
        top_k: size of top-K for overlap metric
        high_drop_threshold: minimum drop to count as "high risk"
    """
    n = len(severity_scores)
    if n < 3:
        return {"spearman": None, "kendall": None, "top_k_overlap": None,
                "high_risk_recall": None, "n": n}

    # Spearman rho
    from scipy.stats import spearmanr, kendalltau
    rho, p_spearman = spearmanr(severity_scores, measured_drops)
    tau, p_kendall = kendalltau(severity_scores, measured_drops)

    # Top-K overlap: how many top-K by severity are also top-K by measured drop
    top_k_sev = set(np.argsort(-severity_scores)[:top_k])
    top_k_drop = set(np.argsort(-measured_drops)[:top_k])
    overlap = len(top_k_sev & top_k_drop) / top_k

    # High-risk recall: among truly high-drop entries, how many are in top-K by severity
    high_risk_indices = set(np.where(measured_drops >= high_drop_threshold)[0])
    if len(high_risk_indices) > 0:
        recall = len(high_risk_indices & top_k_sev) / len(high_risk_indices)
    else:
        recall = None

    return {
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p_spearman), 6),
        "kendall_tau": round(float(tau), 4),
        "kendall_p": round(float(p_kendall), 6),
        "top_k_overlap": round(overlap, 4),
        "high_risk_recall": round(recall, 4) if recall is not None else None,
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate joint severity ranking against real FPE measurements"
    )
    parser.add_argument(
        "--ground-truth", type=str,
        default=os.path.join(os.path.dirname(__file__), "result", "combined_top1_position_long.csv"),
        help="Path to combined_top1_position_long.csv"
    )
    parser.add_argument(
        "--joint-severity", type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "joint_severity_stuck1_ws.json"),
        help="Path to joint severity JSON"
    )
    parser.add_argument(
        "--baseline-top1", type=float, default=None,
        help="No-fault baseline top-1 accuracy. If not set, uses max top1 in the data."
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="K for top-K overlap metric"
    )
    parser.add_argument(
        "--high-drop-threshold", type=float, default=0.05,
        help="Minimum accuracy drop to classify as high-risk"
    )
    parser.add_argument(
        "--per-bit-diagnostics", action="store_true", default=True,
        help="Print per-bit position diagnostics"
    )
    args = parser.parse_args()

    # Load data
    gt = load_ground_truth(args.ground_truth)
    joint = load_joint_severity(args.joint_severity)
    pos_cov_agg = load_position_coverage_aggregated(
        os.path.join(os.path.dirname(__file__), "config", "position_severity_ws.json"),
        ["attention", "intermediate", "output"],
    )

    # Determine baseline
    if args.baseline_top1 is not None:
        baseline = args.baseline_top1
        print(f"[INFO] Using provided baseline top1: {baseline:.4f}")
    else:
        baseline = max(r["top1"] for r in gt)
        print(f"[INFO] Estimated baseline top1 from data: {baseline:.4f}")

    # Join ground truth with predictions
    matched: list[dict] = []
    missed = 0
    for r in gt:
        key = (r["type"], r["bit"], r["pe_row"], r["pe_col"])
        pred = joint.get(key)
        if pred is None:
            missed += 1
            continue
        r["acc_drop"] = baseline - r["top1"]
        r["joint_mean"] = pred["joint_mean"]
        r["joint_max"] = pred["joint_max"]
        r["joint_weighted_mean"] = pred.get("joint_weighted_mean", pred["joint_mean"])
        matched.append(r)
    print(f"[INFO] Matched {len(matched)} rows, missed {missed}")

    # Also load bit-only severity for comparison
    # Build bit-only: average severity across PE positions for each bit
    bit_only: dict[tuple, float] = defaultdict(list)
    pos_only: dict[tuple, float] = defaultdict(list)
    for r in matched:
        bkey = (r["type"], r["bit"])
        bit_only[bkey].append(r["joint_mean"])
        pkey = (r["type"], r["pe_row"], r["pe_col"])
        pos_only[pkey].append(r["acc_drop"])

    bit_only_mean = {k: np.mean(v) for k, v in bit_only.items()}
    bit_only_vals = np.array([bit_only_mean[(r["type"], r["bit"])] for r in matched])

    # Assign position-only coverage: use aggregated coverage across operators
    pos_vals = []
    for r in matched:
        pkey = (r["type"], r["pe_row"], r["pe_col"])
        cov = pos_cov_agg.get(pkey, 1.0)
        pos_vals.append(cov)
    pos_vals = np.array(pos_vals)

    joint_vals_mean = np.array([r["joint_mean"] for r in matched])
    joint_vals_max = np.array([r["joint_max"] for r in matched])
    acc_drops = np.array([r["acc_drop"] for r in matched])

    # ======================================================
    # Overall comparison (all data pooled)
    # ======================================================
    print("\n" + "=" * 70)
    print("PREDICTOR COMPARISON (all rows)")
    print("=" * 70)

    predictors = {
        "bit_only": bit_only_vals,
        "position_only": pos_vals,
        "joint_mean": joint_vals_mean,
        "joint_max": joint_vals_max,
    }

    print(f"{'Predictor':<18s} {'Spearman':>10s} {'Kendall':>10s} {'Top-{0}'.format(args.top_k):>10s} {'HR-Recall':>10s} {'n':>6s}")
    print("-" * 70)
    results_all = {}
    for name, vals in predictors.items():
        metrics = compute_metrics(vals, acc_drops, top_k=args.top_k,
                                  high_drop_threshold=args.high_drop_threshold)
        results_all[name] = metrics
        print(f"{name:<18s} {str(metrics['spearman_rho']):>10s} "
              f"{str(metrics['kendall_tau']):>10s} "
              f"{str(metrics['top_k_overlap']):>10s} "
              f"{str(metrics['high_risk_recall'] or 'N/A'):>10s} "
              f"{metrics['n']:>6d}")

    # ======================================================
    # Per-type comparison
    # ======================================================
    for exp_type in ["input", "weight"]:
        idx = [i for i, r in enumerate(matched) if r["type"] == exp_type]
        if len(idx) < 3:
            print(f"\n[WARNING] Too few rows for type={exp_type} (n={len(idx)}), skipping")
            continue

        print(f"\n" + "=" * 70)
        print(f"PREDICTOR COMPARISON (type={exp_type})")
        print("=" * 70)
        print(f"{'Predictor':<18s} {'Spearman':>10s} {'Kendall':>10s} {'Top-{0}'.format(args.top_k):>10s} {'HR-Recall':>10s} {'n':>6s}")
        print("-" * 70)

        type_drops = acc_drops[idx]
        for name, vals in predictors.items():
            type_vals = vals[idx]
            metrics = compute_metrics(type_vals, type_drops, top_k=args.top_k,
                                      high_drop_threshold=args.high_drop_threshold)
            print(f"{name:<18s} {str(metrics['spearman_rho']):>10s} "
                  f"{str(metrics['kendall_tau']):>10s} "
                  f"{str(metrics['top_k_overlap']):>10s} "
                  f"{str(metrics['high_risk_recall'] or 'N/A'):>10s} "
                  f"{metrics['n']:>6d}")

    # ======================================================
    # Per-bit position diagnostics
    # ======================================================
    if args.per_bit_diagnostics:
        from scipy.stats import pearsonr

        print("\n" + "=" * 70)
        print("PER-BIT POSITION DIAGNOSTICS")
        print("=" * 70)
        print(f"{'type':>6s} {'bit':>4s} {'n':>5s} {'r(position_cov, acc_drop)':>28s} "
              f"{'top1_min':>9s} {'top1_max':>9s} {'drop_range':>12s}")
        print("-" * 85)

        for exp_type in ["input", "weight"]:
            type_matched = [r for r in matched if r["type"] == exp_type]
            bits = sorted(set(r["bit"] for r in type_matched))
            for bit in bits:
                bit_rows = [r for r in type_matched if r["bit"] == bit]
                if len(bit_rows) < 5:
                    continue
                pos_covs = []
                drops = []
                for r in bit_rows:
                    pkey = (r["type"], r["pe_row"], r["pe_col"])
                    cov = pos_cov_agg.get(pkey, 1.0)
                    if cov > 0:
                        pos_covs.append(cov)
                        drops.append(r["acc_drop"])
                if len(pos_covs) < 5:
                    continue
                r_val, _ = pearsonr(pos_covs, drops)
                top1_vals = [r["top1"] for r in bit_rows]
                print(f"{exp_type:>6s} {bit:4d} {len(bit_rows):5d} "
                      f"{r_val:28.4f} "
                      f"{min(top1_vals):9.4f} {max(top1_vals):9.4f} "
                      f"{max(top1_vals)-min(top1_vals):12.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

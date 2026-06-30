"""
Compare joint severity predictions against per-PE accuracy measurements.

Loads pe_accuracy_*.csv (ground truth) and joint_severity_*.json (predictions),
computes Spearman rho / Kendall tau at per-(type, bit, pe_col) level.

This validates whether v5's reach multiplier improves PE-level ranking quality.

Usage:
  uv run python projects/bert/compare_pe_ranking.py
"""

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, kendalltau

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config")
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")

CSV_MAP = {
    ("input", 0): "pe_accuracy_input_stuck0.csv",
    ("input", 1): "pe_accuracy_input_stuck1.csv",
    ("weight", 0): "pe_accuracy_weight_stuck0.csv",
    ("weight", 1): "pe_accuracy_weight_stuck1.csv",
}


def _parse_formula(path: str) -> str:
    basename = os.path.basename(path).replace(".json", "")
    parts = basename.split("_")
    if len(parts) > 4 and parts[4] not in ("ws",):
        return "_".join(parts[4:])
    return "v3"


def _parse_stuck(path: str) -> int:
    basename = os.path.basename(path)
    for part in basename.split("_"):
        if part.startswith("stuck"):
            return int(part.replace("stuck", ""))
    return 1


def load_pe_accuracy(exp_type: str, stuck_value: int) -> dict[tuple, float]:
    """Return {(bit, pe_col): mean_acc_drop} from per-PE CSV."""
    key = (exp_type, stuck_value)
    if key not in CSV_MAP:
        return {}
    path = os.path.join(RESULT_DIR, CSV_MAP[key])
    if not os.path.exists(path):
        print(f"  [WARNING] Missing CSV: {path}")
        return {}

    by_key = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["mode"] != exp_type:
                continue
            bit = int(row["bit"])
            pe_col = int(row["pe_col"])
            drop = float(row["acc_drop"])
            by_key[(bit, pe_col)].append(drop)

    return {k: float(np.mean(v)) for k, v in by_key.items()}


def load_joint_by_bit_col(path: str, exp_type: str) -> dict[tuple, float]:
    """Load joint severity, aggregate to per-(bit, pe_col).

    Takes mean across rows (all rows in a column have same reach) and
    across operators for all_entries, or mean of operator_entries.
    """
    with open(path) as f:
        data = json.load(f)

    by_key = defaultdict(list)

    # Use operator_entries for per-operator granularity
    entries = data.get("operator_entries", [])
    for entry in entries:
        if entry.get("type") != exp_type:
            continue
        bit = entry["bit"]
        pe_col = entry["pe_col"]
        js = entry.get("joint_severity", 0)
        if js == 0:
            continue
        by_key[(bit, pe_col)].append(js)

    return {k: float(np.mean(v)) for k, v in by_key.items()}


def compute_correlation(
    sev_pred: dict[tuple, float],
    acc_true: dict[tuple, float],
) -> tuple[float, float, int]:
    """Spearman rho, Kendall tau over common (bit, pe_col) keys."""
    common = sorted(set(sev_pred.keys()) & set(acc_true.keys()))
    if len(common) < 5:
        return 0.0, 0.0, 0

    sev_vals = [sev_pred[k] for k in common]
    acc_vals = [acc_true[k] for k in common]

    rho, _ = spearmanr(sev_vals, acc_vals)
    tau, _ = kendalltau(sev_vals, acc_vals)
    return round(rho, 4), round(tau, 4), len(common)


def main():
    # Discover joint severity tables
    pattern = os.path.join(CONFIG_DIR, "joint_severity_stuck*_ws*.json")
    paths = sorted(glob.glob(pattern))

    print("=" * 80)
    print("PE-Level Ranking Comparison: Joint Severity vs Measured acc_drop")
    print(f"  Joint severity tables: {len(paths)}")
    print(f"  PE accuracy CSVs: {list(CSV_MAP.values())}")
    print("=" * 80)

    all_results: dict[str, list[dict]] = defaultdict(list)

    for csv_key, csv_name in CSV_MAP.items():
        exp_type, stuck_value = csv_key
        acc_data = load_pe_accuracy(exp_type, stuck_value)
        if not acc_data:
            print(f"\n[SKIP] {exp_type} sa{stuck_value}: no CSV data")
            continue

        print(f"\n{'='*60}")
        print(f"[{exp_type} sa{stuck_value}]  PE measurements: {len(acc_data)} (bit, col) entries")
        print(f"{'='*60}")

        for path in paths:
            if _parse_stuck(path) != stuck_value:
                continue
            formula = _parse_formula(path)
            sev_data = load_joint_by_bit_col(path, exp_type)
            if not sev_data:
                continue

            rho, tau, n = compute_correlation(sev_data, acc_data)
            tag = f"{exp_type}_sa{stuck_value}"
            all_results[tag].append({
                "formula": formula,
                "rho": rho,
                "tau": tau,
                "n": n,
            })
            print(f"  {formula:>8s}: rho={rho:+.4f}  tau={tau:+.4f}  n={n}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary: Best formula per category (by Spearman rho)")
    print("=" * 80)
    for tag, rows in sorted(all_results.items()):
        best = max(rows, key=lambda r: r["rho"])
        rows_sorted = sorted(rows, key=lambda r: r["rho"], reverse=True)
        print(f"\n  {tag}:")
        for r in rows_sorted:
            marker = " <-- best" if r == best else ""
            print(f"    {r['formula']:>8s}  rho={r['rho']:+.4f}  tau={r['tau']:+.4f}  n={r['n']}{marker}")


if __name__ == "__main__":
    main()

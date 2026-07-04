"""
Compare joint severity formula rankings against measured accuracy drop.

Aggregates per-(type,bit,pe) joint_severity down to per-bit scores using
mean / max / top-k mean, then computes Spearman rho and Kendall tau against
CSV ground-truth accuracy_drop.

Usage:
  uv run python projects/bert/compare_formula_ranking.py
  uv run python projects/bert/compare_formula_ranking.py --top-k 20
  uv run python projects/bert/compare_formula_ranking.py --top-k 15%
"""

import argparse
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

# Maps (type, stuck_value) → CSV filename
CSV_MAP = {
    ("input", 0): "accuracy_drop_perbit_input_stuck_0.csv",
    ("input", 1): "accuracy_drop_perbit_input_stuck_1.csv",
    ("weight", 0): "accuracy_drop_perbit_weight_stuck_0.csv",
    ("weight", 1): "accuracy_drop_perbit_weight_stuck_1.csv",
}


def discover_tables() -> dict[str, list[str]]:
    """Find all joint severity JSON files, grouped by (type, stuck_value).

    Returns {(stub_key, stuck_value): [path, ...]} where stub_key is e.g. 'joint_severity_stuck1_ws'.
    """
    pattern = os.path.join(CONFIG_DIR, "joint_severity_stuck*_ws*.json")
    paths = sorted(glob.glob(pattern))

    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for p in paths:
        basename = os.path.basename(p)
        # Parse: joint_severity_stuck{val}_ws{_formula}.json
        name = basename.replace(".json", "")
        parts = name.split("_")  # ['joint', 'severity', 'stuck0', 'ws'] or ['joint', 'severity', 'stuck1', 'ws', 'v4']
        stuck_part = parts[2]  # 'stuck0' or 'stuck1'
        stuck_value = int(stuck_part.replace("stuck", ""))

        # Derive stub key: everything before formula suffix
        # e.g. 'joint_severity_stuck1_ws'
        if len(parts) > 4:
            # has formula suffix
            stub = "_".join(parts[:4])
        else:
            stub = "_".join(parts)

        groups[(stub, stuck_value)].append(p)

    return dict(groups)


def parse_formula_name(path: str) -> str:
    """Extract formula name from joint severity file path.

    joint_severity_stuck1_ws.json        → v3 (default when no suffix)
    joint_severity_stuck1_ws_v4.json     → v4
    joint_severity_stuck1_ws_v5_lin.json → v5_lin
    joint_severity_stuck1_ws_v5_exp.json → v5_exp
    """
    basename = os.path.basename(path).replace(".json", "")
    parts = basename.split("_")
    # parts: ['joint', 'severity', 'stuckX', 'ws'] or ['joint', 'severity', 'stuckX', 'ws', 'v4']
    if len(parts) > 4:
        formula = "_".join(parts[4:])  # 'v4' or 'v5_lin' or 'v5_exp'
        return formula
    return "v3"


def load_accuracy_drops(exp_type: str, stuck_value: int) -> dict[int, float]:
    """Load per-bit accuracy_drop from CSV. Returns {bit: acc_drop}."""
    csv_key = (exp_type, stuck_value)
    if csv_key not in CSV_MAP:
        print(f"  [WARNING] No CSV mapping for type={exp_type} stuck={stuck_value}")
        return {}
    csv_path = os.path.join(RESULT_DIR, CSV_MAP[csv_key])
    if not os.path.exists(csv_path):
        print(f"  [WARNING] CSV not found: {csv_path}")
        return {}
    result = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        # Try exact column name first, then fall back to suffix match
        if "acc_drop" in reader.fieldnames:
            acc_drop_col = "acc_drop"
        else:
            acc_drop_col = [c for c in reader.fieldnames if c.endswith("_acc_drop")][0]
        for row in reader:
            bit = int(row["bit"])
            result[bit] = float(row[acc_drop_col])
    return result


def aggregate_per_bit(
    all_entries: list[dict],
    exp_type: str,
    method: str,
    top_k: int,
) -> dict[int, float]:
    """Aggregate joint_severity from all_entries down to per-bit scores.

    Filters out entries with joint_severity == 0 and wrong type.

    method: 'mean', 'max', 'topk'
    """
    # Group by bit
    bit_sevs: dict[int, list[float]] = defaultdict(list)
    for entry in all_entries:
        if entry.get("type") != exp_type:
            continue
        js = entry.get("joint_weighted_mean", entry.get("joint_severity", 0))
        if js == 0:
            continue
        bit_sevs[entry["bit"]].append(js)

    result: dict[int, float] = {}
    for bit, sevs in bit_sevs.items():
        if method == "mean":
            result[bit] = float(np.mean(sevs))
        elif method == "max":
            result[bit] = float(np.max(sevs))
        elif method == "topk":
            sevs_sorted = sorted(sevs, reverse=True)
            k = min(top_k, len(sevs_sorted))
            result[bit] = float(np.mean(sevs_sorted[:k])) if k > 0 else 0.0
    return result


def make_rankings(
    sev_scores: dict[int, float],
    acc_drops: dict[int, float],
) -> tuple[list[int], list[int], set[int]]:
    """Build paired rankings for bits present in both severity and accuracy_drop.

    Lower rank number = higher severity / larger accuracy drop (i.e. rank 1 = most severe).
    Uses descending ordering for both, then assigns ranks via scipy-compatible ordinal index.

    Returns (sev_ranks_list, acc_ranks_list, common_bits).
    """
    common = sorted(set(sev_scores.keys()) & set(acc_drops.keys()))
    if len(common) < 2:
        return [], [], set()

    # Sort bits by severity descending → lower rank = more severe
    bits_by_sev = sorted(common, key=lambda b: sev_scores[b], reverse=True)
    # Sort bits by accuracy_drop descending → lower rank = larger drop
    bits_by_acc = sorted(common, key=lambda b: acc_drops[b], reverse=True)

    sev_rank_map = {b: i + 1 for i, b in enumerate(bits_by_sev)}
    acc_rank_map = {b: i + 1 for i, b in enumerate(bits_by_acc)}

    sev_ranks = [sev_rank_map[b] for b in common]
    acc_ranks = [acc_rank_map[b] for b in common]
    return sev_ranks, acc_ranks, set(common)


def parse_top_k(raw: str, total_pe: int = 1024) -> int:
    """Parse --top-k argument: '10%' → int(10% of total_pe), '100' → 100."""
    raw = raw.strip()
    if raw.endswith("%"):
        pct = float(raw[:-1]) / 100.0
        return max(1, int(total_pe * pct))
    return max(1, int(raw))


def main():
    parser = argparse.ArgumentParser(
        description="Compare joint severity formulas against measured accuracy drop"
    )
    parser.add_argument("--methods", type=str, nargs="+",
                        choices=["mean", "max", "topk"],
                        default=["mean", "max", "topk"],
                        help="Aggregation methods to compare")
    parser.add_argument("--top-k", type=str, default="10%",
                        help="Top-k for topk method: percentage (10%%) or count (100)")
    parser.add_argument("--tables", type=str, nargs="+", default=None,
                        help="Specific table files to compare (default: auto-discover)")
    args = parser.parse_args()

    top_k = parse_top_k(args.top_k)

    # Discover or use specified tables
    if args.tables:
        table_groups = {"custom": args.tables}
    else:
        table_groups = discover_tables()

    if not table_groups:
        print("No joint severity tables found.")
        return

    print("=" * 80)
    print("Formula Ranking Comparison")
    print(f"  Aggregation methods: {args.methods}")
    if "topk" in args.methods:
        print(f"  Top-k: {args.top_k} → {top_k} PEs")
    print(f"  Tables discovered:")
    for key, paths in sorted(table_groups.items()):
        print(f"    {key}: {len(paths)} file(s)")
    print("=" * 80)

    # Accumulate results per (exp_type, stuck_value, method)
    all_results: dict[tuple[str, int, str], list[dict]] = defaultdict(list)

    for group_key, paths in sorted(table_groups.items()):
        if isinstance(group_key, tuple):
            _, stuck_value = group_key
        else:
            stuck_value = None

        for path in sorted(paths):
            formula = parse_formula_name(path)
            print(f"\nLoading: {os.path.basename(path)}  (formula={formula})")
            with open(path) as f:
                data = json.load(f)

            if stuck_value is None:
                sv = data.get("stuck_value", 1)
            else:
                sv = stuck_value

            all_entries = data.get("all_entries", [])
            exp_types = sorted(set(e["type"] for e in all_entries))

            for exp_type in exp_types:
                acc_drops = load_accuracy_drops(exp_type, sv)
                if not acc_drops:
                    continue

                for method in args.methods:
                    sev_scores = aggregate_per_bit(all_entries, exp_type, method, top_k)
                    sev_ranks, acc_ranks, common = make_rankings(sev_scores, acc_drops)

                    if len(common) < 2:
                        print(f"  {exp_type:>6s} {method:>5s}: too few common bits ({len(common)})")
                        continue

                    rho, p_s = spearmanr(sev_ranks, acc_ranks)
                    tau, p_k = kendalltau(sev_ranks, acc_ranks)

                    all_results[(exp_type, sv, method)].append({
                        "formula": formula,
                        "spearman_r": round(rho, 4),
                        "spearman_p": round(p_s, 4),
                        "kendall_tau": round(tau, 4),
                        "kendall_p": round(p_k, 4),
                        "n_bits": len(common),
                    })

    # Print results grouped by (exp_type, stuck_value, method)
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    for (exp_type, sv, method), rows in sorted(all_results.items()):
        print(f"\ntype={exp_type}, stuck={sv}  (aggregation={method}):")
        # Header
        print(f"  {'formula':>8s}  {'spearman_r':>11s}  {'kendall_tau':>11s}  {'n_bits':>6s}")
        print(f"  {'-'*8}  {'-'*11}  {'-'*11}  {'-'*6}")

        # Sort by spearman_r descending
        for row in sorted(rows, key=lambda r: r["spearman_r"], reverse=True):
            print(f"  {row['formula']:>8s}  {row['spearman_r']:11.4f}  {row['kendall_tau']:11.4f}  {row['n_bits']:6d}")

    # Summary: best formula per (type, sv, method) by spearman_r
    print("\n" + "=" * 80)
    print("Best formula per category (by Spearman rho)")
    print("=" * 80)
    for (exp_type, sv, method), rows in sorted(all_results.items()):
        best = max(rows, key=lambda r: r["spearman_r"])
        print(f"  {exp_type:>6s} stuck={sv} {method:>5s}: {best['formula']:>8s} "
              f"(rho={best['spearman_r']:.4f}, tau={best['kendall_tau']:.4f})")


if __name__ == "__main__":
    main()

"""Compare joint severity vs true single-PE accuracy measurements."""
import csv, json, os, glob
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, kendalltau

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "result")

def load_pe_accuracy(fname):
    by_key = defaultdict(list)
    with open(os.path.join(RESULT_DIR, fname)) as f:
        for row in csv.DictReader(f):
            by_key[(int(row["bit"]), int(row["pe_row"]), int(row["pe_col"]))].append(
                float(row["acc_drop"])
            )
    return {k: float(np.mean(v)) for k, v in by_key.items()}

def load_joint_severity(path, exp_type):
    with open(path) as f:
        data = json.load(f)
    by_key = defaultdict(list)
    for entry in data.get("operator_entries", []):
        if entry.get("type") != exp_type:
            continue
        js = entry.get("joint_severity", 0)
        if js == 0:
            continue
        by_key[(entry["bit"], entry["pe_row"], entry["pe_col"])].append(js)
    return {k: float(np.mean(v)) for k, v in by_key.items()}

for fname, stuck in [("pe_accuracy_input_stuck1_single.csv", 1),
                       ("pe_accuracy_input_stuck0_single.csv", 0)]:
    acc = load_pe_accuracy(fname)
    print(f"\n=== input sa{stuck} === ({len(acc)} PE measurements)")

    results = []
    for formula in ["v3", "v4", "v5_lin", "v5_exp"]:
        if formula == "v3":
            sv_path = os.path.join(CONFIG_DIR, f"joint_severity_stuck{stuck}_ws.json")
        else:
            sv_path = os.path.join(CONFIG_DIR, f"joint_severity_stuck{stuck}_ws_{formula}.json")
        if not os.path.exists(sv_path):
            print(f"  {formula:>8s}: file not found")
            continue
        sev = load_joint_severity(sv_path, "input")
        common = sorted(set(sev.keys()) & set(acc.keys()))
        if len(common) < 5:
            print(f"  {formula:>8s}: too few common ({len(common)})")
            continue
        sev_vals = [sev[k] for k in common]
        acc_vals = [acc[k] for k in common]
        rho, _ = spearmanr(sev_vals, acc_vals)
        tau, _ = kendalltau(sev_vals, acc_vals)
        results.append((formula, rho, tau, len(common)))
        print(f"  {formula:>8s}: rho={rho:+.4f}  tau={tau:+.4f}  n={len(common)}")

    best = max(results, key=lambda r: r[1])
    print(f"  best: {best[0]} (rho={best[1]:.4f})")

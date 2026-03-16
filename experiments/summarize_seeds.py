"""
experiments/summarize_seeds.py

Aggregate results across seeds using results/summary.csv.

Input:
    results/summary.csv   (created by experiments/summarize.py)

Output:
    results/summary_mean_std.csv

Aggregation key:
    (experiment_name, group, variant)

Usage:
    python experiments/summarize_seeds.py
"""

# --- Import safety block ---
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- Standard imports ---
import csv
import math


def is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def std(values):
    # sample std (n-1). If n<2 -> 0.0
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def main():
    results_dir = Path("results")
    in_path = results_dir / "summary.csv"
    if not in_path.exists():
        print("[ERR] results/summary.csv not found. Run experiments/summarize.py first.")
        sys.exit(1)

    # Read summary.csv
    with in_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("[ERR] summary.csv is empty.")
        sys.exit(1)

    # Determine metric columns (everything except these)
    base_cols = {"experiment_name", "group", "variant", "seed", "run_dir"}
    all_cols = list(rows[0].keys())
    metric_cols = [c for c in all_cols if c not in base_cols]

    # Group rows by (experiment_name, group, variant)
    groups = {}
    for r in rows:
        key = (r.get("experiment_name"), r.get("group"), r.get("variant"))
        groups.setdefault(key, []).append(r)

    # Build aggregated rows
    out_rows = []
    for (exp, grp, var), items in sorted(groups.items()):
        out = {
            "experiment_name": exp,
            "group": grp,
            "variant": var,
            "n_runs": str(len(items)),
        }

        for mc in metric_cols:
            vals = []
            for it in items:
                v = it.get(mc, "")
                if v is None or v == "":
                    continue
                if is_number(v):
                    vals.append(float(v))

            if vals:
                out[f"{mc}_mean"] = f"{mean(vals):.6g}"
                out[f"{mc}_std"] = f"{std(vals):.6g}"
            else:
                out[f"{mc}_mean"] = ""
                out[f"{mc}_std"] = ""

        out_rows.append(out)

    # Write aggregated CSV
    out_path = results_dir / "summary_mean_std.csv"

    fieldnames = ["experiment_name", "group", "variant", "n_runs"]
    for mc in metric_cols:
        fieldnames += [f"{mc}_mean", f"{mc}_std"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print("[OK] Wrote:", out_path)
    print("[OK] Groups:", len(out_rows))


if __name__ == "__main__":
    main()

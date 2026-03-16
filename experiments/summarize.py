"""
experiments/summarize.py

Collect all results from results/*/metrics.json and write:
- results/summary.csv

Usage:
    python experiments/summarize.py
"""

# --- Import safety block (prevents import error if executed from subdirectory) ---
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- Standard imports ---
import json
import csv


def main():
    results_dir = Path("results")
    if not results_dir.exists():
        print("[ERR] results/ not found. Run experiments first.")
        sys.exit(1)

    metric_files = sorted(results_dir.glob("*/metrics.json"))
    if not metric_files:
        print("[ERR] No metrics.json found under results/*/.")
        sys.exit(1)

    rows = []
    metric_keys = set()

    # Load all metrics
    for mf in metric_files:
        data = json.loads(mf.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})

        # Collect metric keys (dynamic)
        for k in metrics.keys():
            metric_keys.add(k)

        row = {
            "experiment_name": data.get("experiment_name"),
            "group": data.get("group"),
            "variant": data.get("variant"),
            "seed": data.get("seed"),
            "run_dir": data.get("run_dir"),
        }

        for k, v in metrics.items():
            row[k] = v

        rows.append(row)

    # Stable column order
    metric_keys = sorted(metric_keys)
    fieldnames = ["experiment_name", "group", "variant", "seed"] + metric_keys + ["run_dir"]

    out_path = results_dir / "summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[OK] Wrote:", out_path)
    print("[OK] Rows:", len(rows))


if __name__ == "__main__":
    main()

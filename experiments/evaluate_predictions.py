"""Create condition-wise fall-detection metrics from a prediction CSV.

Required columns: label, score, condition

Example:
    python experiments/evaluate_predictions.py predictions.csv --seed 0 \
        --output-dir results/b1_seed0 --window-seconds 6.4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation import evaluate_tradeoff, worst_condition_recall


def read_predictions(path: Path) -> tuple[list[int], list[float], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "score", "condition"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("prediction CSV requires label, score, condition columns")
        rows = list(reader)
    if not rows:
        raise ValueError("prediction CSV is empty")
    return (
        [int(row["label"]) for row in rows],
        [float(row["score"]) for row in rows],
        [row["condition"] for row in rows],
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-id", required=True, choices=["B1", "B2", "B3", "B4", "P1", "P2"])
    parser.add_argument("--window-seconds", type=float)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels, scores, conditions = read_predictions(args.predictions)
    from src.evaluation import default_thresholds

    tradeoff = evaluate_tradeoff(
        labels,
        scores,
        conditions,
        default_thresholds(args.threshold_step),
        window_seconds=args.window_seconds,
    )
    for row in tradeoff:
        row["seed"] = args.seed
        row["model_id"] = args.model_id

    worst = worst_condition_recall(tradeoff)
    for row in worst:
        row["seed"] = args.seed
        row["model_id"] = args.model_id

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "tradeoff.csv", tradeoff)
    write_csv(args.output_dir / "worst_condition.csv", worst)
    manifest = {
        "model_id": args.model_id,
        "seed": args.seed,
        "prediction_file": str(args.predictions.resolve()),
        "prediction_rows": len(labels),
        "threshold_step": args.threshold_step,
        "window_seconds": args.window_seconds,
        "false_alert_unit": "negative_window",
        "ceiling_results_are_provisional": args.model_id == "B4",
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[OK] Wrote evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()

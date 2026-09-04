"""Condition-wise evaluation for fall-detection experiments.

The functions in this module operate on exported labels and fall scores.  They
do not depend on a particular model, which keeps B1--B4 and P1--P2 comparable.
"""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Iterable, Mapping, Sequence

import numpy as np


OVERALL_CONDITION = "__overall__"


def default_thresholds(step: float = 0.01) -> list[float]:
    """Return fixed thresholds from 0 to 1, including both endpoints."""
    if not 0.0 < step <= 1.0:
        raise ValueError("step must be in (0, 1]")
    count = int(round(1.0 / step))
    values = [min(i * step, 1.0) for i in range(count + 1)]
    if values[-1] != 1.0:
        values.append(1.0)
    return values


def evaluate_tradeoff(
    labels: Sequence[int] | np.ndarray,
    fall_scores: Sequence[float] | np.ndarray,
    conditions: Sequence[str] | np.ndarray | None = None,
    thresholds: Iterable[float] | None = None,
    *,
    window_seconds: float | None = None,
    include_overall: bool = True,
) -> list[dict]:
    """Evaluate recall and false alerts for fixed thresholds.

    ``labels`` must be binary (fall=1, non-fall=0).  A false alert is one
    negative window whose score is greater than or equal to the threshold.
    Consequently, ``false_alerts_per_hour`` is a window-level rate, not an
    event-level alarm rate.  It is only emitted when ``window_seconds`` is
    supplied.
    """
    y_true = np.asarray(labels)
    scores = np.asarray(fall_scores, dtype=float)
    if y_true.ndim != 1 or scores.ndim != 1:
        raise ValueError("labels and fall_scores must be one-dimensional")
    if len(y_true) == 0 or len(y_true) != len(scores):
        raise ValueError("labels and fall_scores must have the same non-zero length")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("labels must contain only 0 and 1")
    if not np.all(np.isfinite(scores)):
        raise ValueError("fall_scores must be finite")

    if conditions is None:
        condition_values = np.full(len(y_true), OVERALL_CONDITION, dtype=object)
        include_overall = False
    else:
        condition_values = np.asarray(conditions, dtype=object)
        if condition_values.ndim != 1 or len(condition_values) != len(y_true):
            raise ValueError("conditions must have the same length as labels")
        if any(value is None or str(value).strip() == "" for value in condition_values):
            raise ValueError("condition names must be non-empty")
        condition_values = np.asarray([str(value) for value in condition_values])

    if window_seconds is not None and window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    threshold_values = list(default_thresholds() if thresholds is None else thresholds)
    if not threshold_values:
        raise ValueError("at least one threshold is required")
    threshold_values = [float(value) for value in threshold_values]
    if not np.all(np.isfinite(threshold_values)):
        raise ValueError("thresholds must be finite")

    names = sorted(set(condition_values.tolist()))
    groups: list[tuple[str, np.ndarray]] = [
        (name, condition_values == name) for name in names
    ]
    if include_overall and OVERALL_CONDITION not in names:
        groups.insert(0, (OVERALL_CONDITION, np.ones(len(y_true), dtype=bool)))

    rows: list[dict] = []
    for threshold in threshold_values:
        predicted = scores >= threshold
        for condition, mask in groups:
            group_labels = y_true[mask]
            group_predictions = predicted[mask]
            positive = group_labels == 1
            negative = ~positive

            true_positives = int(np.sum(group_predictions & positive))
            false_negatives = int(np.sum(~group_predictions & positive))
            false_positives = int(np.sum(group_predictions & negative))
            true_negatives = int(np.sum(~group_predictions & negative))
            positive_count = true_positives + false_negatives
            negative_count = false_positives + true_negatives

            row = {
                "condition": condition,
                "threshold": threshold,
                "recall": (
                    true_positives / positive_count if positive_count else None
                ),
                "false_alerts": false_positives,
                "false_alert_rate": (
                    false_positives / negative_count if negative_count else None
                ),
                "true_positives": true_positives,
                "false_negatives": false_negatives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "positive_windows": positive_count,
                "negative_windows": negative_count,
            }
            if window_seconds is not None:
                negative_hours = negative_count * window_seconds / 3600.0
                row["false_alerts_per_hour"] = (
                    false_positives / negative_hours if negative_hours else None
                )
            rows.append(row)
    return rows


def worst_condition_recall(rows: Sequence[Mapping]) -> list[dict]:
    """Return the lowest recall and its condition for every threshold."""
    by_threshold: dict[float, list[Mapping]] = defaultdict(list)
    for row in rows:
        if row["condition"] != OVERALL_CONDITION and row.get("recall") is not None:
            by_threshold[float(row["threshold"])].append(row)

    result = []
    for threshold in sorted(by_threshold):
        candidates = by_threshold[threshold]
        if not candidates:
            continue
        worst = min(candidates, key=lambda row: (float(row["recall"]), row["condition"]))
        result.append(
            {
                "threshold": threshold,
                "worst_condition_recall": float(worst["recall"]),
                "worst_condition": worst["condition"],
            }
        )
    return result


def aggregate_seed_curves(
    rows: Sequence[Mapping], *, expected_seed_count: int | None = 5
) -> list[dict]:
    """Aggregate condition/threshold rows as mean and sample standard deviation.

    Each input row must contain ``seed``, ``condition``, ``threshold``,
    ``recall``, and ``false_alert_rate``.  By default exactly five distinct
    seeds are required for every condition/threshold point.
    """
    grouped: dict[tuple[str, float], list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), float(row["threshold"]))].append(row)

    output: list[dict] = []
    for (condition, threshold), items in sorted(grouped.items()):
        seeds = {int(item["seed"]) for item in items}
        if len(seeds) != len(items):
            raise ValueError(f"duplicate seed for {condition} at threshold {threshold}")
        if expected_seed_count is not None and len(seeds) != expected_seed_count:
            raise ValueError(
                f"expected {expected_seed_count} seeds for {condition} at "
                f"threshold {threshold}, found {len(seeds)}"
            )

        row = {
            "condition": condition,
            "threshold": threshold,
            "n_seeds": len(seeds),
        }
        for metric in ("recall", "false_alert_rate"):
            values = [item.get(metric) for item in items]
            if any(value is None for value in values):
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                continue
            numeric = [float(value) for value in values]
            mean = sum(numeric) / len(numeric)
            variance = (
                sum((value - mean) ** 2 for value in numeric) / (len(numeric) - 1)
                if len(numeric) > 1
                else 0.0
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = sqrt(variance)
        output.append(row)
    return output

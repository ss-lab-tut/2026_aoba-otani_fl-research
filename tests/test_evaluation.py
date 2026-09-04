import math
import unittest

from src.evaluation import (
    OVERALL_CONDITION,
    aggregate_seed_curves,
    default_thresholds,
    evaluate_tradeoff,
    worst_condition_recall,
)


class EvaluateTradeoffTests(unittest.TestCase):
    def test_known_confusion_counts_and_rates(self):
        rows = evaluate_tradeoff(
            labels=[1, 1, 0, 0],
            fall_scores=[0.9, 0.4, 0.8, 0.1],
            conditions=["room_a"] * 4,
            thresholds=[0.5],
            window_seconds=6.0,
        )
        room = next(row for row in rows if row["condition"] == "room_a")
        self.assertEqual(room["true_positives"], 1)
        self.assertEqual(room["false_negatives"], 1)
        self.assertEqual(room["false_positives"], 1)
        self.assertEqual(room["true_negatives"], 1)
        self.assertEqual(room["recall"], 0.5)
        self.assertEqual(room["false_alert_rate"], 0.5)
        self.assertEqual(room["false_alerts_per_hour"], 300.0)

        overall = next(row for row in rows if row["condition"] == OVERALL_CONDITION)
        self.assertEqual(overall["recall"], room["recall"])

    def test_worst_condition_excludes_overall(self):
        rows = evaluate_tradeoff(
            labels=[1, 1, 1, 1, 0, 0],
            fall_scores=[0.9, 0.8, 0.6, 0.1, 0.7, 0.2],
            conditions=["good", "good", "bad", "bad", "good", "bad"],
            thresholds=[0.5],
        )
        worst = worst_condition_recall(rows)
        self.assertEqual(worst, [{
            "threshold": 0.5,
            "worst_condition_recall": 0.5,
            "worst_condition": "bad",
        }])

    def test_missing_class_is_reported_as_none(self):
        rows = evaluate_tradeoff(
            labels=[0, 0],
            fall_scores=[0.7, 0.2],
            conditions=["negative_only", "negative_only"],
            thresholds=[0.5],
        )
        room = next(row for row in rows if row["condition"] == "negative_only")
        self.assertIsNone(room["recall"])
        self.assertEqual(room["false_alert_rate"], 0.5)
        self.assertEqual(worst_condition_recall(rows), [])

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            evaluate_tradeoff([], [])
        with self.assertRaises(ValueError):
            evaluate_tradeoff([2], [0.5])
        with self.assertRaises(ValueError):
            evaluate_tradeoff([1], [float("nan")])
        with self.assertRaises(ValueError):
            evaluate_tradeoff([1], [0.5], [""])
        with self.assertRaises(ValueError):
            default_thresholds(0)


class SeedAggregationTests(unittest.TestCase):
    def test_five_seed_mean_and_sample_std(self):
        rows = []
        recalls = [0.8, 0.9, 1.0, 0.9, 0.9]
        for seed, recall in enumerate(recalls):
            rows.append({
                "seed": seed,
                "condition": "room_a",
                "threshold": 0.5,
                "recall": recall,
                "false_alert_rate": 0.1 + seed * 0.01,
            })
        result = aggregate_seed_curves(rows)
        self.assertEqual(result[0]["n_seeds"], 5)
        self.assertAlmostEqual(result[0]["recall_mean"], 0.9)
        self.assertAlmostEqual(result[0]["recall_std"], math.sqrt(0.005))

    def test_default_rejects_incomplete_seed_set(self):
        row = {
            "seed": 0,
            "condition": "room_a",
            "threshold": 0.5,
            "recall": 1.0,
            "false_alert_rate": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "expected 5 seeds"):
            aggregate_seed_curves([row])

    def test_rejects_duplicate_seed(self):
        row = {
            "seed": 0,
            "condition": "room_a",
            "threshold": 0.5,
            "recall": 1.0,
            "false_alert_rate": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            aggregate_seed_curves([row, dict(row)], expected_seed_count=None)


if __name__ == "__main__":
    unittest.main()

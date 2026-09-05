import unittest

import numpy as np

from experiments.condition_threshold_validation import (
    fold_keys, metrics, roc_auc, select_threshold,
)


class ThresholdValidationTests(unittest.TestCase):
    def test_ties_never_break_false_positive_budget(self):
        y = [0, 0, 0, 1, 1]
        s = [0.8, 0.8, 0.1, 0.8, 0.9]
        threshold = select_threshold(y, s, ["a"] * 5, 1 / 3)
        self.assertGreater(threshold, 0.8)
        self.assertEqual(metrics(y, s, threshold)["false_positives"], 0)
        self.assertEqual(metrics(y, s, threshold)["recall"], 0.5)

    def test_constraint_is_per_group_not_pooled(self):
        y = [0, 0, 1, 0, 0, 1]
        s = [0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
        groups = np.array(["a"] * 3 + ["b"] * 3)
        t = select_threshold(y, s, groups, 0.5)
        for group in ("a", "b"):
            mask = groups == group
            self.assertLessEqual(metrics(np.array(y)[mask], np.array(s)[mask], t)["false_positive_rate"], 0.5)

    def test_extreme_budgets(self):
        self.assertGreater(select_threshold([0, 1], [1, 1], ["a", "a"], 0), 1)
        self.assertEqual(select_threshold([0, 1], [0, 1], ["a", "a"], 1), 0)

    def test_selected_threshold_matches_exhaustive_feasible_recall(self):
        rng = np.random.default_rng(42)
        labels = np.tile([0, 0, 1, 1], 2)
        groups = np.repeat(["a", "b"], 4)
        for _ in range(50):
            scores = rng.integers(0, 5, size=8) / 4
            budget = 0.5
            chosen = select_threshold(labels, scores, groups, budget)
            candidates = np.r_[scores.min(), np.nextafter(np.unique(scores), np.inf)]
            feasible = [t for t in candidates if all(
                metrics(labels[groups == g], scores[groups == g], t)["false_positive_rate"] <= budget
                for g in ("a", "b"))]
            self.assertEqual(metrics(labels, scores, chosen)["recall"],
                             max(metrics(labels, scores, t)["recall"] for t in feasible))

    def test_auc_includes_ties(self):
        self.assertEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1)
        self.assertEqual(roc_auc([0, 1], [1, 0]), 0)
        self.assertEqual(roc_auc([0, 0, 1, 1], [0, 1, 1, 2]), 0.875)

    def test_split_excludes_outer_geometry_and_seed(self):
        seeds = list(range(5))
        geometries = ["a", "b", "c", "d", "e"]
        for seed in seeds:
            for geometry in geometries:
                train, val, test = fold_keys(seeds, geometries, seed, geometry)
                self.assertEqual(len(train), 12)
                self.assertEqual(len(val), 4)
                self.assertEqual(test, (seed, geometry))
                self.assertTrue({s for s, _ in train}.isdisjoint({s for s, _ in val}))
                self.assertTrue(all(s != seed and g != geometry for s, g in train + val))

    def test_reject_invalid_inputs(self):
        for labels, scores in [([0], [0.2]), ([0, 1], [0, np.nan]), ([0, 2], [0, 1])]:
            with self.assertRaises(ValueError):
                roc_auc(labels, scores)
        with self.assertRaises(ValueError):
            select_threshold([0, 1], [0, 1], ["a", "b"], 0.1)
        with self.assertRaises(ValueError):
            select_threshold([0, 1], [0, 1], ["a", "a"], -0.1)


if __name__ == "__main__":
    unittest.main()

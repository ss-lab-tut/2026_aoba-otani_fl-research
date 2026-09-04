import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.audit_dataset import audit_dataset, sha256_file


class DatasetAuditTests(unittest.TestCase):
    def make_dataset(self, root: Path, *, mismatch: bool = False) -> None:
        np.save(root / "X_train.npy", np.zeros((3, 2, 2, 2)))
        np.save(root / "X_val_real.npy", np.zeros((2, 2, 2, 2)))
        np.save(root / "y_train_bin.npy", np.array([0, 1, 0]))
        np.save(root / "y_train_sub.npy", np.array([0, 3, 2]))
        np.save(root / "y_val_bin.npy", np.array([1, 0]))
        np.save(root / "y_val_sub.npy", np.array([0 if mismatch else 3, 1]))

    def test_consistent_dataset_is_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            report = audit_dataset(root)
            self.assertTrue(report["official_evaluation_ready"])
            self.assertEqual(report["issues"], [])
            self.assertEqual(
                report["arrays"]["y_train_sub.npy"]["class_counts"],
                {"0": 1, "2": 1, "3": 1},
            )
            self.assertEqual(len(report["arrays"]["X_train.npy"]["sha256"]), 64)

    def test_label_mismatch_is_reported_without_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root, mismatch=True)
            before = sha256_file(root / "y_val_sub.npy")
            report = audit_dataset(root, include_hash=False)
            after = sha256_file(root / "y_val_sub.npy")
            self.assertFalse(report["official_evaluation_ready"])
            self.assertEqual(before, after)
            issue = report["issues"][0]
            self.assertEqual(issue["type"], "binary_subclass_mismatch")
            self.assertEqual(issue["items"], [{
                "index": 0,
                "binary_label": 1,
                "subclass_label": 0,
            }])

    def test_missing_file_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "missing dataset files"):
                audit_dataset(Path(directory))


if __name__ == "__main__":
    unittest.main()

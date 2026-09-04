"""Audit local LiDAR arrays without modifying them.

The generated report records shapes, dtypes, class counts, SHA-256 digests and
cross-label consistency.  Large feature arrays are opened with mmap so the
audit does not load the full dataset into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FEATURE_FILES = ("X_train.npy", "X_val_real.npy")
LABEL_FILES = (
    "y_train_bin.npy",
    "y_train_sub.npy",
    "y_val_bin.npy",
    "y_val_sub.npy",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_array(path: Path, *, include_hash: bool = True) -> dict:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    record = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if include_hash:
        record["sha256"] = sha256_file(path)
    if array.ndim == 1:
        values, counts = np.unique(array, return_counts=True)
        record["class_counts"] = {
            str(value.item()): int(count) for value, count in zip(values, counts)
        }
    return record


def audit_dataset(root: Path, *, include_hash: bool = True) -> dict:
    root = root.resolve()
    required = FEATURE_FILES + LABEL_FILES
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing dataset files: {', '.join(missing)}")

    arrays = {
        name: inspect_array(root / name, include_hash=include_hash)
        for name in required
    }
    issues: list[dict] = []

    expected_pairs = (
        ("X_train.npy", "y_train_bin.npy"),
        ("X_train.npy", "y_train_sub.npy"),
        ("X_val_real.npy", "y_val_bin.npy"),
        ("X_val_real.npy", "y_val_sub.npy"),
    )
    for feature_name, label_name in expected_pairs:
        feature_rows = arrays[feature_name]["shape"][0]
        label_rows = arrays[label_name]["shape"][0]
        if feature_rows != label_rows:
            issues.append({
                "type": "row_count_mismatch",
                "feature_file": feature_name,
                "label_file": label_name,
                "feature_rows": feature_rows,
                "label_rows": label_rows,
            })

    for split in ("train", "val"):
        binary_name = f"y_{split}_bin.npy"
        subclass_name = f"y_{split}_sub.npy"
        binary = np.load(root / binary_name, mmap_mode="r", allow_pickle=False)
        subclass = np.load(root / subclass_name, mmap_mode="r", allow_pickle=False)
        if len(binary) != len(subclass):
            continue
        mismatch_indices = np.flatnonzero(binary != (subclass == 3))
        if len(mismatch_indices):
            issues.append({
                "type": "binary_subclass_mismatch",
                "split": split,
                "rule": "binary_label == (subclass_label == 3)",
                "count": int(len(mismatch_indices)),
                "items": [
                    {
                        "index": int(index),
                        "binary_label": int(binary[index]),
                        "subclass_label": int(subclass[index]),
                    }
                    for index in mismatch_indices
                ],
            })

    return {
        "schema_version": 1,
        "dataset_root": str(root),
        "arrays": arrays,
        "label_rule": "fall is subclass 3",
        "issues": issues,
        "official_evaluation_ready": len(issues) == 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-hash", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero exit status when consistency issues are found",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_dataset(args.data_dir, include_hash=not args.skip_hash)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"[OK] Wrote dataset audit to {args.output}")
    if args.strict and report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

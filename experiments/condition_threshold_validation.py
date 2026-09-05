"""Nested, geometry/seed held-out threshold pilot; not fall-detection accuracy.

Uses the adjacent testbed's observable spatial/temporal features. Source hashes
are recorded because the testbed can contain uncommitted research changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def validate_binary(labels, scores):
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) == 0 or len(y) != len(s):
        raise ValueError("labels and scores must be equal nonempty 1-D arrays")
    if not np.all(np.isin(y, [0, 1])) or not np.all(np.isfinite(s)):
        raise ValueError("binary labels and finite scores required")
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("both classes required for recall/FPR evaluation")
    return y.astype(bool), s


def metrics(labels, scores, threshold):
    y, s = validate_binary(labels, scores)
    predicted = s >= threshold
    tp = int(np.sum(predicted & y))
    fp = int(np.sum(predicted & ~y))
    recall = tp / int(y.sum())
    fpr = fp / int((~y).sum())
    return dict(recall=recall, false_positive_rate=fpr,
                balanced_accuracy=(recall + 1 - fpr) / 2,
                true_positives=tp, false_positives=fp,
                positives=int(y.sum()), negatives=int((~y).sum()))


def roc_auc(labels, scores):
    """Mann--Whitney AUC with half credit for tied scores."""
    y, s = validate_binary(labels, scores)
    order = np.argsort(s, kind="stable")
    ordered = s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1]
    ends = np.r_[starts[1:], len(s)]
    rank_sum = sum(float(np.sum(y[order[a:b]])) * (a + 1 + b) / 2
                   for a, b in zip(starts, ends))
    n_pos = int(y.sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * int((~y).sum()))


def select_threshold(labels, scores, groups, max_fpr):
    """Lowest threshold satisfying empirical FPR bound in EVERY valid group.

    Uses only calibration scores. Inclusive score >= threshold and tied scores
    are handled explicitly. No deployment guarantee is implied by this bound.
    """
    y, s = validate_binary(labels, scores)
    g = np.asarray(groups)
    if g.ndim != 1 or len(g) != len(y):
        raise ValueError("groups must match labels")
    if not np.isfinite(max_fpr) or not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be in [0, 1]")
    bounds = []
    for group in np.unique(g):
        mask = g == group
        negatives = np.sort(s[mask & ~y])[::-1]
        if len(negatives) == 0:
            raise ValueError("each calibration group needs negative examples")
        allowed = int(np.floor(max_fpr * len(negatives)))
        bounds.append(float(np.min(s)) if allowed == len(negatives)
                      else float(np.nextafter(negatives[allowed], np.inf)))
    return max(bounds)


def fold_keys(seeds, geometry_names, held_seed, held_geometry):
    """Calibration gets one whole seed; test seed/geometry never enters fit."""
    validation_seed = seeds[(seeds.index(held_seed) + 1) % len(seeds)]
    train = [(s, g) for s in seeds for g in geometry_names
             if s not in (held_seed, validation_seed) and g != held_geometry]
    calibration = [(validation_seed, g) for g in geometry_names if g != held_geometry]
    return train, calibration, (held_seed, held_geometry)


def source_manifest(testbed):
    paths = sorted((testbed / "heterosense").rglob("*.py"))
    paths += [testbed / "pyproject.toml"]
    return {str(path.relative_to(testbed)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in paths}


def git_info(directory):
    def run(*args):
        return subprocess.check_output(["git", "-C", str(directory), *args],
                                       text=True, encoding="utf-8").strip()
    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def run_experiment(testbed, n_steps=800, calibration_steps=100):
    if not 0 < calibration_steps < n_steps:
        raise ValueError("require 0 < calibration_steps < n_steps")
    sys.path.insert(0, str(testbed))
    from heterosense import FEATURE_NAMES, ObservableConditionEstimator
    from heterosense._scripts.structured_occlusion_pilot import (
        FEATURE_SETS, GEOMETRIES, build_geometry_dataset, trailing_mean,
    )

    seeds = list(range(5))
    names = [g.name for g in GEOMETRIES]
    columns = [FEATURE_NAMES.index(name) for name in FEATURE_SETS["sector_counts_only"]]
    manifest = source_manifest(testbed)
    testbed_version = git_info(testbed)
    datasets = {}
    for seed in seeds:
        for geometry in GEOMETRIES:
            item = build_geometry_dataset(seed, geometry, n_steps,
                                          calibration_steps, 0.10,
                                          "per_sequence_observed")
            datasets[seed, geometry.name] = (
                np.vstack([trailing_mean(seq[:, columns], 5)
                           for seq in item["feature_sequences"]]),
                (item["targets"] == "degraded").astype(int),
            )
        print(f"Generated seed {seed + 1}/5", flush=True)

    def combine(keys):
        return (np.vstack([datasets[k][0] for k in keys]),
                np.concatenate([datasets[k][1] for k in keys]),
                np.concatenate([np.full(len(datasets[k][1]), k[1]) for k in keys]))

    folds = []
    for held_seed in seeds:
        for held_geometry in names:
            train_keys, val_keys, test_key = fold_keys(seeds, names, held_seed, held_geometry)
            x_train, y_train, _ = combine(train_keys)
            x_val, y_val, g_val = combine(val_keys)
            x_test, y_test = datasets[test_key]
            model = ObservableConditionEstimator().fit(x_train, y_train)
            positive_column = list(model.classes_).index("1")
            val_scores = model.predict_proba(x_val)[:, positive_column]
            test_scores = model.predict_proba(x_test)[:, positive_column]
            policies = {"fixed_0.5": 0.5}
            for budget in (0.05, 0.10, 0.20):
                policies[f"calibrated_fpr_{budget:.2f}"] = select_threshold(
                    y_val, val_scores, g_val, budget)
            results = {}
            for policy, threshold in policies.items():
                val_metrics = {name: metrics(y_val[g_val == name],
                                             val_scores[g_val == name], threshold)
                               for name in np.unique(g_val)}
                results[policy] = {
                    "threshold": threshold,
                    "calibration_worst_fpr": max(m["false_positive_rate"]
                                                  for m in val_metrics.values()),
                    "calibration_by_geometry": val_metrics,
                    "test": metrics(y_test, test_scores, threshold),
                }
            folds.append({
                "held_out_seed": held_seed, "held_out_geometry": held_geometry,
                "train_keys": train_keys, "calibration_keys": val_keys,
                "test_auc": roc_auc(y_test, test_scores), "policies": results,
                "test_curve_diagnostic_only": [dict(threshold=float(t), **metrics(y_test, test_scores, t))
                                               for t in np.linspace(0, 1, 101)],
            })
        print(f"Evaluated held-out seed {held_seed + 1}/5", flush=True)

    summary = {}
    for policy in policies:
        summary[policy] = {}
        for geometry in ["all", *names]:
            selected = [f for f in folds if geometry == "all" or f["held_out_geometry"] == geometry]
            result = {}
            for key in ("recall", "false_positive_rate", "balanced_accuracy"):
                values = [f["policies"][policy]["test"][key] for f in selected]
                result[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
            result["auc_mean"] = float(np.mean([f["test_auc"] for f in selected]))
            result["minimum_recall"] = min(f["policies"][policy]["test"]["recall"] for f in selected)
            summary[policy][geometry] = result
    if source_manifest(testbed) != manifest:
        raise RuntimeError("testbed source changed during experiment")
    return {
        "status": "provisional_simulation_condition_estimation_not_fall_detection",
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "testbed": testbed_version, "testbed_source_sha256": manifest,
        "testbed_source_hash_normalization": "CRLF replaced with LF before SHA-256",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": {"seeds": seeds, "n_steps": n_steps, "calibration_steps": calibration_steps,
                     "minimum_loss_ratio": 0.10, "trailing_window": 5,
                     "features": list(FEATURE_SETS["sector_counts_only"]),
                     "baseline_mode": "per_sequence_observed"},
        "protocol": "25 outer seed x geometry folds; next seed for threshold calibration; remaining 3 seeds for fitting; outer geometry excluded from both; no refit",
        "limitations": [
            "Degradation labels describe paired point loss, not falls or fall alerts.",
            "Calibration FPR constraints are empirical; no guarantee on unseen geometry.",
            "Correlated frames and overlapping training folds; SD is descriptive, not a confidence interval.",
            "Clear sequences repeat across geometry pairs, preserving the previous pilot mixture.",
            "Feature candidate was chosen in prior experiments on these seeds/geometries; exploratory, not a pristine final test.",
            "Testbed contains existing uncommitted changes; exact source hashes recorded.",
            "Python 3.14 is outside testbed listed classifiers; real LiDAR validation pending.",
        ],
        "folds": folds, "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testbed", type=Path, default=ROOT / "heterosense-fl-testbed")
    parser.add_argument("--n-steps", type=int, default=800)
    parser.add_argument("--calibration-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "results/condition_threshold_validation.json")
    args = parser.parse_args()
    report = run_experiment(args.testbed.resolve(), args.n_steps, args.calibration_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    patch = subprocess.check_output([
        "git", "-C", str(args.testbed.resolve()), "diff", "--binary", "HEAD", "--",
        "heterosense", "pyproject.toml",
    ])
    patch_path = args.output.with_suffix(".testbed.patch")
    patch_path.write_bytes(patch)
    report["testbed_source_patch"] = {
        "file": patch_path.name, "sha256": hashlib.sha256(patch).hexdigest(),
        "instructions": "Apply to a clean checkout of testbed.commit; verify all source hashes before rerunning.",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

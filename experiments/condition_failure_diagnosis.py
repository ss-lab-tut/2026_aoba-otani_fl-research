"""Observable-feature and numerical ablations for condition-estimator failures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "heterosense-fl-testbed"))
from experiments.condition_threshold_validation import fold_keys, metrics, roc_auc, select_threshold, source_manifest
from heterosense import FEATURE_NAMES, ObservableConditionEstimator, ObservationBaseline, ObservationStatisticsExtractor
from heterosense._scripts.structured_occlusion_pilot import GEOMETRIES, FEATURE_SETS, _build, trailing_mean
from src.robust_condition import fit_mean_spatial_baseline


def stable_margin(model, features):
    """Binary centroid log-odds without exponentiation or distance cancellation."""
    x = (np.asarray(features) - model.mean_) / model.scale_
    negative, positive = [list(model.classes_).index(c) for c in ("0", "1")]
    c0, c1 = model.centroids_[[negative, positive]]
    return (2 * x @ (c1 - c0) + np.sum(c0 * c0 - c1 * c1)) / (x.shape[1] * model.temperature)


def build_cache(path, seeds, geometries=GEOMETRIES):
    arrays = {}
    baseline_rows = []
    for seed in seeds:
        clear = _build(seed, 800, ())
        for geometry in geometries:
            occluded = _build(seed, 800, geometry.occluders)
            xs, mean_xs, ys, states = [], [], [], []
            for kind, frames in (("clear", clear), ("occluded", occluded)):
                baseline = ObservationBaseline.fit(frames[:100])
                xs.append(ObservationStatisticsExtractor(baseline).transform(frames[100:]))
                mean_baseline = fit_mean_spatial_baseline(frames[:100])
                mean_xs.append(ObservationStatisticsExtractor(mean_baseline).transform(frames[100:]))
                states.extend([f.semantic_state for f in frames[100:]])
                counts = np.asarray([len(f.lidar) for f in frames[100:]])
                clear_counts = np.asarray([len(f.lidar) for f in clear[100:]])
                ys.extend(((clear_counts - counts) / np.maximum(clear_counts, 1) >= 0.10).astype(int))
                baseline_rows.append(dict(seed=seed, geometry=geometry.name, kind=kind,
                    point_count=baseline.point_count, sector_counts=baseline.sector_point_counts,
                    mean_sector_counts=mean_baseline.sector_point_counts,
                    absent_fraction=float(np.mean([f.semantic_state == "ABSENT" for f in frames[:100]]))))
            key = f"{seed}_{geometry.name}"
            arrays[key + "_x"] = np.vstack(xs)
            arrays[key + "_mean_x"] = np.vstack(mean_xs)
            arrays[key + "_y"] = np.array(ys)
            arrays[key + "_state"] = np.array(states)
        print(f"Cached seed {seed}", flush=True)
    arrays["baseline_json"] = np.array(json.dumps(baseline_rows))
    arrays["source_json"] = np.array(json.dumps(source_manifest(ROOT / "heterosense-fl-testbed")))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_datasets(path, seeds, geometries=GEOMETRIES):
    with np.load(path, allow_pickle=False) as cache:
        if json.loads(str(cache["source_json"])) != source_manifest(ROOT / "heterosense-fl-testbed"):
            raise ValueError("cache source hash mismatch")
        data = {(s, g.name): (cache[f"{s}_{g.name}_x"], cache[f"{s}_{g.name}_y"], cache[f"{s}_{g.name}_state"])
                for s in seeds for g in geometries}
        return data, json.loads(str(cache["baseline_json"]))


def summarize(folds):
    out = {}
    for variant in folds[0]["variants"]:
        out[variant] = {}
        for group in ["all", *sorted({f["geometry"] for f in folds})]:
            rows = [f["variants"][variant] for f in folds if group == "all" or f["geometry"] == group]
            out[variant][group] = {m: float(np.mean([r[m] for r in rows]))
                                   for m in ("recall", "false_positive_rate", "balanced_accuracy", "auc")}
            out[variant][group]["zero_recall_folds"] = sum(r["recall"] == 0 for r in rows)
    return out


def diagnose(data, baselines):
    seeds = list(range(5))
    names = [g.name for g in GEOMETRIES]
    columns = [FEATURE_NAMES.index(n) for n in FEATURE_SETS["sector_counts_only"]]
    transformed = {}
    for key, (x, y, _) in data.items():
        sectors = x[:, columns]
        transformed[key] = (np.vstack([trailing_mean(a, 5) for a in np.split(sectors, 2)]), y)
    def combine(keys):
        return (np.vstack([transformed[k][0] for k in keys]),
                np.concatenate([transformed[k][1] for k in keys]),
                np.concatenate([np.full(len(transformed[k][1]), k[1]) for k in keys]))
    folds = []
    for seed in seeds:
        for geometry in names:
            train, val, test = fold_keys(seeds, names, seed, geometry)
            xt, yt, _ = combine(train); xv, yv, gv = combine(val); xe, ye = transformed[test]
            model = ObservableConditionEstimator().fit(xt, yt)
            pv = model.predict_proba(xv)[:, 1]; pe = model.predict_proba(xe)[:, 1]
            mv, me = stable_margin(model, xv), stable_margin(model, xe)
            variants = {}
            for name, sv, se, default in (("probability", pv, pe, .5), ("stable_margin", mv, me, 0.0)):
                for policy, threshold in (("fixed", default), ("calibrated05", select_threshold(yv, sv, gv, .05))):
                    result = metrics(ye, se, threshold)
                    result.update(auc=roc_auc(ye, se), threshold=threshold)
                    variants[name + "_" + policy] = result
            folds.append(dict(seed=seed, geometry=geometry, variants=variants,
                calibration_saturation=float(np.mean((pv == 0) | (pv == 1))),
                test_saturation=float(np.mean((pe == 0) | (pe == 1))),
                calibration_negative_margin_quantiles=np.quantile(mv[yv == 0], [0,.5,.95,1]).tolist(),
                test_negative_margin_quantiles=np.quantile(me[ye == 0], [0,.5,.95,1]).tolist(),
                test_feature_max=float(np.max(xe)), train_feature_max=float(np.max(xt))))
    return dict(status="exploratory_condition_diagnosis", baselines=baselines, folds=folds, summary=summarize(folds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "results/local_condition_cache.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/condition_failure_diagnosis.json")
    args = parser.parse_args()
    if not args.cache.exists():
        build_cache(args.cache, list(range(5)))
    data, baselines = load_datasets(args.cache, list(range(5)))
    report = diagnose(data, baselines)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

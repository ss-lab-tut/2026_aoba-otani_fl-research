"""Paired threshold policies on frozen fall and observable-condition scores."""
from collections import defaultdict

import numpy as np

from experiments.condition_threshold_validation import roc_auc, select_threshold

METRICS = ('recall', 'precision', 'f1', 'auroc', 'margin_auroc', 'false_positive_rate')


def operating_metrics(labels, scores, thresholds):
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    t = np.broadcast_to(np.asarray(thresholds, dtype=float), s.shape)
    if y.shape != s.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('nonempty binary labels and matching scores required')
    if not np.isfinite(s).all() or not np.isfinite(t).all():
        raise ValueError('finite scores and thresholds required')
    p = s >= t
    tp, fp = int(np.sum(p & (y == 1))), int(np.sum(p & (y == 0)))
    fn, tn = int(np.sum(~p & (y == 1))), int(np.sum(~p & (y == 0)))
    both = len(np.unique(y)) == 2
    return dict(n=len(y), tp=tp, fp=fp, fn=fn, tn=tn,
                recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                false_positive_rate=fp/(fp+tn) if fp+tn else None,
                auroc=float(roc_auc(y, s)) if both else None,
                margin_auroc=float(roc_auc(y, s-t)) if both else None)


def fit_policies(calibration, budget, condition_cutoff):
    """Fit on calibration only. P1 routes by the frozen estimator's output."""
    y = np.array([r['label'] for r in calibration])
    s1 = np.array([r['b1_score'] for r in calibration])
    s2 = np.array([r['augmented_score'] for r in calibration])
    facilities = np.array([r['facility'] for r in calibration])
    states = np.array([int(r['condition_score'] >= condition_cutoff) for r in calibration])

    def threshold(mask, score):
        return select_threshold(y[mask], score[mask], np.full(mask.sum(), 'all'), budget)

    all_rows = np.ones(len(y), dtype=bool)
    common = threshold(all_rows, s2)
    fitted = dict(B1=threshold(all_rows, s1), B2=common, B3={}, P1={}, fallbacks=[])
    for name, groups in [('B3', facilities), ('P1', states)]:
        for group in np.unique(groups):
            mask = groups == group
            if len(np.unique(y[mask])) < 2:
                fitted['fallbacks'].append(f'{name}:{group}:calibration_missing_class')
                continue
            fitted[name][str(group)] = threshold(mask, s2)
    fitted['condition_cutoff'] = condition_cutoff
    return fitted


def apply_policies(rows, fitted):
    return {
        'B1': np.full(len(rows), fitted['B1']),
        'B2': np.full(len(rows), fitted['B2']),
        'B3': np.array([fitted['B3'].get(r['facility'], fitted['B2']) for r in rows]),
        'P1': np.array([fitted['P1'].get(str(int(r['condition_score'] >= fitted['condition_cutoff'])),
                                      fitted['B2']) for r in rows]),
    }


def evaluate_groups(rows, fitted, seen_geometries):
    thresholds = apply_policies(rows, fitted)
    groups = defaultdict(list)
    predictions = []
    for i, r in enumerate(rows):
        scope = 'seen' if r['geometry'] in seen_geometries else 'unseen'
        for key in [('all', '__all__', '__all__'), (scope, '__all__', '__all__'),
                    (scope, r['facility'], '__all__'), (scope, '__all__', r['geometry']),
                    (scope, r['facility'], r['geometry'])]:
            groups[key].append(i)
        for method, values in thresholds.items():
            score = r['b1_score'] if method == 'B1' else r['augmented_score']
            predictions.append(dict(seed=r['seed'], sample_id=r['sample_id'], facility=r['facility'],
                                    geometry=r['geometry'], scope=scope, method=method, label=r['label'],
                                    score=score, threshold=float(values[i]), prediction=int(score >= values[i]),
                                    condition_score=r['condition_score']))
    metrics = []
    for (scope, facility, geometry), ids in sorted(groups.items()):
        y = [rows[i]['label'] for i in ids]
        for method, values in thresholds.items():
            score = [rows[i]['b1_score' if method == 'B1' else 'augmented_score'] for i in ids]
            metrics.append(dict(seed=rows[0]['seed'], method=method, scope=scope,
                                facility=facility, geometry=geometry,
                                **operating_metrics(y, score, values[ids])))
    return metrics, predictions


def summarize(rows, seeds):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['method'], r['scope'], r['facility'], r['geometry'])].append(r)
    result = []
    for (method, scope, facility, geometry), items in sorted(groups.items()):
        if len(items) != len(seeds) or {r['seed'] for r in items} != set(seeds):
            raise ValueError(f'incomplete or duplicate seeds for {method}/{scope}/{facility}/{geometry}')
        out = dict(method=method, scope=scope, facility=facility, geometry=geometry, n_seeds=len(seeds))
        for metric in METRICS:
            values = [r[metric] for r in items]
            out[metric+'_n_defined'] = sum(v is not None for v in values)
            out[metric+'_mean'] = float(np.mean(values)) if all(v is not None for v in values) else None
            out[metric+'_std'] = float(np.std(values, ddof=1)) if len(values)>1 and all(v is not None for v in values) else None
        result.append(out)
    return result


def paired_deltas(rows):
    reference = {(r['seed'], r['scope'], r['facility'], r['geometry']): r for r in rows if r['method']=='B3'}
    result = []
    for row in rows:
        if row['method'] != 'P1':
            continue
        other = reference[(row['seed'], row['scope'], row['facility'], row['geometry'])]
        out = {k: row[k] for k in ('seed', 'scope', 'facility', 'geometry')}
        out['method'] = 'P1_minus_B3'
        out.update({m: row[m]-other[m] if row[m] is not None and other[m] is not None else None for m in METRICS})
        result.append(out)
    return result

"""Run B1/B2/B3/P1 calibration and paired evaluation for five frozen seeds."""
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fall_comparison import fit_policies, evaluate_groups, summarize, paired_deltas


def write_csv(path, rows):
    if not rows:
        raise ValueError('empty output')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate(rows, cfg):
    seeds = cfg['seeds']
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError('exactly five distinct seeds required')
    if {r['seed'] for r in rows} != set(seeds):
        raise ValueError('input seed set differs from config')
    if not np.isfinite(cfg['condition_cutoff']) or not 0 <= cfg['calibration_fpr_budget'] < 1:
        raise ValueError('invalid operating point')
    for seed in seeds:
        subset = [r for r in rows if r['seed'] == seed]
        if len({r['sample_id'] for r in subset}) != len(subset):
            raise ValueError('duplicate sample or cross-split sample leakage')
        if {r['split'] for r in subset} != {'train', 'calibration', 'test'}:
            raise ValueError('train, calibration and test identities required')
        owners = {}
        for r in subset:
            for k in ('sample_id', 'recording_id', 'facility', 'geometry'):
                if not r[k].strip() or r[k] == '__all__':
                    raise ValueError(f'missing/reserved {k}')
            previous = owners.setdefault(r['recording_id'], r['split'])
            if previous != r['split']:
                raise ValueError('recording leakage across splits')
            if r['label'] not in (0, 1):
                raise ValueError('binary fall labels required')
            if r['split'] != 'train' and not all(np.isfinite(r[k]) for k in ('b1_score','augmented_score','condition_score')):
                raise ValueError('finite frozen scores required')
        provenance = cfg['estimator_provenance'][str(seed)]
        if not provenance['artifact_sha256'] or not provenance['training_geometries'] or not provenance['training_recording_ids']:
            raise ValueError('frozen estimator provenance required')
        if set(provenance['training_recording_ids']) & {r['recording_id'] for r in subset if r['split'] != 'train'}:
            raise ValueError('estimator training overlaps calibration/test recordings')


def run(input_path, config_path, output):
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    with input_path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['seed'], r['label'] = int(r['seed']), int(r['label'])
        for k in ('b1_score', 'augmented_score', 'condition_score'):
            r[k] = float(r[k]) if r[k] else float('nan')
    validate(rows, cfg)
    metrics, predictions, thresholds = [], [], {}
    for seed in cfg['seeds']:
        subset = [r for r in rows if r['seed'] == seed]
        fitted = fit_policies([r for r in subset if r['split']=='calibration'],
                              cfg['calibration_fpr_budget'], cfg['condition_cutoff'])
        seen = {r['geometry'] for r in subset if r['split'] != 'test'}
        seen.update(cfg['estimator_provenance'][str(seed)]['training_geometries'])
        test = [r for r in subset if r['split']=='test']
        if cfg.get('require_unseen_geometry', True) and not any(r['geometry'] not in seen for r in test):
            raise ValueError('no geometry held out from CNN training, estimator training AND calibration')
        m, p = evaluate_groups(test, fitted, seen)
        metrics.extend(m)
        predictions.extend(p)
        fitted['seen_geometries'] = sorted(seen)
        fitted['unseen_facilities_using_B2'] = sorted({r['facility'] for r in test} - set(fitted['B3']))
        thresholds[str(seed)] = fitted
    summary = summarize(metrics, cfg['seeds'])
    delta = paired_deltas(metrics)
    delta_summary = summarize(delta, cfg['seeds'])
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [('metrics', metrics), ('summary', summary), ('predictions', predictions),
                         ('paired_P1_minus_B3', delta), ('paired_P1_minus_B3_summary', delta_summary)]:
        write_csv(output/(name+'.csv'), values)
    manifest = dict(status='frozen_score_evaluation_provenance_declared_by_input_producer', config=cfg,
                    input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (Path(__file__), ROOT/'src/fall_comparison.py', ROOT/'experiments/condition_threshold_validation.py')},
                    python=platform.python_version(), numpy=np.__version__, thresholds=thresholds,
                    unit='window', auroc='raw fall score', margin_auroc='fall score minus applied threshold',
                    limitations=['Input producer must verify checkpoints, training-only preprocessing, causal estimator inference and source identities.',
                                 'Recording disjointness is checked; subject independence is not established.',
                                 'Calibration FPR is an empirical constraint, not a test/deployment guarantee.'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.config, args.output_dir)

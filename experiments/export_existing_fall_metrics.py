"""Export existing B1/B2-like diagnostics; do not invent facility or P1 data."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.compare_fall_policies import write_csv
from experiments.condition_threshold_validation import select_threshold
from src.fall_comparison import operating_metrics, summarize
from src.temporal_integrity import verify_no_frame_overlap
from hashlib import sha256


def main():
    reports = {name:json.loads((ROOT/'results'/path).read_text(encoding='utf-8')) for name,path in
               [('B1', 'real_fall_baseline.json'), ('B2', 'real_fall_augmented.json')]}
    y = np.r_[np.load(ROOT/'y_train_bin.npy'), np.load(ROOT/'y_val_bin.npy')]
    rows, predictions, hashes = [], [], {}
    for method, report in reports.items():
        if [f['seed'] for f in report['folds']] != list(range(5)):
            raise ValueError('five seeds 0..4 required')
        for key in ('epochs','batch_size','learning_rate','input_sha256'):
            if report[key] != reports['B1'][key]:
                raise ValueError(f'unpaired training setting {key}')
        for name, expected in report['input_sha256'].items():
            if name not in hashes:
                hashes[name] = sha256((ROOT/name).read_bytes()).hexdigest()
            if hashes[name] != expected:
                raise ValueError(f'changed input {name}')
        for fold in report['folds']:
            seed = fold['seed']
            ref = reports['B1']['folds'][seed]
            if fold['split'] != ref['split']:
                raise ValueError('unpaired splits')
            verify_no_frame_overlap(report['chronological_order'], fold['split'])
            checkpoint = ROOT/'results'/fold.get('checkpoint_file', f'local_real_fall_seed{seed}.pt')
            if sha256(checkpoint.read_bytes()).hexdigest() != fold['checkpoint_sha256']:
                raise ValueError('checkpoint changed')
            tag = 'real_fall' if method=='B1' else 'real_fall_augmented'
            path = ROOT/'results'/f'local_{tag}_seed{seed}.npz'
            hashes[path.name] = sha256(path.read_bytes()).hexdigest()
            with np.load(path) as data:
                for part in ('calibration', 'test'):
                    np.testing.assert_array_equal(data[part+'_indices'], fold['split'][part])
                cal, scores = data['calibration_scores'], data['test_scores']
                threshold = select_threshold(y[fold['split']['calibration']], cal, np.full(len(cal),'all'), .05)
                np.testing.assert_equal(threshold, fold['selected_threshold'])
                ids = fold['split']['test']
                rows.append(dict(seed=seed, method=method, scope='existing_purged_diagnostic',
                                 facility='unknown', geometry='unknown', **operating_metrics(y[ids], scores, threshold)))
                predictions.extend(dict(seed=seed,method=method,sample_id=i,label=int(y[i]),
                                        score=float(score),threshold=threshold) for i,score in zip(ids,scores))
    output = ROOT/'results/fall_comparison_existing'
    output.mkdir(exist_ok=True)
    write_csv(output/'metrics.csv', rows)
    write_csv(output/'summary.csv', summarize(rows, list(range(5))))
    write_csv(output/'predictions.csv', predictions)
    manifest = dict(status='exploratory_existing_B1_B2_like_diagnostic_only', input_sha256=hashes,
                    data_provenance='unverified_arrays; user confirmed no measured data exists on 2026-09-06',
                    seeds=list(range(5)), calibration_fpr_budget=.05, unit='window',
                    unavailable=['B3: facility identity absent', 'P1: compatible estimator outputs absent',
                                 'unseen geometry: geometry identity absent'],
                    limitations=reports['B1']['limitations'] + [
                        'Seeds also rotate temporal folds: standard deviation mixes initialization and split variation.',
                        'Previously examined test windows; not a new confirmatory test.',
                        'B2 uses existing angular-mask augmentation, not the unrecovered historical B2 setup.',
                        'Scores are cached; checkpoint inference was verified by the preceding verification experiment.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summarize(rows, list(range(5))), indent=2))


if __name__ == '__main__':
    main()

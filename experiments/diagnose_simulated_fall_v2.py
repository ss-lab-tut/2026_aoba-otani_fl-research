"""Diagnose frozen CNN fit without retraining or selecting test thresholds."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_simulated_fall_p1_v2 import read_csv, infer, digest, LidarCNN
from experiments.compare_fall_policies import write_csv
from src.fall_comparison import operating_metrics, fit_policies


def run():
    source = ROOT / 'results/local_simulated_fall_p1_v2'
    output = ROOT / 'results/cnn_fit_v2'
    protocol = json.loads((source / 'protocol.json').read_text(encoding='utf-8'))
    if protocol['status'] != 'complete_simulation_only':
        raise ValueError('Completed experiment required')
    for name, expected in protocol['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Producer changed: {name}')
    for name, expected in [('observations.npz', protocol['observations_sha256']),
                           ('scores.csv', protocol['scores_sha256'])]:
        if digest(source / name) != expected:
            raise ValueError(f'Artifact changed: {name}')
    verification = json.loads((ROOT / 'results/simulated_fall_p1_v2/verification.json').read_text())
    if digest(source / 'windows.csv') != verification['input_sha256']['windows.csv']:
        raise ValueError('Window metadata changed')
    windows = {r['sample_id']: r for r in read_csv(source / 'windows.csv')}
    scores = read_csv(source / 'scores.csv')
    for r in scores:
        r['seed'], r['label'] = int(r['seed']), int(r['label'])
        for key in ('condition_score', 'b1_score', 'augmented_score'):
            r[key] = float(r[key]) if r[key] else float('nan')
    training = json.loads((source / 'training.json').read_text())
    if {(r['seed'], r['method']) for r in training} != {(s,m) for s in range(5) for m in ('B1','B2')} or len(training) != 10:
        raise ValueError('Ten distinct checkpoints required')
    with np.load(source / 'observations.npz') as cache:
        x = cache['x']
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    result, exported_scores = [], []
    for item in training:
        seed, method = item['seed'], item['method']
        checkpoint = source / f'{method}_seed{seed}.pt'
        if digest(checkpoint) != item['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        model = LidarCNN()
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        rows = [windows[k] for k in item['training_windows']]
        if any(r['split'] != 'train' for r in rows) or len(rows) != len({r['sample_id'] for r in rows}):
            raise ValueError('Invalid training selection')
        train_scores = infer(model, x, rows, 16)
        cal = [r for r in scores if r['seed'] == seed and r['split'] == 'calibration']
        key = 'b1_score' if method == 'B1' else 'augmented_score'
        np.testing.assert_allclose(infer(model, x, [windows[r['sample_id']] for r in cal[:16]],16),
                                   [r[key] for r in cal[:16]],rtol=1e-6,atol=1e-7)
        threshold = fit_policies(cal, 0.05, 0.)[method]
        groups = [('train_actual', rows, train_scores)]
        for split in ('calibration', 'test'):
            for scope in ('clear', 'seen', 'unseen', 'all'):
                selected = [r for r in scores if r['seed'] == seed and r['split'] == split and
                            (scope == 'all' or (scope == 'clear' and r['geometry'] == 'clear') or
                             (scope == 'seen' and r['geometry'] in protocol['config']['known_geometries']) or
                             (scope == 'unseen' and r['geometry'] in protocol['config']['unseen_geometries']))]
                if selected:
                    groups.append((split+'_'+scope,selected,[r[key] for r in selected]))
        for scope, selected, values in groups:
            y = np.array([int(r['label']) for r in selected])
            s = np.array(values)
            result.append(dict(seed=seed,method=method,scope=scope,threshold=threshold,
                               positive_score_mean=float(s[y==1].mean()),negative_score_mean=float(s[y==0].mean()),
                               positive_score_std=float(s[y==1].std()),negative_score_std=float(s[y==0].std()),
                               **operating_metrics(y,s,threshold)))
        exported_scores.extend(dict(seed=seed,method=method,sample_id=r['sample_id'],label=r['label'],score=s)
                               for r,s in zip(rows,train_scores))
        print(f'Diagnosed {method} seed {seed}: train AUROC {result[-len(groups)]["auroc"]:.3f}',flush=True)
    output.mkdir(exist_ok=True)
    write_csv(output/'metrics.csv',result)
    write_csv(output/'train_scores.csv',exported_scores)
    lines=['# Frozen CNN fit diagnosis', '',
           'No retraining. Operating metrics use the original calibration-only B1/B2 thresholds (budget 0.05).',
           'Train rows are the actual windows used by each model. Seeds share trajectories; these are descriptive diagnostics.', '',
           '| Method | Scope | AUROC mean ± SD | Recall mean ± SD | FPR mean ± SD |', '|---|---|---|---|---|']
    for method in ('B1','B2'):
        for scope in sorted({r['scope'] for r in result}):
            selected=[r for r in result if r['method']==method and r['scope']==scope]
            if len(selected)!=5: raise ValueError('Incomplete diagnostic group')
            cells=[f'{np.mean([r[m] for r in selected]):.3f} ± {np.std([r[m] for r in selected],ddof=1):.3f}'
                   for m in ('auroc','recall','false_positive_rate')]
            lines.append('| '+' | '.join([method,scope,*cells])+' |')
    lines += ['', 'Train/calibration/test mixtures differ: compare clear test with B1 training, and known-geometry test with B2 training cautiously.',
              'Training performance is descriptive and is not an estimate of generalization. A low training AUROC alone does not identify an optimization or representation cause.',
              'Post-hoc diagnostics do not change the primary comparison or select an improved model.', '', '## Verified input hashes', '']
    for name in ('training.json','windows.csv','scores.csv','observations.npz'):
        lines.append(f'- `{name}`: `{digest(source/name)}`')
    lines.append(f'- Diagnostic script: `{digest(Path(__file__))}`')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    run()

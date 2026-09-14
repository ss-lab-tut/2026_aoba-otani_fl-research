"""Verify and report paired six/eighteen epoch policies on the same new data."""
import gzip
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_simulated_fall_p1_v2 import read_csv,digest
from experiments.report_simulated_fall_p1 import report
from experiments.verify_simulated_fall_v2 import verify
from experiments.compare_fall_policies import write_csv
from src.fall_comparison import METRICS,summarize


def run():
    sources={n:ROOT/f'results/local_simulated_fall_p1_v2_fresh{n}' for n in (6,18)}
    output=ROOT/'results/fresh_trajectory_comparison'
    protocols={n:json.loads((s/'protocol.json').read_text()) for n,s in sources.items()}
    for n,p in protocols.items():
        if p['status']!='complete_simulation_only' or p['config']['epochs']!=n:raise ValueError('Incomplete run')
    cfgs=[{k:v for k,v in protocols[n]['config'].items() if k not in ('epochs','status')} for n in (6,18)]
    if cfgs[0]!=cfgs[1]:raise ValueError('Conditions differ beyond epochs')
    for name in ('observations.npz','windows.csv','selected_endpoints.csv','condition_model.npz'):
        if digest(sources[6]/name)!=digest(sources[18]/name):raise ValueError('Unpaired evaluation data')
    training={n:json.loads((s/'training.json').read_text()) for n,s in sources.items()}
    for item in training[6]:
        other=next(r for r in training[18] if (r['seed'],r['method'])==(item['seed'],item['method']))
        if item['training_windows']!=other['training_windows']:raise ValueError('Training selections differ')
        np.testing.assert_allclose(item['losses'],other['losses'][:6],rtol=1e-10,atol=1e-10)
    outputs={}
    for n,source in sources.items():
        public=output/f'epochs{n}'
        report(source,public);verify(source,public)
        (public/'frozen_scores.csv.gz').write_bytes(gzip.compress((source/'scores.csv').read_bytes(),mtime=0))
        outputs[n]=read_csv(public/'metrics.csv')
    keys=('seed','method','scope','facility','geometry')
    old={tuple(r[k] for k in keys):r for r in outputs[6]}
    if set(old)!={tuple(r[k] for k in keys) for r in outputs[18]}:raise ValueError('Unpaired metric groups')
    delta=[]
    for r in outputs[18]:
        previous=old[tuple(r[k] for k in keys)]
        if r['n']!=previous['n']:raise ValueError('Unpaired counts')
        item={k:r[k] for k in keys};item['seed']=int(item['seed'])
        item.update({m:float(r[m])-float(previous[m]) if r[m] and previous[m] else None for m in METRICS})
        delta.append(item)
    summary=summarize(delta,list(range(5)))
    write_csv(output/'paired_epochs18_minus_6.csv',delta)
    write_csv(output/'paired_epochs18_minus_6_summary.csv',summary)
    lines=['# Fresh-trajectory comparison: fixed six and eighteen epochs','',
           'Additional analysis; original main results are preserved. New calibration seeds 13001/13002, test seeds 14001/14002.',
           'Same training inputs, initializations and first-six-epoch losses verified. Same new evaluation windows and frozen condition estimator.',
           'Mean ± sample SD across five model seeds sharing trajectories. Calibration FPR budget 0.05.', '',
           '| Epochs | Scope | Policy | Recall | Precision | F1 | AUROC | FPR |','|---|---|---|---|---|---|---|---|']
    def fmt(r,m):
        return f'{float(r[m+"_mean"]):.3f} ± {float(r[m+"_std"]):.3f}' if r[m+'_mean'] not in ('',None) else 'undefined'
    for n in (6,18):
        for r in read_csv(output/f'epochs{n}'/'summary.csv'):
            if r['facility']==r['geometry']=='__all__' and r['scope'] in ('all','unseen'):
                lines.append('| '+' | '.join([str(n),r['scope'],r['method'],*[fmt(r,m) for m in ('recall','precision','f1','auroc','false_positive_rate')]])+' |')
    lines+=['','## P1 minus B3 within each epoch setting','',
            '| Epochs | Scope | Recall difference | F1 difference | FPR difference |','|---|---|---|---|---|']
    for n in (6,18):
        for r in read_csv(output/f'epochs{n}'/'paired_P1_minus_B3_summary.csv'):
            if r['facility']==r['geometry']=='__all__' and r['scope'] in ('all','unseen'):
                lines.append('| '+' | '.join([str(n),r['scope'],*[fmt(r,m) for m in ('recall','f1','false_positive_rate')]])+' |')
    lines+=['','## Epoch 18 minus epoch 6 on the same new data','',
            '| Scope | Policy | Recall difference | F1 difference | FPR difference |','|---|---|---|---|---|']
    for r in summary:
        if r['facility']==r['geometry']=='__all__' and r['scope'] in ('all','unseen'):
            lines.append('| '+' | '.join([r['scope'],r['method'],*[fmt(r,m) for m in ('recall','f1','false_positive_rate')]])+' |')
    lines+=['','These are synthetic endpoint metrics, not measured-data or continuous-monitoring results.',
            'Equal calibration FPR budgets do not imply equal test FPR. All four policies and both durations are retained.',
            'No best seed, epoch or operating point was selected using these test results.', '',
            'Verification: 20 checkpoints, 1,240 grouped confusion-count rows, 16 calibration inferences per checkpoint, identical training/evaluation data and matching six-epoch loss prefixes.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Fresh-trajectory comparison verified and published',flush=True)


if __name__=='__main__':run()

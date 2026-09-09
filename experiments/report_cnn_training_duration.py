"""Summarize the predeclared paired training-duration check."""
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'fall-paper-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'results/cnn_training_duration'
    with (source/'metrics.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    with (source/'paired_18_minus_6.csv').open(newline='') as f: paired=list(csv.DictReader(f))
    local=ROOT/'results/local_simulated_fall_p1_v2_duration'
    protocol=json.loads((source/'protocol.json').read_text())
    for name,expected_hash in protocol['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected_hash:
            raise ValueError('Development source changed')
    for seed in range(5):
        saved=json.loads((local/f'seed{seed}.json').read_text())
        if len(saved['checkpoints'])!=2: raise ValueError('Two checkpoints per seed required')
        for name,expected_hash in saved['checkpoints'].items():
            if hashlib.sha256((local/name).read_bytes()).hexdigest()!=expected_hash:
                raise ValueError('Development checkpoint changed')
        for original in saved['metrics']:
            exported=[r for r in rows if int(r['seed'])==seed and int(r['epoch'])==original['epoch'] and r['scope']==original['scope']]
            if len(exported)!=1: raise ValueError('Missing exported metric')
            for key in ('n','auroc','cross_entropy'):
                if float(exported[0][key])!=original[key]: raise ValueError('Metric export changed')
    for r in paired:
        points={int(m['epoch']):m for m in rows if m['seed']==r['seed'] and m['scope']==r['scope']}
        for metric in ('auroc','cross_entropy'):
            if float(r[metric+'_delta'])!=float(points[18][metric])-float(points[6][metric]):
                raise ValueError('Paired contrast changed')
    epochs=(1,3,6,12,18)
    expected={(str(s),str(e),scope) for s in range(5) for e in epochs for scope in ('fit','development')}
    if len(rows)!=len(expected) or {(r['seed'],r['epoch'],r['scope']) for r in rows}!=expected:
        raise ValueError('Incomplete development run')
    lines=['# B2 training duration: development-only check', '',
           'Five model seeds; fit records 10001–10003, development record 10004 in each facility.',
           'The primary comparison was fixed as 18 minus 6 epochs before execution. No calibration/test evaluation or best-epoch selection.', '',
           '| Scope | Epoch | AUROC mean ± SD | Cross entropy mean ± SD |', '|---|---|---|---|']
    for scope in ('fit','development'):
        for epoch in epochs:
            group=[r for r in rows if r['scope']==scope and int(r['epoch'])==epoch]
            cells=[f'{np.mean([float(r[m]) for r in group]):.4f} ± {np.std([float(r[m]) for r in group],ddof=1):.4f}' for m in ('auroc','cross_entropy')]
            lines.append('| '+' | '.join([scope,str(epoch),*cells])+' |')
    lines+=['', '## Paired epoch 18 minus epoch 6', '', '| Scope | AUROC difference | Cross entropy difference |', '|---|---|---|']
    for scope in ('fit','development'):
        group=[r for r in paired if r['scope']==scope]
        if len(group)!=5 or {r['seed'] for r in group}!={str(s) for s in range(5)}: raise ValueError('Unpaired seeds')
        cells=[f'{np.mean([float(r[m]) for r in group]):+.4f} ± {np.std([float(r[m]) for r in group],ddof=1):.4f}' for m in ('auroc_delta','cross_entropy_delta')]
        lines.append('| '+' | '.join([scope,*cells])+' |')
    lines+=['','AUROC: higher is better. Cross entropy: lower is better. Standard deviations describe model seeds, not independent data replications.',
            'This smaller fit partition is not directly comparable with the original full-training 6-epoch experiment.',
            'No threshold policy or condition estimator was altered, and these are not new B3/P1 test results.', '',
            'Verified: recorded producer hashes, all ten checkpoint hashes, all 50 exported metric rows and all paired differences.', '',
            '![Development learning curves](learning_curves.png)']
    (source/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,metric,title in zip(axes,('auroc','cross_entropy'),('Raw-score AUROC','Three-class cross entropy')):
        for scope,color in [('fit','#2476b4'),('development','#d76d20')]:
            means=[];stds=[]
            for epoch in epochs:
                values=[float(r[metric]) for r in rows if r['scope']==scope and int(r['epoch'])==epoch]
                means.append(np.mean(values));stds.append(np.std(values,ddof=1))
            ax.errorbar(epochs,means,yerr=stds,marker='o',capsize=3,label=scope,color=color)
        ax.axvline(6,color='gray',ls='--',lw=1);ax.set_xticks(epochs);ax.set_xlabel('Epoch');ax.set_title(title);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylim(0,1.05)
    fig.suptitle('B2: fixed development split, five seeds (mean ± sample SD)')
    for ext in ('png','svg'):fig.savefig(source/f'learning_curves.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Verified 10 checkpoints and 50 metric rows; report and learning curves saved.')


if __name__=='__main__':main()

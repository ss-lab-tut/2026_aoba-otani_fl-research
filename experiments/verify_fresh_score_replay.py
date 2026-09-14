"""Reproduce both published comparisons using Git-sized score artifacts only."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.compare_fall_policies import run


def read(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def verify_table(expected,actual):
    keys=('method','scope','facility','geometry')
    if 'seed' in expected[0]:keys=('seed',)+keys
    reference={tuple(r[k] for k in keys):r for r in expected}
    candidate={tuple(r[k] for k in keys):r for r in actual}
    if len(reference)!=len(expected) or len(candidate)!=len(actual) or reference.keys()!=candidate.keys():
        raise ValueError('Unpaired or duplicate rows in replay')
    for key,r in reference.items():
        other=candidate[key]
        if r.keys()!=other.keys():raise ValueError('Replayed columns differ')
        for column,value in r.items():
            if column in keys or value=='':
                if value!=other[column]:raise ValueError('Replayed metadata/undefined value differs')
            elif not math.isclose(float(value),float(other[column]),rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('Replayed number differs')


def main():
    source=ROOT/'results/fresh_trajectory_comparison'
    lines=['# Frozen-score replay verification', '',
           'Recomputed from the compressed scores and protocol saved in Git. No local observations or CNN checkpoints are read.', '',
           '| Epochs | Metric rows | Summary rows | Paired rows | Paired summary rows |', '|---|---|---|---|---|']
    for epoch in (6,18):
        folder=source/f'epochs{epoch}'
        audit=json.loads((folder/'audit.json').read_text())
        p=audit['protocol']
        raw=gzip.decompress((folder/'frozen_scores.csv.gz').read_bytes())
        if hashlib.sha256(raw).hexdigest()!=p['scores_sha256']:raise ValueError('Frozen scores changed')
        cfg=dict(seeds=p['config']['seeds'],calibration_fpr_budget=.05,condition_cutoff=0.,require_unseen_geometry=True,
                 estimator_provenance={str(s):p['condition_provenance'] for s in p['config']['seeds']})
        if p['config']['calibration_fpr_budget']!=.05:raise ValueError('Unexpected published budget')
        counts=[]
        with tempfile.TemporaryDirectory(prefix='fall-replay-') as temporary:
            work=Path(temporary)
            (work/'scores.csv').write_bytes(raw)
            (work/'config.json').write_text(json.dumps(cfg))
            run(work/'scores.csv',work/'config.json',work/'comparison')
            for name in ('metrics.csv','summary.csv','paired_P1_minus_B3.csv','paired_P1_minus_B3_summary.csv'):
                if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=audit['exported_sha256'][name]:
                    raise ValueError('Published table changed')
                expected=read(folder/name)
                verify_table(expected,read(work/'comparison'/name));counts.append(len(expected))
        lines.append('| '+' | '.join(map(str,[epoch,*counts]))+' |')
        print(f'Verified {epoch}-epoch frozen-score replay',flush=True)
    lines+=['', 'All confusion counts, operating metrics, seed summaries and P1-minus-B3 differences match the published tables (numeric tolerance 1e-12).',
            'The input score hashes and all four exported-table hashes were checked against each audit record.',
            'This validates score-to-table reproduction; it does not retrain models or independently validate the simulator.', '',
            'Reproduce with `python experiments/verify_fresh_score_replay.py` (NumPy required).', '',
            f'Script SHA-256: `{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}`']
    (source/'REPLAY_VERIFICATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()

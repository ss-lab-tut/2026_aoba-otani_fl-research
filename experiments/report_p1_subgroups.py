"""Report every facility/geometry subgroup for the fixed fresh 18-epoch run."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'results/fresh_trajectory_comparison/epochs18'
    audit=json.loads((source/'audit.json').read_text())
    data={}
    for name in ('summary.csv','paired_P1_minus_B3.csv','paired_P1_minus_B3_summary.csv'):
        path=source/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=audit['exported_sha256'][name]:
            raise ValueError('Verified comparison table changed')
        with path.open(newline='',encoding='utf-8') as f:data[name]=list(csv.DictReader(f))
    changes=data['paired_P1_minus_B3_summary.csv']
    per_seed=data['paired_P1_minus_B3.csv']
    def values(r):
        return [f'{float(r[m+"_mean"]):+.4f}' for m in ('recall','f1','false_positive_rate')]
    lines=['# P1 versus B3: all facility and geometry subgroups', '',
           'Fresh-trajectory 18-epoch comparison, five model seeds. Differences are P1 minus B3.',
           'Positive recall/F1 and negative FPR differences are favorable. No subgroups are selected for model tuning.', '',
           '## Aggregate and facility groups', '',
           '| Scope | Facility | Recall difference | F1 difference | FPR difference |', '|---|---|---|---|---|']
    for r in changes:
        if r['geometry']=='__all__':
            lines.append('| '+' | '.join([r['scope'],r['facility'],*values(r)])+' |')
    lines+=['','## All facility × geometry groups','',
            'Wins count strict within-seed improvements; ties are not wins. Five seeds share trajectories.', '',
            '| Scope | Facility | Geometry | Recall difference | F1 difference | FPR difference | Recall wins / 5 | F1 wins / 5 | FPR reductions / 5 |',
            '|---|---|---|---|---|---|---|---|---|']
    groups=[]
    for r in changes:
        if r['geometry']=='__all__' or r['facility']=='__all__':continue
        group=[s for s in per_seed if all(s[k]==r[k] for k in ('scope','facility','geometry'))]
        if len(group)!=5 or {s['seed'] for s in group}!={str(i) for i in range(5)}:
            raise ValueError('Incomplete subgroup seeds')
        wins=[sum(float(s[m])>0 for s in group) for m in ('recall','f1')]
        wins.append(sum(float(s['false_positive_rate'])<0 for s in group))
        lines.append('| '+' | '.join([r['scope'],r['facility'],r['geometry'],*values(r),*[str(n) for n in wins]])+' |')
        groups.append(r)
    if len(groups)!=16:raise ValueError('Expected 2 facilities by 8 geometries')
    lines+=['','## Mean subgroup directions','',
            '| Scope | Subgroups | Recall positive / zero / negative | F1 positive / zero / negative | FPR lower / equal / higher |',
            '|---|---|---|---|---|']
    for scope in ('seen','unseen'):
        selected=[r for r in groups if r['scope']==scope]
        cells=[]
        for metric,direction in [('recall',1),('f1',1),('false_positive_rate',-1)]:
            v=[direction*float(r[metric+'_mean']) for r in selected]
            cells.append(' / '.join(str(sum(check(x) for x in v)) for check in (lambda x:x>1e-12,lambda x:abs(x)<=1e-12,lambda x:x< -1e-12)))
        lines.append('| '+' | '.join([scope,str(len(selected)),*cells])+' |')
    lines+=['','Group-level counts are descriptive, not independent statistical trials. Positive aggregate differences do not imply uniform benefits.',
            'The common calibration FPR budget is 0.05; it is not a matched test-FPR comparison.',
            'See the original grouped CSVs for policy values, sample standard deviations and all seeds.']
    target=ROOT/'results/fresh_trajectory_comparison/SUBGROUPS.md'
    target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Verified and reported all 16 facility/geometry groups')


if __name__=='__main__':main()

"""Export manuscript tables from the verified five-seed comparison."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/simulated_fall_p1_v2'
METRICS = ('recall', 'precision', 'f1', 'auroc', 'false_positive_rate')


def main():
    audit = json.loads((SOURCE / 'audit.json').read_text(encoding='utf-8'))
    tables = {}
    hashes = {}
    for name in ('summary.csv', 'paired_P1_minus_B3_summary.csv'):
        path = SOURCE / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[name] != audit['exported_sha256'][name]:
            raise ValueError(f'Published table hash mismatch: {name}')
        with path.open(encoding='utf-8', newline='') as stream:
            tables[name] = list(csv.DictReader(stream))
    output = ROOT / 'paper'
    output.mkdir(exist_ok=True)
    lines = ['# Tables generated from the verified comparison', '',
             'Mean ± sample standard deviation over five model seeds sharing fixed trajectories.',
             'Main calibration FPR budget: 0.05. These are simulated endpoint metrics.', '']
    for scope in ('all', 'seen', 'unseen'):
        rows = [r for r in tables['summary.csv'] if r['scope'] == scope
                and r['facility'] == r['geometry'] == '__all__']
        if len(rows) != 4 or {r['method'] for r in rows} != {'B1', 'B2', 'B3', 'P1'}:
            raise ValueError('Incomplete policy table')
        lines += [f'## {scope}', '', '| Method | Recall | Precision | F1 | AUROC | FPR |',
                  '|---|---|---|---|---|---|']
        for r in rows:
            if int(r['n_seeds']) != 5 or any(int(r[m+'_n_defined']) != 5 for m in METRICS):
                raise ValueError('Incomplete seed metrics')
            cells = [f'{float(r[m+"_mean"]):.3f} ± {float(r[m+"_std"]):.3f}' for m in METRICS]
            lines.append('| ' + ' | '.join([r['method'], *cells]) + ' |')
        lines.append('')
    lines += ['## Paired P1 minus B3', '',
              'Differences are computed within each seed before averaging. Negative FPR differences mean fewer false positives.', '',
              '| Scope | Recall difference | Precision difference | F1 difference | AUROC difference | FPR difference |',
              '|---|---|---|---|---|---|']
    for r in tables['paired_P1_minus_B3_summary.csv']:
        if r['facility'] != '__all__' or r['geometry'] != '__all__':
            continue
        cells = [f'{float(r[m+"_mean"]):+.4f} ± {float(r[m+"_std"]):.4f}' for m in METRICS]
        lines.append('| ' + ' | '.join([r['scope'], *cells]) + ' |')
    lines += ['', '## Source SHA-256', '']
    lines += [f'- `{name}`: `{value}`' for name, value in hashes.items()]
    (output / 'TABLES.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Exported all/seen/unseen policy tables and paired differences to paper/TABLES.md')


if __name__ == '__main__':
    main()

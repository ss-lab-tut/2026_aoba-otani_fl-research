"""Compare calibration-selected operating points without retraining or test tuning."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.fall_comparison import fit_policies,apply_policies,evaluate_groups,summarize,paired_deltas,operating_metrics
from experiments.compare_fall_policies import write_csv,validate


def read_scores(path,expected_hash):
    raw=gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest()!=expected_hash:raise ValueError('frozen scores changed')
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    for r in rows:
        r['seed'],r['label']=int(r['seed']),int(r['label'])
        for key in ('b1_score','augmented_score','condition_score'):
            r[key]=float(r[key]) if r[key] else float('nan')
    return rows


def calibration_check(rows,fitted,budget):
    thresholds=apply_policies(rows,fitted)
    records=[]
    for method,values in thresholds.items():
        group=np.array([r['facility'] if method=='B3' else str(int(r['condition_score']>=fitted['condition_cutoff']))
                        if method=='P1' else 'all' for r in rows])
        for name in np.unique(group):
            ids=np.flatnonzero(group==name)
            m=operating_metrics([rows[i]['label'] for i in ids],
                                [rows[i]['b1_score' if method=='B1' else 'augmented_score'] for i in ids],values[ids])
            fallback=method in ('B3','P1') and name not in fitted[method]
            if not fallback and m['false_positive_rate'] is not None and m['false_positive_rate']>budget+1e-12:
                raise ValueError('calibration FPR constraint violated')
            records.append(dict(seed=rows[0]['seed'],budget=budget,method=method,calibration_group=str(name),
                                fallback=fallback,**m))
    return records


def main(source,config,output):
    experiment=json.loads((source/'audit.json').read_text(encoding='utf-8'))['protocol']
    cfg=json.loads(config.read_text(encoding='utf-8'))
    budgets=cfg['calibration_fpr_budgets']
    if len(set(budgets))!=len(budgets) or cfg['main_budget'] not in budgets or not all(0<=b<1 for b in budgets):
        raise ValueError('unique valid budgets and original main budget required')
    rows=read_scores(source/'frozen_scores.csv.gz',experiment['scores_sha256'])
    seeds=experiment['config']['seeds']
    provenance=experiment['condition_provenance']
    validate(rows,dict(seeds=seeds,condition_cutoff=0.,calibration_fpr_budget=cfg['main_budget'],
                      estimator_provenance={str(s):provenance for s in seeds}))
    all_metrics,all_summary,all_deltas,delta_summary,calibration=[],[],[],[],[]
    fitted_manifest={}
    for budget in budgets:
        metrics=[];fitted_manifest[str(budget)]={}
        for seed in seeds:
            selected=[r for r in rows if r['seed']==seed]
            cal=[r for r in selected if r['split']=='calibration']
            fitted=fit_policies(cal,budget,0.)
            calibration.extend(calibration_check(cal,fitted,budget))
            seen={r['geometry'] for r in selected if r['split']!='test'}|set(provenance['training_geometries'])
            test=[r for r in selected if r['split']=='test']
            m,_=evaluate_groups(test,fitted,seen)
            metrics.extend(m);fitted_manifest[str(budget)][str(seed)]=fitted
        ds=paired_deltas(metrics)
        for target,data in ((all_metrics,metrics),(all_summary,summarize(metrics,seeds)),
                            (all_deltas,ds),(delta_summary,summarize(ds,seeds))):
            target.extend(dict(budget=budget,**r) for r in data)
    # The main operating point must exactly recover the already published counts/metrics.
    with (source/'metrics.csv').open(encoding='utf-8',newline='') as f:old=list(csv.DictReader(f))
    keys=('seed','method','scope','facility','geometry')
    reference={tuple(str(r[k]) for k in keys):r for r in old}
    original=[r for r in all_metrics if r['budget']==cfg['main_budget']]
    if len(original)!=len(reference):raise ValueError('main result group mismatch')
    for r in original:
        ref=reference[tuple(str(r[k]) for k in keys)]
        for k,v in r.items():
            if k=='budget' or k in keys:continue
            if v is None:
                if ref[k]!='':raise ValueError('undefined metric changed')
            else:np.testing.assert_allclose(float(v),float(ref[k]),rtol=1e-12,atol=1e-12)
    output.mkdir(parents=True,exist_ok=True)
    for name,data in [('metrics',all_metrics),('summary',all_summary),('paired_P1_minus_B3',all_deltas),
                      ('paired_P1_minus_B3_summary',delta_summary),('calibration_checks',calibration)]:
        write_csv(output/(name+'.csv'),data)
    manifest=dict(status='descriptive_calibration_budget_sensitivity_not_test_optimization',config=cfg,
                  scores_sha256=experiment['scores_sha256'],main_analysis_reproduced=True,thresholds=fitted_manifest,
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (Path(__file__),ROOT/'src/fall_comparison.py',ROOT/'experiments/condition_threshold_validation.py')})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    lines=['# 固定スコアでの動作点比較', '',
           '全thresholdをcalibrationだけで選択。主解析の予算0.05は変更せず、他の予算は記述的な感度分析。',
           '**同じcalibration FPR予算でも、実際のtest FPRが一致するわけではない。**', '',
           '| 範囲 | calibration FPR予算 | 方法 | test Recall平均 | test FPR平均 | test F1平均 |',
           '|---|---:|---|---:|---:|---:|']
    for r in all_summary:
        if r['scope'] in ('all','unseen') and r['method'] in ('B3','P1') and r['facility']==r['geometry']=='__all__':
            lines.append(f'| {r["scope"]} | {r["budget"]:.3f} | {r["method"]} | {r["recall_mean"]:.3f} | {r["false_positive_rate_mean"]:.3f} | {r["f1_mean"]:.3f} |')
    lines += ['', '5 seedsの標本標準偏差・施設/geometry別の値・対応差はCSVに保存。',
              '0.05の元の主解析と全groupの混同行列・指標が一致することを検証した。',
              'このtest曲線から都合のよい予算を選び、新しい汎化性能として報告しない。',
              '判定時点指定・正例率1/3のシミュレーションという元実験の制約を引き継ぐ。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps([r for r in delta_summary if r['scope']=='unseen' and r['facility']==r['geometry']=='__all__'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'results/simulated_fall_p1_v2')
    p.add_argument('--config',type=Path,default=ROOT/'configs/fall_fpr_sweep.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/fall_operating_points')
    a=p.parse_args();main(a.source,a.config,a.output_dir)

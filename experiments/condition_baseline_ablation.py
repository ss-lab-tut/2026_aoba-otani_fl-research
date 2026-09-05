"""Controlled replacement of the spatial reference, holding splits fixed."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.condition_failure_diagnosis import (
    GEOMETRIES, FEATURE_NAMES, FEATURE_SETS, ObservableConditionEstimator,
    build_cache, fold_keys, stable_margin, metrics, roc_auc, select_threshold,
    summarize, trailing_mean,
)
from src.robust_condition import bounded_contrast


def run(cache_path):
    seeds = list(range(5))
    names = [g.name for g in GEOMETRIES]
    cols = [FEATURE_NAMES.index(n) for n in FEATURE_SETS['sector_counts_only']]
    variants = ('legacy', 'mean_baseline', 'bounded_mean', 'shape_only')
    shape_cols = [FEATURE_NAMES.index(n) for n in ('lidar_available', 'z_mean','z_std','z_q10','z_q50','z_q90','floor02_ratio','floor04_ratio','x_std','y_std','xy_spread')]
    data = {}
    with np.load(cache_path, allow_pickle=False) as cache:
        for s in seeds:
            for g in names:
                k = f'{s}_{g}'
                x, xm, y = cache[k+'_x'], cache[k+'_mean_x'], cache[k+'_y']
                matrices = [x[:, cols], xm[:, cols], bounded_contrast(xm[:, cols]), x[:,shape_cols]]
                data[s,g] = {name: (np.vstack([trailing_mean(a,5) for a in np.split(matrix,2)]),y)
                             for name,matrix in zip(variants,matrices)}
        baselines = json.loads(str(cache['baseline_json']))
    folds=[]
    for s in seeds:
        for g in names:
            train,val,test=fold_keys(seeds,names,s,g)
            result={}
            for v in variants:
                def join(keys):
                    return (np.vstack([data[k][v][0] for k in keys]),np.concatenate([data[k][v][1] for k in keys]),
                            np.concatenate([np.full(len(data[k][v][1]),k[1]) for k in keys]))
                xt,yt,_=join(train); xv,yv,gv=join(val); xe,ye=data[test][v]
                model=ObservableConditionEstimator().fit(xt,yt)
                sv,se=stable_margin(model,xv),stable_margin(model,xe)
                for policy,t in [('fixed',0),('calibrated05',select_threshold(yv,sv,gv,.05)),('calibrated10',select_threshold(yv,sv,gv,.1))]:
                    m=metrics(ye,se,t); m.update(auc=roc_auc(ye,se),threshold=t)
                    result[v+'_'+policy]=m
            folds.append(dict(seed=s,geometry=g,variants=result))
    return dict(status='exploratory_baseline_ablation',folds=folds,summary=summarize(folds),baselines=baselines)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,default=ROOT/'results/local_condition_robust_cache.npz')
    p.add_argument('--output',type=Path,default=ROOT/'results/condition_baseline_ablation.json')
    args=p.parse_args()
    if not args.cache.exists(): build_cache(args.cache,list(range(5)))
    r=run(args.cache)
    args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v['all'] for k,v in r['summary'].items()},indent=2))

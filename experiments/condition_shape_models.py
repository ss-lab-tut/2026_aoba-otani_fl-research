"""Compare fixed shape-score candidates on development folds only."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.condition_failure_diagnosis import (
    GEOMETRIES, FEATURE_NAMES, ObservableConditionEstimator, fold_keys,
    stable_margin, metrics, roc_auc, select_threshold, summarize, trailing_mean,
)
from src.robust_condition import QuadraticRidgeScore

SHAPE_NAMES=('lidar_available','z_mean','z_std','z_q10','z_q50','z_q90',
             'floor02_ratio','floor04_ratio','x_std','y_std','xy_spread')


def prepare(x,window):
    columns=[FEATURE_NAMES.index(n) for n in SHAPE_NAMES]
    return np.vstack([trailing_mean(a[:,columns],window) for a in np.split(x,2)])


def run():
    seeds=list(range(5)); names=[g.name for g in GEOMETRIES]
    with np.load(ROOT/'results/local_condition_robust_cache.npz') as c:
        data={(s,g):(c[f'{s}_{g}_x'],c[f'{s}_{g}_y']) for s in seeds for g in names}
    folds=[]
    for s in seeds:
        for g in names:
            train,val,test=fold_keys(seeds,names,s,g)
            variants={}
            for window in (1,5):
                def join(keys):
                    return (np.vstack([prepare(data[k][0],window) for k in keys]),
                            np.concatenate([data[k][1] for k in keys]),
                            np.concatenate([np.full(len(data[k][1]),k[1]) for k in keys]))
                xt,yt,_=join(train); xv,yv,gv=join(val)
                xe=prepare(data[test][0],window); ye=data[test][1]
                for kind in ('centroid','quadratic'):
                    if kind=='quadratic':
                        model=QuadraticRidgeScore().fit(xt,yt)
                        sv,se=model.score(xv),model.score(xe)
                    else:
                        model=ObservableConditionEstimator().fit(xt,yt)
                        sv,se=stable_margin(model,xv),stable_margin(model,xe)
                    for budget in (.05,.1):
                        t=select_threshold(yv,sv,gv,budget)
                        m=metrics(ye,se,t);m.update(auc=roc_auc(ye,se),threshold=t)
                        variants[f'{kind}_window{window}_fpr{budget}']=m
            folds.append(dict(seed=s,geometry=g,variants=variants))
        print(f'Evaluated development seed {s}',flush=True)
    return dict(status='exploratory_model_selection_not_final_test',folds=folds,summary=summarize(folds))


if __name__=='__main__':
    r=run()
    (ROOT/'results/condition_shape_models.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v['all'] for k,v in r['summary'].items()},indent=2))

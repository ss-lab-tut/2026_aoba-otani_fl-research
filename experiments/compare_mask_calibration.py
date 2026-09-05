"""Compare single-position and multiple-position calibration on the same masks."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_real_fall_baseline import tensors,scores,sha256
from experiments.condition_threshold_validation import select_threshold
from src.input_quality import angular_zero_mask,observed_quality_groups,route_thresholds
from src.evaluation import evaluate_tradeoff
from model import LidarCNN


def main():
    torch.set_num_threads(4)
    cfgpath=ROOT/'configs/real_2d_mask_calibration.json'
    cfg=json.loads(cfgpath.read_text())
    baseline=json.loads((ROOT/cfg['baseline']).read_text())
    previous=json.loads((ROOT/cfg['previous_comparison']).read_text())
    x=np.concatenate([np.load(ROOT/'X_train.npy'),np.load(ROOT/'X_val_real.npy')])
    y=np.r_[np.load(ROOT/'y_train_bin.npy'),np.load(ROOT/'y_val_bin.npy')]
    folds=[]
    for fold in baseline['folds']:
        seed=fold['seed'];model=LidarCNN()
        checkpoint=ROOT/'results'/f'local_real_fall_seed{seed}.pt'
        if sha256(checkpoint)!=fold['checkpoint_sha256']:raise ValueError('checkpoint changed')
        model.load_state_dict(torch.load(checkpoint,weights_only=True))
        def collect(partition,angles):
            ids=fold['split'][partition];result=[[],[],[],[]]
            conditions=[(0.,0.)]+[(fraction,angle) for fraction in cfg['mask_fractions'] for angle in angles]
            for fraction,angle in conditions:
                z=angular_zero_mask(x[ids],fraction,angle)
                ds=tensors(z,np.zeros(len(z),dtype=int),range(len(z)))
                values=[scores(model,ds,32),y[ids],observed_quality_groups(z),
                        np.full(len(z),f'mask{fraction}_start{angle}')]
                for output,value in zip(result,values):output.extend(value)
            return tuple(np.array(v) for v in result)
        cs,cy,cg,cc=collect('calibration',cfg['calibration_start_fractions'])
        budget=cfg['calibration_fpr_limit_per_mask_position']
        common=select_threshold(cy,cs,cc,budget)
        conditional={str(g):select_threshold(cy[cg==g],cs[cg==g],cc[cg==g],budget) for g in np.unique(cg)}
        # Thresholds fixed before collecting scores for the new stress positions.
        ts,ty,tg,tc=collect('test',cfg['test_start_fractions'])
        old=next(f for f in previous['folds'] if f['seed']==seed)
        thresholds={
            'single_angle_quality':route_thresholds(tg,old['quality_thresholds'],old['common_threshold']),
            'multi_angle_common':np.full(len(ts),common),
            'multi_angle_quality':route_thresholds(tg,conditional,common),
        }
        policies={name:evaluate_tradeoff(ty,ts-values,tc,thresholds=[0.]) for name,values in thresholds.items()}
        folds.append(dict(seed=seed,common_threshold=common,quality_thresholds=conditional,policies=policies))
        print(f'Multi-position calibration seed {seed} finished',flush=True)
    summary={}
    for policy in folds[0]['policies']:
        summary[policy]={}
        for condition in [r['condition'] for r in folds[0]['policies'][policy]]:
            rows=[next(r for r in f['policies'][policy] if r['condition']==condition) for f in folds]
            summary[policy][condition]={m:float(np.mean([r[m] for r in rows])) for m in ('recall','false_alert_rate')}
    q=summary['multi_angle_quality'];c=summary['multi_angle_common']['__overall__']
    checks=dict(mean_fpr=q['__overall__']['false_alert_rate']<=cfg['acceptance']['quality_mean_fpr_max'],
                per_scenario_fpr=all(v['false_alert_rate']<=cfg['acceptance']['each_test_scenario_mean_fpr_max'] for k,v in q.items() if k!='__overall__'),
                recall=q['__overall__']['recall']>=c['recall'])
    r=dict(status=cfg['status'],config=cfg,config_sha256=sha256(cfgpath),
           baseline_sha256=sha256(ROOT/cfg['baseline']),previous_comparison_sha256=sha256(ROOT/cfg['previous_comparison']),
           folds=folds,summary=summary,acceptance=dict(passed=all(checks.values()),checks=checks))
    (ROOT/'results/real_2d_mask_calibration.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'overall':{k:v['__overall__'] for k,v in summary.items()},'acceptance':r['acceptance']},indent=2))


if __name__=='__main__':main()

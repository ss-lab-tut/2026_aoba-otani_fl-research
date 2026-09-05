"""Frozen-CNN comparison on artificial input masking, not physical occlusion."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_real_fall_baseline import tensors,scores,sha256
from experiments.condition_threshold_validation import select_threshold
from src.evaluation import evaluate_tradeoff
from src.input_quality import angular_zero_mask,observed_quality_groups,route_thresholds
from model import LidarCNN


def main():
    torch.set_num_threads(4)
    baseline=json.loads((ROOT/'results/real_fall_baseline.json').read_text())
    x=np.concatenate([np.load(ROOT/'X_train.npy'),np.load(ROOT/'X_val_real.npy')])
    y=np.r_[np.load(ROOT/'y_train_bin.npy'),np.load(ROOT/'y_val_bin.npy')]
    # Fixed before running stress evaluation: test masks use different angles.
    scenarios=[('unmasked',0.,0.,0.),('mask15',.15,.10,.65),('mask35',.35,.30,.80)]
    folds=[]
    for fold in baseline['folds']:
        model=LidarCNN()
        checkpoint=ROOT/'results'/f"local_real_fall_seed{fold['seed']}.pt"
        if sha256(checkpoint)!=fold['checkpoint_sha256']:raise ValueError('checkpoint changed')
        model.load_state_dict(torch.load(checkpoint,weights_only=True))
        outputs={}
        for partition,angle_index in [('calibration',2),('test',3)]:
            ids=fold['split'][partition];pred=[];labels=[];groups=[];conditions=[]
            for scenario in scenarios:
                z=angular_zero_mask(x[ids],scenario[1],scenario[angle_index])
                ds=tensors(z,np.zeros(len(z),dtype=int),list(range(len(z))))
                pred.extend(scores(model,ds,32));labels.extend(y[ids])
                groups.extend(observed_quality_groups(z));conditions.extend([scenario[0]]*len(z))
            outputs[partition]=(np.array(pred),np.array(labels),np.array(groups),np.array(conditions))
        cs,cy,cg,cc=outputs['calibration'];ts,ty,tg,tc=outputs['test']
        common=select_threshold(cy,cs,cg,.05)
        conditional={str(g):select_threshold(cy[cg==g],cs[cg==g],cg[cg==g],.05) for g in np.unique(cg)}
        decisions={'clean_calibration_threshold':np.full(len(ts),fold['selected_threshold']),
                   'stress_common_threshold':np.full(len(ts),common),
                   'observed_quality_thresholds':route_thresholds(tg,conditional,common)}
        policies={}
        for name,thresholds in decisions.items():
            # A zero threshold on score-minus-operating-point preserves inclusive ties.
            policies[name]=evaluate_tradeoff(ty,ts-thresholds,tc,thresholds=[0.])
        folds.append(dict(seed=fold['seed'],common_threshold=common,quality_thresholds=conditional,policies=policies))
        print(f"Stress evaluation seed {fold['seed']} finished",flush=True)
    summary={}
    for policy in folds[0]['policies']:
        summary[policy]={}
        for condition in [r['condition'] for r in folds[0]['policies'][policy]]:
            rows=[next(r for r in f['policies'][policy] if r['condition']==condition) for f in folds]
            summary[policy][condition]={m:float(np.mean([r[m] for r in rows])) for m in ('recall','false_alert_rate')}
    report=dict(status='exploratory_2d_zero_mask_stress_not_physical_occlusion_or_P1_validation',
                baseline_file='real_fall_baseline.json',scenarios=scenarios,folds=folds,summary=summary,
                limitations=['Masks replace already normalized values with zero, not a modeled sensor ray or actual missing return.',
                             'Quality routing measures observable zero fraction, not the prior z-based degradation estimator.',
                             'Facilities, subjects, source preprocessing and event identities remain unverified.',
                             'No claim of physical occlusion robustness or event false-alert reduction.'])
    (ROOT/'results/real_2d_input_stress.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

"""Frozen B1-like versus masked-training CNN diagnostic on artificial masks."""
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
    cfgpath=ROOT/'configs/real_2d_augmentation_comparison.json';cfg=json.loads(cfgpath.read_text())
    reports={name:json.loads((ROOT/'results'/file).read_text()) for name,file in
             [('baseline','real_fall_baseline.json'),('augmented','real_fall_augmented.json')]}
    if any(r['epochs']!=cfg['epochs'] or [f['seed'] for f in r['folds']]!=cfg['seeds'] for r in reports.values()):
        raise ValueError('epoch and seed settings must match the frozen comparison')
    x=np.concatenate([np.load(ROOT/'X_train.npy'),np.load(ROOT/'X_val_real.npy')])
    y=np.r_[np.load(ROOT/'y_train_bin.npy'),np.load(ROOT/'y_val_bin.npy')]
    folds=[]
    for seed in cfg['seeds']:
        rows={};operating_points={}
        for kind,report in reports.items():
            fold=next(f for f in report['folds'] if f['seed']==seed)
            reference=next(f for f in reports['baseline']['folds'] if f['seed']==seed)
            if fold['split']!=reference['split']:raise ValueError('models must use identical splits')
            checkpoint=ROOT/'results'/fold.get('checkpoint_file',f'local_real_fall_seed{seed}.pt')
            if sha256(checkpoint)!=fold['checkpoint_sha256']:raise ValueError('checkpoint changed')
            model=LidarCNN();model.load_state_dict(torch.load(checkpoint,weights_only=True))
            def collect(partition,angles):
                ids=fold['split'][partition];items=[[],[],[],[]]
                scenarios=[(0.,0.)]+[(f,a) for f in cfg['test_mask_fractions'] for a in angles]
                for fraction,angle in scenarios:
                    z=angular_zero_mask(x[ids],fraction,angle)
                    ds=tensors(z,np.zeros(len(z),dtype=int),range(len(z)))
                    values=[scores(model,ds,32),y[ids],observed_quality_groups(z),
                            np.full(len(z),f'mask{fraction}_start{angle}')]
                    for output,value in zip(items,values):output.extend(value)
                return tuple(np.array(v) for v in items)
            cs,cy,cg,cc=collect('calibration',cfg['calibration_start_fractions'])
            common=select_threshold(cy,cs,cc,cfg['calibration_fpr_budget'])
            conditional={str(g):select_threshold(cy[cg==g],cs[cg==g],cc[cg==g],cfg['calibration_fpr_budget']) for g in np.unique(cg)}
            ts,ty,tg,tc=collect('test',cfg['test_start_fractions'])
            for policy,thresholds in [('common',np.full(len(ts),common)),('quality',route_thresholds(tg,conditional,common))]:
                rows[kind+'_'+policy]=evaluate_tradeoff(ty,ts-thresholds,tc,thresholds=[0.])
            operating_points[kind]=dict(common=common,conditional=conditional)
        folds.append(dict(seed=seed,policies=rows,operating_points=operating_points))
        print(f'Augmentation comparison seed {seed} finished',flush=True)
    summary={}
    for policy in folds[0]['policies']:
        summary[policy]={}
        for condition in [r['condition'] for r in folds[0]['policies'][policy]]:
            points=[next(r for r in f['policies'][policy] if r['condition']==condition) for f in folds]
            summary[policy][condition]={m:float(np.mean([p[m] for p in points])) for m in ('recall','false_alert_rate')}
    proposed=summary['augmented_quality'];reference=summary['baseline_quality'];limits=cfg['positive_result_requirements']
    checks=dict(recall=proposed['__overall__']['recall']>=reference['__overall__']['recall'],
                mean_fpr=proposed['__overall__']['false_alert_rate']<=limits['overall_fpr_at_most'],
                each_scenario_recall=all(v['recall']>=limits['each_scenario_mean_recall_at_least'] for k,v in proposed.items() if k!='__overall__'),
                each_scenario_fpr=all(v['false_alert_rate']<=limits['each_scenario_mean_fpr_at_most'] for k,v in proposed.items() if k!='__overall__'))
    report=dict(status=cfg['status'],config=cfg,config_sha256=sha256(cfgpath),folds=folds,summary=summary,
                acceptance=dict(passed=all(checks.values()),checks=checks),
                limitations=cfg['restrictions']+['Temporal test blocks have been examined in preceding exploratory diagnostics.'])
    (ROOT/'results/real_2d_augmentation_comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'overall':{k:v['__overall__'] for k,v in summary.items()},'acceptance':report['acceptance']},indent=2))


if __name__=='__main__':main()

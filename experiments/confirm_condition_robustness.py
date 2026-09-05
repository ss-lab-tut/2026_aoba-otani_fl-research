"""One-shot confirmation using the protocol frozen before test generation."""
import hashlib
import argparse
import json
import platform
import sys
import subprocess
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.condition_failure_diagnosis import (
    GEOMETRIES, FEATURE_NAMES, FEATURE_SETS, ObservableConditionEstimator,
    build_cache, load_datasets, stable_margin, metrics, roc_auc, select_threshold,
    source_manifest, summarize, trailing_mean,
)
from heterosense._scripts.structured_occlusion_pilot import Geometry
from experiments.condition_shape_models import prepare, SHAPE_NAMES
from experiments.condition_threshold_validation import git_info
from src.robust_condition import QuadraticRidgeScore


def acceptance(summary,folds,limits):
    overall=summary['all']
    groups=[v for k,v in summary.items() if k!='all']
    checks={
        'mean_recall':overall['recall']>=limits['mean_recall_min'],
        'mean_fpr':overall['false_positive_rate']<=limits['mean_fpr_max'],
        'each_geometry_mean_recall':all(g['recall']>=limits['each_geometry_mean_recall_min'] for g in groups),
        'each_geometry_mean_fpr':all(g['false_positive_rate']<=limits['each_geometry_mean_fpr_max'] for g in groups),
        'each_seed_geometry_recall':all(f['variants']['shape_quadratic']['recall']>=limits['each_seed_geometry_recall_min'] for f in folds),
    }
    return dict(passed=all(checks.values()),checks=checks)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/condition_robustness_holdout.json')
    args=parser.parse_args()
    config_path=args.config.resolve()
    cfg=json.loads(config_path.read_text())
    if (cfg['n_steps'],cfg['calibration_steps'],cfg['minimum_point_loss_ratio'])!=(800,100,.1):
        raise ValueError('cache builder requires frozen 800/100/.1 settings')
    if (cfg['model'],cfg['ridge_strength'],cfg['standardized_clip'],cfg['trailing_window'])!=('QuadraticRidgeScore',1.0,8.0,5):
        raise ValueError('runner implements only the frozen quadratic model')
    if set(cfg['train_seeds']) & set(cfg['threshold_calibration_seeds']) or set(cfg['test_seeds']) & set(cfg['train_seeds']+cfg['threshold_calibration_seeds']):
        raise ValueError('seed split overlap')
    new_geometries=tuple(Geometry(g['name'],tuple(g['occluders'])) for g in cfg['test_geometries'])
    train_data,_=load_datasets(ROOT/'results/local_condition_robust_cache.npz',cfg['train_seeds']+cfg['threshold_calibration_seeds'])
    training_geometries=list(GEOMETRIES)
    if cfg.get('augmented_training_geometries'):
        augmented=tuple(Geometry(g['name'],tuple(g['occluders'])) for g in cfg['augmented_training_geometries'])
        augmented_cache=ROOT/'results/local_condition_augmented_cache.npz'
        if not augmented_cache.exists(): build_cache(augmented_cache,cfg['train_seeds']+cfg['threshold_calibration_seeds'],augmented)
        augmented_data,_=load_datasets(augmented_cache,cfg['train_seeds']+cfg['threshold_calibration_seeds'],augmented)
        train_data.update(augmented_data)
        training_geometries.extend(augmented)
    legacy_cols=[FEATURE_NAMES.index(n) for n in FEATURE_SETS['sector_counts_only']]
    def transform(x,kind):
        if kind=='shape': return prepare(x,cfg['trailing_window'])
        return np.vstack([trailing_mean(a[:,legacy_cols],5) for a in np.split(x,2)])
    def join(seeds,kind):
        keys=[(s,g.name) for s in seeds for g in training_geometries]
        return (np.vstack([transform(train_data[k][0],kind) for k in keys]),
                np.concatenate([train_data[k][1] for k in keys]),
                np.concatenate([np.full(len(train_data[k][1]),k[1]) for k in keys]))
    fitted={}
    for kind in ('shape','legacy'):
        xt,yt,_=join(cfg['train_seeds'],kind); xv,yv,gv=join(cfg['threshold_calibration_seeds'],kind)
        model=QuadraticRidgeScore().fit(xt,yt) if kind=='shape' else ObservableConditionEstimator().fit(xt,yt)
        sv=model.score(xv) if kind=='shape' else stable_margin(model,xv)
        threshold=select_threshold(yv,sv,gv,cfg['calibration_max_fpr_per_geometry'])
        fitted[kind]=(model,threshold,{g:metrics(yv[gv==g],sv[gv==g],threshold) for g in np.unique(gv)})
    # Models and operating points are fully fixed before opening any test data.
    test_cache=ROOT/'results'/cfg.get('test_cache','local_condition_confirmation_cache.npz')
    if not test_cache.exists(): build_cache(test_cache,cfg['test_seeds'],new_geometries)
    test_data,_=load_datasets(test_cache,cfg['test_seeds'],new_geometries)
    folds=[]
    for (seed,geometry),(x,y,states) in test_data.items():
        variants={}
        for kind in ('shape','legacy'):
            model,t,_=fitted[kind]; z=transform(x,kind)
            scores=model.score(z) if kind=='shape' else stable_margin(model,z)
            policies=[('shape_quadratic',t)] if kind=='shape' else [('legacy_calibrated',t),('legacy_fixed',0)]
            for name,threshold in policies:
                row=metrics(y,scores,threshold)
                row.update(auc=roc_auc(y,scores),threshold=threshold)
                variants[name]=row
        folds.append(dict(seed=seed,geometry=geometry,variants=variants))
    summary=summarize(folds)
    report=dict(status='frozen_unseen_seed_and_geometry_simulation_confirmation_not_fall_detection',
        config=cfg,config_sha256=hashlib.sha256(config_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        frozen_protocol_commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(config_path)],cwd=ROOT,text=True).strip(),
        environment=dict(python=platform.python_version(),numpy=np.__version__),
        testbed=git_info(ROOT/'heterosense-fl-testbed'),testbed_source_sha256=source_manifest(ROOT/'heterosense-fl-testbed'),
        source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in [Path(__file__),ROOT/'src/robust_condition.py',ROOT/'experiments/condition_shape_models.py',ROOT/'experiments/condition_failure_diagnosis.py',ROOT/'experiments/condition_threshold_validation.py']},
        input_features=SHAPE_NAMES,calibration={k:dict(threshold=v[1],by_geometry=v[2]) for k,v in fitted.items()},
        folds=folds,summary=summary,acceptance=acceptance(summary['shape_quadratic'],folds,cfg['exploratory_engineering_acceptance']),
        limitations=['Simulated 2.5-D occlusion and synthetic 3-coordinate point clouds; not validated on physical 2D LiDAR.',
                     'No claim about fall detection, event alerts, arbitrary geometry, sensor noise/domain shifts, or real-world deployment.',
                     'Repeated clear trajectories and temporally dependent frames; averages are descriptive.',
                     'The observed shape can carry synthetic generator artifacts even though raw counts and hidden parameters are excluded.'])
    path=ROOT/'results'/(config_path.stem+'.json')
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary={k:v['all'] for k,v in summary.items()},acceptance=report['acceptance']),indent=2))


if __name__=='__main__': main()

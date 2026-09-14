"""Evaluate original six-epoch checkpoints on the fixed new calibration/test."""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_simulated_fall_p1_v2 import read_csv, infer, digest, LidarCNN
from experiments.compare_fall_policies import write_csv, run as compare


def run():
    old=ROOT/'results/local_simulated_fall_p1_v2'
    fresh=ROOT/'results/local_simulated_fall_p1_v2_fresh18'
    output=ROOT/'results/local_simulated_fall_p1_v2_fresh6'
    if output.exists() and any(output.iterdir()):raise ValueError('Fresh output required')
    p=json.loads((fresh/'protocol.json').read_text())
    original=json.loads((old/'protocol.json').read_text())
    if p['status'] not in ('prepared','complete_simulation_only'):raise ValueError('Fresh data not yet prepared')
    for protocol in (p,original):
        for name,expected in protocol['source_sha256'].items():
            if digest(ROOT/name)!=expected:raise ValueError('Producer source changed')
    for directory,protocol in ((old,original),(fresh,p)):
        if digest(directory/'observations.npz')!=protocol['observations_sha256']:raise ValueError('Observation cache changed')
    rows=read_csv(fresh/'windows.csv');oldrows=read_csv(old/'windows.csv')
    with np.load(fresh/'observations.npz') as values:x=values['x']
    with np.load(old/'observations.npz') as values:oldx=values['x']
    lookup={r['sample_id']:r for r in rows if r['split']=='train'}
    oldtrain={r['sample_id']:r for r in oldrows if r['split']=='train'}
    if set(lookup)!=set(oldtrain):raise ValueError('Training windows differ')
    for key,r in lookup.items():
        previous=oldtrain[key]
        for field in ('label','geometry','recording_id','endpoint','condition_score'):
            if r[field]!=previous[field]:raise ValueError('Training metadata differs')
        if not np.array_equal(x[int(r['array_index'])],oldx[int(previous['array_index'])]):
            raise ValueError('Training observation differs')
    del oldx
    oldids={r['recording_id'] for r in oldrows}
    if oldids&{r['recording_id'] for r in rows if r['split']!='train'}:
        raise ValueError('Evaluation trajectories reused')
    output.mkdir(exist_ok=True)
    for name in ('observations.npz','windows.csv','condition_model.npz','selected_endpoints.csv','event_preflight.json'):
        shutil.copyfile(fresh/name,output/name)
    cfg=dict(p['config'],epochs=6,status='original_six_epoch_models_on_fresh_trajectories')
    config=output/'run_config.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
    training=json.loads((old/'training.json').read_text())
    if len(training)!=10 or {(r['seed'],r['method']) for r in training}!={(s,m) for s in range(5) for m in ('B1','B2')}:
        raise ValueError('Ten original models required')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    evaluation=[r for r in rows if r['split']!='train'];all_scores=[]
    for seed in range(5):
        scores=[dict(seed=seed,**{k:r[k] for k in ('sample_id','recording_id','split','facility','geometry','label','condition_score')}) for r in evaluation]
        used=set()
        for method in ('B1','B2'):
            item=next(r for r in training if r['seed']==seed and r['method']==method)
            checkpoint=old/f'{method}_seed{seed}.pt'
            if digest(checkpoint)!=item['checkpoint_sha256']:raise ValueError('Original checkpoint changed')
            if len(item['losses'])!=6:raise ValueError('Not a six-epoch checkpoint')
            used.update(item['training_windows'])
            model=LidarCNN();model.load_state_dict(torch.load(checkpoint,weights_only=True))
            for row,value in zip(scores,infer(model,x,evaluation,16)):
                row['b1_score' if method=='B1' else 'augmented_score']=value
            shutil.copyfile(checkpoint,output/checkpoint.name)
            print(f'Rescored original {method} seed {seed}',flush=True)
        for key in sorted(used):
            r=lookup[key]
            scores.append(dict(seed=seed,**{k:r[k] for k in ('sample_id','recording_id','split','facility','geometry','label','condition_score')},b1_score='',augmented_score=''))
        all_scores.extend(scores)
    write_csv(output/'scores.csv',all_scores)
    (output/'training.json').write_text(json.dumps(training,indent=2)+'\n')
    comparison=output/'comparison_config.json'
    comparison.write_text(json.dumps(dict(seeds=list(range(5)),calibration_fpr_budget=.05,condition_cutoff=0.,require_unseen_geometry=True,
        estimator_provenance={str(s):p['condition_provenance'] for s in range(5)}),indent=2)+'\n')
    compare(output/'scores.csv',comparison,output/'comparison')
    p.update(config=cfg,config_sha256=digest(config),scores_sha256=digest(output/'scores.csv'),status='complete_simulation_only',
             checkpoint_origin='Original six-epoch run; no retraining',original_training_arrays_identical=True,
             original_training_sha256=digest(old/'training.json'))
    p['source_sha256'][str(Path(__file__).relative_to(ROOT))]=digest(Path(__file__))
    (output/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    print('Completed original six-epoch evaluation on fresh trajectories',flush=True)


if __name__=='__main__':run()

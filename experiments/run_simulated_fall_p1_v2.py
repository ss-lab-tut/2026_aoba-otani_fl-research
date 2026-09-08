"""Corrected target/time/data protocol; unchanged CNN and frozen condition model."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_simulated_fall_p1 import (
    ConfigurationManager, ObservationBaseline, ObservationStatisticsExtractor, FEATURE_NAMES,
    GEOMETRIES, Geometry, _client, trailing_mean, SHAPE_NAMES, QuadraticRidgeScore, LidarCNN,
    rasterize, digest, source_manifest, git_info,
)
from heterosense._core._behavior_model import BehaviorModel, SemanticState
from heterosense._core._observation_model import ObservationModel
from experiments.audit_simulated_cnn_inputs import episodes
from experiments.compare_fall_policies import write_csv, run as compare


def stable_seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')


def set_persistence(behavior, probability):
    if not 0<probability<1:
        raise ValueError('persistence must be strictly between zero and one')
    row=behavior.transition_matrix[SemanticState.ABNORMAL]
    old=row[SemanticState.ABNORMAL]
    behavior.transition_matrix[SemanticState.ABNORMAL]={k:probability if k==SemanticState.ABNORMAL else v*(1-probability)/(1-old)
                                                       for k,v in row.items()}


def candidates(latent,record,minimum_end=163):
    states=np.array([s.state.value for s in latent])
    result={'fall':[],'near_fall':[],'adl':[]}
    for onset,end in episodes(states):
        subtype=latent[onset].abnormal_type.value
        if subtype in ('FALL','RECOVERED_FALL') and end-onset>=3:
            endpoint=onset+2
            kind='fall'
        elif subtype=='NEAR_FALL':
            endpoint=end+1
            kind='near_fall'
            if endpoint>=len(latent) or states[endpoint]=='ABNORMAL':
                continue
        else:
            continue
        if endpoint<minimum_end:
            continue
        result[kind].append(dict(recording_id=record,endpoint=endpoint,event_id=f'{record}_event{onset}',
                                 event_onset=onset,event_end=end,event_subtype=subtype,label=int(kind=='fall'),kind=kind))
    for endpoint in range(minimum_end,len(latent),64):
        if np.any(states[endpoint-4:endpoint+1]=='ABNORMAL'):
            continue
        result['adl'].append(dict(recording_id=record,endpoint=endpoint,event_id='',event_onset=-1,event_end=-1,
                                event_subtype=latent[endpoint].state.value,label=0,kind='adl'))
    return result


def choose_balanced(per_record,quotas,seed,minimum_strict):
    rng=np.random.default_rng(seed)
    selected=[]
    used=set()
    for kind,quota in quotas.items():
        queues=[]
        for record in sorted(per_record):
            items=per_record[record][kind]
            queues.append([items[i] for i in rng.permutation(len(items))])
        # Round-robin records so one long trajectory cannot supply the whole set.
        chosen=[]
        while len(chosen)<quota and any(queues):
            for queue in queues:
                while queue:
                    item=queue.pop()
                    key=(item['recording_id'],item['endpoint'])
                    if key in used:
                        continue
                    chosen.append(item);used.add(key)
                    break
                if len(chosen)==quota:
                    break
        if len(chosen)!=quota:
            raise ValueError(f'insufficient {kind} candidates: {len(chosen)} < {quota}')
        selected.extend(chosen)
    if sum(r['event_subtype']=='FALL' for r in selected)<minimum_strict:
        raise ValueError('insufficient strict FALL episodes; do not train')
    return selected


class ClipDataset(Dataset):
    def __init__(self,x,rows):
        self.x,self.rows=x,rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        return torch.from_numpy(self.x[int(r['array_index'])].copy()),torch.tensor(2 if int(r['label']) else 0,dtype=torch.long)


def infer(model,x,rows,batch):
    out=[];model.eval()
    with torch.inference_mode():
        for features,_ in DataLoader(ClipDataset(x,rows),batch_size=batch):
            out.extend(torch.softmax(model(features),dim=1)[:,2].tolist())
    return out


def read_csv(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def prepare(cfg,output):
    old=ROOT/cfg['frozen_condition_source']
    prior=json.loads((old/'protocol.json').read_text(encoding='utf-8'))
    if digest(old/'condition_model.npz')!=prior['condition_provenance']['artifact_sha256']:
        raise ValueError('frozen condition artifact changed')
    shutil.copyfile(old/'condition_model.npz',output/'condition_model.npz')
    with np.load(output/'condition_model.npz') as values:
        condition=QuadraticRidgeScore()
        condition.mean_,condition.scale_,condition.coef_=values['mean'],values['scale'],values['coef']
        cutoff=float(values['cutoff'])
    cc=json.loads((ROOT/cfg['geometry_config']).read_text(encoding='utf-8'))
    geometries={'clear':Geometry('clear',()),**{g.name:g for g in GEOMETRIES}}
    geometries.update({g['name']:Geometry(g['name'],tuple(g['occluders'])) for g in cc['test_geometries']})
    cols=[FEATURE_NAMES.index(n) for n in SHAPE_NAMES]
    recordings={};selected=[];counts=[]
    # Labels and event counts are fixed before observing/scoring any clips.
    for split,seeds in cfg['trajectory_seeds'].items():
        for facility in cfg['facilities']:
            pools={}
            for seed in seeds:
                record=f'v2_{facility["name"]}_{seed}'
                client=_client(())
                client['client_id']='fall_'+facility['name']
                client['sensor_noise_level']=facility['noise']
                sim=ConfigurationManager.from_clients([client],n_steps=cfg['n_steps'],random_seed=seed).to_sim_config()
                sim.delta_t=cfg['delta_t_seconds']
                behavior=BehaviorModel(sim.clients[0],np.random.default_rng(stable_seed(record+'_behavior')))
                set_persistence(behavior,cfg['abnormal_persistence'])
                latent=behavior.generate(cfg['n_steps'])
                recordings[record]=(sim,latent,split,facility['name'])
                pools[record]=candidates(latent,record,cfg['baseline_steps']+63)
            chosen=choose_balanced(pools,cfg['quotas_per_facility'][split],stable_seed(split+'_'+facility['name']),cfg['minimum_strict_fall_per_facility'])
            selected.extend(chosen)
            counts.append(dict(split=split,facility=facility['name'],records=len(pools),
                selected=dict(Counter(r['kind'] for r in chosen)),subtypes=dict(Counter(r['event_subtype'] for r in chosen)),
                candidates={kind:sum(len(p[kind]) for p in pools.values()) for kind in ('fall','near_fall','adl')}))
            print(f'preflight {split} {facility["name"]}: {counts[-1]["subtypes"]}',flush=True)
    (output/'event_preflight.json').write_text(json.dumps(counts,indent=2)+'\n',encoding='utf-8')
    write_csv(output/'selected_endpoints.csv',selected)
    arrays=[];rows=[]
    for record,(sim,latent,split,facility) in recordings.items():
        items=sorted([r for r in selected if r['recording_id']==record],key=lambda r:r['endpoint'])
        if not items:
            raise ValueError('selected data omit a planned record')
        needed=sorted(set(range(cfg['baseline_steps'])) | {t for r in items for t in range(r['endpoint']-63,r['endpoint']+1)})
        names=cfg['known_geometries']+(cfg['unseen_geometries'] if split=='test' else [])
        for name in names:
            from dataclasses import replace
            from heterosense._core._config_schema import OccluderConfig
            client=replace(sim.clients[0],lidar_occluders=tuple(OccluderConfig.from_dict(o) for o in geometries[name].occluders))
            observer=ObservationModel(client,np.random.default_rng(stable_seed(record+'_observation')))
            frames=[observer.observe(latent[t],t*sim.delta_t) for t in needed]
            positions={t:i for i,t in enumerate(needed)}
            raster=rasterize(frames,cfg['raster'])
            baseline=ObservationBaseline.fit(frames[:cfg['baseline_steps']])
            features=ObservationStatisticsExtractor(baseline).transform(frames)[:,cols]
            for item in items:
                endpoint=item['endpoint']
                indices=[positions[t] for t in range(endpoint-63,endpoint+1)]
                shape=trailing_mean(features[indices],5)[-1:]
                margin=float(condition.score(shape)[0]-cutoff)
                clip=raster[indices].transpose(1,0,2).copy()
                sample=f'{record}_{name}_{endpoint}'
                rows.append(dict(sample_id=sample,recording_id=record,split=split,facility=facility,geometry=name,
                    label=item['label'],condition_score=margin,array_index=len(arrays),endpoint=endpoint,
                    timestamp_seconds=endpoint*sim.delta_t,event_id=item['event_id'],event_onset=item['event_onset'],
                    event_end=item['event_end'],event_subtype=item['event_subtype'],kind=item['kind']))
                arrays.append(clip)
        print(f'observed {record}: {len(items)} clips x {len(names)} geometries',flush=True)
    x=np.stack(arrays)
    if not np.isfinite(x).all() or x.shape[1:]!=(2,64,430):
        raise ValueError('invalid generated inputs')
    np.savez_compressed(output/'observations.npz',x=x)
    write_csv(output/'windows.csv',rows)
    return x,rows,prior['condition_provenance']


def run(config,output,resume=False):
    cfg=json.loads(config.read_text(encoding='utf-8'))
    seeds=sum(cfg['trajectory_seeds'].values(),[])
    if len(set(seeds))!=len(seeds) or len(set(cfg['seeds']))!=5:
        raise ValueError('disjoint trajectory sets and five model seeds required')
    source={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/'model.py',ROOT/'experiments/run_simulated_fall_p1.py',
        ROOT/'experiments/audit_simulated_cnn_inputs.py',ROOT/'src/fall_comparison.py',ROOT/'experiments/compare_fall_policies.py')}
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    if resume:
        protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
        if protocol['config_sha256']!=digest(config) or protocol['source_sha256']!=source or protocol['testbed_source_sha256']!=source_manifest(ROOT/'heterosense-fl-testbed'):
            raise ValueError('resume protocol/source mismatch')
        if digest(output/'observations.npz')!=protocol['observations_sha256']:
            raise ValueError('resume observation cache changed')
        with np.load(output/'observations.npz') as cache:x=cache['x']
        rows=read_csv(output/'windows.csv');provenance=protocol['condition_provenance']
        for r in rows:
            for key in ('label','array_index','endpoint'):r[key]=int(r[key])
            r['condition_score']=float(r['condition_score'])
    else:
        if output.exists() and any(output.iterdir()):raise ValueError('fresh output required, or use --resume')
        output.mkdir(parents=True,exist_ok=True)
        protocol=dict(config=cfg,config_sha256=digest(config),source_sha256=source,
            testbed_source_sha256=source_manifest(ROOT/'heterosense-fl-testbed'),git=git_info(ROOT),
            python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,status='preparing')
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
        x,rows,provenance=prepare(cfg,output)
        protocol.update(observations_sha256=digest(output/'observations.npz'),condition_provenance=provenance,status='prepared')
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    evaluation=[r for r in rows if r['split']!='train']
    clear=[r for r in rows if r['split']=='train' and r['geometry']=='clear']
    lookup={(r['recording_id'],r['endpoint'],r['geometry']):r for r in rows if r['split']=='train'}
    all_scores=[];training=[]
    for seed in cfg['seeds']:
        score_path=output/f'scores_seed{seed}.csv';train_path=output/f'training_seed{seed}.json'
        if resume and score_path.exists() and train_path.exists():
            saved=json.loads(train_path.read_text(encoding='utf-8'))
            for item in saved:
                if digest(output/f'{item["method"]}_seed{seed}.pt')!=item['checkpoint_sha256']:raise ValueError('checkpoint changed')
            all_scores.extend(read_csv(score_path));training.extend(saved)
            print(f'resumed finished seed={seed}',flush=True);continue
        rng=np.random.default_rng(seed)
        augmented=[lookup[(r['recording_id'],r['endpoint'],str(rng.choice(cfg['known_geometries'])))] for r in clear]
        scores=[dict(seed=seed,**{k:r[k] for k in ('sample_id','recording_id','split','facility','geometry','label','condition_score')}) for r in evaluation]
        seed_training=[]
        for method,train in [('B1',clear),('B2',augmented)]:
            torch.manual_seed(seed);model=LidarCNN()
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
            loader=DataLoader(ClipDataset(x,train),batch_size=cfg['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
            losses=[]
            for epoch in range(cfg['epochs']):
                model.train();total=0.
                for features,labels in loader:
                    optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(features),labels)
                    loss.backward();optimizer.step();total+=float(loss.detach())*len(labels)
                losses.append(total/len(train))
                print(f'seed={seed} {method} epoch={epoch+1}/{cfg["epochs"]} loss={losses[-1]:.5f}',flush=True)
            checkpoint=output/f'{method}_seed{seed}.pt';torch.save(model.state_dict(),checkpoint)
            for row,value in zip(scores,infer(model,x,evaluation,cfg['batch_size'])):
                row['b1_score' if method=='B1' else 'augmented_score']=value
            seed_training.append(dict(seed=seed,method=method,losses=losses,checkpoint_sha256=digest(checkpoint),training_windows=[r['sample_id'] for r in train]))
        for r in {r['sample_id']:r for r in clear+augmented}.values():
            scores.append(dict(seed=seed,**{k:r[k] for k in ('sample_id','recording_id','split','facility','geometry','label','condition_score')},b1_score='',augmented_score=''))
        write_csv(score_path,scores)
        train_path.write_text(json.dumps(seed_training,indent=2)+'\n',encoding='utf-8')
        all_scores.extend(scores);training.extend(seed_training)
        print(f'completed seed={seed}',flush=True)
    (output/'training.json').write_text(json.dumps(training,indent=2)+'\n',encoding='utf-8')
    cp=output/'comparison_config.json'
    cp.write_text(json.dumps(dict(seeds=cfg['seeds'],calibration_fpr_budget=cfg['calibration_fpr_budget'],condition_cutoff=0.,
        require_unseen_geometry=True,estimator_provenance={str(s):provenance for s in cfg['seeds']}),indent=2)+'\n',encoding='utf-8')
    write_csv(output/'scores.csv',all_scores)
    summary=compare(output/'scores.csv',cp,output/'comparison')
    protocol.update(status='complete_simulation_only',scores_sha256=digest(output/'scores.csv'))
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([r for r in summary if r['facility']==r['geometry']=='__all__'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/simulated_fall_p1_v2.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/local_simulated_fall_p1_v2')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.config,a.output_dir,a.resume)

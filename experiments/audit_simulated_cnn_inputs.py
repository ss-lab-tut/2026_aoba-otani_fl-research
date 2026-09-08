"""Audit frozen CNN input/label alignment and simulated trajectory composition.

Regenerates clear observations and latent labels for verification, never trains
or selects a model. Latent information is used only for the audit tables.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_simulated_fall_p1 import (
    ConfigurationManager, DatasetBuilder, _client, rasterize, digest, source_manifest,
)
from experiments.compare_fall_policies import write_csv
from heterosense._core._behavior_model import BehaviorModel


def episodes(states):
    positive=np.asarray(states)=='ABNORMAL'
    starts=np.flatnonzero(positive & ~np.r_[False,positive[:-1]])
    ends=np.flatnonzero(positive & ~np.r_[positive[1:],False])+1
    return list(zip(starts.tolist(),ends.tolist()))


def label_details(states, start, width):
    positive=np.asarray(states)=='ABNORMAL'
    segment=positive[start:start+width]
    if len(segment)!=width:
        raise ValueError('window outside recording')
    return dict(label=int(segment.any()), abnormal_frames=int(segment.sum()),
                endpoint_abnormal=int(segment[-1]), last5_abnormal=int(segment[-5:].any()),
                first_abnormal_offset=int(np.flatnonzero(segment)[0]) if segment.any() else -1,
                last_abnormal_offset=int(np.flatnonzero(segment)[-1]) if segment.any() else -1)


def input_stats(raster):
    x=np.asarray(raster)
    if x.ndim!=3 or x.shape[1:]!=(2,430) or not np.isfinite(x).all() or np.any((x<0)|(x>1)):
        raise ValueError('invalid CNN input')
    occupied=x[:,0]>0
    return dict(occupied_bin_fraction=float(occupied.mean()),
                always_empty_bin_fraction=float(np.mean(~occupied.any(axis=0))),
                range_saturated_fraction=float(np.mean(x[:,0]==1)),
                height_saturated_fraction=float(np.mean(x[:,1]==1)),
                empty_frames=int(np.sum(~occupied.any(axis=1))))


def read(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def run(source,output):
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    cfg=protocol['config']
    if protocol['testbed_source_sha256']!=source_manifest(ROOT/'heterosense-fl-testbed'):
        raise ValueError('simulator source changed')
    if digest(source/'observations.npz')!=protocol['observations_sha256']:
        raise ValueError('observation cache changed')
    if digest(source/'scores.csv')!=protocol['scores_sha256']:
        raise ValueError('score cache changed')
    for name,expected in protocol['source_sha256'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError(f'producer source changed: {name}')
    original_hashes={name:digest(source/name) for name in ('observations.npz','scores.csv','condition_model.npz','training.json')}
    windows=read(source/'windows.csv')
    recording_rows,event_rows,window_rows,input_rows=[],[],[],[]
    with np.load(source/'observations.npz') as cache:
        for split,seeds in cfg['trajectory_seeds'].items():
            for facility in cfg['facilities']:
                for seed in seeds:
                    client=_client(())
                    client['client_id']='fall_'+facility['name']
                    client['sensor_noise_level']=facility['noise']
                    manager=ConfigurationManager.from_clients([client],n_steps=cfg['n_steps'],random_seed=seed)
                    simcfg=manager.to_sim_config()
                    offset=int.from_bytes(hashlib.sha256(client['client_id'].encode()).digest()[:4],'big')
                    latent=BehaviorModel(simcfg.clients[0],np.random.default_rng(seed+offset)).generate(cfg['n_steps'])
                    frames=DatasetBuilder(simcfg).build()[client['client_id']]
                    states=np.array([f.semantic_state for f in frames])
                    np.testing.assert_array_equal(states,[s.state.value for s in latent])
                    key=f'{split}_{facility["name"]}_{seed}_clear'
                    np.testing.assert_array_equal(rasterize(frames,cfg['raster']),cache[key])
                    record=f'{facility["name"]}_{seed}'
                    selected=[r for r in windows if r['split']==split and r['recording_id']==record and r['geometry']=='clear']
                    expected_starts=list(range(cfg['baseline_steps'],cfg['n_steps']-cfg['window_frames']+1,cfg['stride_frames']))
                    if [int(r['start']) for r in selected]!=expected_starts:
                        raise ValueError('window indexing mismatch')
                    ep=episodes(states)
                    eligible=[(a,b) for a,b in ep if any(a<int(r['start'])+64 and b>int(r['start']) for r in selected)]
                    details=[]
                    for r in selected:
                        start=int(r['start'])
                        detail=label_details(states,start,64)
                        if detail['label']!=int(r['label']):
                            raise ValueError('window/label mismatch')
                        detail.update(split=split,facility=facility['name'],recording_id=record,start=start,
                                      last_abnormal_age_frames=(63-detail['last_abnormal_offset']) if detail['label'] else -1,
                                      strict_fall_frames=sum(s.abnormal_type.value=='FALL' for s in latent[start:start+64]))
                        window_rows.append(detail)
                        details.append(detail)
                    subtype_counts=Counter()
                    for a,b in eligible:
                        subtype=latent[a].abnormal_type.value
                        subtype_counts[subtype]+=1
                        event_rows.append(dict(split=split,facility=facility['name'],recording_id=record,
                                               start=a,end_exclusive=b,duration_frames=b-a,subtype=subtype,
                                               overlapping_windows=sum(a<int(r['start'])+64 and b>int(r['start']) for r in selected)))
                    positives=[r for r in details if r['label']]
                    raw_count=np.array([len(f.lidar) for f in frames])
                    raster=cache[key]
                    recording_rows.append(dict(split=split,facility=facility['name'],recording_id=record,
                        delta_t_seconds=simcfg.delta_t,window_duration_seconds=64*simcfg.delta_t,
                        windows=len(details),positive_windows=len(positives),positive_rate=len(positives)/len(details),
                        abnormal_episodes=len(eligible),strict_fall_episodes=subtype_counts['FALL'],
                        near_fall_episodes=subtype_counts['NEAR_FALL'],recovered_fall_episodes=subtype_counts['RECOVERED_FALL'],
                        positive_endpoint_negative=sum(not r['endpoint_abnormal'] for r in positives),
                        positive_last5_negative=sum(not r['last5_abnormal'] for r in positives),
                        median_abnormal_frames=float(np.median([r['abnormal_frames'] for r in positives])) if positives else None,
                        mean_points_per_occupied_bin=float(np.sum(raw_count)/np.sum(raster[:,0]>0)),
                        **input_stats(raster)))
                    for rkey in cache.files:
                        if rkey.startswith(f'{split}_{facility["name"]}_{seed}_'):
                            geometry=rkey.removeprefix(f'{split}_{facility["name"]}_{seed}_')
                            rs=[r for r in windows if r['key']==rkey]
                            np.testing.assert_array_equal([int(r['label']) for r in rs],[r['label'] for r in details])
                            input_rows.append(dict(split=split,facility=facility['name'],recording_id=record,
                                                   geometry=geometry,**input_stats(cache[rkey])))
                    print(f'audited {split} {record}',flush=True)
    for name,expected in original_hashes.items():
        if digest(source/name)!=expected:
            raise ValueError('original artifact modified during audit')
    summary={}
    for split in cfg['trajectory_seeds']:
        rs=[r for r in recording_rows if r['split']==split]
        ws=[r for r in window_rows if r['split']==split and r['label']]
        es=[r for r in event_rows if r['split']==split]
        summary[split]=dict(recordings=len(rs),unique_windows=sum(r['windows'] for r in rs),positive_windows=len(ws),
            abnormal_episodes=len(es),episode_subtypes=dict(Counter(r['subtype'] for r in es)),
            median_episode_frames=float(np.median([r['duration_frames'] for r in es])),
            median_positive_abnormal_frames=float(np.median([r['abnormal_frames'] for r in ws])),
            positive_endpoint_negative_fraction=float(np.mean([not r['endpoint_abnormal'] for r in ws])),
            positive_last5_negative_fraction=float(np.mean([not r['last5_abnormal'] for r in ws])),
            positive_without_strict_fall_fraction=float(np.mean([r['strict_fall_frames']==0 for r in ws])),
            mean_occupied_bin_fraction=float(np.mean([r['occupied_bin_fraction'] for r in rs])),
            mean_always_empty_bin_fraction=float(np.mean([r['always_empty_bin_fraction'] for r in rs])))
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('recordings',recording_rows),('episodes',event_rows),('window_labels',window_rows),('input_geometry',input_rows)]:
        write_csv(output/(name+'.csv'),rows)
    audit=dict(status='read_only_input_label_data_audit_no_training_or_selection',summary=summary,
               original_artifacts_sha256=original_hashes,script_sha256=digest(Path(__file__)),
               checks=dict(clear_rasters_exactly_reproduced=True,labels_exactly_reproduced=True,
                           geometry_labels_identical=True,finite_normalized_inputs=True,original_artifacts_unchanged=True),
               limitations=['Subtype is simulator post-hoc ground truth, used for audit only.',
                            'Counts deduplicate geometry variants but not individual persons; people are synthetic.',
                            'Old test is inspected only diagnostically, not used to select a revised model.'])
    (output/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'results/local_simulated_fall_p1')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/simulated_cnn_input_audit')
    args=p.parse_args()
    run(args.source,args.output_dir)

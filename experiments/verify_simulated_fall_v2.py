"""Verify corrected endpoint labels, paired data and actual CNN score exports."""
import argparse
from collections import Counter, defaultdict
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_simulated_fall_p1_v2 import read_csv,infer,digest,LidarCNN


def verify(source,output):
    p=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if p.get('status')!='complete_simulation_only':raise ValueError('complete run required')
    cfg=p['config']
    for path,expected in p['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError(f'changed producer source: {path}')
    if digest(source/'observations.npz')!=p['observations_sha256']:raise ValueError('observations changed')
    if digest(source/'condition_model.npz')!=p['condition_provenance']['artifact_sha256']:raise ValueError('condition model changed')
    endpoints=read_csv(source/'selected_endpoints.csv')
    windows=read_csv(source/'windows.csv')
    groups=defaultdict(list)
    for r in windows:groups[r['split'],r['facility'],r['geometry']].append(r)
    for (split,facility,geometry),items in groups.items():
        if Counter(r['kind'] for r in items)!=cfg['quotas_per_facility'][split]:raise ValueError('unmatched class quotas')
        if sum(r['event_subtype']=='FALL' for r in items)<cfg['minimum_strict_fall_per_facility']:raise ValueError('strict FALL shortage')
        if geometry in cfg['unseen_geometries'] and split!='test':raise ValueError('unseen geometry leakage')
    owners={}
    selected={(r['recording_id'],int(r['endpoint'])):r for r in endpoints}
    if len(selected)!=len(endpoints):raise ValueError('duplicate selected endpoints')
    for r in windows:
        record=r['recording_id'];endpoint=int(r['endpoint'])
        if owners.setdefault(record,r['split'])!=r['split']:raise ValueError('cross-split recording')
        original=selected[record,endpoint]
        for key in ('label','kind','event_subtype','event_id','event_onset','event_end'):
            if original[key]!=r[key]:raise ValueError('endpoint metadata mismatch')
        if r['label']=='1':
            if r['event_subtype'] not in ('FALL','RECOVERED_FALL') or endpoint!=int(r['event_onset'])+2 or endpoint>=int(r['event_end']):
                raise ValueError('positive must be actual third episode frame')
        elif r['kind']=='near_fall' and (r['event_subtype']!='NEAR_FALL' or endpoint!=int(r['event_end'])+1):
            raise ValueError('near-fall negative alignment mismatch')
    with np.load(source/'observations.npz') as cache:x=cache['x']
    if x.shape!=(len(windows),2,64,430) or not np.isfinite(x).all():raise ValueError('invalid arrays')
    if [int(r['array_index']) for r in windows]!=list(range(len(windows))):raise ValueError('array alignment mismatch')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    training=json.loads((source/'training.json').read_text(encoding='utf-8'))
    scores=read_csv(source/'scores.csv')
    window_lookup={r['sample_id']:r for r in windows}
    verified=[]
    for item in training:
        seed,method=item['seed'],item['method']
        model=LidarCNN();checkpoint=source/f'{method}_seed{seed}.pt'
        if digest(checkpoint)!=item['checkpoint_sha256']:raise ValueError('checkpoint changed')
        model.load_state_dict(torch.load(checkpoint,weights_only=True))
        selected_scores=[r for r in scores if int(r['seed'])==seed and r['split']=='calibration'][:16]
        result=infer(model,x,[window_lookup[r['sample_id']] for r in selected_scores],16)
        key='b1_score' if method=='B1' else 'augmented_score'
        np.testing.assert_allclose(result,[float(r[key]) for r in selected_scores],rtol=1e-6,atol=1e-7)
        verified.append(dict(seed=seed,method=method,reproduced_windows=len(result)))
        print(f'verified {method} seed={seed}',flush=True)
    if len(verified)!=10:raise ValueError('ten checkpoints required')
    output.mkdir(parents=True,exist_ok=True)
    for name in ('event_preflight.json','selected_endpoints.csv'):
        shutil.copyfile(source/name,output/name)
    report=dict(status='verified_corrected_endpoint_comparison',checks=dict(
        source_hashes=True,condition_artifact_unchanged=True,quotas_per_facility_geometry=True,
        target_subtypes=True,endpoint_alignment=True,disjoint_recordings=True,array_alignment=True),
        model_score_reproduction=verified,windows=len(windows),unique_endpoints=len(endpoints),
        input_sha256={name:digest(source/name) for name in ('observations.npz','windows.csv','scores.csv','selected_endpoints.csv','event_preflight.json')},
        verifier_sha256=digest(Path(__file__)))
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/local_simulated_fall_p1_v2')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/simulated_fall_p1_v2')
    args=parser.parse_args();verify(args.source,args.output_dir)

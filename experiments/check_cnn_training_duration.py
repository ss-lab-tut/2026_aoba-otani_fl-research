"""Predefined 6-versus-18 epoch check on original training recordings only."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_simulated_fall_p1_v2 import read_csv, digest, ClipDataset, LidarCNN
from experiments.compare_fall_policies import write_csv
from experiments.condition_threshold_validation import roc_auc

EPOCHS=(1,3,6,12,18)
GEOMETRIES=('clear','vertical_mid','central_furniture')


def partition(rows):
    fit,dev=[],[]
    for r in rows:
        if r['split']!='train':
            continue
        if r['geometry'] not in GEOMETRIES:
            raise ValueError('Unexpected training geometry')
        trajectory=int(r['recording_id'].rsplit('_',1)[1])
        if trajectory in (10001,10002,10003): fit.append(r)
        elif trajectory==10004: dev.append(r)
        else: raise ValueError('Unexpected training recording')
    if not fit or not dev: raise ValueError('Empty development partition')
    if {r['recording_id'] for r in fit}&{r['recording_id'] for r in dev}:
        raise ValueError('Recording leakage')
    for group in (fit,dev):
        for facility in ('standard','harsh'):
            for geometry in GEOMETRIES:
                labels={int(r['label']) for r in group if r['facility']==facility and r['geometry']==geometry}
                if labels!={0,1}: raise ValueError('Missing class in facility/geometry')
    return fit,dev


def evaluate(model,x,rows):
    model.eval();losses=[];labels=[];scores=[]
    with torch.inference_mode():
        for features,y in DataLoader(ClipDataset(x,rows),batch_size=16):
            logits=model(features)
            losses.extend(torch.nn.functional.cross_entropy(logits,y,reduction='none').tolist())
            labels.extend((y==2).tolist());scores.extend(torch.softmax(logits,dim=1)[:,2].tolist())
    return dict(n=len(rows),auroc=float(roc_auc(np.array(labels),np.array(scores))),cross_entropy=float(np.mean(losses)))


def run(resume=False):
    source=ROOT/'results/local_simulated_fall_p1_v2'
    output=ROOT/'results/local_simulated_fall_p1_v2_duration'
    published=ROOT/'results/cnn_training_duration'
    original=json.loads((source/'protocol.json').read_text())
    verification=json.loads((ROOT/'results/simulated_fall_p1_v2/verification.json').read_text())
    for name,expected in [('observations.npz',original['observations_sha256']),('windows.csv',verification['input_sha256']['windows.csv'])]:
        if digest(source/name)!=expected: raise ValueError('Input changed')
    rows=read_csv(source/'windows.csv')
    fit,dev=partition(rows)
    sources=[Path(__file__),ROOT/'model.py',ROOT/'experiments/run_simulated_fall_p1_v2.py',ROOT/'docs/CNN_TRAINING_CHECK.md']
    protocol=dict(status='development_only',epochs=list(EPOCHS),seeds=list(range(5)),
                  learning_rate=.001,batch_size=16,primary_contrast='epoch18_minus_epoch6',
                  fit_records=sorted({r['recording_id'] for r in fit}),development_records=sorted({r['recording_id'] for r in dev}),
                  observations_sha256=digest(source/'observations.npz'),windows_sha256=digest(source/'windows.csv'),
                  source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
                  torch=torch.__version__,numpy=np.__version__)
    if resume:
        if json.loads((output/'protocol.json').read_text())!=protocol: raise ValueError('Resume protocol mismatch')
    elif output.exists() and any(output.iterdir()): raise ValueError('Use --resume for existing run')
    output.mkdir(exist_ok=True)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with np.load(source/'observations.npz') as data: x=data['x']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    lookup={(r['recording_id'],r['endpoint'],r['geometry']):r for r in fit}
    clear=[r for r in fit if r['geometry']=='clear']
    all_metrics=[]
    for seed in range(5):
        cache=output/f'seed{seed}.json'
        if resume and cache.exists():
            saved=json.loads(cache.read_text())
            for name,expected in saved['checkpoints'].items():
                if digest(output/name)!=expected: raise ValueError('Checkpoint changed')
            all_metrics.extend(saved['metrics']);print(f'Resumed seed {seed}',flush=True);continue
        rng=np.random.default_rng(seed)
        selected=[lookup[(r['recording_id'],r['endpoint'],str(rng.choice(GEOMETRIES)))] for r in clear]
        torch.manual_seed(seed);model=LidarCNN()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        loader=DataLoader(ClipDataset(x,selected),batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(seed))
        metrics=[];checkpoints={}
        for epoch in range(1,19):
            model.train();total=0.
            for features,y in loader:
                optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(features),y)
                loss.backward();optimizer.step();total+=float(loss.detach())*len(y)
            print(f'Seed {seed} epoch {epoch}/18 loss {total/len(selected):.4f}',flush=True)
            if epoch in EPOCHS:
                for scope,items in [('fit',selected),('development',dev)]:
                    m=dict(seed=seed,epoch=epoch,scope=scope,**evaluate(model,x,items))
                    metrics.append(m);print(f'  {scope}: AUROC {m["auroc"]:.4f} CE {m["cross_entropy"]:.4f}',flush=True)
            if epoch in (6,18):
                name=f'B2_seed{seed}_epoch{epoch}.pt';torch.save(model.state_dict(),output/name);checkpoints[name]=digest(output/name)
        saved=dict(metrics=metrics,checkpoints=checkpoints,fit_samples=[r['sample_id'] for r in selected])
        cache.write_text(json.dumps(saved,indent=2)+'\n');all_metrics.extend(metrics)
    published.mkdir(exist_ok=True)
    write_csv(published/'metrics.csv',all_metrics)
    contrasts=[]
    for seed in range(5):
        for scope in ('fit','development'):
            points={r['epoch']:r for r in all_metrics if r['seed']==seed and r['scope']==scope}
            contrasts.append(dict(seed=seed,scope=scope,auroc_delta=points[18]['auroc']-points[6]['auroc'],
                                  cross_entropy_delta=points[18]['cross_entropy']-points[6]['cross_entropy']))
    write_csv(published/'paired_18_minus_6.csv',contrasts)
    (published/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('Completed all five development seeds',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true')
    run(parser.parse_args().resume)

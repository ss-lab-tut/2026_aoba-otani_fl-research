"""Purged within-series CNN diagnostic on provided preprocessed LiDAR arrays.

Not a subject-independent benchmark. Source preprocessing and event identities
are unavailable. Ambiguous labels are quarantined, not repaired by inference.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import LidarCNN
from src.temporal_integrity import reconstruct_order, purged_split
from src.evaluation import evaluate_tradeoff
from experiments.condition_threshold_validation import select_threshold


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def tensors(x,ys,ids):
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(x[ids].transpose(0,3,1,2),dtype=np.float32)),
                         torch.from_numpy(ys[ids].astype(np.int64)))


def scores(model,dataset,batch_size):
    out=[]
    model.eval()
    with torch.inference_mode():
        for x,_ in DataLoader(dataset,batch_size=batch_size):
            out.append(torch.softmax(model(x),dim=1)[:,2].numpy())
    return np.concatenate(out)


def run(epochs=6,seeds=(0,1,2,3,4),batch_size=32,augment=False):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    files=('X_train.npy','X_val_real.npy','y_train_sub.npy','y_val_sub.npy','y_train_bin.npy','y_val_bin.npy','meta.json','scaler.json')
    fingerprints={name:sha256(ROOT/name) for name in files}
    x=np.concatenate([np.load(ROOT/name) for name in files[:2]])
    subclass=np.concatenate([np.load(ROOT/name) for name in files[2:4]])
    y=np.concatenate([np.load(ROOT/name) for name in files[4:6]])
    uncertain=np.flatnonzero(y!=(subclass==3))
    order=reconstruct_order(x)
    y3=np.where(subclass==3,2,np.where(subclass==0,0,1))
    folds=[]
    for seed in seeds:
        np.random.seed(seed);torch.manual_seed(seed)
        split=purged_split(order,seed%5,uncertain)
        for name in ('train','calibration','test'):
            if set(np.unique(y[split[name]]))!={0,1}:
                raise ValueError(f'{name} lacks a class')
        datasets={name:tensors(x,y3,split[name]) for name in ('train','calibration','test')}
        loader=DataLoader(datasets['train'],batch_size=batch_size,shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
        model=LidarCNN()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        loss_fn=nn.CrossEntropyLoss()
        losses=[]
        for epoch in range(epochs):
            model.train();total=0.
            for features,labels in loader:
                if augment:
                    features=features.clone()
                    beams=features.shape[-1]
                    for row in range(len(features)):
                        if torch.rand(()).item()<.5:
                            width=int(beams*(.05+.30*torch.rand(()).item()))
                            start=int(torch.randint(beams,()).item())
                            columns=(start+torch.arange(width))%beams
                            features[row,:,:,columns]=0
                optimizer.zero_grad();loss=loss_fn(model(features),labels)
                loss.backward();optimizer.step();total+=float(loss.detach())*len(labels)
            losses.append(total/len(datasets['train']))
            print(f'seed={seed} epoch={epoch+1}/{epochs} loss={losses[-1]:.4f}',flush=True)
        cal=scores(model,datasets['calibration'],batch_size)
        test=scores(model,datasets['test'],batch_size)
        threshold=select_threshold(y[split['calibration']],cal,np.full(len(cal),'all'),.05)
        rows=evaluate_tradeoff(y[split['test']],test,thresholds=[.5,threshold])
        tag='real_fall_augmented' if augment else 'real_fall'
        checkpoint=ROOT/'results'/f'local_{tag}_seed{seed}.pt'
        torch.save(model.state_dict(),checkpoint)
        prediction_path=ROOT/'results'/f'local_{tag}_seed{seed}.npz'
        np.savez_compressed(prediction_path,calibration_scores=cal,test_scores=test,
                            calibration_indices=split['calibration'],test_indices=split['test'])
        folds.append(dict(seed=seed,split=split,losses=losses,selected_threshold=threshold,
                          metrics=rows,checkpoint_sha256=sha256(checkpoint),
                          checkpoint_file=checkpoint.name,
                          test_subclass_counts={str(k):int(np.sum(subclass[split['test']]==k)) for k in range(4)}))
    summary={}
    for i,name in enumerate(('fixed_0.5','calibrated_fpr_0.05')):
        summary[name]={key:dict(mean=float(np.mean([f['metrics'][i][key] for f in folds])),
                               std=float(np.std([f['metrics'][i][key] for f in folds],ddof=1)) if len(folds)>1 else None)
                       for key in ('recall','false_alert_rate')}
    return dict(status='exploratory_purged_within_series_diagnostic_not_official_generalization',
                environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu'),
                input_sha256=fingerprints,epochs=epochs,batch_size=batch_size,learning_rate=.001,
                augmentation=({'probability':.5,'beam_fraction_min':.05,'beam_fraction_max':.35,'value':0.,'training_only':True} if augment else None),
                chronological_order=order.tolist(),quarantined_global_indices=uncertain.tolist(),
                folds=folds,summary=summary,
                limitations=['No subject/recording/site identities; temporal separation is not independence of subjects or events.',
                             'Supplied inputs are already scaled and clipped; training-only fitting of the original scaler cannot be verified or undone.',
                             'Binary/subclass disagreement quarantined; true label remains unresolved.',
                             'One contiguous class-3 region is not evidence of many independent fall events.',
                             'Window-level false-positive rate only, not event alerts/hour.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--output',type=Path,default=ROOT/'results/real_fall_baseline.json')
    p.add_argument('--augment',action='store_true')
    args=p.parse_args()
    if args.augment and args.output==ROOT/'results/real_fall_baseline.json':
        raise ValueError('augmented run requires a separate --output path')
    r=run(args.epochs,tuple(args.seeds),augment=args.augment)
    args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(r['summary'],indent=2))

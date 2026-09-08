"""Verify data immutability, purged splits, and replay saved CNN inference."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_real_fall_baseline import tensors,scores,sha256
from src.temporal_integrity import verify_no_frame_overlap
from src.evaluation import evaluate_tradeoff
from model import LidarCNN


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('report',type=Path)
    args=parser.parse_args()
    r=json.loads(args.report.read_text())
    for name,expected in r['input_sha256'].items():
        if sha256(ROOT/name)!=expected:raise ValueError(f'input changed: {name}')
    torch.set_num_threads(4)
    x=np.concatenate([np.load(ROOT/'X_train.npy'),np.load(ROOT/'X_val_real.npy')])
    y=np.r_[np.load(ROOT/'y_train_bin.npy'),np.load(ROOT/'y_val_bin.npy')]
    for f in r['folds']:
        verify_no_frame_overlap(r['chronological_order'],f['split'])
        checkpoint=ROOT/'results'/f.get('checkpoint_file',f"local_real_fall_seed{f['seed']}.pt")
        if sha256(checkpoint)!=f['checkpoint_sha256']:raise ValueError('checkpoint changed')
        model=LidarCNN();model.load_state_dict(torch.load(checkpoint,weights_only=True))
        ids=f['split']['test']
        actual=scores(model,tensors(x,np.zeros(len(x),dtype=int),ids),r['batch_size'])
        predictions=checkpoint.with_suffix('.npz')
        with np.load(predictions) as cached:
            np.testing.assert_array_equal(actual,cached['test_scores'])
        rows=evaluate_tradeoff(y[ids],actual,thresholds=[.5,f['selected_threshold']])
        if rows!=f['metrics']:raise ValueError('metrics replay mismatch')
        print(f"Verified seed {f['seed']}: exact inference, metrics, and no cross-split frames",flush=True)
    print('All source dataset hashes unchanged')


if __name__=='__main__':main()

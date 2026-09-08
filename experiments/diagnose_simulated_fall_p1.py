"""Read-only train/calibration/test diagnosis of frozen simulation checkpoints."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_simulated_fall_p1 import Windows, score, digest
from experiments.compare_fall_policies import write_csv
from src.fall_comparison import operating_metrics, summarize
from model import LidarCNN


def read(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def run(source, output):
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if protocol.get('status')!='complete_simulation_only':
        raise ValueError('finished experiment required')
    for path, expected in protocol['source_sha256'].items():
        if digest(ROOT/path)!=expected:
            raise ValueError(f'experiment source changed: {path}')
    if digest(source/'observations.npz')!=protocol['observations_sha256']:
        raise ValueError('observation cache changed')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    windows={r['sample_id']:r for r in read(source/'windows.csv')}
    rows=read(source/'scores.csv')
    training=json.loads((source/'training.json').read_text(encoding='utf-8'))
    with np.load(source/'observations.npz') as data:
        records={k:dict(raster=data[k]) for k in data.files}
    result=[]
    for item in training:
        seed,method=item['seed'],item['method']
        checkpoint=source/f'{method}_seed{seed}.pt'
        if digest(checkpoint)!=item['checkpoint_sha256']:
            raise ValueError('checkpoint changed')
        model=LidarCNN()
        model.load_state_dict(torch.load(checkpoint,weights_only=True))
        selected=[windows[k] for k in item['training_windows']]
        def infer(selected):
            dataset=Windows(records,[(r['key'],int(r['start']),int(r['label'])) for r in selected],64)
            return score(model,dataset,16)
        train_scores=infer(selected)
        result.append(dict(seed=seed,method=method,scope='train',facility='__all__',geometry='__all__',
                           **operating_metrics([int(r['label']) for r in selected],train_scores,.5)))
        for split in ('calibration','test'):
            subset=[r for r in rows if int(r['seed'])==seed and r['split']==split]
            key='b1_score' if method=='B1' else 'augmented_score'
            if split=='calibration':
                reproduced=infer([windows[r['sample_id']] for r in subset[:16]])
                np.testing.assert_allclose(reproduced,[float(r[key]) for r in subset[:16]],rtol=1e-6,atol=1e-7)
            for facility in ('__all__','standard','harsh'):
                for geometry in ('__all__','clear'):
                    group=[r for r in subset if (facility=='__all__' or r['facility']==facility)
                           and (geometry=='__all__' or r['geometry']==geometry)]
                    result.append(dict(seed=seed,method=method,scope=split,facility=facility,geometry=geometry,
                                       **operating_metrics([int(r['label']) for r in group],[float(r[key]) for r in group],.5)))
        print(f'diagnosed {method} seed={seed}; cached score reproduction passed',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    summary=summarize(result,protocol['config']['seeds'])
    write_csv(output/'diagnostic_metrics.csv',result)
    write_csv(output/'diagnostic_summary.csv',summary)
    (output/'diagnostic_audit.json').write_text(json.dumps(dict(status='read_only_diagnosis_no_tuning',
        source_sha256=digest(Path(__file__)),experiment_protocol_sha256=digest(source/'protocol.json'),
        checkpoint_count=len(training),score_reproduction_samples_per_checkpoint=16,
        threshold=.5,limitations=['Training scores measure resubstitution, not generalization.',
        'Test inspection is diagnostic; no model or threshold selected from these results.']),indent=2)+'\n',encoding='utf-8')
    print(json.dumps([r for r in summary if r['facility']==r['geometry']=='__all__'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'results/local_simulated_fall_p1')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/simulated_fall_p1')
    args=p.parse_args()
    run(args.source,args.output_dir)

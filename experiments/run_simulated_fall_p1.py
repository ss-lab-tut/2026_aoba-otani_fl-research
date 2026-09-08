"""Generate simulated observations, reuse LidarCNN, and compare B1/B2/B3/P1."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'heterosense-fl-testbed'))
from heterosense import ConfigurationManager, DatasetBuilder, ObservationBaseline, ObservationStatisticsExtractor, FEATURE_NAMES
from heterosense._scripts.structured_occlusion_pilot import GEOMETRIES, Geometry, _client, trailing_mean
from experiments.condition_failure_diagnosis import load_datasets
from experiments.condition_shape_models import SHAPE_NAMES, prepare
from experiments.condition_threshold_validation import select_threshold, source_manifest, git_info
from experiments.compare_fall_policies import run as compare, write_csv
from src.robust_condition import QuadraticRidgeScore
from model import LidarCNN


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rasterize(frames, cfg):
    """Observed angular-bin mean range/height; no label or geometry access."""
    beams = cfg['beams']
    result = np.zeros((len(frames), 2, beams), dtype=np.float32)
    for i, frame in enumerate(frames):
        if frame.lidar is None or not len(frame.lidar):
            continue
        p = np.asarray(frame.lidar)
        angle = np.mod(np.arctan2(p[:, 1], p[:, 0]), 2*np.pi)
        bins = np.minimum((angle/(2*np.pi)*beams).astype(int), beams-1)
        count = np.bincount(bins, minlength=beams)
        valid = count > 0
        for channel, (value, scale) in enumerate(((np.linalg.norm(p[:, :2], axis=1), cfg['range_scale_metres']),
                                                  (p[:, 2], cfg['height_scale_metres']))):
            total = np.bincount(bins, weights=value, minlength=beams)
            result[i, channel, valid] = np.clip(total[valid]/count[valid]/scale, 0, 1)
    return result


def window_labels(states, starts, width):
    return np.array([int(np.any(np.asarray(states)[s:s+width]=='ABNORMAL')) for s in starts])


def freeze_condition(cfg, output):
    path = ROOT/cfg['condition_config']
    cc = json.loads(path.read_text(encoding='utf-8'))
    geometries = list(GEOMETRIES)
    data, _ = load_datasets(ROOT/'results/local_condition_robust_cache.npz', cc['train_seeds']+cc['threshold_calibration_seeds'])
    extra = tuple(Geometry(g['name'], tuple(g['occluders'])) for g in cc['augmented_training_geometries'])
    more, _ = load_datasets(ROOT/'results/local_condition_augmented_cache.npz', cc['train_seeds']+cc['threshold_calibration_seeds'], extra)
    data.update(more)
    geometries.extend(extra)
    def join(seeds):
        keys = [(s, g.name) for s in seeds for g in geometries]
        return (np.vstack([prepare(data[k][0], cc['trailing_window']) for k in keys]),
                np.concatenate([data[k][1] for k in keys]),
                np.concatenate([np.full(len(data[k][1]), k[1]) for k in keys]))
    x, y, _ = join(cc['train_seeds'])
    model = QuadraticRidgeScore().fit(x, y)
    x, y, g = join(cc['threshold_calibration_seeds'])
    cutoff = select_threshold(y, model.score(x), g, cc['calibration_max_fpr_per_geometry'])
    previous = json.loads((ROOT/'results/condition_robustness_holdout_v2.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(cutoff, previous['calibration']['shape']['threshold'], rtol=1e-10, atol=1e-10)
    artifact = output/'condition_model.npz'
    np.savez_compressed(artifact, mean=model.mean_, scale=model.scale_, coef=model.coef_, cutoff=cutoff)
    provenance = dict(artifact_sha256=digest(artifact), training_geometries=[g.name for g in geometries],
                      training_recording_ids=[f'condition_paired_site_{s}' for s in cc['train_seeds']+cc['threshold_calibration_seeds']],
                      condition_config_sha256=digest(path), frozen_cutoff=cutoff,
                      robust_cache_sha256=digest(ROOT/'results/local_condition_robust_cache.npz'),
                      augmented_cache_sha256=digest(ROOT/'results/local_condition_augmented_cache.npz'))
    return model, cutoff, provenance, cc


class Windows(Dataset):
    def __init__(self, recordings, items, width):
        self.recordings, self.items, self.width = recordings, items, width

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        key, start, label = self.items[i]
        # Existing CNN expects two channels, 64 frames and 430 angular bins.
        value = self.recordings[key]['raster'][start:start+self.width].transpose(1, 0, 2).copy()
        return torch.from_numpy(value), torch.tensor(2 if label else 0, dtype=torch.long)


def generate(cfg, condition, cutoff, condition_cfg, output):
    geometries = {'clear': Geometry('clear', ()), **{g.name:g for g in GEOMETRIES}}
    geometries.update({g['name']:Geometry(g['name'], tuple(g['occluders'])) for g in condition_cfg['test_geometries']})
    records, index = {}, []
    cols = [FEATURE_NAMES.index(n) for n in SHAPE_NAMES]
    for split, seeds in cfg['trajectory_seeds'].items():
        names = cfg['known_geometries'] + (cfg['unseen_geometries'] if split=='test' else [])
        for facility in cfg['facilities']:
            for seed in seeds:
                reference = None
                for name in names:
                    client = _client(geometries[name].occluders)
                    client['client_id'] = 'fall_'+facility['name']
                    client['sensor_noise_level'] = facility['noise']
                    manager = ConfigurationManager.from_clients([client], n_steps=cfg['n_steps'], random_seed=seed)
                    frames = DatasetBuilder(manager.to_sim_config()).build()[client['client_id']]
                    states = np.array([f.semantic_state for f in frames])
                    if reference is None:
                        reference = states
                    np.testing.assert_array_equal(states, reference)
                    baseline = ObservationBaseline.fit(frames[:cfg['baseline_steps']])
                    features = ObservationStatisticsExtractor(baseline).transform(frames[cfg['baseline_steps']:])
                    margins = condition.score(trailing_mean(features[:, cols], 5))-cutoff
                    starts = np.arange(cfg['baseline_steps'], len(frames)-cfg['window_frames']+1, cfg['stride_frames'])
                    labels = window_labels(states, starts, cfg['window_frames'])
                    key = f'{split}_{facility["name"]}_{seed}_{name}'
                    records[key] = dict(raster=rasterize(frames, cfg['raster']), starts=starts, labels=labels,
                                        condition=margins[starts+cfg['window_frames']-1-cfg['baseline_steps']])
                    for start, label, score in zip(starts, labels, records[key]['condition']):
                        index.append(dict(key=key, start=int(start), label=int(label), condition_score=float(score),
                                          split=split, facility=facility['name'], geometry=name,
                                          recording_id=f'{facility["name"]}_{seed}', sample_id=f'{key}_{start}'))
                print(f'generated {split} {facility["name"]} trajectory={seed}', flush=True)
    # Cache observed rasters and window metadata for reproducible later runs.
    np.savez_compressed(output/'observations.npz', **{k:v['raster'] for k,v in records.items()})
    write_csv(output/'windows.csv', index)
    return records, index


def score(model, data, batch_size):
    model.eval()
    values = []
    with torch.inference_mode():
        for x, _ in DataLoader(data, batch_size=batch_size):
            values.extend(torch.softmax(model(x), dim=1)[:, 2].tolist())
    return values


def execute(config, output):
    cfg = json.loads(config.read_text(encoding='utf-8'))
    if cfg['window_frames'] != 64 or cfg['raster']['beams'] != 430 or cfg['label'] != 'any_ABNORMAL_in_window':
        raise ValueError('fixed existing CNN adapter/label required')
    all_seeds = sum(cfg['trajectory_seeds'].values(), [])
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError('trajectory overlap across splits')
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a fresh output directory to preserve prior experiments')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    source = {str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__), ROOT/'model.py', ROOT/'src/robust_condition.py',
              ROOT/'src/fall_comparison.py', ROOT/'experiments/compare_fall_policies.py', ROOT/'experiments/condition_shape_models.py']}
    protocol = dict(config=cfg, config_sha256=digest(config), source_sha256=source,
                    testbed_source_sha256=source_manifest(ROOT/'heterosense-fl-testbed'),
                    git=git_info(ROOT), python=platform.python_version(), numpy=np.__version__, torch=torch.__version__)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    condition, cutoff, provenance, cc = freeze_condition(cfg, output)
    records, index = generate(cfg, condition, cutoff, cc, output)
    eval_rows = [r for r in index if r['split'] != 'train']
    clear_rows = [r for r in index if r['split']=='train' and r['geometry']=='clear']
    lookup = {(r['recording_id'],r['start'],r['geometry']):r for r in index if r['split']=='train'}
    def dataset(rows):
        return Windows(records, [(r['key'],r['start'],r['label']) for r in rows], cfg['window_frames'])
    evaluation = dataset(eval_rows)
    all_rows, training = [], []
    for seed in cfg['seeds']:
        seed_rows = [dict(seed=seed, **{k:r[k] for k in ('sample_id','recording_id','split','facility','geometry','label','condition_score')}) for r in eval_rows]
        rng = np.random.default_rng(seed)
        augmented = [lookup[(r['recording_id'],r['start'],str(rng.choice(cfg['known_geometries'])))] for r in clear_rows]
        for method, train_rows in [('B1',clear_rows), ('B2',augmented)]:
            torch.manual_seed(seed)
            model = LidarCNN()
            loader = DataLoader(dataset(train_rows), batch_size=cfg['batch_size'], shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
            losses = []
            for epoch in range(cfg['epochs']):
                model.train()
                total = 0.
                for x, y in loader:
                    optimizer.zero_grad()
                    loss = torch.nn.functional.cross_entropy(model(x), y)
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach())*len(y)
                losses.append(total/len(train_rows))
                print(f'seed={seed} {method} epoch={epoch+1}/{cfg["epochs"]} loss={losses[-1]:.5f}', flush=True)
            checkpoint = output/f'{method}_seed{seed}.pt'
            torch.save(model.state_dict(), checkpoint)
            predicted = score(model, evaluation, cfg['batch_size'])
            for row, value in zip(seed_rows, predicted):
                row['b1_score' if method=='B1' else 'augmented_score'] = value
            training.append(dict(seed=seed, method=method, losses=losses, checkpoint_sha256=digest(checkpoint),
                                 training_windows=[r['sample_id'] for r in train_rows]))
        # Include union of both models' actual training windows in the split audit.
        train_union = {r['sample_id']:r for r in clear_rows+augmented}
        for r in train_union.values():
            seed_rows.append(dict(seed=seed, **{k:r[k] for k in ('sample_id','recording_id','split','facility','geometry','label','condition_score')},
                                  b1_score='', augmented_score=''))
        write_csv(output/f'scores_seed{seed}.csv', seed_rows)
        all_rows.extend(seed_rows)
        (output/'training.json').write_text(json.dumps(training, indent=2)+'\n', encoding='utf-8')
        print(f'seed={seed} score export complete', flush=True)
    comparison_cfg = dict(seeds=cfg['seeds'], calibration_fpr_budget=cfg['calibration_fpr_budget'],
                          condition_cutoff=0., require_unseen_geometry=True,
                          estimator_provenance={str(s):provenance for s in cfg['seeds']})
    cp = output/'comparison_config.json'
    cp.write_text(json.dumps(comparison_cfg, indent=2)+'\n', encoding='utf-8')
    write_csv(output/'scores.csv', all_rows)
    summary = compare(output/'scores.csv', cp, output/'comparison')
    protocol['observations_sha256'] = digest(output/'observations.npz')
    protocol['scores_sha256'] = digest(output/'scores.csv')
    protocol['condition_provenance'] = provenance
    protocol['status'] = 'complete_simulation_only'
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    print(json.dumps([r for r in summary if r['facility']==r['geometry']=='__all__'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/simulated_fall_p1.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/local_simulated_fall_p1')
    args = parser.parse_args()
    execute(args.config, args.output_dir)

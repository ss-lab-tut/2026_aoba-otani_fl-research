import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.fall_comparison import operating_metrics, fit_policies, apply_policies, summarize
from experiments.compare_fall_policies import run, validate


def fixture():
    rows = []
    for seed in range(5):
        for split in ('train', 'calibration', 'test'):
            for i in range(8):
                rows.append(dict(seed=seed, sample_id=f'{split}-{i}', recording_id=split,
                                 split=split, facility='f1' if i<4 else 'f2',
                                 geometry='heldout' if split=='test' else 'known',
                                 label=i%2, b1_score=.8 if i%2 else .3,
                                 augmented_score=.7 if i%2 else .2,
                                 condition_score=float(i>=4)))
    cfg = dict(seeds=list(range(5)), condition_cutoff=.5, calibration_fpr_budget=.05,
               estimator_provenance={str(s):dict(artifact_sha256='fixture_only',
                    training_geometries=['known'], training_recording_ids=['train']) for s in range(5)})
    return rows, cfg


class FallComparisonTests(unittest.TestCase):
    def test_known_metrics_and_auc_ties(self):
        m = operating_metrics([0, 1, 0, 1], [.1, .7, .7, .9], .7)
        self.assertEqual((m['tp'], m['fp'], m['fn'], m['tn']), (2, 1, 0, 1))
        self.assertEqual(m['recall'], 1)
        self.assertAlmostEqual(m['precision'], 2/3)
        self.assertAlmostEqual(m['f1'], .8)
        self.assertAlmostEqual(m['auroc'], .875)
        self.assertIsNone(operating_metrics([0, 0], [.1, .2], .5)['auroc'])
        self.assertIsNone(operating_metrics([0, 1], [.1, .2], .5)['precision'])

    def test_policy_routing_and_missing_facility(self):
        rows, _ = fixture()
        cal = [r for r in rows if r['seed']==0 and r['split']=='calibration']
        for r in cal:
            if r['facility']=='f2':
                r['augmented_score'] += .4
        fitted = fit_policies(cal, .05, .5)
        self.assertGreater(fitted['B3']['f2'], fitted['B3']['f1'])
        test = copy.deepcopy(cal)
        test[0]['facility'] = 'new'
        t = apply_policies(test, fitted)
        self.assertEqual(t['B3'][0], fitted['B2'])
        self.assertGreater(t['P1'][4], t['P1'][0])
        # Decision-time policies must not read test labels.
        for r in test:
            r['label'] = 1-r['label']
        for key, value in apply_policies(test, fitted).items():
            np.testing.assert_array_equal(value, t[key])

    def test_leakage_and_missing_seed_rejected(self):
        rows, cfg = fixture()
        rows[8]['recording_id'] = 'train'
        with self.assertRaisesRegex(ValueError, 'recording leakage'):
            validate(rows, cfg)
        rows, cfg = fixture()
        cfg['estimator_provenance']['0']['training_recording_ids'] = ['test']
        with self.assertRaisesRegex(ValueError, 'estimator training overlaps'):
            validate(rows, cfg)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            summarize([dict(seed=0,method='B1',scope='all',facility='x',geometry='y')], range(5))

    def test_full_five_seed_csv_and_unseen_guard(self):
        rows, cfg = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'input.csv'
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            config = root/'config.json'
            config.write_text(json.dumps(cfg))
            summary = run(path, config, root/'out')
            self.assertEqual({r['method'] for r in summary}, {'B1','B2','B3','P1'})
            self.assertTrue(all(r['n_seeds']==5 for r in summary))
            self.assertTrue((root/'out/paired_P1_minus_B3_summary.csv').exists())
            for p in cfg['estimator_provenance'].values():
                p['training_geometries'].append('heldout')
            config.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'no geometry held out'):
                run(path, config, root/'rejected')


if __name__ == '__main__':
    unittest.main()

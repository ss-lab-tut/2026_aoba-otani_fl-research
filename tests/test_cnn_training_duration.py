import unittest
from experiments.check_cnn_training_duration import partition, GEOMETRIES


class DevelopmentPartitionTests(unittest.TestCase):
    def rows(self):
        return [dict(split='train',recording_id=f'v2_{facility}_{seed}',facility=facility,
                     geometry=geometry,label=str(label))
                for facility in ('standard','harsh') for seed in (10001,10002,10003,10004)
                for geometry in GEOMETRIES for label in (0,1)]

    def test_recordings_disjoint_and_nontraining_excluded(self):
        rows=self.rows()+[dict(split='test'),dict(split='calibration')]
        fit,dev=partition(rows)
        self.assertTrue(all(r['recording_id'].endswith('10004') for r in dev))
        self.assertFalse({r['recording_id'] for r in fit}&{r['recording_id'] for r in dev})
        self.assertEqual(len(fit)+len(dev),len(self.rows()))

    def test_unknown_training_record_rejected(self):
        rows=self.rows();rows[0]['recording_id']='v2_standard_99999'
        with self.assertRaises(ValueError): partition(rows)

    def test_missing_development_class_rejected(self):
        rows=[r for r in self.rows() if not (r['recording_id']=='v2_standard_10004' and r['label']=='1')]
        with self.assertRaises(ValueError): partition(rows)

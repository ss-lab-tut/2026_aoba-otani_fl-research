import gzip
import hashlib
from pathlib import Path
import tempfile
import unittest

from experiments.sweep_fall_operating_points import read_scores,calibration_check
from src.fall_comparison import fit_policies


class OperatingPointTests(unittest.TestCase):
    def test_modified_scores_rejected_before_evaluation(self):
        raw=b'seed,label,b1_score,augmented_score,condition_score\n0,1,0.7,0.8,0.1\n'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'scores.csv.gz';p.write_bytes(gzip.compress(raw,mtime=0))
            h=hashlib.sha256(raw).hexdigest()
            self.assertEqual(read_scores(p,h)[0]['augmented_score'],.8)
            p.write_bytes(gzip.compress(raw.replace(b'0.8',b'0.9'),mtime=0))
            with self.assertRaisesRegex(ValueError,'changed'):read_scores(p,h)

    def test_empirical_group_budget_violation_detected(self):
        rows=[dict(seed=0,label=i%2,b1_score=.8 if i%2 else .2,augmented_score=.8 if i%2 else .2,
                   facility='a' if i<4 else 'b',condition_score=float(i>=4)) for i in range(8)]
        fitted=fit_policies(rows,.05,.5)
        result=calibration_check(rows,fitted,.05)
        self.assertTrue(all(r['false_positive_rate']==0 for r in result))
        fitted['B3']['a']=0.
        with self.assertRaisesRegex(ValueError,'constraint violated'):
            calibration_check(rows,fitted,.05)


if __name__=='__main__':unittest.main()

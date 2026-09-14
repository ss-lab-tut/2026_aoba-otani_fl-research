import unittest
from experiments.verify_fresh_score_replay import verify_table


class ReplayTableTests(unittest.TestCase):
    def rows(self):
        return [dict(seed='0',method='P1',scope='unseen',facility='standard',geometry='wide',n='24',recall='0.5',precision='')]

    def test_changed_confusion_count_rejected(self):
        expected=self.rows();actual=self.rows();actual[0]['n']='25'
        with self.assertRaises(ValueError):verify_table(expected,actual)

    def test_undefined_metric_not_silently_zero(self):
        expected=self.rows();actual=self.rows();actual[0]['precision']='0'
        with self.assertRaises(ValueError):verify_table(expected,actual)

    def test_duplicate_group_rejected(self):
        expected=self.rows();actual=self.rows()*2
        with self.assertRaises(ValueError):verify_table(expected,actual)

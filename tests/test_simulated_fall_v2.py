import copy
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.run_simulated_fall_p1_v2 import (
    BehaviorModel, ConfigurationManager, _client, SemanticState, candidates, set_persistence, choose_balanced,
)
from heterosense._core._behavior_model import AbnormalType


def state(name='WALKING',subtype='NONE'):
    return SimpleNamespace(state=SemanticState(name),abnormal_type=AbnormalType(subtype))


class CorrectedFallTests(unittest.TestCase):
    def test_persistence_changes_only_abnormal_row(self):
        cfg=ConfigurationManager.from_clients([_client(())],n_steps=10,random_seed=0).to_sim_config()
        model=BehaviorModel(cfg.clients[0],np.random.default_rng(0))
        old=copy.deepcopy(model.transition_matrix)
        set_persistence(model,.8)
        for key,row in model.transition_matrix.items():
            self.assertAlmostEqual(sum(row.values()),1)
            if key!=SemanticState.ABNORMAL:self.assertEqual(row,old[key])
        self.assertEqual(model.transition_matrix[SemanticState.ABNORMAL][SemanticState.ABNORMAL],.8)

    def test_near_fall_is_not_positive_and_endpoint_is_third_frame(self):
        latent=[state() for _ in range(230)]
        latent[170:174]=[state('ABNORMAL','FALL') for _ in range(4)]
        latent[190]=state('ABNORMAL','NEAR_FALL')
        latent[210:213]=[state('ABNORMAL','RECOVERED_FALL') for _ in range(3)]
        data=candidates(latent,'r')
        self.assertEqual([r['endpoint'] for r in data['fall']],[172,212])
        self.assertEqual([r['label'] for r in data['near_fall']],[0])
        self.assertEqual(data['near_fall'][0]['endpoint'],192)
        self.assertTrue(all(latent[r['endpoint']].state!=SemanticState.ABNORMAL for r in data['near_fall']))

    def test_event_shortage_fails_before_training(self):
        pools={'r':{'fall':[],'near_fall':[],'adl':[]}}
        with self.assertRaisesRegex(ValueError,'insufficient fall'):
            choose_balanced(pools,{'fall':1},0,1)
        pools['r']['fall']=[dict(recording_id='r',endpoint=100,event_subtype='RECOVERED_FALL')]
        with self.assertRaisesRegex(ValueError,'strict FALL'):
            choose_balanced(pools,{'fall':1},0,1)


if __name__=='__main__':unittest.main()

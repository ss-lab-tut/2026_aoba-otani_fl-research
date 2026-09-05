import unittest

import numpy as np

from src.temporal_integrity import reconstruct_order,purged_split,verify_no_frame_overlap
from src.input_quality import angular_zero_mask,observed_quality_groups,route_thresholds
from experiments.audit_temporal_overlap import overlap_components


class IntegrityTests(unittest.TestCase):
    def test_reconstruct_and_purge_shuffled_overlapping_windows(self):
        frames=np.arange(250*3*2,dtype=float).reshape(250,3,2)
        windows=np.array([frames[i:i+8] for i in range(0,242,2)])
        shuffle=np.random.default_rng(7).permutation(len(windows))
        order=reconstruct_order(windows[shuffle],hop=2)
        np.testing.assert_array_equal(shuffle[order],np.arange(len(windows)))
        for fold in range(5):
            split=purged_split(order,fold,excluded=[0],hop=2,window_frames=8)
            verify_no_frame_overlap(order,split,hop=2,window_frames=8)
            self.assertNotIn(0,split['train']+split['calibration']+split['test'])
            self.assertEqual(len(split['train']+split['calibration']+split['test']+split['dropped']),len(order))

    def test_reject_ambiguous_order_and_shared_frames(self):
        with self.assertRaises(ValueError):reconstruct_order(np.zeros((3,8,3,2)),2)
        with self.assertRaises(ValueError):verify_no_frame_overlap([0,1],{'train':[0],'calibration':[],'test':[1]},2,8)

    def test_static_blocks_do_not_establish_recording_identity(self):
        r=overlap_components([np.zeros((1,16,3,2)),np.zeros((1,16,3,2))])
        self.assertEqual(r['n_components'],2)
        self.assertFalse(r['shared_cross_split_blocks'])

    def test_mask_routes_only_by_observed_tensor(self):
        x=np.ones((2,64,100,2))
        masked=angular_zero_mask(x,.35,.8)
        self.assertTrue(np.all(x==1))
        self.assertTrue(np.all(observed_quality_groups(masked)=='high_zero_fraction'))
        np.testing.assert_array_equal(route_thresholds(['new','known'],{'known':.2},.7),[.7,.2])


if __name__=='__main__':unittest.main()

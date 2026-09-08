import unittest

import numpy as np

from experiments.audit_simulated_cnn_inputs import episodes,label_details,input_stats


class AuditTests(unittest.TestCase):
    def test_episodes_do_not_duplicate_contiguous_frames(self):
        self.assertEqual(episodes(['ABNORMAL','ABNORMAL','WALKING','ABNORMAL']),[(0,2),(3,4)])
        self.assertEqual(episodes(['WALKING']),[])

    def test_old_event_does_not_imply_endpoint_event(self):
        states=['ABNORMAL']+['WALKING']*63
        result=label_details(states,0,64)
        self.assertEqual(result['label'],1)
        self.assertEqual(result['abnormal_frames'],1)
        self.assertEqual(result['last5_abnormal'],0)
        self.assertEqual(result['endpoint_abnormal'],0)
        with self.assertRaises(ValueError):
            label_details(states,1,64)

    def test_missing_input_and_saturation_audit(self):
        raster=np.zeros((2,2,430),dtype=np.float32)
        raster[0,0,0]=1
        m=input_stats(raster)
        self.assertEqual(m['empty_frames'],1)
        self.assertAlmostEqual(m['occupied_bin_fraction'],1/860)
        raster[0,0,0]=np.nan
        with self.assertRaises(ValueError):
            input_stats(raster)


if __name__=='__main__':
    unittest.main()

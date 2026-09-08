import unittest
from types import SimpleNamespace

import numpy as np

from experiments.run_simulated_fall_p1 import rasterize, window_labels, Windows


class SimulatedFallTests(unittest.TestCase):
    def test_raster_uses_only_observations(self):
        cfg = dict(beams=430, range_scale_metres=8., height_scale_metres=2.5)
        points = np.array([[1.,0.,1.], [3.,0.,2.]])
        a = SimpleNamespace(lidar=points, semantic_state='ABNORMAL')
        b = SimpleNamespace(lidar=points, semantic_state='WALKING')
        raster = rasterize([a,b,SimpleNamespace(lidar=None)], cfg)
        np.testing.assert_array_equal(raster[0], raster[1])
        self.assertAlmostEqual(float(raster[0,0,0]), .25)
        self.assertAlmostEqual(float(raster[0,1,0]), .6)
        self.assertEqual(raster[2].sum(), 0)

    def test_window_label_causality_and_cnn_shape(self):
        states = ['WALKING']*128
        states[64] = 'ABNORMAL'
        np.testing.assert_array_equal(window_labels(states, [0,1,64], 64), [0,1,1])
        recordings = {'r':dict(raster=np.zeros((128,2,430), dtype=np.float32))}
        data = Windows(recordings, [('r',0,0),('r',1,1)],64)
        x,y = data[1]
        self.assertEqual(tuple(x.shape), (2,64,430))
        self.assertEqual(int(y),2)
        x[:] = 1
        self.assertEqual(recordings['r']['raster'].sum(),0)


if __name__ == '__main__':
    unittest.main()

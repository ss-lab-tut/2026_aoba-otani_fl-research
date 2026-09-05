import unittest

import numpy as np

from src.data_partition import client_indices


class PartitionTests(unittest.TestCase):
    def test_all_955_samples_are_used_exactly_once(self):
        parts=[client_indices(955,i,2) for i in range(2)]
        self.assertEqual([len(p) for p in parts],[478,477])
        np.testing.assert_array_equal(np.concatenate(parts),np.arange(955))

    def test_reject_empty_and_invalid_clients(self):
        for args in [(1,0,2),(3,-1,2),(3,2,2),(3,0,0)]:
            with self.assertRaises(ValueError):client_indices(*args)


if __name__=='__main__':unittest.main()

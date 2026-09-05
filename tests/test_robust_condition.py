import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'heterosense-fl-testbed'))
from heterosense import ObservationBaseline, ObservableConditionEstimator
from src.robust_condition import bounded_contrast, fit_mean_spatial_baseline, QuadraticRidgeScore
from experiments.condition_failure_diagnosis import stable_margin
from experiments.confirm_condition_robustness import acceptance


class RobustConditionTests(unittest.TestCase):
    def test_acceptance_cannot_hide_one_failed_condition_behind_average(self):
        limits={'mean_recall_min':.9,'mean_fpr_max':.1,
                'each_geometry_mean_recall_min':.8,'each_geometry_mean_fpr_max':.1,
                'each_seed_geometry_recall_min':.7}
        summary={'all':{'recall':.95,'false_positive_rate':.01},
                 'difficult':{'recall':.6,'false_positive_rate':.01}}
        folds=[{'variants':{'shape_quadratic':{'recall':.6}}}]
        result=acceptance(summary,folds,limits)
        self.assertFalse(result['passed'])
        self.assertTrue(result['checks']['mean_recall'])
        self.assertFalse(result['checks']['each_geometry_mean_recall'])

    def test_quadratic_score_learns_interaction_without_changing_scaler(self):
        x=np.tile([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],(10,1))
        y=(x[:,0]*x[:,1]>0).astype(int)
        model=QuadraticRidgeScore().fit(x,y)
        self.assertTrue(np.array_equal(model.score(x)>=.5,y))
        before=model.mean_.copy()
        model.score(np.array([[100.,100.]]))
        np.testing.assert_array_equal(before,model.mean_)
        with self.assertRaises(ValueError): model.score([[np.nan,0]])

    def test_reference_conserves_mass_for_moving_person(self):
        frames=[]
        for i in range(9):
            x,y=divmod(i,3)
            cloud=np.tile([x,y,1.0],(90,1)).astype(float)
            frames.append(SimpleNamespace(lidar=cloud,pressure=None))
        old=ObservationBaseline.fit(frames)
        new=fit_mean_spatial_baseline(frames)
        self.assertEqual(sum(old.sector_point_counts),0)
        self.assertAlmostEqual(sum(new.sector_point_counts),90)
        self.assertTrue(all(v>0 for v in new.sector_point_counts))

    def test_reference_uses_only_provided_observations(self):
        frames=[SimpleNamespace(lidar=np.array([[0.,0.,1.],[1.,1.,1.],[2.,2.,1.]]),pressure=None)]
        first=fit_mean_spatial_baseline(frames)
        frames[0].semantic_state='changed_ground_truth'
        second=fit_mean_spatial_baseline(frames)
        np.testing.assert_array_equal(first.sector_point_counts,second.sector_point_counts)
        self.assertIsNone(fit_mean_spatial_baseline([]).sector_point_counts)

    def test_bounded_contrast(self):
        values=bounded_contrast([0,1,10,1e300])
        self.assertEqual(values[0],-1)
        self.assertEqual(values[1],0)
        self.assertTrue(np.all(np.diff(values)>0))
        self.assertTrue(np.all(np.abs(values)<=1))
        with self.assertRaises(ValueError): bounded_contrast([-1])

    def test_margin_matches_log_odds_and_avoids_saturation(self):
        model=ObservableConditionEstimator().fit(np.array([[-1.],[1.]]),np.array([0,1]))
        x=np.array([[-.3],[.2]])
        p=model.predict_proba(x)
        np.testing.assert_allclose(stable_margin(model,x),np.log(p[:,1]/p[:,0]))
        extreme=np.array([[1000.],[1001.]])
        self.assertTrue(np.all(model.predict_proba(extreme)[:,1]==1))
        self.assertGreater(np.diff(stable_margin(model,extreme))[0],0)


if __name__=='__main__': unittest.main()

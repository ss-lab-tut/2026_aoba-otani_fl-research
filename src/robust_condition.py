"""Spatial reference fitting which conserves observed point mass.

Only observed calibration frames are used; labels and simulator settings are
not accessed. A per-cell temporal median need not sum to the total point count
and can erase cells visited less than half the time. Means avoid that failure.
"""
from dataclasses import replace

import numpy as np


def fit_mean_spatial_baseline(frames):
    from heterosense import ObservationBaseline
    from heterosense.observation_statistics import _sector_values
    frames = list(frames)
    baseline = ObservationBaseline.fit(frames)
    if baseline.x_sector_edges is None:
        return baseline
    counts = []
    for frame in frames:
        if frame.lidar is not None and len(frame.lidar):
            c, _ = _sector_values(frame.lidar, baseline.x_sector_edges, baseline.y_sector_edges)
            counts.append(c)
    return replace(baseline, sector_point_counts=tuple(np.mean(counts, axis=0).tolist()))


def bounded_contrast(ratios):
    """Map nonnegative ratios to [-1, 1], preserving the reference value 1."""
    x = np.asarray(ratios, dtype=float)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("ratios must be finite and nonnegative")
    return 1 - 2 / (1 + x)


class QuadraticRidgeScore:
    """Regularized quadratic binary regression; scores are not probabilities.

    Scaling and all polynomial coefficients are fitted on training data only.
    Fixed ridge strength 1 and clipping bound 8 are exploratory defaults.
    """

    def _design(self, features):
        x=np.asarray(features,dtype=float)
        if x.ndim!=2 or x.shape[1]!=len(self.mean_) or not np.isfinite(x).all():
            raise ValueError('invalid feature matrix')
        z=np.clip((x-self.mean_)/self.scale_,-8,8)
        i,j=np.triu_indices(z.shape[1])
        return np.column_stack([np.ones(len(z)),z,z[:,i]*z[:,j]])

    def fit(self,features,labels):
        x=np.asarray(features,dtype=float); y=np.asarray(labels)
        if x.ndim!=2 or not len(x) or not np.isfinite(x).all():
            raise ValueError('finite nonempty matrix required')
        if y.shape!=(len(x),) or set(np.unique(y))!={0,1}:
            raise ValueError('matching binary labels with both classes required')
        self.mean_=x.mean(axis=0)
        self.scale_=np.maximum(x.std(axis=0),1e-8)
        design=self._design(x)
        weights=np.where(y==1,len(y)/(2*np.sum(y==1)),len(y)/(2*np.sum(y==0)))
        weighted=design*weights[:,None]
        penalty=np.eye(design.shape[1]); penalty[0,0]=0
        self.coef_=np.linalg.solve(design.T@weighted+penalty,weighted.T@y)
        return self

    def score(self,features):
        return self._design(features)@self.coef_

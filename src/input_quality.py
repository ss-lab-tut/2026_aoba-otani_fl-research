"""Quality routing for explicit zero-masked tensor stress tests only.

Zero is a normalized median, not necessarily a missing physical LiDAR return.
This protocol must not be presented as real occlusion estimation.
"""
import numpy as np


def angular_zero_mask(windows,fraction,start_fraction):
    x=np.array(windows,copy=True)
    if x.ndim!=4 or not 0<=fraction<1 or not 0<=start_fraction<1:
        raise ValueError('invalid windows or mask fractions')
    beams=x.shape[2];width=int(round(beams*fraction));start=int(beams*start_fraction)
    indices=(start+np.arange(width))%beams
    x[:,:,indices,:]=0
    return x


def observed_quality_groups(windows):
    x=np.asarray(windows)
    if x.ndim!=4 or not np.isfinite(x).all():
        raise ValueError('finite N x time x beam x channel required')
    fraction=np.mean(np.all(x==0,axis=-1),axis=(1,2))
    return np.where(fraction<.10,'low_zero_fraction',np.where(fraction<.25,'medium_zero_fraction','high_zero_fraction'))


def route_thresholds(groups,thresholds,default):
    """Unknown observed groups use the declared common threshold, not labels."""
    return np.array([thresholds.get(str(g),default) for g in groups],dtype=float)

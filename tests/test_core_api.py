"""Contract tests for the stable ``kernelcal.core`` facade."""

import numpy as np

from kernelcal.core import FixedPointDetector, KernelTrajectory, MaxCalSampler


def test_core_exports_are_importable():
    assert FixedPointDetector is not None
    assert KernelTrajectory is not None
    assert MaxCalSampler is not None


def test_core_fixed_point_detector_smoke():
    K = np.eye(4)
    fp = FixedPointDetector(tol=1e-9, window=2)
    fp.update(K).update(K).update(K)
    assert fp.is_fixed_point()
    assert fp.candidate_fixed_point() is not None


def test_core_maxcal_sampler_smoke():
    locs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    sampler = MaxCalSampler(locs)
    sampler.update()
    p = sampler.distribution()
    assert p.shape == (4,)
    assert np.isclose(np.sum(p), 1.0)

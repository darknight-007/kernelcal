"""Ensure legacy ``deepgis_kernelcal`` re-exports match ``kernelcal.geo3d``."""

import numpy as np

import deepgis_kernelcal
from kernelcal import geo3d


def test_deepgis_kernelcal_reexports_geo3d_api():
    assert deepgis_kernelcal.compress_point_cloud is geo3d.compress_point_cloud
    assert deepgis_kernelcal.decompress_to_kernel is geo3d.decompress_to_kernel
    assert deepgis_kernelcal.compress_mesh_geometry is geo3d.compress_mesh_geometry
    assert deepgis_kernelcal.compress_temporal_clouds is geo3d.compress_temporal_clouds
    assert deepgis_kernelcal.mesh_combinatorial_laplacian is geo3d.mesh_combinatorial_laplacian


def test_deepgis_kernelcal_smoke_from_old_tests():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((80, 3))
    c = deepgis_kernelcal.compress_point_cloud(pts, max_points=64, n_modes=20, k_neighbors=6, seed=1)
    K = deepgis_kernelcal.decompress_to_kernel(c)
    assert K.shape == (64, 64)
    assert np.max(np.abs(K - K.T)) < 1e-9

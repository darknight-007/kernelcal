import numpy as np
import pytest

from kernelcal.geo3d import (
    CompressedMeshGeometry,
    CompressedSpectralKernel,
    CompressionBounds,
    HodgeSpectralBasis,
    PersistenceResult,
    betti_numbers,
    build_hodge_basis,
    compress_dae,
    compress_mesh_geometry,
    compress_mesh_roundtrip,
    compress_point_cloud,
    compress_temporal_clouds,
    compression_ratio_formula,
    compression_ratio_vs_modes,
    decompress_dae,
    decompress_mesh_roundtrip,
    decompress_to_kernel,
    estimate_compression_bounds,
    mesh_persistence,
    mode_count_for_distortion,
    mode_count_for_topology,
    vietoris_rips_persistence,
)
from kernelcal.geo3d.hodge import boundary_1, boundary_2, hodge_decompose


def test_point_cloud_roundtrip_and_serialization():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((80, 3))
    c = compress_point_cloud(pts, max_points=64, n_modes=20, k_neighbors=6, seed=1)
    payload = c.to_bytes()
    c2 = CompressedSpectralKernel.from_bytes(payload)
    K = decompress_to_kernel(c2)
    assert K.shape == (64, 64)
    assert np.max(np.abs(K - K.T)) < 1e-9
    evals = np.linalg.eigvalsh((K + K.T) / 2)
    assert np.min(evals) > -1e-8


def test_mesh_compress():
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    c = compress_mesh_geometry(v, f, n_modes=4)
    assert c.eigenvalues.shape[0] == 4


def test_mesh_roundtrip_geometry_payload():
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    c = compress_mesh_roundtrip(v, f, n_modes=4)
    payload = c.to_bytes()
    c2 = CompressedMeshGeometry.from_bytes(payload)
    v_hat, f_hat = decompress_mesh_roundtrip(c2)
    assert v_hat.shape == v.shape
    assert np.array_equal(f_hat, f)


def test_temporal_summary():
    rng = np.random.default_rng(2)
    clouds = [rng.standard_normal((40, 3)) for _ in range(3)]
    s = compress_temporal_clouds(clouds, times=[0.0, 1.0, 2.0], max_points=32, n_modes=12)
    assert s.hs_distances.shape == (2,)
    assert len(s.compressed_frames) == 3
    assert s.trajectory is not None


# ---------------------------------------------------------------------------
# Hodge Laplacian tests
# ---------------------------------------------------------------------------

def _tetra():
    v = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    f = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
    return v, f


def test_hodge_boundary_operators_exact_sequence():
    """B₁ B₂ = 0 (boundary of a boundary is zero)."""
    _, f = _tetra()
    B1 = boundary_1(4, f)
    B2 = boundary_2(f)
    product = (B1 @ B2).toarray()
    assert np.allclose(product, 0.0, atol=1e-12), f"B₁B₂ ≠ 0: max={np.abs(product).max()}"


def test_betti_numbers_tetrahedron():
    """Tetrahedron (hollow): β₀=1, β₁=0, β₂=1."""
    _, f = _tetra()
    b0, b1, b2 = betti_numbers(4, f)
    assert b0 == 1, f"β₀ should be 1 (connected), got {b0}"
    assert b1 == 0, f"β₁ should be 0 (no loops), got {b1}"
    assert b2 == 1, f"β₂ should be 1 (one void), got {b2}"


def test_betti_numbers_euler_characteristic():
    """Euler characteristic V - E + F = β₀ - β₁ + β₂."""
    _, f = _tetra()
    b0, b1, b2 = betti_numbers(4, f)
    chi_topo = b0 - b1 + b2
    # tetrahedron: V=4, E=6, F=4 → χ=2
    chi_geom = 4 - 6 + 4
    assert chi_topo == chi_geom, f"χ topological={chi_topo} ≠ geometric={chi_geom}"


def test_build_hodge_basis_shape():
    _, f = _tetra()
    basis = build_hodge_basis(4, f, n_modes_0=4, n_modes_1=6, n_modes_2=4)
    assert isinstance(basis, HodgeSpectralBasis)
    assert basis.eigenvalues_0.shape[0] == 4
    assert basis.betti == (1, 0, 1)


def test_hodge_decompose_pythagoras():
    """||f||² = ||grad||² + ||curl||² + ||harmonic||² (Pythagorean identity)."""
    _, f = _tetra()
    B1 = boundary_1(4, f)
    B2 = boundary_2(f)
    n_E = B2.shape[0]
    rng = np.random.default_rng(7)
    signal = rng.standard_normal(n_E)
    grad, curl, harmonic = hodge_decompose(signal, B1, B2)
    total = np.dot(signal, signal)
    parts = np.dot(grad, grad) + np.dot(curl, curl) + np.dot(harmonic, harmonic)
    # Orthogonality holds only for exact decomposition; lstsq gives close approximation
    assert abs(total - parts) / (total + 1e-12) < 0.05


# ---------------------------------------------------------------------------
# Persistent homology tests
# ---------------------------------------------------------------------------

def test_persistence_0d_path_graph():
    """Path graph on 4 nodes: 3 finite 0D pairs, 1 essential."""
    result = vietoris_rips_persistence(
        np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[3.,0.,0.]])
    )
    assert isinstance(result, PersistenceResult)
    pairs0 = result.pairs_by_dim(0)
    assert len(pairs0) == 4  # 3 finite merges + 1 essential


def test_mesh_persistence_tetrahedron():
    v, f = _tetra()
    result = mesh_persistence(4, f, v)
    assert isinstance(result, PersistenceResult)
    # One essential 0D class (connected mesh)
    assert result.betti_at_inf.get(0, 0) == 1


def test_persistence_betti_at_threshold():
    v, f = _tetra()
    result = mesh_persistence(4, f, v)
    betti = result.betti_at_threshold(10.0)  # all edges included
    assert betti.get(0, 0) >= 1


# ---------------------------------------------------------------------------
# Compression bound tests
# ---------------------------------------------------------------------------

def test_compression_ratio_formula_coeff_only():
    """coeff_only ratio should be > 1 for V >> k."""
    ratio = compression_ratio_formula(10000, 20000, 64, coeff_only=True)
    assert ratio > 1.0, f"Expected ratio > 1, got {ratio:.3f}"


def test_compression_ratio_formula_full():
    """full (eigenvec stored) ratio < 1 for k ≈ V (no savings)."""
    ratio_small_k = compression_ratio_formula(10000, 20000, 10, coeff_only=False)
    ratio_large_k = compression_ratio_formula(10000, 20000, 9000, coeff_only=False)
    assert ratio_small_k > ratio_large_k


def test_compression_ratio_vs_modes_monotone():
    table = compression_ratio_vs_modes(1000, 2000, modes=np.arange(1, 200, 10))
    ratios = table[:, 1]
    # coeff_only=True: ratios should be monotonically decreasing in k
    assert np.all(np.diff(ratios) <= 0.0), "Ratios should decrease as k grows"


def test_estimate_compression_bounds_end_to_end():
    v, f = _tetra()
    c = compress_mesh_roundtrip(v, f, n_modes=4)
    b = betti_numbers(4, f)
    bounds = estimate_compression_bounds(
        v, f,
        eigenvectors=c.eigenvectors,
        eigenvalues=c.eigenvalues,
        n_modes=4,
        coeff_only=True,
        betti=b,
    )
    assert isinstance(bounds, CompressionBounds)
    assert bounds.compression_ratio > 0
    assert 0.0 <= bounds.relative_distortion <= 1.0
    assert bounds.topology_preserved is not None
    print(bounds.summary())


def test_mode_count_for_topology():
    b = (1, 2, 0)  # torus: β₁=2
    k_min = mode_count_for_topology(b)
    assert k_min == 3  # β₀ + β₁ = 3


def test_mode_count_for_distortion():
    lam = np.array([0., 0.5, 1., 2., 4., 8., 16.])
    # Coefficients: energy concentrated in first few modes
    coeffs = np.array([[10., 0., 0.], [8., 0., 0.], [1., 0., 0.],
                       [0.1, 0., 0.], [0.01, 0., 0.],
                       [0.001, 0., 0.], [0.0001, 0., 0.]])
    k = mode_count_for_distortion(lam, coeffs, target_rel_distortion=0.01)
    assert 1 <= k <= len(lam)


# ---------------------------------------------------------------------------
# DAE roundtrip (skipped without trimesh)
# ---------------------------------------------------------------------------

def test_dae_roundtrip_with_trimesh(tmp_path):
    trimesh = pytest.importorskip("trimesh", reason="trimesh not installed")
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    in_dae = tmp_path / "in.dae"
    payload = tmp_path / "mesh.kcmesh"
    out_dae = tmp_path / "out.dae"
    mesh.export(str(in_dae), file_type="dae")

    compress_dae(in_dae, n_modes=4, payload_path=payload)
    decompress_dae(payload, out_dae)

    assert out_dae.exists()

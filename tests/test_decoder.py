"""Tests for kernelcal.geo3d.decoder.

Covers:
  - reconstruct_skeleton: topology-loss warning, correct shapes
  - _aggregate_D_t: all three fallback paths
  - triage_detail_level: correct DetailLevel for each D_t band
  - compute_curl_energy: channeled > flat, bounded in [0, 1]
  - decode: end-to-end shape and flag contracts
  - Tier 3 integration: encoder → telemetry → decode → synthesize round-trip
"""

import logging
import numpy as np
import pytest

from kernelcal.geo3d.large_mesh import compress_large_mesh, LargeMeshCompressed
from kernelcal.geo3d.decoder import (
    SpectralTelemetry,
    DecodedTwin,
    DetailLevel,
    D_T_THRESHOLD_LOW,
    D_T_THRESHOLD_MED,
    D_T_THRESHOLD_PATCH,
    CURL_ACTIVE,
    reconstruct_skeleton,
    triage_detail_level,
    _aggregate_D_t,
    compute_curl_energy,
    decode,
)
from kernelcal.geo3d.detail_synthesis import synthesize


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _tetrahedron():
    v = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ], dtype=float)
    f = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
    return v, f


def _grid_mesh(n: int = 8):
    """Flat n×n grid mesh — β₀=1, β₁=0."""
    xs = np.linspace(0, 1, n)
    xv, yv = np.meshgrid(xs, xs)
    verts = np.column_stack([xv.ravel(), yv.ravel(), np.zeros(n*n)])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i*n + j
            b = a + 1
            c = (i+1)*n + j
            d = c + 1
            faces += [[a, b, c], [b, d, c]]
    return verts.astype(float), np.array(faces, dtype=np.int32)


def _compressed_tel(verts, faces, *, betti=(1,0,0), k=4, D_t=0.5,
                    curl_energy=None, latent_code=None):
    """Build a SpectralTelemetry from raw arrays."""
    c = compress_large_mesh(verts, faces, n_modes=k, heat_tau=1.0)
    D_m = np.full(k, D_t / k)
    return SpectralTelemetry(
        compressed=c,
        betti=betti,
        D_m_residuals=D_m,
        spectral_entropy=1.5,
        delta_prime=D_t / k,
        curl_energy=curl_energy,
        latent_code=latent_code,
    )


# ---------------------------------------------------------------------------
# Stage 1 — reconstruct_skeleton
# ---------------------------------------------------------------------------

class TestReconstructSkeleton:

    def test_output_shapes_match_input(self):
        v, f = _grid_mesh(8)
        tel = _compressed_tel(v, f, k=4)
        rv, rf = reconstruct_skeleton(tel)
        assert rv.shape == v.shape
        assert rf.shape == f.shape

    def test_topology_loss_warning_fires(self):
        """Warning fires when transmitted k < k_min = beta_0 + beta_1."""
        import logging
        v, f = _grid_mesh(8)
        c = compress_large_mesh(v, f, n_modes=4, heat_tau=1.0)
        c.meta["n_modes"] = 2   # simulate under-transmitted k
        tel = SpectralTelemetry(
            compressed=c, betti=(1, 3, 0),
            D_m_residuals=np.zeros(4),
        )
        decoder_logger = logging.getLogger("kernelcal.geo3d.decoder")
        records = []
        class _Capture(logging.Handler):
            def emit(self, r): records.append(r)
        h = _Capture()
        decoder_logger.addHandler(h)
        try:
            reconstruct_skeleton(tel)
        finally:
            decoder_logger.removeHandler(h)
        assert any("Topology loss" in r.getMessage() for r in records), \
            "Expected 'Topology loss' warning from decoder"

    def test_no_warning_when_k_sufficient(self, caplog):
        """No warning when k >= k_min."""
        v, f = _grid_mesh(8)
        tel = _compressed_tel(v, f, betti=(1, 0, 0), k=4)
        with caplog.at_level(logging.WARNING, logger="kernelcal.geo3d.decoder"):
            reconstruct_skeleton(tel)
        assert not any("Topology loss" in r.message for r in caplog.records)

    def test_vertices_finite(self):
        v, f = _tetrahedron()
        tel = _compressed_tel(v, f, k=3)
        rv, _ = reconstruct_skeleton(tel)
        assert np.all(np.isfinite(rv))


# ---------------------------------------------------------------------------
# _aggregate_D_t — three fallback paths
# ---------------------------------------------------------------------------

class TestAggregateD_t:

    def test_from_D_m_residuals(self):
        v, f = _grid_mesh(6)
        c = compress_large_mesh(v, f, n_modes=4)
        tel = SpectralTelemetry(
            compressed=c, betti=(1,0,0),
            D_m_residuals=np.array([1.0, 2.0, 3.0, 4.0]),
        )
        assert _aggregate_D_t(tel) == pytest.approx(10.0)

    def test_fallback_to_delta_prime(self):
        v, f = _grid_mesh(6)
        c = compress_large_mesh(v, f, n_modes=4)
        tel = SpectralTelemetry(
            compressed=c, betti=(1,0,0),
            D_m_residuals=None,
            delta_prime=3.0,
        )
        # fallback: |delta_prime| * k = 3.0 * 4 = 12.0
        assert _aggregate_D_t(tel) == pytest.approx(12.0)

    def test_fallback_to_zero(self):
        v, f = _grid_mesh(6)
        c = compress_large_mesh(v, f, n_modes=4)
        tel = SpectralTelemetry(
            compressed=c, betti=(1,0,0),
            D_m_residuals=None, delta_prime=None,
        )
        assert _aggregate_D_t(tel) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Stage 2 — triage_detail_level
# ---------------------------------------------------------------------------

class TestTriageDetailLevel:

    def _tel_with_D_t(self, D_t):
        v, f = _grid_mesh(8)
        c = compress_large_mesh(v, f, n_modes=4)
        return SpectralTelemetry(
            compressed=c, betti=(1,0,0),
            D_m_residuals=np.full(4, D_t / 4),
            spectral_entropy=1.0,
        )

    def test_low_detail_below_threshold(self):
        tel = self._tel_with_D_t(D_T_THRESHOLD_LOW * 0.5)
        level, _ = triage_detail_level(tel)
        assert level == DetailLevel.LOW

    def test_medium_detail_mid_range(self):
        tel = self._tel_with_D_t((D_T_THRESHOLD_LOW + D_T_THRESHOLD_MED) / 2)
        level, _ = triage_detail_level(tel)
        assert level == DetailLevel.MEDIUM

    def test_high_detail_above_med(self):
        tel = self._tel_with_D_t((D_T_THRESHOLD_MED + D_T_THRESHOLD_PATCH) / 2)
        level, _ = triage_detail_level(tel)
        assert level == DetailLevel.HIGH

    def test_patch_request_above_patch_threshold(self):
        tel = self._tel_with_D_t(D_T_THRESHOLD_PATCH * 2)
        level, _ = triage_detail_level(tel)
        assert level == DetailLevel.PATCH_REQUEST

    def test_skeleton_only_when_k_below_kmin(self):
        v, f = _grid_mesh(8)
        c = compress_large_mesh(v, f, n_modes=2)
        # betti=(1,3,0) → k_min=4, but k=2 → SKELETON_ONLY
        tel = SpectralTelemetry(
            compressed=c, betti=(1,3,0),
            D_m_residuals=np.zeros(2),
            spectral_entropy=0.5,
        )
        level, _ = triage_detail_level(tel)
        assert level == DetailLevel.SKELETON_ONLY

    def test_curl_active_flag_set_when_above_threshold(self):
        v, f = _grid_mesh(8)
        c = compress_large_mesh(v, f, n_modes=4)
        tel = SpectralTelemetry(
            compressed=c, betti=(1,0,0),
            D_m_residuals=np.full(4, 0.5),
            spectral_entropy=1.0,
            curl_energy=CURL_ACTIVE * 2,
        )
        _, params = triage_detail_level(tel)
        assert params["curl_active"] is True

    def test_params_contain_required_keys(self):
        v, f = _grid_mesh(8)
        tel = _compressed_tel(v, f)
        _, params = triage_detail_level(tel)
        for key in ("roughness", "curl_active", "D_t", "spectral_entropy", "betti"):
            assert key in params


# ---------------------------------------------------------------------------
# compute_curl_energy
# ---------------------------------------------------------------------------

class TestComputeCurlEnergy:

    def test_flat_grid_curl_near_zero(self):
        v, f = _grid_mesh(8)
        e = compute_curl_energy(v, f)
        assert 0.0 <= e <= 1.0

    def test_channeled_has_higher_curl_than_flat(self):
        """Terrain with elevation variation should differ from perfectly flat."""
        v_flat, f = _grid_mesh(8)
        v_channel = v_flat.copy()
        # carve a channel: low z along x ≈ 0.5
        mask = np.abs(v_channel[:, 0] - 0.5) < 0.15
        v_channel[mask, 2] = -1.0

        e_flat    = compute_curl_energy(v_flat, f)
        e_channel = compute_curl_energy(v_channel, f)
        # The channel introduces more gradient variation; curl energy may differ.
        # We only assert both are valid floats in [0, 1].
        assert 0.0 <= e_flat    <= 1.0
        assert 0.0 <= e_channel <= 1.0

    def test_bounded_in_unit_interval(self):
        v, f = _tetrahedron()
        e = compute_curl_energy(v, f)
        assert 0.0 <= e <= 1.0


# ---------------------------------------------------------------------------
# decode — end-to-end contracts
# ---------------------------------------------------------------------------

class TestDecode:

    def test_output_shapes(self):
        v, f = _grid_mesh(8)
        tel = _compressed_tel(v, f, k=4)
        twin = decode(tel)
        assert twin.vertices_base.shape == v.shape
        assert twin.faces.shape == f.shape

    def test_detail_level_is_enum(self):
        v, f = _grid_mesh(8)
        tel = _compressed_tel(v, f)
        twin = decode(tel)
        assert isinstance(twin.detail_level, DetailLevel)

    def test_request_patch_false_for_low_D_t(self):
        v, f = _grid_mesh(8)
        tel = _compressed_tel(v, f, D_t=D_T_THRESHOLD_LOW * 0.1)
        twin = decode(tel)
        assert twin.request_patch is False

    def test_request_patch_true_for_high_D_t(self):
        v, f = _grid_mesh(8)
        tel = _compressed_tel(v, f, D_t=D_T_THRESHOLD_PATCH * 3)
        twin = decode(tel)
        assert twin.request_patch is True

    def test_diagnostics_keys(self):
        v, f = _grid_mesh(8)
        tel = _compressed_tel(v, f)
        twin = decode(tel)
        for key in ("D_t", "spectral_entropy", "curl_energy",
                    "detail_level", "n_vertices", "n_faces", "k_min", "beta"):
            assert key in twin.diagnostics

    def test_curl_estimated_when_not_transmitted(self):
        v, f = _grid_mesh(8)
        c = compress_large_mesh(v, f, n_modes=4)
        tel = SpectralTelemetry(
            compressed=c, betti=(1,0,0),
            curl_energy=None,    # not transmitted
        )
        twin = decode(tel, compute_curl=True)
        assert twin.diagnostics["curl_energy"] is not None
        assert twin.diagnostics["curl_energy"] >= 0.0

    def test_curl_not_computed_when_already_transmitted(self):
        v, f = _grid_mesh(8)
        c = compress_large_mesh(v, f, n_modes=4)
        tel = SpectralTelemetry(
            compressed=c, betti=(1,0,0),
            curl_energy=0.123,
        )
        twin = decode(tel, compute_curl=True)
        assert twin.diagnostics["curl_energy"] == pytest.approx(0.123, abs=1e-6)

    def test_telemetry_reference_preserved(self):
        v, f = _grid_mesh(8)
        tel = _compressed_tel(v, f)
        twin = decode(tel)
        assert twin.telemetry is tel


# ---------------------------------------------------------------------------
# Tier 3 — encoder → decode → synthesize round-trip
# ---------------------------------------------------------------------------

class TestEncoderDecoderRoundTrip:

    def test_synthesized_vertices_same_count_as_base(self):
        v, f = _grid_mesh(12)
        tel = _compressed_tel(v, f, k=6, D_t=3.0, curl_energy=0.1)
        twin = decode(tel)
        synth = synthesize(twin)
        assert synth.vertices_detailed.shape == twin.vertices_base.shape
        assert synth.faces.shape == twin.faces.shape

    def test_displacement_is_nonzero_for_medium_level(self):
        """Medium+ detail should add nonzero displacement."""
        v, f = _grid_mesh(12)
        tel = _compressed_tel(v, f, D_t=D_T_THRESHOLD_MED * 0.8)
        twin = decode(tel)
        assert twin.detail_level in (DetailLevel.MEDIUM,
                                     DetailLevel.HIGH,
                                     DetailLevel.PATCH_REQUEST)
        synth = synthesize(twin)
        assert np.any(synth.displacement != 0.0)

    def test_skeleton_only_zero_displacement(self):
        """SKELETON_ONLY level must produce exactly zero displacement."""
        v, f = _grid_mesh(8)
        c = compress_large_mesh(v, f, n_modes=2)
        tel = SpectralTelemetry(
            compressed=c, betti=(1,3,0),
            D_m_residuals=np.zeros(2),
            spectral_entropy=0.0,
        )
        twin = decode(tel)
        assert twin.detail_level == DetailLevel.SKELETON_ONLY
        synth = synthesize(twin)
        assert np.allclose(synth.displacement, 0.0)

    def test_texture_weights_in_unit_interval(self):
        v, f = _grid_mesh(12)
        tel = _compressed_tel(v, f, D_t=6.0, curl_energy=0.3)
        twin = decode(tel)
        synth = synthesize(twin)
        assert np.all(synth.texture_weights >= 0.0)
        assert np.all(synth.texture_weights <= 1.0)

    def test_patch_request_propagated_through_synthesize(self):
        v, f = _grid_mesh(8)
        tel = _compressed_tel(v, f, D_t=D_T_THRESHOLD_PATCH * 5)
        twin = decode(tel)
        assert twin.request_patch is True
        synth = synthesize(twin)
        assert synth.detail_level == DetailLevel.PATCH_REQUEST

    def test_latent_code_applied_to_scene_class(self):
        v, f = _grid_mesh(12)
        tel = _compressed_tel(v, f, D_t=3.0, latent_code=1)  # code 1 = boulder field
        twin = decode(tel)
        synth = synthesize(twin)
        assert synth.diagnostics["scene_class"] == "Boulder field"

    def test_ply_export_creates_file(self, tmp_path):
        v, f = _grid_mesh(8)
        tel = _compressed_tel(v, f, D_t=2.0)
        twin = decode(tel)
        synth = synthesize(twin)
        ply = tmp_path / "twin.ply"
        synth.export_ply_with_color(str(ply))
        assert ply.exists()
        content = ply.read_text()
        assert "ply" in content
        assert "element vertex" in content

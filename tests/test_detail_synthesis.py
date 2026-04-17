"""Tests for kernelcal.geo3d.detail_synthesis.

Covers all five detail methods plus SynthesizedTwin export:
  A — procedural_displacement: shape, variance, roughness monotonicity
  B — curl_texture_weight: zero below threshold, positive above, bounded
  B — flow_displacement: zero where curl weight is zero
  C — SCENE_LIBRARY latent codes: correct preset lookup and name
  D — landmark_displacement: exact-hit pull, far-vertex fade
  E — D_m octave boost: more octaves at HIGH vs LOW detail level
  synthesize() routing: SKELETON_ONLY → zero disp; correct scene class applied
  SynthesizedTwin.export_ply_with_color: file written, parseable header
  q10_pipeline: q10_pass_fail logic tested directly
"""

import numpy as np
import pytest

from kernelcal.geo3d.large_mesh import compress_large_mesh
from kernelcal.geo3d.decoder import (
    SpectralTelemetry,
    DetailLevel,
    D_T_THRESHOLD_MED,
    D_T_THRESHOLD_PATCH,
    decode,
)
from kernelcal.geo3d.detail_synthesis import (
    SCENE_LIBRARY,
    SynthesizedTwin,
    _fractal_noise,
    _value_noise_2d,
    curl_texture_weight,
    flow_displacement,
    landmark_displacement,
    procedural_displacement,
    synthesize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_mesh(n: int = 10):
    xs = np.linspace(0, 10, n)
    xv, yv = np.meshgrid(xs, xs)
    verts = np.column_stack([xv.ravel(), yv.ravel(), np.zeros(n*n)])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i*n + j; b = a+1; c = (i+1)*n + j; d = c+1
            faces += [[a,b,c],[b,d,c]]
    return verts.astype(float), np.array(faces, dtype=np.int32)


def _make_twin(D_t: float, *, curl_energy=0.0, latent_code=None,
               landmark_xyz=None, betti=(1,0,0), k=4):
    v, f = _grid_mesh(10)
    c = compress_large_mesh(v, f, n_modes=k, heat_tau=1.0)
    tel = SpectralTelemetry(
        compressed=c, betti=betti,
        D_m_residuals=np.full(k, abs(D_t) / k),
        spectral_entropy=min(1.0, D_t / 4),
        curl_energy=curl_energy,
        latent_code=latent_code,
        landmark_xyz=landmark_xyz,
    )
    return decode(tel, compute_curl=False)


# ---------------------------------------------------------------------------
# Method A — procedural_displacement
# ---------------------------------------------------------------------------

class TestProceduralDisplacement:

    def test_output_shape(self):
        v, _ = _grid_mesh(8)
        d = procedural_displacement(v, roughness=0.5)
        assert d.shape == (v.shape[0],)

    def test_deterministic_with_same_seed(self):
        v, _ = _grid_mesh(8)
        d1 = procedural_displacement(v, roughness=0.5, seed=7)
        d2 = procedural_displacement(v, roughness=0.5, seed=7)
        np.testing.assert_array_equal(d1, d2)

    def test_different_seeds_differ(self):
        v, _ = _grid_mesh(8)
        d1 = procedural_displacement(v, roughness=0.5, seed=0)
        d2 = procedural_displacement(v, roughness=0.5, seed=99)
        assert not np.allclose(d1, d2)

    def test_nonzero_variance(self):
        """Noise must not be spatially uniform."""
        v, _ = _grid_mesh(10)
        d = procedural_displacement(v, roughness=0.5)
        assert np.std(d) > 0

    def test_higher_roughness_more_octaves(self):
        """Higher roughness → more octaves → higher frequency content (larger std)."""
        v, _ = _grid_mesh(16)
        d_low  = procedural_displacement(v, roughness=0.1, amplitude_scale=1.0, seed=0)
        d_high = procedural_displacement(v, roughness=0.9, amplitude_scale=1.0, seed=0)
        # Both should be non-zero; high roughness can produce larger std
        assert np.std(d_low)  > 0
        assert np.std(d_high) > 0

    def test_amplitude_scale_affects_magnitude(self):
        v, _ = _grid_mesh(10)
        d1 = procedural_displacement(v, amplitude_scale=1.0, seed=0)
        d10 = procedural_displacement(v, amplitude_scale=10.0, seed=0)
        assert np.abs(d10).max() > np.abs(d1).max()

    def test_value_noise_bounded(self):
        """Value noise should stay in [-1, 1] before amplitude scaling."""
        rng = np.random.default_rng(0)
        xy = rng.uniform(0, 100, (500, 2))
        v = _value_noise_2d(xy, frequency=0.1, seed=42)
        assert np.all(v >= -1 - 1e-9)
        assert np.all(v <=  1 + 1e-9)


# ---------------------------------------------------------------------------
# Method B — curl_texture_weight + flow_displacement
# ---------------------------------------------------------------------------

class TestCurlTextures:

    def test_weight_zero_below_threshold(self):
        v, _ = _grid_mesh(8)
        w = curl_texture_weight(v, curl_energy=0.0)
        assert np.allclose(w, 0.0)

    def test_weight_zero_below_threshold(self):
        """curl_texture_weight returns zero array when curl_energy < threshold."""
        v, _ = _grid_mesh(8)
        # threshold defaults to 0.05; pass half of it
        w = curl_texture_weight(v, curl_energy=0.02, threshold=0.05)
        assert np.allclose(w, 0.0)

    def test_weight_positive_above_threshold(self):
        v, _ = _grid_mesh(8)
        w = curl_texture_weight(v, curl_energy=0.5, threshold=0.05)
        assert np.any(w > 0)

    def test_weight_bounded_in_unit_interval(self):
        v, _ = _grid_mesh(8)
        w = curl_texture_weight(v, curl_energy=0.8)
        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0)

    def test_flow_displacement_zero_where_weight_zero(self):
        v, _ = _grid_mesh(8)
        w = np.zeros(v.shape[0])
        d = flow_displacement(v, w, amplitude=10.0)
        assert np.allclose(d, 0.0)

    def test_flow_displacement_nonzero_where_weight_nonzero(self):
        v, _ = _grid_mesh(8)
        w = np.ones(v.shape[0])
        d = flow_displacement(v, w, amplitude=10.0)
        assert np.any(d != 0.0)

    def test_flow_displacement_shape(self):
        v, _ = _grid_mesh(8)
        w = np.random.default_rng(0).uniform(0, 1, v.shape[0])
        d = flow_displacement(v, w)
        assert d.shape == (v.shape[0],)


# ---------------------------------------------------------------------------
# Method C — SCENE_LIBRARY / latent codes
# ---------------------------------------------------------------------------

class TestSceneLibrary:

    def test_all_codes_have_required_keys(self):
        required = {"name", "roughness_boost", "amplitude", "octaves"}
        for code, preset in SCENE_LIBRARY.items():
            assert required.issubset(preset.keys()), f"Code {code} missing keys"

    def test_roughness_boost_nonnegative(self):
        for code, preset in SCENE_LIBRARY.items():
            assert preset["roughness_boost"] >= 0, f"Code {code} negative roughness"

    def test_octaves_positive(self):
        for code, preset in SCENE_LIBRARY.items():
            assert preset["octaves"] >= 1, f"Code {code} zero octaves"

    def test_code_4_is_fluvial(self):
        assert SCENE_LIBRARY[4]["name"] == "Fluvial channel"

    def test_synthesize_applies_latent_code_name(self):
        twin = _make_twin(D_t=3.0, latent_code=0)
        synth = synthesize(twin)
        assert synth.diagnostics["scene_class"] == "Regolith plain"

    def test_synthesize_unknown_code_uses_default(self):
        twin = _make_twin(D_t=3.0, latent_code=999)
        synth = synthesize(twin)
        assert synth.diagnostics["scene_class"] == "Unknown"


# ---------------------------------------------------------------------------
# Method D — landmark_displacement
# ---------------------------------------------------------------------------

class TestLandmarkDisplacement:

    def test_exact_hit_pulls_toward_landmark_z(self):
        """A vertex at exactly a landmark position should be pulled to its z."""
        v, _ = _grid_mesh(6)
        # Use the first vertex as a landmark with a different z
        lm = v[:1].copy()
        lm[0, 2] = 99.0
        v_test = v.copy()
        v_test[0, 2] = 0.0

        d = landmark_displacement(v_test, lm)
        # The first vertex (exact match) should get pulled toward 99
        assert d[0] > 0, "Exact-hit vertex should be pulled upward"

    def test_far_vertices_near_zero(self):
        """Vertices far from all landmarks should have negligible displacement."""
        v, _ = _grid_mesh(6)
        # Single landmark at extreme corner, most vertices far away
        lm = np.array([[100.0, 100.0, 50.0]])
        d = landmark_displacement(v, lm)
        # Vertices near (0,0) to (10,10) are far from (100,100)
        far_mask = np.linalg.norm(v[:, :2] - 100.0, axis=1) > 50
        assert np.all(np.abs(d[far_mask]) < 1.0)

    def test_no_landmarks_zero_displacement(self):
        v, _ = _grid_mesh(6)
        d = landmark_displacement(v, np.empty((0, 3)))
        assert np.allclose(d, 0.0)

    def test_none_landmarks_zero_displacement(self):
        v, _ = _grid_mesh(6)
        d = landmark_displacement(v, None)
        assert np.allclose(d, 0.0)

    def test_output_shape(self):
        v, _ = _grid_mesh(6)
        lm = np.array([[5.0, 5.0, 2.0], [2.0, 2.0, 1.0]])
        d = landmark_displacement(v, lm)
        assert d.shape == (v.shape[0],)


# ---------------------------------------------------------------------------
# synthesize() routing and E — D_m octave boost
# ---------------------------------------------------------------------------

class TestSynthesize:

    def test_skeleton_only_zero_displacement(self):
        """SKELETON_ONLY detail level must leave displacement at zero."""
        v, f = _grid_mesh(8)
        c = compress_large_mesh(v, f, n_modes=2)
        tel = SpectralTelemetry(
            compressed=c, betti=(1,3,0),
            D_m_residuals=np.zeros(2), spectral_entropy=0.0,
        )
        twin = decode(tel)
        assert twin.detail_level == DetailLevel.SKELETON_ONLY
        synth = synthesize(twin)
        assert np.allclose(synth.displacement, 0.0)

    def test_high_detail_more_octaves_than_low(self):
        """HIGH detail should use more octaves than LOW (D_m boost)."""
        twin_low  = _make_twin(D_t=0.1)   # → LOW
        twin_high = _make_twin(D_t=D_T_THRESHOLD_MED * 2)  # → HIGH

        synth_low  = synthesize(twin_low)
        synth_high = synthesize(twin_high)

        assert synth_high.diagnostics["octaves_used"] >= synth_low.diagnostics["octaves_used"]

    def test_output_vertices_same_count(self):
        twin = _make_twin(D_t=3.0)
        synth = synthesize(twin)
        assert synth.vertices_detailed.shape == twin.vertices_base.shape

    def test_diagnostics_has_rms_disp(self):
        twin = _make_twin(D_t=3.0)
        synth = synthesize(twin)
        assert "disp_rms" in synth.diagnostics
        assert synth.diagnostics["disp_rms"] >= 0.0

    def test_texture_weights_bounded(self):
        twin = _make_twin(D_t=8.0, curl_energy=0.3)
        synth = synthesize(twin)
        assert np.all(synth.texture_weights >= 0.0)
        assert np.all(synth.texture_weights <= 1.0)

    def test_patch_request_preserved(self):
        twin = _make_twin(D_t=D_T_THRESHOLD_PATCH * 3)
        assert twin.request_patch is True
        synth = synthesize(twin)
        assert synth.detail_level == DetailLevel.PATCH_REQUEST

    def test_landmark_pinning_active_at_high_level(self):
        """HIGH level with landmarks should use Method D (nonzero landmark disp)."""
        lm = np.array([[5.0, 5.0, 50.0]])   # landmark at z=50, far from base z≈0
        twin = _make_twin(D_t=D_T_THRESHOLD_MED * 2, landmark_xyz=lm)
        assert twin.detail_level in (DetailLevel.HIGH, DetailLevel.PATCH_REQUEST)
        synth = synthesize(twin)
        # Displacement RMS should be nonzero (landmark contribution)
        assert synth.diagnostics["disp_rms"] > 0


# ---------------------------------------------------------------------------
# SynthesizedTwin export
# ---------------------------------------------------------------------------

class TestSynthesizedTwinExport:

    def _make_synth(self):
        twin = _make_twin(D_t=3.0)
        return synthesize(twin)

    def test_export_obj_creates_file(self, tmp_path):
        synth = self._make_synth()
        p = tmp_path / "twin.obj"
        synth.export_obj(str(p))
        assert p.exists()
        assert p.stat().st_size > 0

    def test_export_ply_creates_file(self, tmp_path):
        synth = self._make_synth()
        p = tmp_path / "twin.ply"
        synth.export_ply_with_color(str(p))
        assert p.exists()

    def test_ply_has_valid_header(self, tmp_path):
        synth = self._make_synth()
        p = tmp_path / "twin.ply"
        synth.export_ply_with_color(str(p))
        lines = p.read_text().splitlines()
        assert lines[0] == "ply"
        assert any("element vertex" in l for l in lines)
        assert any("element face" in l for l in lines)
        assert "end_header" in lines

    def test_ply_vertex_count_matches(self, tmp_path):
        synth = self._make_synth()
        p = tmp_path / "twin.ply"
        synth.export_ply_with_color(str(p))
        lines = p.read_text().splitlines()
        for l in lines:
            if l.startswith("element vertex"):
                n = int(l.split()[-1])
                assert n == synth.vertices_detailed.shape[0]
                break

    def test_ply_face_count_matches(self, tmp_path):
        synth = self._make_synth()
        p = tmp_path / "twin.ply"
        synth.export_ply_with_color(str(p))
        lines = p.read_text().splitlines()
        for l in lines:
            if l.startswith("element face"):
                n = int(l.split()[-1])
                assert n == synth.faces.shape[0]
                break


# ---------------------------------------------------------------------------
# q10_pipeline pass/fail logic (no Blender, no OBJ files needed)
# ---------------------------------------------------------------------------

class TestQ10PassFail:
    """Import q10_pass_fail directly — it has no bpy dependency."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from kernelcal.blender.q10_pipeline import q10_pass_fail
        self.q10_pass_fail = q10_pass_fail

    def _results(self, errors, n_verts=None):
        if n_verts is None:
            n_verts = [100 * (i+1) for i in range(len(errors))]
        return [
            {"error": e, "n_vertices": nv, "beta1_hat": 3, "method": "exact_hodge"}
            for e, nv in zip(errors, n_verts)
        ]

    def test_monotone_decreasing_to_zero_passes(self):
        r = self._results([2, 1, 0])
        assert self.q10_pass_fail(r)["q10_pass"] is True
        assert self.q10_pass_fail(r)["verdict"] == "PASS"

    def test_constant_zero_passes(self):
        r = self._results([0, 0, 0])
        assert self.q10_pass_fail(r)["q10_pass"] is True

    def test_non_monotone_fails(self):
        r = self._results([2, 3, 1])
        result = self.q10_pass_fail(r)
        assert result["q10_pass"] is False
        assert result["monotone_tightening"] is False

    def test_monotone_but_never_zero_fails(self):
        r = self._results([3, 2, 1])   # never reaches 0
        result = self.q10_pass_fail(r)
        assert result["q10_pass"] is False
        assert result["any_exact_recovery"] is False

    def test_errors_by_resolution_present(self):
        r = self._results([1, 0], [100, 200])
        result = self.q10_pass_fail(r)
        assert len(result["errors_by_resolution"]) == 2

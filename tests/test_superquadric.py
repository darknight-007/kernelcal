"""Tests for ``kernelcal.distinction_game.geometry`` (PR-5).

Coverage:

* Math sanity: inside-outside, signed distance, volume, bbox, tessellation.
* Factory constructors (box, cylinder, ellipsoid, sphere) and their limits.
* Pose round-trip (``to_dict``/``from_dict``, ``transformed``).
* Quantized binary codec round-trip.
* SQ fitting recovers ellipsoid / cuboid / cylinder limits (shape-invariant
  checks: surface residual, volume, sorted-scale).
* Robustness to noise + outliers via the Cauchy loss path.
* Tree fit splits a synthetic tree into trunk + crown with parent_id link.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kernelcal.distinction_game.geometry import (
    EPS_MAX,
    EPS_MIN,
    PACKED_BYTES,
    Superquadric,
    fit_superquadric,
    fit_tree,
    pack_superquadric,
    superquadric_box,
    superquadric_cylinder,
    superquadric_ellipsoid,
    superquadric_sphere,
    unpack_superquadric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _surface_points(sq: Superquadric, n_lat: int = 32, n_lon: int = 64) -> np.ndarray:
    return sq.surface_points(n_lat=n_lat, n_lon=n_lon)


def _shape_match_residual(fitted: Superquadric, true_sq: Superquadric) -> float:
    """Mean ``|F_fitted^eps1 - 1|`` evaluated on points sampled from ``true_sq``.

    Shape-invariant: doesn't care about (a, eps, R) parameter
    permutations as long as the fitted surface coincides with the true
    surface.
    """
    true_pts = _surface_points(true_sq, n_lat=32, n_lon=64)
    F = fitted.inside_outside(true_pts)
    F_eps1 = np.power(np.maximum(F, 1e-12), float(fitted.epsilon[0]))
    return float(np.mean(np.abs(F_eps1 - 1.0)))


# ---------------------------------------------------------------------------
# Math sanity
# ---------------------------------------------------------------------------


class TestSuperquadricMath:
    def test_unit_sphere_F_is_unit_on_axes(self):
        sq = superquadric_sphere(center=[0, 0, 0], radius=1.0)
        pts = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0]])
        F = sq.inside_outside(pts)
        assert np.allclose(F, 1.0, atol=1e-9)

    def test_unit_sphere_F_is_quarter_at_half_radius(self):
        sq = superquadric_sphere(center=[0, 0, 0], radius=1.0)
        F = sq.inside_outside(np.array([[0.5, 0, 0]]))
        assert F[0] == pytest.approx(0.25, abs=1e-9)

    def test_unit_sphere_volume_matches_analytic(self):
        sq = superquadric_sphere(center=[0, 0, 0], radius=1.0)
        assert sq.volume() == pytest.approx(4.0 / 3.0 * math.pi, rel=1e-6)

    def test_ellipsoid_volume_matches_analytic(self):
        sq = superquadric_ellipsoid(center=[0, 0, 0], axes=[2.0, 1.5, 1.0])
        analytic = 4.0 / 3.0 * math.pi * 2.0 * 1.5 * 1.0
        assert sq.volume() == pytest.approx(analytic, rel=1e-6)

    def test_signed_distance_sphere(self):
        sq = superquadric_sphere(center=[0, 0, 0], radius=1.0)
        # Along an axis, signed distance ~ |p| - 1.
        pts = np.array([[2.0, 0, 0], [0.5, 0, 0]])
        d = sq.signed_distance(pts)
        assert d[0] == pytest.approx(1.0, abs=1e-6)
        assert d[1] == pytest.approx(-0.5, abs=1e-6)

    def test_contains_returns_bool_mask(self):
        sq = superquadric_sphere(center=[0, 0, 0], radius=1.0)
        pts = np.array([[0.5, 0, 0], [2.0, 0, 0]])
        mask = sq.contains(pts)
        assert mask.tolist() == [True, False]

    def test_bbox_3d_axis_aligned_box(self):
        sq = superquadric_box(center=[1, 2, 3], size=[4, 2, 6])
        lo, hi = sq.bbox_3d()
        np.testing.assert_allclose(lo, [-1.0, 1.0, 0.0])
        np.testing.assert_allclose(hi, [3.0, 3.0, 6.0])

    def test_tessellate_returns_closed_mesh(self):
        sq = superquadric_ellipsoid(center=[0, 0, 0], axes=[2.0, 1.5, 1.0])
        verts, faces = sq.tessellate(n_lat=12, n_lon=24)
        # Two triangles per quad on a (n_lat-1) x n_lon grid:
        assert faces.shape[0] == (12 - 1) * 24 * 2
        assert verts.shape[1] == 3
        assert np.isfinite(verts).all()
        # Vertices should sit inside the bbox, with epsilon for fp drift.
        assert np.all(np.abs(verts) <= np.array([2.0, 1.5, 1.0]) + 1e-6)

    def test_tessellate_handles_extreme_eps(self):
        sq = superquadric_box(center=[0, 0, 0], size=[2, 2, 2], eps=EPS_MIN)
        verts, faces = sq.tessellate(n_lat=12, n_lon=24)
        assert np.isfinite(verts).all()
        # All cuboid corners should appear within numerical tolerance.
        max_coord = float(np.max(np.abs(verts)))
        assert 0.95 <= max_coord <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Factory constructors
# ---------------------------------------------------------------------------


class TestFactories:
    def test_box_size_to_semi_axes(self):
        sq = superquadric_box(center=[0, 0, 0], size=[4, 6, 2])
        np.testing.assert_allclose(sq.scale, [2.0, 3.0, 1.0])

    def test_cylinder_axis_alignment_z(self):
        sq = superquadric_cylinder(
            base=[0, 0, 0], axis=[0, 0, 1], radius=0.5, height=4.0
        )
        # body-z column of R should equal +z axis.
        np.testing.assert_allclose(sq.R[:, 2], [0, 0, 1], atol=1e-9)

    def test_cylinder_axis_alignment_arbitrary(self):
        axis = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
        sq = superquadric_cylinder(base=[0, 0, 0], axis=axis, radius=0.5, height=2.0)
        np.testing.assert_allclose(sq.R[:, 2], axis, atol=1e-9)

    def test_ellipsoid_eps_is_one(self):
        sq = superquadric_ellipsoid(center=[0, 0, 0], axes=[1, 2, 3])
        np.testing.assert_allclose(sq.epsilon, [1.0, 1.0])

    def test_sphere_is_uniform(self):
        sq = superquadric_sphere(center=[0, 0, 0], radius=1.5)
        np.testing.assert_allclose(sq.scale, [1.5, 1.5, 1.5])
        np.testing.assert_allclose(sq.epsilon, [1.0, 1.0])

    def test_invalid_scale_rejected(self):
        with pytest.raises(ValueError):
            Superquadric(scale=[0.0, 1.0, 1.0], epsilon=[1, 1])

    def test_invalid_epsilon_rejected(self):
        with pytest.raises(ValueError):
            Superquadric(scale=[1, 1, 1], epsilon=[0.0, 1.0])
        with pytest.raises(ValueError):
            Superquadric(scale=[1, 1, 1], epsilon=[5.0, 1.0])


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_from_dict_round_trip(self):
        original = superquadric_ellipsoid(
            center=[3, 1, 2],
            axes=[2.0, 1.5, 1.0],
            attributes={"class": "tree_crown", "p": 0.91},
        )
        d = original.to_dict()
        recovered = Superquadric.from_dict(d)
        np.testing.assert_allclose(recovered.scale, original.scale)
        np.testing.assert_allclose(recovered.epsilon, original.epsilon)
        np.testing.assert_allclose(recovered.R, original.R)
        np.testing.assert_allclose(recovered.t, original.t)
        assert recovered.attributes == original.attributes

    def test_packed_size_is_advertised(self):
        sq = superquadric_ellipsoid(center=[0, 0, 0], axes=[1, 1, 1])
        buf = pack_superquadric(sq, class_idx=0)
        assert len(buf) == PACKED_BYTES

    def test_codec_round_trip_ellipsoid(self):
        original = superquadric_ellipsoid(
            center=[12.345, -5.678, 2.1],
            axes=[2.0, 1.5, 1.0],
        )
        buf = pack_superquadric(original, class_idx=42)
        recovered, meta = unpack_superquadric(buf)
        # Quantization budget: ~0.1% on scale, ~2% on eps, ~1mm on t.
        np.testing.assert_allclose(recovered.scale, original.scale, rtol=2e-3)
        np.testing.assert_allclose(recovered.epsilon, original.epsilon, atol=0.02)
        np.testing.assert_allclose(recovered.t, original.t, atol=2e-3)
        assert meta["class_idx"] == 42
        assert meta["parent_hash"] is None

    def test_codec_round_trip_cylinder_with_rotation(self):
        original = superquadric_cylinder(
            base=[10.0, 5.0, 0.0],
            axis=[1, 1, 1],
            radius=0.3,
            height=4.0,
        )
        buf = pack_superquadric(original, class_idx=7)
        recovered, _ = unpack_superquadric(buf)
        # Verify shape coincidence on a sample of true surface points.
        residual = _shape_match_residual(recovered, original)
        assert residual < 0.05, f"shape residual {residual:.3f} too high"

    def test_codec_with_parent_hash(self):
        sq = superquadric_sphere(center=[0, 0, 0], radius=1.0)
        ph = 0x1234_5678_9ABC_DEF0
        buf = pack_superquadric(sq, parent_hash=ph)
        assert len(buf) == PACKED_BYTES + 8
        _, meta = unpack_superquadric(buf)
        assert meta["parent_hash"] == ph

    def test_codec_translation_extreme_range(self):
        # 1 km offset still round-trips cleanly at 1 mm resolution.
        original = superquadric_box(center=[1234.567, -987.654, 50.5], size=[2, 2, 2])
        buf = pack_superquadric(original)
        recovered, _ = unpack_superquadric(buf)
        np.testing.assert_allclose(recovered.t, original.t, atol=2e-3)


# ---------------------------------------------------------------------------
# Pose updates
# ---------------------------------------------------------------------------


class TestPoseUpdates:
    def test_transformed_composition(self):
        sq = superquadric_box(center=[1, 0, 0], size=[2, 2, 2])
        # 90 deg rotation about world-z, then translate by (3, 0, 0).
        c, s = math.cos(math.pi / 2.0), math.sin(math.pi / 2.0)
        R_world = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        t_world = np.array([3.0, 0.0, 0.0])
        new = sq.transformed(R_world, t_world)
        # Original center (1, 0, 0) -> R(1, 0, 0) + (3, 0, 0) = (3 + 0, 0 + 1, 0)
        np.testing.assert_allclose(new.t, [3.0, 1.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Fit recovery
# ---------------------------------------------------------------------------


class TestFitRecovery:
    def test_fit_recovers_ellipsoid(self):
        rng = np.random.default_rng(42)
        true_sq = superquadric_ellipsoid(center=[3, 1, 2], axes=[2.0, 1.5, 1.0])
        pts = _surface_points(true_sq, n_lat=24, n_lon=48)
        pts += rng.normal(0, 0.02, pts.shape)
        fit = fit_superquadric(pts, robust=True)
        residual = _shape_match_residual(fit.superquadric, true_sq)
        assert residual < 0.05, f"surface residual {residual:.3f} too high"
        # Sorted scale should match.
        np.testing.assert_allclose(
            np.sort(fit.superquadric.scale), np.sort(true_sq.scale), rtol=0.05
        )
        # Eps should be close to (1, 1).
        np.testing.assert_allclose(fit.superquadric.epsilon, [1.0, 1.0], atol=0.1)
        # Center should be close.
        np.testing.assert_allclose(fit.superquadric.t, true_sq.t, atol=0.05)
        assert fit.diagnostics.converged
        assert fit.diagnostics.rms_residual < 0.5

    def test_fit_recovers_cuboid_limit(self):
        rng = np.random.default_rng(0)
        true_box = superquadric_box(center=[0, 0, 0], size=[4, 2, 6], eps=0.1)
        pts = _surface_points(true_box, n_lat=24, n_lon=48)
        pts += rng.normal(0, 0.02, pts.shape)
        fit = fit_superquadric(pts, robust=True)
        residual = _shape_match_residual(fit.superquadric, true_box)
        assert residual < 0.10
        # Sorted scale equals sorted true.
        np.testing.assert_allclose(
            np.sort(fit.superquadric.scale), np.sort(true_box.scale), rtol=0.05
        )
        # Both epsilons should be small (cuboid-ish).
        assert (fit.superquadric.epsilon < 0.3).all()

    def test_fit_recovers_volume(self):
        rng = np.random.default_rng(7)
        true_sq = superquadric_ellipsoid(center=[0, 0, 0], axes=[3.0, 2.0, 1.0])
        pts = _surface_points(true_sq, n_lat=24, n_lon=48)
        pts += rng.normal(0, 0.01, pts.shape)
        fit = fit_superquadric(pts, robust=True)
        assert fit.superquadric.volume() == pytest.approx(true_sq.volume(), rel=0.1)

    def test_fit_with_outliers(self):
        rng = np.random.default_rng(13)
        true_sq = superquadric_ellipsoid(center=[0, 0, 0], axes=[2.0, 1.0, 1.5])
        pts = _surface_points(true_sq, n_lat=24, n_lon=48)
        pts += rng.normal(0, 0.02, pts.shape)
        # Inject 10% outliers far from the surface.
        n_out = int(0.1 * pts.shape[0])
        outliers = rng.uniform(low=-10, high=10, size=(n_out, 3))
        pts_with_outliers = np.vstack([pts, outliers])
        fit_robust = fit_superquadric(pts_with_outliers, robust=True)
        residual = _shape_match_residual(fit_robust.superquadric, true_sq)
        # Robust fit should reject outliers and still recover surface.
        assert residual < 0.20, f"robust fit failed with outliers: {residual:.3f}"

    def test_fit_returns_covariance(self):
        rng = np.random.default_rng(99)
        true_sq = superquadric_sphere(center=[0, 0, 0], radius=1.5)
        pts = _surface_points(true_sq, n_lat=24, n_lon=48)
        pts += rng.normal(0, 0.02, pts.shape)
        fit = fit_superquadric(pts, robust=True, return_covariance=True)
        cov = fit.superquadric.covariance
        assert cov is not None
        assert cov.shape == (11, 11)
        # Diagonal should be positive.
        diag = np.diag(cov)
        assert np.all(diag >= -1e-9)

    def test_fit_too_few_points_raises(self):
        with pytest.raises(ValueError):
            fit_superquadric(np.zeros((5, 3)))

    def test_fit_with_init_uses_it(self):
        rng = np.random.default_rng(2)
        true_sq = superquadric_ellipsoid(center=[5, 0, 0], axes=[2.0, 2.0, 2.0])
        pts = _surface_points(true_sq, n_lat=20, n_lon=40)
        pts += rng.normal(0, 0.01, pts.shape)
        # Init far from truth, but same shape — should still converge.
        init = superquadric_ellipsoid(center=[4.5, 0.5, -0.5], axes=[1.5, 1.5, 1.5])
        fit = fit_superquadric(pts, init=init, robust=True)
        residual = _shape_match_residual(fit.superquadric, true_sq)
        assert residual < 0.10


# ---------------------------------------------------------------------------
# Tree fit
# ---------------------------------------------------------------------------


class TestTreeFit:
    def _synthetic_tree_points(self, rng: np.random.Generator) -> np.ndarray:
        """Trunk + ellipsoidal crown synthetic tree."""
        # Trunk: vertical cylinder, radius 0.15, height 3 m, base at origin.
        trunk_template = superquadric_cylinder(
            base=[0, 0, 0], axis=[0, 0, 1], radius=0.15, height=3.0
        )
        trunk_pts = _surface_points(trunk_template, n_lat=12, n_lon=32)
        # Crown: ellipsoid centered at (0, 0, 4), axes (1.5, 1.5, 1.0).
        crown_template = superquadric_ellipsoid(
            center=[0, 0, 4.0], axes=[1.5, 1.5, 1.0]
        )
        crown_pts = _surface_points(crown_template, n_lat=16, n_lon=32)
        pts = np.vstack([trunk_pts, crown_pts])
        pts += rng.normal(0, 0.02, pts.shape)
        return pts, trunk_template, crown_template

    def test_fit_tree_separates_trunk_and_crown(self):
        rng = np.random.default_rng(31)
        pts, trunk_template, crown_template = self._synthetic_tree_points(rng)
        result = fit_tree(pts)
        # Crown has parent_id linking to trunk.
        assert result.crown.parent_id == result.trunk.id
        # Trunk should be near the bottom; crown near the top in z.
        assert result.trunk.t[2] < result.crown.t[2]
        # Trunk attributes should mark it as a trunk.
        assert result.trunk.attributes.get("part") == "trunk"
        assert result.crown.attributes.get("part") == "crown"

    def test_fit_tree_recovers_approximate_geometry(self):
        rng = np.random.default_rng(101)
        pts, trunk_template, crown_template = self._synthetic_tree_points(rng)
        result = fit_tree(pts)
        # Crown bbox should overlap the true crown extent significantly.
        crown_lo, crown_hi = result.crown.bbox_3d()
        true_lo, true_hi = crown_template.bbox_3d()
        # Allow some looseness; trunk-extraction biased the crown points.
        # We only require the recovered crown's centre to lie within
        # the true crown's bbox-padded by 0.5 m.
        assert np.all(result.crown.t >= true_lo - 0.5)
        assert np.all(result.crown.t <= true_hi + 0.5)

    def test_fit_tree_too_few_points_raises(self):
        with pytest.raises(ValueError):
            fit_tree(np.zeros((10, 3)))


# ---------------------------------------------------------------------------
# Distinction-game integration: top-level imports
# ---------------------------------------------------------------------------


class TestDistinctionGameIntegration:
    def test_top_level_imports(self):
        from kernelcal.distinction_game import (
            EPS_MAX,
            EPS_MIN,
            PACKED_BYTES,
            Superquadric,
            fit_superquadric,
            fit_tree,
            pack_superquadric,
            superquadric_box,
            superquadric_cylinder,
            superquadric_ellipsoid,
            superquadric_sphere,
            unpack_superquadric,
        )
        assert EPS_MIN > 0
        assert EPS_MAX > EPS_MIN
        assert PACKED_BYTES == 32
        # Quick smoke: build a primitive via the top-level surface.
        sq = superquadric_box(center=[0, 0, 0], size=[2, 2, 2])
        assert isinstance(sq, Superquadric)

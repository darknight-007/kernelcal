"""Tests for ``kernelcal.distinction_game.geometry.frames``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kernelcal.distinction_game.geometry import (
    FrameSpec,
    Superquadric,
    ecef_to_enu,
    ecef_to_geodetic,
    enu_basis_at,
    enu_to_ecef,
    geodetic_to_ecef,
    geodetic_to_utm,
    ned_to_enu,
    superquadric_box,
    superquadric_cylinder,
    superquadric_ellipsoid,
    transform_point,
    transform_pose,
    transform_superquadric,
    utm_to_geodetic,
    utm_zone_for,
)
from kernelcal.distinction_game.geometry.frames import (
    WGS84_A,
    WGS84_E2,
)

try:  # pragma: no cover
    import pyproj  # type: ignore  # noqa: F401

    _HAVE_PYPROJ = True
except Exception:  # pragma: no cover
    _HAVE_PYPROJ = False


# ---------------------------------------------------------------------------
# FrameSpec basics
# ---------------------------------------------------------------------------


class TestFrameSpec:
    def test_kind_validation(self):
        with pytest.raises(ValueError):
            FrameSpec(kind="bogus")

    def test_utm_validation(self):
        with pytest.raises(ValueError):
            FrameSpec.utm(zone=0)
        with pytest.raises(ValueError):
            FrameSpec.utm(zone=61)
        with pytest.raises(ValueError):
            FrameSpec.utm(zone=12, hemisphere="X")
        f = FrameSpec.utm(zone=12, hemisphere="N", name="map")
        assert f.params["zone"] == 12
        assert f.name == "map"

    def test_enu_local_validation(self):
        with pytest.raises(ValueError):
            FrameSpec.enu_local(origin_lla=(0.0, 0.0))  # type: ignore[arg-type]
        f = FrameSpec.enu_local(origin_lla=(33.42, -111.94, 350.0))
        assert f.params["origin_lla"] == (33.42, -111.94, 350.0)

    def test_round_trip_dict(self):
        f = FrameSpec.utm(zone=12, hemisphere="N", name="map")
        d = f.to_dict()
        f2 = FrameSpec.from_dict(d)
        assert f == f2

    def test_equality(self):
        a = FrameSpec.ecef()
        b = FrameSpec.ecef()
        assert a == b
        c = FrameSpec.utm(zone=12)
        assert a != c


# ---------------------------------------------------------------------------
# Geodetic <-> ECEF
# ---------------------------------------------------------------------------


class TestGeodeticEcef:
    def test_origin_round_trip(self):
        # Equator + prime meridian on the surface = (a, 0, 0).
        ecef = geodetic_to_ecef(np.array([0.0, 0.0, 0.0]))
        assert ecef[0] == pytest.approx(WGS84_A, abs=1e-6)
        assert ecef[1] == pytest.approx(0.0, abs=1e-9)
        assert ecef[2] == pytest.approx(0.0, abs=1e-9)
        lla = ecef_to_geodetic(ecef)
        assert lla[0] == pytest.approx(0.0, abs=1e-9)
        assert lla[1] == pytest.approx(0.0, abs=1e-9)
        assert lla[2] == pytest.approx(0.0, abs=1e-6)

    def test_pole_round_trip(self):
        # North pole, on the surface: Z = b = a*(1-f) ~= 6356752.3 m.
        ecef = geodetic_to_ecef(np.array([90.0, 0.0, 0.0]))
        assert abs(ecef[0]) < 1e-6
        assert abs(ecef[1]) < 1e-6
        # Z = a/sqrt(1-e^2) * (1-e^2) = a*sqrt(1-e^2) = b.
        b = WGS84_A * math.sqrt(1.0 - WGS84_E2)
        assert ecef[2] == pytest.approx(b, abs=1e-6)
        lla = ecef_to_geodetic(ecef)
        assert lla[0] == pytest.approx(90.0, abs=1e-6)

    def test_phoenix_round_trip(self):
        # Tempe, AZ at ground level.
        lla = np.array([33.4255, -111.9400, 350.0])
        ecef = geodetic_to_ecef(lla)
        lla2 = ecef_to_geodetic(ecef)
        assert lla2[0] == pytest.approx(lla[0], abs=1e-7)
        assert lla2[1] == pytest.approx(lla[1], abs=1e-7)
        assert lla2[2] == pytest.approx(lla[2], abs=1e-3)

    def test_vectorized(self):
        lla = np.array(
            [
                [0.0, 0.0, 0.0],
                [33.0, -112.0, 100.0],
                [-23.5, 138.0, 50.0],
                [60.0, 5.0, 1500.0],
            ]
        )
        ecef = geodetic_to_ecef(lla)
        assert ecef.shape == (4, 3)
        lla2 = ecef_to_geodetic(ecef)
        # Sub-millimetre vertical, sub-microdegree horizontal.
        np.testing.assert_allclose(lla2[:, 0], lla[:, 0], atol=1e-7)
        np.testing.assert_allclose(lla2[:, 1], lla[:, 1], atol=1e-7)
        np.testing.assert_allclose(lla2[:, 2], lla[:, 2], atol=1e-3)


# ---------------------------------------------------------------------------
# ENU
# ---------------------------------------------------------------------------


class TestEnu:
    def test_basis_orthonormal(self):
        R = enu_basis_at(33.4, -111.9)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

    def test_origin_is_zero(self):
        origin = (33.42, -111.94, 350.0)
        ecef_origin = geodetic_to_ecef(np.array(origin))
        enu = ecef_to_enu(ecef_origin, origin)
        np.testing.assert_allclose(enu, np.zeros(3), atol=1e-9)

    def test_round_trip_local_point(self):
        origin = (33.42, -111.94, 350.0)
        # 100 m east, 50 m north, 10 m up.
        enu = np.array([100.0, 50.0, 10.0])
        ecef = enu_to_ecef(enu, origin)
        enu2 = ecef_to_enu(ecef, origin)
        np.testing.assert_allclose(enu2, enu, atol=1e-9)

    def test_north_displacement_is_along_local_north(self):
        origin = (33.42, -111.94, 350.0)
        # 1 km purely north should land ~1 km north and same up.
        enu = np.array([0.0, 1000.0, 0.0])
        ecef = enu_to_ecef(enu, origin)
        lla = ecef_to_geodetic(ecef)
        # Latitude increases, longitude unchanged.
        assert lla[0] > origin[0]
        assert lla[1] == pytest.approx(origin[1], abs=1e-5)


# ---------------------------------------------------------------------------
# NED
# ---------------------------------------------------------------------------


class TestNed:
    def test_axis_remap(self):
        # NED (1, 0, 0) = north, ENU (0, 1, 0) = north.
        v_enu = ned_to_enu(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(v_enu, [0.0, 1.0, 0.0], atol=1e-12)
        # NED (0, 0, 1) = down, ENU (0, 0, -1).
        v_enu = ned_to_enu(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(v_enu, [0.0, 0.0, -1.0], atol=1e-12)

    def test_full_pipe_ned_to_ecef(self):
        origin = (33.42, -111.94, 350.0)
        src = FrameSpec.ned_local(origin_lla=origin)
        dst = FrameSpec.ecef()
        # 1 m north in NED.
        p = transform_point(np.array([1.0, 0.0, 0.0]), src, dst)
        # Round trip.
        p2 = transform_point(p, dst, src)
        np.testing.assert_allclose(p2, [1.0, 0.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# UTM (skipped without pyproj)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_PYPROJ, reason="pyproj not available")
class TestUtm:
    def test_zone_for_phoenix(self):
        zone, hemi = utm_zone_for(lon_deg=-111.94, lat_deg=33.42)
        assert zone == 12
        assert hemi == "N"

    def test_round_trip_phoenix(self):
        lla = np.array([33.42, -111.94, 350.0])
        utm = geodetic_to_utm(lla, zone=12, hemisphere="N")
        # easting ~= 412k m, northing ~= 3.7M m around there.
        assert 200_000.0 < utm[0] < 800_000.0
        assert 3_500_000.0 < utm[1] < 4_000_000.0
        assert utm[2] == pytest.approx(350.0, abs=1e-6)
        lla2 = utm_to_geodetic(utm, zone=12, hemisphere="N")
        np.testing.assert_allclose(lla2[:2], lla[:2], atol=1e-6)
        assert lla2[2] == pytest.approx(lla[2], abs=1e-6)

    def test_southern_hemisphere(self):
        # Sydney
        lla = np.array([-33.86, 151.21, 50.0])
        zone, hemi = utm_zone_for(151.21, -33.86)
        utm = geodetic_to_utm(lla, zone=zone, hemisphere=hemi)
        lla2 = utm_to_geodetic(utm, zone=zone, hemisphere=hemi)
        np.testing.assert_allclose(lla2[:2], lla[:2], atol=1e-6)


# ---------------------------------------------------------------------------
# transform_point through ECEF pivot
# ---------------------------------------------------------------------------


class TestTransformPoint:
    def test_identity(self):
        ecef = FrameSpec.ecef()
        p = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(transform_point(p, ecef, ecef), p)

    def test_lla_to_ecef_and_back(self):
        lla_frame = FrameSpec.wgs84_lla()
        ecef_frame = FrameSpec.ecef()
        p = np.array([33.42, -111.94, 350.0])
        q = transform_point(p, lla_frame, ecef_frame)
        p2 = transform_point(q, ecef_frame, lla_frame)
        np.testing.assert_allclose(p2[:2], p[:2], atol=1e-7)
        assert p2[2] == pytest.approx(p[2], abs=1e-3)

    def test_enu_to_lla_round_trip(self):
        origin = (33.42, -111.94, 350.0)
        enu = FrameSpec.enu_local(origin_lla=origin)
        lla = FrameSpec.wgs84_lla()
        p_enu = np.array([10.0, 20.0, 5.0])
        p_lla = transform_point(p_enu, enu, lla)
        p_enu2 = transform_point(p_lla, lla, enu)
        np.testing.assert_allclose(p_enu2, p_enu, atol=1e-6)

    def test_batch(self):
        origin = (33.42, -111.94, 350.0)
        enu = FrameSpec.enu_local(origin_lla=origin)
        ecef = FrameSpec.ecef()
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        ecef_pts = transform_point(pts, enu, ecef)
        enu_pts = transform_point(ecef_pts, ecef, enu)
        np.testing.assert_allclose(enu_pts, pts, atol=1e-6)


# ---------------------------------------------------------------------------
# Pose round trips
# ---------------------------------------------------------------------------


class TestTransformPose:
    def test_identity(self):
        ecef = FrameSpec.ecef()
        t = np.array([1.0, 2.0, 3.0])
        R = np.eye(3)
        t2, R2 = transform_pose(t, R, ecef, ecef)
        np.testing.assert_allclose(t2, t, atol=1e-12)
        np.testing.assert_allclose(R2, R, atol=1e-12)

    def test_pose_round_trip_enu_ecef(self):
        origin = (33.42, -111.94, 350.0)
        enu = FrameSpec.enu_local(origin_lla=origin)
        ecef = FrameSpec.ecef()
        # Body frame: rotated 30 deg about ENU "Up".
        theta = math.radians(30.0)
        R_enu = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        t_enu = np.array([100.0, 50.0, 10.0])
        t_ecef, R_ecef = transform_pose(t_enu, R_enu, enu, ecef)
        t_enu2, R_enu2 = transform_pose(t_ecef, R_ecef, ecef, enu)
        np.testing.assert_allclose(t_enu2, t_enu, atol=1e-6)
        np.testing.assert_allclose(R_enu2, R_enu, atol=1e-9)

    def test_rotation_orthonormal_after_transform(self):
        origin = (33.42, -111.94, 350.0)
        enu = FrameSpec.enu_local(origin_lla=origin)
        ecef = FrameSpec.ecef()
        rng = np.random.default_rng(0)
        # Random rotation via QR.
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        t_enu = np.array([5.0, -3.0, 2.0])
        t_ecef, R_ecef = transform_pose(t_enu, Q, enu, ecef)
        np.testing.assert_allclose(R_ecef.T @ R_ecef, np.eye(3), atol=1e-12)
        assert np.linalg.det(R_ecef) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Superquadric pose transforms
# ---------------------------------------------------------------------------


class TestTransformSuperquadric:
    def test_box_round_trip_enu_ecef(self):
        origin = (33.42, -111.94, 350.0)
        enu = FrameSpec.enu_local(origin_lla=origin)
        ecef = FrameSpec.ecef()
        sq = superquadric_box(center=(10.0, 20.0, 5.0), size=(4.0, 3.0, 8.0))
        # Rotate 45 deg about up.
        theta = math.radians(45.0)
        sq.R = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        sq_ecef = transform_superquadric(sq, enu, ecef)
        sq_back = transform_superquadric(sq_ecef, ecef, enu)
        np.testing.assert_allclose(sq_back.t, sq.t, atol=1e-6)
        np.testing.assert_allclose(sq_back.R, sq.R, atol=1e-9)
        # Intrinsic shape preserved.
        np.testing.assert_allclose(sq_back.scale, sq.scale, atol=1e-12)
        np.testing.assert_allclose(sq_back.epsilon, sq.epsilon, atol=1e-12)
        assert sq_back.id == sq.id

    def test_tree_via_utm_to_lla_chain(self):
        if not _HAVE_PYPROJ:
            pytest.skip("pyproj not available")
        # earth_rover slams in UTM zone 12N around Phoenix
        utm = FrameSpec.utm(zone=12, hemisphere="N", name="map")
        lla = FrameSpec.wgs84_lla(name="cesium")
        # Tree at UTM (412345, 3700000, 350) -- roughly tempe.
        trunk = superquadric_cylinder(
            base=(412_345.0, 3_700_000.0, 350.0),
            axis=(0.0, 0.0, 1.0),
            radius=0.2,
            height=4.0,
        )
        crown = superquadric_ellipsoid(
            center=(412_345.0, 3_700_000.0, 354.0),
            axes=(2.0, 2.0, 1.5),
        )
        crown.parent_id = trunk.id
        # Transform to WGS84-LLA so a Cesium client can render.
        trunk_lla = transform_superquadric(trunk, utm, lla)
        crown_lla = transform_superquadric(crown, utm, lla)
        # Lat in Phoenix region.
        assert 33.0 < trunk_lla.t[0] < 34.0
        assert -113.0 < trunk_lla.t[1] < -110.0
        # Round-trip back to UTM should recover original within mm.
        trunk2 = transform_superquadric(trunk_lla, lla, utm)
        np.testing.assert_allclose(trunk2.t[:2], trunk.t[:2], atol=1e-3)
        # Parent linkage preserved.
        assert crown_lla.parent_id == trunk.id

    def test_intrinsic_geometry_preserved_under_transform(self):
        # The whole point: scale and epsilon must not change.
        origin = (33.42, -111.94, 350.0)
        enu = FrameSpec.enu_local(origin_lla=origin)
        ecef = FrameSpec.ecef()
        sq = superquadric_ellipsoid(center=(0.0, 0.0, 0.0), axes=(1.0, 2.0, 3.0))
        sq2 = transform_superquadric(sq, enu, ecef)
        np.testing.assert_allclose(sq2.scale, sq.scale, atol=1e-12)
        np.testing.assert_allclose(sq2.epsilon, sq.epsilon, atol=1e-12)

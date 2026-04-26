"""Coordinate frames + transforms for the earth_rover -> deepgis pipeline.

The pipeline crosses several frames:

* **WGS84-LLA** (geodetic latitude, longitude, ellipsoidal altitude) --
  the canonical interchange format for Cesium / OSM clients and the
  schema used by ``GET /api/v1/scene-graph``.
* **ECEF** (Earth-Centered Earth-Fixed, metres) -- the canonical
  server-side storage form for fused superquadric maps because it is
  geocentric, isotropic, and free of zone discontinuities.
* **UTM** (zone, hemisphere, easting, northing, ellipsoidal altitude) --
  what the earth_rover's onboard SLAM produces on a Pixhawk PX4 / GPS
  fix.  UTM is locally Cartesian within a zone, which is what graph SLAM
  back-ends expect.
* **ENU local tangent** (East-North-Up, metres, anchored at an LLA
  origin) -- a per-window local frame for short-baseline fusion or for
  rendering a small bbox of fused SQs as a flat scene.
* **NED local tangent** (North-East-Down, metres) -- PX4 / aviation
  convention; identical to ENU up to an axis re-map.

All frame conversions go *through ECEF* as the pivot.  The frames module
is intentionally pure numpy + (optional) pyproj; superquadric-aware
helpers live alongside it.

Conventions
-----------
* All angles in this module's public API are degrees; internally the
  helpers use radians.
* Quaternions are XYZW order, consistent with the wire codec.
* A pose ``(t, R)`` carried in frame A means: a body whose origin is at
  ``t`` in frame A's coordinates, with body-frame axes whose columns are
  expressed in A's basis vectors.  Re-expressing the pose in frame B is
  ``R_B = R_AB @ R_A`` and ``t_B = R_AB @ t_A + t_AB`` where
  ``(R_AB, t_AB)`` is the rigid transform from A to B *at the relevant
  origin* (constant for ENU<->ECEF, location-dependent for ECEF<->LLA --
  see the per-function notes).
* For LLA we treat lat/lon as a pseudo-Cartesian ``(lat, lon, alt)`` only
  for the position; orientation is *undefined* for LLA frames because
  rotations on a sphere don't have a single basis.  Routines that
  transform poses from/to LLA route the rotation through an ENU tangent
  frame anchored at the body's own location, which is the standard
  convention used by Cesium ``Transforms.eastNorthUpToFixedFrame``.

References
----------
* WGS84 constants: NIMA TR-8350.2, "Department of Defense World
  Geodetic System 1984", 3rd ed., 1997.
* Bowring (1985) for ECEF -> geodetic; we use the closed-form
  approximation, which is accurate to <1e-6 m for terrestrial altitudes.
* UTM: Snyder (1987) "Map Projections - A Working Manual", USGS
  Professional Paper 1395.  We delegate to ``pyproj`` when available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Optional pyproj for UTM <-> WGS84 (deferred import; we still want the
# rest of the module importable even on systems without pyproj).
try:  # pragma: no cover - import side-effect only
    import pyproj  # type: ignore

    _HAVE_PYPROJ = True
except Exception:  # pragma: no cover
    _HAVE_PYPROJ = False


# ---------------------------------------------------------------------------
# WGS84 ellipsoid constants
# ---------------------------------------------------------------------------

WGS84_A: float = 6_378_137.0  # semi-major axis (m)
WGS84_F: float = 1.0 / 298.257223563  # flattening
WGS84_B: float = WGS84_A * (1.0 - WGS84_F)  # semi-minor axis (m)
WGS84_E2: float = 2.0 * WGS84_F - WGS84_F * WGS84_F  # first eccentricity squared
WGS84_EP2: float = WGS84_E2 / (1.0 - WGS84_E2)  # second eccentricity squared


# ---------------------------------------------------------------------------
# Frame specification
# ---------------------------------------------------------------------------

#: Allowed values of :attr:`FrameSpec.kind`.
FRAME_KINDS = ("wgs84_lla", "ecef", "utm", "enu_local", "ned_local")


@dataclass(frozen=True)
class FrameSpec:
    """Identifier for a coordinate frame.

    Parameters
    ----------
    kind:
        One of :data:`FRAME_KINDS`.
    params:
        Frame-specific parameters:

        * ``"wgs84_lla"``: ``{}``
        * ``"ecef"``: ``{}``
        * ``"utm"``: ``{"zone": int (1..60), "hemisphere": "N" | "S"}``
        * ``"enu_local"``: ``{"origin_lla": (lat_deg, lon_deg, alt_m)}``
        * ``"ned_local"``: ``{"origin_lla": (lat_deg, lon_deg, alt_m)}``

    name:
        Optional human-friendly tag (e.g. ``"map"``, ``"odom"``,
        ``"server-canonical"``).  Not used in math; carried for
        debugging and for round-tripping through observation envelopes.
    """

    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in FRAME_KINDS:
            raise ValueError(
                f"FrameSpec.kind={self.kind!r} not in {FRAME_KINDS}"
            )
        if self.kind == "utm":
            zone = self.params.get("zone")
            hemi = self.params.get("hemisphere", "N")
            if not isinstance(zone, int) or not 1 <= zone <= 60:
                raise ValueError(
                    f"UTM frame requires integer zone in 1..60, got {zone!r}"
                )
            if hemi not in ("N", "S"):
                raise ValueError(
                    f"UTM hemisphere must be 'N' or 'S', got {hemi!r}"
                )
        if self.kind in ("enu_local", "ned_local"):
            origin = self.params.get("origin_lla")
            if (
                origin is None
                or len(origin) != 3
                or not all(isinstance(c, (int, float)) for c in origin)
            ):
                raise ValueError(
                    f"{self.kind} frame requires "
                    f"params['origin_lla']=(lat,lon,alt), got {origin!r}"
                )

    # ---- ergonomic constructors -------------------------------------------

    @classmethod
    def wgs84_lla(cls, name: Optional[str] = None) -> "FrameSpec":
        return cls(kind="wgs84_lla", params={}, name=name)

    @classmethod
    def ecef(cls, name: Optional[str] = None) -> "FrameSpec":
        return cls(kind="ecef", params={}, name=name)

    @classmethod
    def utm(
        cls,
        zone: int,
        hemisphere: str = "N",
        name: Optional[str] = None,
    ) -> "FrameSpec":
        return cls(
            kind="utm",
            params={"zone": int(zone), "hemisphere": hemisphere},
            name=name,
        )

    @classmethod
    def enu_local(
        cls,
        origin_lla: Sequence[float],
        name: Optional[str] = None,
    ) -> "FrameSpec":
        return cls(
            kind="enu_local",
            params={"origin_lla": tuple(float(c) for c in origin_lla)},
            name=name,
        )

    @classmethod
    def ned_local(
        cls,
        origin_lla: Sequence[float],
        name: Optional[str] = None,
    ) -> "FrameSpec":
        return cls(
            kind="ned_local",
            params={"origin_lla": tuple(float(c) for c in origin_lla)},
            name=name,
        )

    # ---- serialization for /api/observe envelopes -------------------------

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind, "params": dict(self.params)}
        if self.name is not None:
            out["name"] = self.name
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FrameSpec":
        return cls(
            kind=d["kind"],
            params=dict(d.get("params", {})),
            name=d.get("name"),
        )


# ---------------------------------------------------------------------------
# WGS84 <-> ECEF (closed-form, vectorized)
# ---------------------------------------------------------------------------


def geodetic_to_ecef(lla: np.ndarray) -> np.ndarray:
    """Convert geodetic ``(lat_deg, lon_deg, alt_m)`` to ECEF ``(X, Y, Z)``.

    Vectorized; accepts shape ``(3,)`` or ``(N, 3)``.
    """
    lla = np.asarray(lla, dtype=float)
    single = lla.ndim == 1
    if single:
        lla = lla.reshape(1, 3)
    lat = np.deg2rad(lla[:, 0])
    lon = np.deg2rad(lla[:, 1])
    alt = lla[:, 2]
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    X = (N + alt) * cos_lat * cos_lon
    Y = (N + alt) * cos_lat * sin_lon
    Z = (N * (1.0 - WGS84_E2) + alt) * sin_lat
    out = np.stack([X, Y, Z], axis=-1)
    return out[0] if single else out


def ecef_to_geodetic(xyz: np.ndarray) -> np.ndarray:
    """Convert ECEF ``(X, Y, Z)`` to geodetic ``(lat_deg, lon_deg, alt_m)``.

    Uses Bowring's closed-form method (1985), accurate to ``<1e-6 m`` for
    altitudes within ``+/-10 km``.  Vectorized.
    """
    xyz = np.asarray(xyz, dtype=float)
    single = xyz.ndim == 1
    if single:
        xyz = xyz.reshape(1, 3)
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]
    p = np.sqrt(X * X + Y * Y)
    # Bowring's auxiliary angle.
    theta = np.arctan2(Z * WGS84_A, p * WGS84_B)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    lat = np.arctan2(
        Z + WGS84_EP2 * WGS84_B * sin_t * sin_t * sin_t,
        p - WGS84_E2 * WGS84_A * cos_t * cos_t * cos_t,
    )
    lon = np.arctan2(Y, X)
    sin_lat = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    # Robust altitude: avoid the cos(lat) singularity at the poles.
    cos_lat = np.cos(lat)
    alt = np.where(
        np.abs(cos_lat) > 1e-9,
        p / cos_lat - N,
        Z / np.where(np.abs(sin_lat) > 1e-9, sin_lat, 1.0)
        - N * (1.0 - WGS84_E2),
    )
    out = np.stack([np.rad2deg(lat), np.rad2deg(lon), alt], axis=-1)
    return out[0] if single else out


# ---------------------------------------------------------------------------
# ECEF <-> ENU (linear, anchored at an LLA origin)
# ---------------------------------------------------------------------------


def enu_basis_at(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Return the 3x3 ``R_ENU_to_ECEF`` matrix at a tangent-plane origin.

    Columns are the ENU basis vectors expressed in ECEF.  This matrix is
    orthonormal and is the canonical "east-north-up to fixed frame"
    rotation used by Cesium's ``Transforms``.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    # Columns: [E, N, U] in ECEF.
    return np.array(
        [
            [-so, -sl * co, cl * co],
            [co, -sl * so, cl * so],
            [0.0, cl, sl],
        ],
        dtype=float,
    )


def ecef_to_enu(
    p_ecef: np.ndarray,
    origin_lla: Sequence[float],
) -> np.ndarray:
    """Project ECEF point(s) to a local ENU tangent frame at ``origin_lla``."""
    p_ecef = np.asarray(p_ecef, dtype=float)
    single = p_ecef.ndim == 1
    if single:
        p_ecef = p_ecef.reshape(1, 3)
    origin_ecef = geodetic_to_ecef(np.asarray(origin_lla, dtype=float))
    R = enu_basis_at(origin_lla[0], origin_lla[1])
    out = (p_ecef - origin_ecef) @ R  # equivalent to R.T @ (p - origin)
    return out[0] if single else out


def enu_to_ecef(
    p_enu: np.ndarray,
    origin_lla: Sequence[float],
) -> np.ndarray:
    """Lift local ENU point(s) at ``origin_lla`` back to ECEF."""
    p_enu = np.asarray(p_enu, dtype=float)
    single = p_enu.ndim == 1
    if single:
        p_enu = p_enu.reshape(1, 3)
    origin_ecef = geodetic_to_ecef(np.asarray(origin_lla, dtype=float))
    R = enu_basis_at(origin_lla[0], origin_lla[1])
    out = p_enu @ R.T + origin_ecef
    return out[0] if single else out


# ---------------------------------------------------------------------------
# ENU <-> NED (constant axis re-map)
# ---------------------------------------------------------------------------

#: ``v_enu = R_NED_TO_ENU @ v_ned``.  NED axes (N, E, D) re-map to
#: ENU axes (E, N, U) by swapping the first two coordinates and
#: negating the third.
R_NED_TO_ENU: np.ndarray = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)
#: Inverse mapping (its own inverse, since it is an involution).
R_ENU_TO_NED: np.ndarray = R_NED_TO_ENU.T  # equals R_NED_TO_ENU itself


def ned_to_enu(p_ned: np.ndarray) -> np.ndarray:
    p_ned = np.asarray(p_ned, dtype=float)
    return p_ned @ R_NED_TO_ENU.T


def enu_to_ned(p_enu: np.ndarray) -> np.ndarray:
    p_enu = np.asarray(p_enu, dtype=float)
    return p_enu @ R_ENU_TO_NED.T


# ---------------------------------------------------------------------------
# UTM <-> WGS84 (delegated to pyproj)
# ---------------------------------------------------------------------------


def _require_pyproj() -> None:
    if not _HAVE_PYPROJ:
        raise ImportError(
            "UTM <-> WGS84 conversion requires the optional dependency "
            "`pyproj`.  Install with `pip install pyproj`."
        )


def _utm_epsg(zone: int, hemisphere: str) -> int:
    if hemisphere == "N":
        return 32600 + int(zone)
    if hemisphere == "S":
        return 32700 + int(zone)
    raise ValueError(f"hemisphere must be 'N' or 'S', got {hemisphere!r}")


def utm_to_geodetic(
    en_alt: np.ndarray,
    zone: int,
    hemisphere: str = "N",
) -> np.ndarray:
    """Convert UTM ``(easting_m, northing_m, alt_m)`` to ``(lat, lon, alt)``."""
    _require_pyproj()
    en_alt = np.asarray(en_alt, dtype=float)
    single = en_alt.ndim == 1
    if single:
        en_alt = en_alt.reshape(1, 3)
    transformer = pyproj.Transformer.from_crs(  # type: ignore[name-defined]
        f"EPSG:{_utm_epsg(zone, hemisphere)}",
        "EPSG:4326",
        always_xy=True,
    )
    lon_deg, lat_deg = transformer.transform(en_alt[:, 0], en_alt[:, 1])
    out = np.stack(
        [np.asarray(lat_deg), np.asarray(lon_deg), en_alt[:, 2]],
        axis=-1,
    )
    return out[0] if single else out


def geodetic_to_utm(
    lla: np.ndarray,
    zone: int,
    hemisphere: str = "N",
) -> np.ndarray:
    """Convert ``(lat, lon, alt)`` to UTM ``(easting_m, northing_m, alt_m)``."""
    _require_pyproj()
    lla = np.asarray(lla, dtype=float)
    single = lla.ndim == 1
    if single:
        lla = lla.reshape(1, 3)
    transformer = pyproj.Transformer.from_crs(  # type: ignore[name-defined]
        "EPSG:4326",
        f"EPSG:{_utm_epsg(zone, hemisphere)}",
        always_xy=True,
    )
    e, n = transformer.transform(lla[:, 1], lla[:, 0])
    out = np.stack([np.asarray(e), np.asarray(n), lla[:, 2]], axis=-1)
    return out[0] if single else out


def utm_zone_for(lon_deg: float, lat_deg: float) -> Tuple[int, str]:
    """Return ``(zone, hemisphere)`` for a given lon/lat (standard UTM)."""
    zone = int(math.floor((lon_deg + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    hemi = "N" if lat_deg >= 0.0 else "S"
    return zone, hemi


# ---------------------------------------------------------------------------
# Generic point/pose transforms via ECEF pivot
# ---------------------------------------------------------------------------


def _to_ecef(p: np.ndarray, frame: FrameSpec) -> np.ndarray:
    if frame.kind == "ecef":
        return np.asarray(p, dtype=float)
    if frame.kind == "wgs84_lla":
        return geodetic_to_ecef(p)
    if frame.kind == "utm":
        zone = int(frame.params["zone"])
        hemi = str(frame.params.get("hemisphere", "N"))
        lla = utm_to_geodetic(p, zone, hemi)
        return geodetic_to_ecef(lla)
    if frame.kind == "enu_local":
        origin = frame.params["origin_lla"]
        return enu_to_ecef(p, origin)
    if frame.kind == "ned_local":
        origin = frame.params["origin_lla"]
        # NED -> ENU (axis remap) -> ECEF.
        p = np.asarray(p, dtype=float)
        single = p.ndim == 1
        p_enu = ned_to_enu(p.reshape(-1, 3))
        out = enu_to_ecef(p_enu, origin)
        return out[0] if single else out
    raise ValueError(f"Unsupported frame.kind={frame.kind!r}")


def _from_ecef(p_ecef: np.ndarray, frame: FrameSpec) -> np.ndarray:
    if frame.kind == "ecef":
        return np.asarray(p_ecef, dtype=float)
    if frame.kind == "wgs84_lla":
        return ecef_to_geodetic(p_ecef)
    if frame.kind == "utm":
        zone = int(frame.params["zone"])
        hemi = str(frame.params.get("hemisphere", "N"))
        lla = ecef_to_geodetic(p_ecef)
        return geodetic_to_utm(lla, zone, hemi)
    if frame.kind == "enu_local":
        origin = frame.params["origin_lla"]
        return ecef_to_enu(p_ecef, origin)
    if frame.kind == "ned_local":
        origin = frame.params["origin_lla"]
        p_ecef = np.asarray(p_ecef, dtype=float)
        single = p_ecef.ndim == 1
        p_enu = ecef_to_enu(p_ecef.reshape(-1, 3), origin)
        p_ned = enu_to_ned(p_enu)
        return p_ned[0] if single else p_ned
    raise ValueError(f"Unsupported frame.kind={frame.kind!r}")


def transform_point(
    p: np.ndarray,
    src: FrameSpec,
    dst: FrameSpec,
) -> np.ndarray:
    """Transform point(s) from ``src`` to ``dst``.

    Accepts ``(3,)`` or ``(N, 3)``.  Always pivots through ECEF.
    """
    if src == dst:
        return np.asarray(p, dtype=float).copy()
    return _from_ecef(_to_ecef(p, src), dst)


# ---------------------------------------------------------------------------
# Pose (translation + rotation) transforms
# ---------------------------------------------------------------------------


def _local_basis_in_ecef(p_ecef: np.ndarray, frame: FrameSpec) -> np.ndarray:
    """Return ``R`` such that ``v_ecef = R @ v_frame_local`` *at p_ecef*.

    For ECEF and UTM, this is the identity (UTM is locally Cartesian and
    ignores the curvature of the projection over the ~metre scale of a
    rigid body, which is overwhelmingly accurate for terrestrial robots).
    For ENU/NED at a fixed origin, it is the constant tangent-plane
    rotation at the origin.  For WGS84-LLA the rotation is undefined
    globally; we use the ENU tangent at the body's *own* lat/lon, which
    matches Cesium's ``eastNorthUpToFixedFrame`` convention.
    """
    if frame.kind in ("ecef", "utm"):
        return np.eye(3)
    if frame.kind == "enu_local":
        origin = frame.params["origin_lla"]
        return enu_basis_at(origin[0], origin[1])
    if frame.kind == "ned_local":
        origin = frame.params["origin_lla"]
        return enu_basis_at(origin[0], origin[1]) @ R_NED_TO_ENU
    if frame.kind == "wgs84_lla":
        lla = ecef_to_geodetic(p_ecef)
        return enu_basis_at(float(lla[0]), float(lla[1]))
    raise ValueError(f"Unsupported frame.kind={frame.kind!r}")


def transform_pose(
    t: np.ndarray,
    R: np.ndarray,
    src: FrameSpec,
    dst: FrameSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a 6-DOF pose from ``src`` to ``dst``.

    Parameters
    ----------
    t:
        Translation in ``src`` (3,).
    R:
        3x3 rotation, body-axes expressed in ``src`` basis.

    Returns
    -------
    (t_dst, R_dst):
        Translation and rotation re-expressed in ``dst``.
    """
    t = np.asarray(t, dtype=float).reshape(3)
    R = np.asarray(R, dtype=float).reshape(3, 3)

    # Translate via ECEF pivot.
    t_ecef = _to_ecef(t, src)
    t_dst = _from_ecef(t_ecef, dst)

    # Rotation: take src basis -> ECEF basis (at this point) -> dst
    # basis (at this point).
    R_src_to_ecef = _local_basis_in_ecef(t_ecef, src)
    R_dst_to_ecef = _local_basis_in_ecef(t_ecef, dst)
    R_src_to_dst = R_dst_to_ecef.T @ R_src_to_ecef
    R_dst = R_src_to_dst @ R
    return t_dst, R_dst


# ---------------------------------------------------------------------------
# Superquadric-aware helpers
# ---------------------------------------------------------------------------


def transform_superquadric(sq, src: FrameSpec, dst: FrameSpec):
    """Return a copy of ``sq`` with its pose re-expressed in ``dst``.

    The superquadric's intrinsic geometry (``scale``, ``epsilon``,
    ``parent_id``, ``attributes``, ``covariance``) is preserved
    bit-for-bit; only ``R`` and ``t`` are re-expressed.
    """
    if src == dst:
        # Return a shallow copy with copied pose so callers can mutate.
        return sq.transformed(np.eye(3), np.zeros(3))
    t_dst, R_dst = transform_pose(sq.t, sq.R, src, dst)
    new = sq.transformed(np.eye(3), np.zeros(3))
    new.R = R_dst
    new.t = t_dst
    return new


def transform_superquadrics(
    sqs: Iterable,
    src: FrameSpec,
    dst: FrameSpec,
) -> List:
    """Vectorized convenience over :func:`transform_superquadric`."""
    return [transform_superquadric(sq, src, dst) for sq in sqs]


__all__ = [
    # Frame spec
    "FrameSpec",
    "FRAME_KINDS",
    # WGS84 constants
    "WGS84_A",
    "WGS84_B",
    "WGS84_F",
    "WGS84_E2",
    # Geodetic primitives
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    # ENU primitives
    "enu_basis_at",
    "ecef_to_enu",
    "enu_to_ecef",
    # NED primitives
    "ned_to_enu",
    "enu_to_ned",
    "R_NED_TO_ENU",
    "R_ENU_TO_NED",
    # UTM primitives
    "utm_to_geodetic",
    "geodetic_to_utm",
    "utm_zone_for",
    # Generic transforms
    "transform_point",
    "transform_pose",
    "transform_superquadric",
    "transform_superquadrics",
]

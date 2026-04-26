"""Superquadric primitive: dataclass, math, factories, serialization.

A superquadric is defined by the implicit *inside-outside* function

    F(x, y, z) = ( |x/a1|^(2/eps2) + |y/a2|^(2/eps2) )^(eps2/eps1)
                 + |z/a3|^(2/eps1)

where (a1, a2, a3) are positive semi-axis lengths and (eps1, eps2) are
shape exponents. F = 1 on the surface, F < 1 inside, F > 1 outside.

eps1 controls the "north-south" roundness (top/bottom edges).
eps2 controls the "east-west" roundness (cross-section corners).

Special cases (degenerate limits):

==================== ============== ============================
 (eps1, eps2)         Shape          Use
==================== ============== ============================
 (1.0, 1.0)            ellipsoid /    tree crown (deciduous),
                       sphere         vehicle silhouette
 (eps -> 0, eps -> 0)  cuboid         OSM building extrusion,
                                      kiosk, signpost backplane
 (1.0, eps -> 0)       cylinder       tree trunk, lamppost, pier
 (eps -> 0, 1.0)       rounded slab   curb, low wall
 (1.0, 0.5)            cone-ish       conifer crown
 (2.0, 2.0)            octahedron     (rare)
==================== ============== ============================

The full posed primitive carries:

* ``scale``    : (3,) semi-axis lengths in meters
* ``epsilon``  : (2,) shape exponents in [EPS_MIN, EPS_MAX]
* ``R``        : (3, 3) rotation, body -> world
* ``t``        : (3,) translation, world position of body origin
* ``parent_id``: optional id of a parent SQ for composite/tree shapes
* ``attributes``: free-form metadata (semantics, source, score, ...)
* ``covariance``: optional (11, 11) Hessian-inverse cov over fit params

Wire footprint (see :mod:`.codec`): ~22 bytes quantized.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Lower clamp on shape exponents (eps -> 0 is the cuboid limit).  Going
#: strictly to zero blows up ``2/eps`` in the inside-outside function;
#: ~0.05 is small enough to look cuboid for all practical purposes.
EPS_MIN: float = 0.05

#: Upper clamp on shape exponents.  Above ~2 you get pinched/octahedral
#: shapes that the literature treats as a separate regime; we reject
#: these to keep fitting well-behaved.
EPS_MAX: float = 2.0

#: Lower clamp on semi-axis lengths (meters).  Below this we can't
#: reliably fit or quantize.
A_MIN: float = 1e-3

#: Upper clamp on semi-axis lengths (meters).  Above this we're outside
#: anything you'd model as a single primitive (use a hierarchy).
A_MAX: float = 1e4

# Tiny floor used inside fractional-power ops to avoid 0**negative.
_TINY: float = 1e-12


# ---------------------------------------------------------------------------
# Helpers: signed power, rotation algebra, axis-angle
# ---------------------------------------------------------------------------


def _signed_pow(x: np.ndarray, p: float) -> np.ndarray:
    """Signed fractional power: ``sign(x) * |x|^p``.

    Used in the parametric form ``cos^eps(theta) := sign(cos t) |cos t|^eps``.
    """
    ax = np.abs(x)
    # avoid 0**0 ambiguity and 0**negative blowups
    return np.sign(x) * np.power(np.maximum(ax, _TINY), p)


def _abs_pow(x: np.ndarray, p: float) -> np.ndarray:
    """``|x|^p`` with a tiny floor to keep fractional powers finite."""
    return np.power(np.maximum(np.abs(x), _TINY), p)


def _so3_from_axis_angle(omega: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map: so(3) tangent vector -> SO(3) matrix.

    ``omega`` is a length-3 vector (axis * angle).  Returns the rotation
    that maps the body frame to the world frame.
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        # 1st-order approximation — exact for theta = 0.
        K = np.array(
            [
                [0.0, -omega[2], omega[1]],
                [omega[2], 0.0, -omega[0]],
                [-omega[1], omega[0], 0.0],
            ],
            dtype=float,
        )
        return np.eye(3) + K
    axis = omega / theta
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    s = math.sin(theta)
    c = math.cos(theta)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _axis_angle_from_so3(R: np.ndarray) -> np.ndarray:
    """Inverse Rodrigues: SO(3) -> so(3) tangent (axis * angle)."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_t = (np.trace(R) - 1.0) * 0.5
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    theta = math.acos(cos_t)
    if theta < 1e-9:
        # Near identity: the off-diagonal antisymmetric part *is* the
        # tangent vector to first order.
        return np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=float,
        ) * 0.5
    if theta > math.pi - 1e-6:
        # Near-pi rotations: extract axis from the symmetric part.
        # R + I is rank-1 with column proportional to axis.
        M = R + np.eye(3)
        col = np.argmax(np.linalg.norm(M, axis=0))
        axis = M[:, col]
        axis = axis / max(np.linalg.norm(axis), _TINY)
        # Disambiguate sign using the antisymmetric part.
        s_vec = np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        )
        if float(np.dot(s_vec, axis)) < 0.0:
            axis = -axis
        return axis * theta
    s = math.sin(theta)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    ) / (2.0 * s)
    return axis * theta


def _project_to_so3(M: np.ndarray) -> np.ndarray:
    """Closest rotation matrix to ``M`` in Frobenius norm (SVD projection)."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        # Flip the smallest singular direction to enforce det = +1.
        Vt[-1, :] *= -1.0
        R = U @ Vt
    return R


# ---------------------------------------------------------------------------
# Superquadric dataclass
# ---------------------------------------------------------------------------


@dataclass
class Superquadric:
    """Posed superquadric with optional fit covariance and provenance.

    Parameters
    ----------
    scale
        Length-3 semi-axis vector ``(a1, a2, a3)`` in meters, all positive.
    epsilon
        Length-2 shape exponents ``(eps1, eps2)`` in
        ``[EPS_MIN, EPS_MAX]``.
    R
        ``(3, 3)`` rotation matrix mapping body frame to world frame.
        Defaults to identity.
    t
        ``(3,)`` translation in meters, world position of the body
        origin.  Defaults to zeros.
    id
        Stable identifier; auto-generated as ``"sq-<uuid>"`` if omitted.
    parent_id
        Optional id of a parent :class:`Superquadric` for composite or
        tree-structured shapes (e.g. tree-canopy rooted at trunk,
        building roof rooted at footprint).
    covariance
        Optional ``(11, 11)`` Hessian-inverse covariance over the fit
        parameters in the order ``[log a1, log a2, log a3, eps1, eps2,
        omega_x, omega_y, omega_z, t_x, t_y, t_z]``.  Emitted by
        :func:`fit_superquadric` when ``return_covariance=True``.
    attributes
        Free-form metadata bag.  Common keys: ``"class"`` (semantic
        label), ``"score"`` (confidence in [0, 1]), ``"source_id"``,
        ``"frame_id"`` (e.g. ``"map"``, ``"odom"``).
    """

    scale: np.ndarray
    epsilon: np.ndarray
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    id: str = field(default_factory=lambda: f"sq-{uuid.uuid4().hex[:12]}")
    parent_id: Optional[str] = None
    covariance: Optional[np.ndarray] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    # ---- Construction -------------------------------------------------

    def __post_init__(self) -> None:
        self.scale = np.asarray(self.scale, dtype=float).reshape(3)
        self.epsilon = np.asarray(self.epsilon, dtype=float).reshape(2)
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=float).reshape(3)
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance, dtype=float).reshape(11, 11)
        # Validate ranges.
        if np.any(self.scale < A_MIN) or np.any(self.scale > A_MAX):
            raise ValueError(
                f"Superquadric.scale out of range [{A_MIN}, {A_MAX}]: "
                f"got {self.scale.tolist()}"
            )
        if np.any(self.epsilon < EPS_MIN) or np.any(self.epsilon > EPS_MAX):
            raise ValueError(
                f"Superquadric.epsilon out of range [{EPS_MIN}, {EPS_MAX}]: "
                f"got {self.epsilon.tolist()}"
            )
        # Make sure R is actually orthonormal (allow tiny SVD drift).
        rtr = self.R.T @ self.R
        if not np.allclose(rtr, np.eye(3), atol=1e-6):
            self.R = _project_to_so3(self.R)
        if abs(np.linalg.det(self.R) - 1.0) > 1e-3:
            self.R = _project_to_so3(self.R)

    # ---- Frame conversions --------------------------------------------

    def world_to_body(self, points: np.ndarray) -> np.ndarray:
        """Rotate+translate ``(N, 3)`` world points into the body frame."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        return (pts - self.t) @ self.R  # R^T applied via right-multiply

    def body_to_world(self, points: np.ndarray) -> np.ndarray:
        """Rotate+translate ``(N, 3)`` body points into the world frame."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        return pts @ self.R.T + self.t

    # ---- Implicit functions -------------------------------------------

    def inside_outside(self, points: np.ndarray) -> np.ndarray:
        """Solina-Bajcsy inside-outside ``F(x,y,z)`` evaluated per point.

        ``F == 1`` on the surface, ``F < 1`` inside, ``F > 1`` outside.
        """
        local = self.world_to_body(points)
        a1, a2, a3 = self.scale
        e1, e2 = self.epsilon
        x = local[:, 0] / a1
        y = local[:, 1] / a2
        z = local[:, 2] / a3
        cross = _abs_pow(x, 2.0 / e2) + _abs_pow(y, 2.0 / e2)
        return np.power(np.maximum(cross, _TINY), e2 / e1) + _abs_pow(z, 2.0 / e1)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Approximate Euclidean signed distance (Solina radial form).

        Negative inside, positive outside.  Not the true Euclidean
        distance to the surface, but smooth, sign-correct, and cheap —
        sufficient for spatial pairwise factors and inclusion tests.
        """
        local = self.world_to_body(points)
        e1, _ = self.epsilon
        F = self.inside_outside(points)
        # Radial distance: |p_local| * (1 - F^(-eps1/2))
        r = np.linalg.norm(local, axis=1)
        scale = 1.0 - np.power(np.maximum(F, _TINY), -0.5 * e1)
        return r * scale

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean ``(N,)`` mask: True for points strictly inside."""
        return self.inside_outside(points) < 1.0

    # ---- Geometry helpers ---------------------------------------------

    def volume(self) -> float:
        """Closed-form interior volume.

        ``V = 2 * a1 * a2 * a3 * eps1 * eps2 * B(eps1/2 + 1, eps1)
                                  * B(eps2/2, eps2/2)``

        where ``B`` is the Euler beta function.  Reduces to ``4/3 pi a1 a2 a3``
        for an ellipsoid (eps1 = eps2 = 1).
        """
        from scipy.special import beta as _beta
        a1, a2, a3 = self.scale
        e1, e2 = self.epsilon
        return float(
            2.0 * a1 * a2 * a3 * e1 * e2 * _beta(e1 / 2.0 + 1.0, e1) * _beta(e2 / 2.0, e2 / 2.0)
        )

    def bbox_3d(self) -> Tuple[np.ndarray, np.ndarray]:
        """World-axis-aligned bounding box ``(p_min, p_max)``.

        Computed by rotating the body-frame AABB ``[-a, a]^3`` corners
        into the world frame.  Tight for cuboid limits; slightly loose
        for ellipsoidal shapes (still O(scale)).
        """
        a = self.scale
        signs = np.array(
            [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
            dtype=float,
        )
        corners_body = signs * a  # (8, 3)
        corners_world = corners_body @ self.R.T + self.t
        return corners_world.min(axis=0), corners_world.max(axis=0)

    # ---- Surface tessellation -----------------------------------------

    def surface_points(
        self,
        n_lat: int = 24,
        n_lon: int = 48,
    ) -> np.ndarray:
        """Sample ``(n_lat * n_lon, 3)`` points on the surface.

        Parametric form (signed-power "trig"):
            x = a1 * sgnpow(cos eta, eps1) * sgnpow(cos omega, eps2)
            y = a2 * sgnpow(cos eta, eps1) * sgnpow(sin omega, eps2)
            z = a3 * sgnpow(sin eta, eps1)

        with eta in [-pi/2, pi/2], omega in [-pi, pi].
        """
        e1, e2 = self.epsilon
        a1, a2, a3 = self.scale
        eta = np.linspace(-math.pi / 2.0, math.pi / 2.0, int(n_lat))
        omega = np.linspace(-math.pi, math.pi, int(n_lon))
        E, O = np.meshgrid(eta, omega, indexing="ij")
        ce = _signed_pow(np.cos(E), e1)
        se = _signed_pow(np.sin(E), e1)
        co = _signed_pow(np.cos(O), e2)
        so_ = _signed_pow(np.sin(O), e2)
        x = a1 * ce * co
        y = a2 * ce * so_
        z = a3 * se
        body_pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        return body_pts @ self.R.T + self.t

    def tessellate(
        self,
        n_lat: int = 24,
        n_lon: int = 48,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Triangle mesh ``(verts (V, 3), faces (F, 3))`` in the world frame.

        Closes the surface by tying the omega ring at +/- pi together.
        """
        n_lat = max(int(n_lat), 4)
        n_lon = max(int(n_lon), 6)
        verts = self.surface_points(n_lat=n_lat, n_lon=n_lon)
        # Build faces over the (n_lat x n_lon) grid; wrap omega.
        faces = []
        for i in range(n_lat - 1):
            for j in range(n_lon):
                jn = (j + 1) % n_lon
                v00 = i * n_lon + j
                v01 = i * n_lon + jn
                v10 = (i + 1) * n_lon + j
                v11 = (i + 1) * n_lon + jn
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])
        return verts, np.asarray(faces, dtype=np.int64)

    # ---- Pose updates -------------------------------------------------

    def transformed(self, R_world: np.ndarray, t_world: np.ndarray) -> "Superquadric":
        """Return a copy with pose updated by an *outer* rigid transform.

        ``new.R = R_world @ self.R``, ``new.t = R_world @ self.t + t_world``.
        Useful when re-expressing a body-frame fit in a world frame.
        """
        R_world = np.asarray(R_world, dtype=float).reshape(3, 3)
        t_world = np.asarray(t_world, dtype=float).reshape(3)
        return Superquadric(
            scale=self.scale.copy(),
            epsilon=self.epsilon.copy(),
            R=R_world @ self.R,
            t=R_world @ self.t + t_world,
            id=self.id,
            parent_id=self.parent_id,
            covariance=None if self.covariance is None else self.covariance.copy(),
            attributes=dict(self.attributes),
        )

    # ---- Serialization ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly view; geometry is fully recoverable from this dict."""
        out: Dict[str, Any] = {
            "id": self.id,
            "kind": "superquadric",
            "scale": self.scale.tolist(),
            "epsilon": self.epsilon.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "attributes": dict(self.attributes),
        }
        if self.parent_id is not None:
            out["parent_id"] = self.parent_id
        if self.covariance is not None:
            out["covariance"] = self.covariance.tolist()
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Superquadric":
        if d.get("kind") not in (None, "superquadric"):
            raise ValueError(f"Expected kind='superquadric', got {d.get('kind')!r}")
        return cls(
            scale=np.asarray(d["scale"], dtype=float),
            epsilon=np.asarray(d["epsilon"], dtype=float),
            R=np.asarray(d.get("R", np.eye(3).tolist()), dtype=float),
            t=np.asarray(d.get("t", np.zeros(3).tolist()), dtype=float),
            id=d.get("id") or f"sq-{uuid.uuid4().hex[:12]}",
            parent_id=d.get("parent_id"),
            covariance=(
                np.asarray(d["covariance"], dtype=float)
                if d.get("covariance") is not None
                else None
            ),
            attributes=dict(d.get("attributes") or {}),
        )

    # ---- Fit-parameter vector -----------------------------------------

    def to_fit_params(self) -> np.ndarray:
        """Pack into the 11-vector used by :func:`fit_superquadric`.

        Order: ``[log a1, log a2, log a3, eps1, eps2,
                  omega_x, omega_y, omega_z, t_x, t_y, t_z]``.
        """
        log_a = np.log(self.scale)
        omega = _axis_angle_from_so3(self.R)
        return np.concatenate([log_a, self.epsilon, omega, self.t])

    @classmethod
    def from_fit_params(
        cls,
        params: np.ndarray,
        *,
        attributes: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        id: Optional[str] = None,
        covariance: Optional[np.ndarray] = None,
    ) -> "Superquadric":
        params = np.asarray(params, dtype=float).reshape(11)
        scale = np.exp(np.clip(params[0:3], math.log(A_MIN), math.log(A_MAX)))
        epsilon = np.clip(params[3:5], EPS_MIN, EPS_MAX)
        R = _so3_from_axis_angle(params[5:8])
        t = params[8:11]
        return cls(
            scale=scale,
            epsilon=epsilon,
            R=R,
            t=t,
            id=id or f"sq-{uuid.uuid4().hex[:12]}",
            parent_id=parent_id,
            covariance=covariance,
            attributes=dict(attributes or {}),
        )


# ---------------------------------------------------------------------------
# Factory constructors
# ---------------------------------------------------------------------------


def superquadric_box(
    center: np.ndarray,
    size: np.ndarray,
    *,
    R: Optional[np.ndarray] = None,
    eps: float = 0.1,
    attributes: Optional[Dict[str, Any]] = None,
) -> Superquadric:
    """OSM-style axis-aligned (or oriented) box.

    ``size`` is the *full* extent (width, depth, height); the SQ
    semi-axes are ``size / 2``.  Default ``eps = 0.1`` looks visually
    cuboid; lower values are crisper but harder to fit.
    """
    center = np.asarray(center, dtype=float).reshape(3)
    size = np.asarray(size, dtype=float).reshape(3)
    a = np.maximum(size * 0.5, A_MIN)
    return Superquadric(
        scale=a,
        epsilon=np.array([eps, eps], dtype=float),
        R=np.eye(3) if R is None else np.asarray(R, dtype=float),
        t=center,
        attributes=dict(attributes or {}),
    )


def superquadric_cylinder(
    base: np.ndarray,
    axis: np.ndarray,
    radius: float,
    height: float,
    *,
    eps2: float = 0.1,
    attributes: Optional[Dict[str, Any]] = None,
) -> Superquadric:
    """Cylinder spanning from ``base`` along ``axis`` for ``height``.

    Body-z is aligned with ``axis``; cross-section radius is ``radius``.
    Encoded as a SQ with ``eps1 = 1.0`` (round caps) and small ``eps2``
    (square-ish cross-section limit -> circular when ``eps2 -> 0``;
    visually circular at ``eps2 = 0.1``).

    Note: a true mathematical circle is the ``eps2 = 1.0`` limit of the
    superellipse; we use ``eps2 = 0.1`` here because the canonical SQ
    cylinder has flat top and bottom, which corresponds to small
    ``eps2`` in the (a1, a2)-plane.  For a smooth-circular cross
    section, use ``eps2 = 1.0``.
    """
    base = np.asarray(base, dtype=float).reshape(3)
    axis = np.asarray(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < _TINY:
        raise ValueError("superquadric_cylinder: axis must be non-zero.")
    z_hat = axis / n
    # Build a rotation whose +z column is z_hat.
    if abs(z_hat[2]) < 0.999:
        x_hat = np.cross(np.array([0.0, 0.0, 1.0]), z_hat)
    else:
        x_hat = np.cross(np.array([1.0, 0.0, 0.0]), z_hat)
    x_hat = x_hat / max(np.linalg.norm(x_hat), _TINY)
    y_hat = np.cross(z_hat, x_hat)
    R = np.column_stack([x_hat, y_hat, z_hat])
    height = max(float(height), 2.0 * A_MIN)
    radius = max(float(radius), A_MIN)
    center = base + 0.5 * height * z_hat
    return Superquadric(
        scale=np.array([radius, radius, height * 0.5], dtype=float),
        epsilon=np.array([1.0, eps2], dtype=float),
        R=R,
        t=center,
        attributes=dict(attributes or {}),
    )


def superquadric_ellipsoid(
    center: np.ndarray,
    axes: np.ndarray,
    *,
    R: Optional[np.ndarray] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Superquadric:
    """Posed ellipsoid (eps = (1, 1)).

    ``axes`` are the semi-axis lengths; ``R`` rotates the body frame
    into world.  Default ``R`` is identity (axis-aligned).
    """
    center = np.asarray(center, dtype=float).reshape(3)
    axes = np.asarray(axes, dtype=float).reshape(3)
    return Superquadric(
        scale=np.maximum(axes, A_MIN),
        epsilon=np.array([1.0, 1.0], dtype=float),
        R=np.eye(3) if R is None else np.asarray(R, dtype=float),
        t=center,
        attributes=dict(attributes or {}),
    )


def superquadric_sphere(
    center: np.ndarray,
    radius: float,
    *,
    attributes: Optional[Dict[str, Any]] = None,
) -> Superquadric:
    """Posed sphere (degenerate ellipsoid)."""
    return superquadric_ellipsoid(
        center=center,
        axes=np.array([radius, radius, radius], dtype=float),
        attributes=attributes,
    )

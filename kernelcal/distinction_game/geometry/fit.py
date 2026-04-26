"""Superquadric fitting from point clouds.

This module implements Solina-Bajcsy least-squares recovery of a
superquadric from an unstructured ``(N, 3)`` point cloud, with optional
robust (Cauchy) loss for outlier resistance and Hessian-inverse
covariance estimation that downstream factor-graph code can consume as
unary evidence weight.

It also provides :func:`fit_tree`, a convenience wrapper that splits a
single-tree point cloud into a trunk (vertical-axis cylinder fit) and a
crown (ellipsoid fit) and links them via ``parent_id``.

Math
----
Given the inside-outside function ``F(x, y, z)`` of a superquadric
:class:`~kernelcal.distinction_game.geometry.superquadric.Superquadric`,
we minimise the per-point residual

    r_i = sqrt(a1 * a2 * a3) * (F_i ** eps1 - 1)

over the 11-vector parameter ``[log a, eps, omega, t]``.  The
``sqrt(a1 a2 a3)`` prefactor prevents the optimiser from collapsing the
SQ to zero volume; the ``F ** eps1`` form was shown by Solina & Bajcsy
to behave approximately as squared Euclidean distance near the surface
(see their §III).

For uncertainty, the asymptotic covariance is

    Cov(theta_hat) approx sigma^2 * (J^T J)^{-1},

where J is the Jacobian at the optimum and sigma^2 is the residual
variance.  This drops directly into a unary perceptual factor as a
Gaussian-on-surface evidence term.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .superquadric import (
    A_MAX,
    A_MIN,
    EPS_MAX,
    EPS_MIN,
    Superquadric,
    _abs_pow,
    _so3_from_axis_angle,
    _TINY,
    superquadric_cylinder,
    superquadric_ellipsoid,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class FitDiagnostics:
    """Per-fit diagnostics surfaced for the factor graph and admin views.

    Attributes
    ----------
    n_points
        Points used in the final fit (after subsampling, if any).
    rms_residual
        Root-mean-square Solina-Bajcsy residual at the optimum.
    inlier_fraction
        Fraction of points with ``|F^eps1 - 1| <= inlier_threshold``.
        ``inlier_threshold`` defaults to ``0.1`` (about 10% of "surface
        thickness" in the parametric sense).
    converged
        ``True`` if the optimiser reported success.
    n_iter
        Iteration count from scipy.
    cost
        Final 1/2 * sum(residual^2) reported by scipy.
    method
        ``"lm"``, ``"trf"``, etc. — the underlying solver.
    """

    n_points: int
    rms_residual: float
    inlier_fraction: float
    converged: bool
    n_iter: int
    cost: float
    method: str = "trf"


@dataclass
class SuperquadricFit:
    """Output of :func:`fit_superquadric`.

    The recovered :class:`Superquadric` carries ``covariance`` populated
    from the Hessian inverse when ``return_covariance=True`` (default).
    ``diagnostics`` summarises fit quality.
    """

    superquadric: Superquadric
    diagnostics: FitDiagnostics


# ---------------------------------------------------------------------------
# Initialisation: PCA-based posed ellipsoid
# ---------------------------------------------------------------------------


def _pca_initial_pose(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(scale_init, R_init, t_init)`` from a PCA of points."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 4:
        raise ValueError(
            f"_pca_initial_pose: need >= 4 points, got {pts.shape[0]}."
        )
    t = pts.mean(axis=0)
    centred = pts - t
    # Symmetric eigendecomposition of the scatter matrix.
    cov = (centred.T @ centred) / max(len(centred) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending so axis 1 = longest extent.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Project points onto principal axes; use half the (5%, 95%) span as
    # the semi-axis init — robust to a few outliers without going full
    # robust statistics here.
    proj = centred @ eigvecs
    lo = np.percentile(proj, 5.0, axis=0)
    hi = np.percentile(proj, 95.0, axis=0)
    span = np.maximum(hi - lo, 4.0 * A_MIN)
    scale_init = np.clip(span * 0.5, A_MIN, A_MAX)
    # Right-handed rotation matrix (cols = principal axes).
    R_init = eigvecs
    if np.linalg.det(R_init) < 0.0:
        R_init[:, -1] *= -1.0
    return scale_init, R_init, t


# ---------------------------------------------------------------------------
# Solina-Bajcsy residual + Jacobian
# ---------------------------------------------------------------------------


# Sentinel for non-finite residuals.  When LM drifts into a bad region
# (e.g. ``|x/a|^(2/eps)`` overflowing float64), we substitute this
# magnitude so scipy gets a strong, signed gradient pulling it back.
# Choose large but well below ``inf`` so the trust-region step
# computation stays finite.
_RES_INF_FALLBACK: float = 1e12


def _solina_residuals(
    params: np.ndarray,
    points: np.ndarray,
    *,
    fix_epsilon: Optional[np.ndarray] = None,
    fix_rotation: bool = False,
) -> np.ndarray:
    """Per-point Solina-Bajcsy residual ``r_i = sqrt(V) * (F_i^eps1 - 1)``.

    ``params`` layout is the canonical 11-vector
    ``[log a1, log a2, log a3, eps1, eps2, omega_x, omega_y, omega_z,
       t_x, t_y, t_z]`` — even when ``fix_*`` overrides apply, we keep
    the layout stable so scipy gets a consistent ``x0``.

    Numerics: the inside-outside ``F`` can overflow when the SQ is
    pathologically small relative to a far-away point and ``eps`` is
    small.  We compute under ``np.errstate(over='ignore')`` and only
    replace explicit ``+/-inf`` / ``nan`` with a sentinel — the
    *finite* magnitude of the residual is preserved so the optimiser
    knows in which direction to retreat.  Capping the magnitude itself
    would create a degenerate global minimum at ``a -> A_MIN`` (the SQ
    collapses to a point and "hides" all outliers behind a saturated
    cost).
    """
    log_a = np.clip(params[0:3], math.log(A_MIN), math.log(A_MAX))
    a1, a2, a3 = np.exp(log_a)
    if fix_epsilon is not None:
        e1, e2 = float(fix_epsilon[0]), float(fix_epsilon[1])
    else:
        e1, e2 = float(np.clip(params[3], EPS_MIN, EPS_MAX)), float(
            np.clip(params[4], EPS_MIN, EPS_MAX)
        )
    if fix_rotation:
        R = np.eye(3)
    else:
        R = _so3_from_axis_angle(params[5:8])
    t = params[8:11]
    local = (points - t) @ R  # body frame
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x = local[:, 0] / a1
        y = local[:, 1] / a2
        z = local[:, 2] / a3
        cross = _abs_pow(x, 2.0 / e2) + _abs_pow(y, 2.0 / e2)
        F = np.power(np.maximum(cross, _TINY), e2 / e1) + _abs_pow(z, 2.0 / e1)
        F_eps1 = np.power(np.maximum(F, _TINY), e1)
        sqrt_vol = math.sqrt(max(a1 * a2 * a3, _TINY))
        resid = sqrt_vol * (F_eps1 - 1.0)
    # Only replace non-finite residuals; preserve magnitude otherwise so
    # the LM solver sees the correct gradient sign.
    resid = np.where(np.isfinite(resid), resid, _RES_INF_FALLBACK)
    return resid


# ---------------------------------------------------------------------------
# Fit driver
# ---------------------------------------------------------------------------


def fit_superquadric(
    points: np.ndarray,
    *,
    init: Optional[Superquadric] = None,
    fix_epsilon: Optional[np.ndarray] = None,
    fix_rotation: bool = False,
    robust: bool = True,
    f_scale: Optional[float] = None,
    max_nfev: int = 200,
    inlier_threshold: float = 0.1,
    max_points: int = 4096,
    rng: Optional[np.random.Generator] = None,
    return_covariance: bool = True,
    attributes: Optional[Dict[str, Any]] = None,
    parent_id: Optional[str] = None,
    id: Optional[str] = None,
) -> SuperquadricFit:
    """Fit a superquadric to ``points`` ``(N, 3)`` via Solina-Bajcsy LM.

    Parameters
    ----------
    points
        ``(N, 3)`` point cloud in world coordinates.  Must have ``N >= 11``
        for the 11-parameter problem to be over-determined.
    init
        Optional initial guess.  If ``None``, PCA on the points is used
        to set ``(scale, R, t)`` and ``epsilon = (1, 1)`` (ellipsoid).
    fix_epsilon
        If provided as ``(eps1, eps2)``, those values are held fixed
        (useful when the semantic class implies a known shape — e.g.
        ``(1.0, 0.1)`` for a tree trunk cylinder).
    fix_rotation
        If ``True``, freeze ``R = I`` and only fit ``(scale, eps, t)`` —
        useful for OSM-aligned axis-aligned building boxes.
    robust
        If ``True`` (default), use Cauchy loss in the LM solver to
        downweight outliers.  Set ``False`` for clean synthetic data.
    f_scale
        Cauchy loss scale (residual at which the loss starts to bend).
        If ``None``, defaults to the median absolute residual from a
        first non-robust pass.
    max_nfev
        Maximum function evaluations passed to scipy.
    inlier_threshold
        Threshold on ``|F^eps1 - 1|`` for the inlier-fraction
        diagnostic.
    max_points
        Subsample if more points than this.  Random uniform subsample
        with a fixed-seed ``rng`` (default ``np.random.default_rng(0)``).
    return_covariance
        If ``True``, populate the returned SQ's ``covariance`` field
        from the Jacobian at the optimum.
    attributes / parent_id / id
        Forwarded to the resulting :class:`Superquadric`.
    """
    from scipy.optimize import least_squares

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 11:
        raise ValueError(
            f"fit_superquadric: need >= 11 points for an 11-parameter fit, got {pts.shape[0]}."
        )

    rng = rng or np.random.default_rng(0)
    if pts.shape[0] > max_points:
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    if init is None:
        scale_init, R_init, t_init = _pca_initial_pose(pts)
        eps_init = np.array([1.0, 1.0], dtype=float)
        x0 = np.concatenate(
            [
                np.log(scale_init),
                eps_init,
                np.zeros(3),  # rotation as so(3); we re-rotate points below
                t_init,
            ]
        )
        # Pre-rotate points into the PCA frame so the optimiser starts
        # with omega = 0 (avoids the wrap-around at theta = pi).
        pts_init = (pts - t_init) @ R_init + t_init
        scratch_R0 = R_init
    else:
        x0 = init.to_fit_params()
        pts_init = pts
        scratch_R0 = np.eye(3)

    # First pass: non-robust LM, to get a reasonable f_scale for Cauchy.
    bounds_lo = np.array(
        [
            math.log(A_MIN), math.log(A_MIN), math.log(A_MIN),
            EPS_MIN, EPS_MIN,
            -math.pi, -math.pi, -math.pi,
            -np.inf, -np.inf, -np.inf,
        ]
    )
    bounds_hi = np.array(
        [
            math.log(A_MAX), math.log(A_MAX), math.log(A_MAX),
            EPS_MAX, EPS_MAX,
            math.pi, math.pi, math.pi,
            np.inf, np.inf, np.inf,
        ]
    )

    def _residuals(p: np.ndarray) -> np.ndarray:
        return _solina_residuals(
            p, pts_init, fix_epsilon=fix_epsilon, fix_rotation=fix_rotation
        )

    res_pre = least_squares(
        _residuals,
        x0=x0,
        bounds=(bounds_lo, bounds_hi),
        method="trf",
        max_nfev=max(40, max_nfev // 4),
        x_scale="jac",
    )

    if robust:
        if f_scale is not None:
            cauchy_scale = float(f_scale)
        else:
            # Robust scale heuristic: use the 25th percentile of |residuals|
            # rather than the median.  When the pre-pass is pulled by
            # outliers the *upper* half of |residuals| is dominated by
            # the misfit at outlier points; the lower half is closer to
            # the true inlier surface error.  Empirically this recovers
            # the surface even with 10-20% gross outliers.
            absr = np.abs(res_pre.fun)
            q25 = float(np.percentile(absr, 25.0))
            cauchy_scale = max(q25, 1e-3)
        # Run robust LM, then optionally one more refinement pass with a
        # tightened scale derived from the robust solution's *inlier*
        # residuals — this protects against the rare case where the
        # initial Cauchy scale is itself contaminated.
        res = least_squares(
            _residuals,
            x0=res_pre.x,
            bounds=(bounds_lo, bounds_hi),
            method="trf",
            max_nfev=max_nfev,
            loss="cauchy",
            f_scale=cauchy_scale,
            x_scale="jac",
        )
        if f_scale is None:
            absr2 = np.abs(res.fun)
            q25_2 = float(np.percentile(absr2, 25.0))
            tightened = max(q25_2, 1e-3)
            # Only re-run if the tightened scale is meaningfully smaller
            # than the previous one; otherwise we're already converged.
            if tightened < 0.5 * cauchy_scale:
                res = least_squares(
                    _residuals,
                    x0=res.x,
                    bounds=(bounds_lo, bounds_hi),
                    method="trf",
                    max_nfev=max(40, max_nfev // 2),
                    loss="cauchy",
                    f_scale=tightened,
                    x_scale="jac",
                )
    else:
        res = res_pre

    # Recover SQ in original world frame.  We pre-rotated points by
    # ``scratch_R0`` (PCA frame); the optimiser worked in that frame, so
    # we now bake the pre-rotation back into the SQ.
    sq_in_init_frame = Superquadric.from_fit_params(
        res.x,
        attributes=attributes,
        parent_id=parent_id,
        id=id or f"sq-{uuid.uuid4().hex[:12]}",
    )
    if init is None:
        # The pre-rotation was: pts_init = (pts - t_init) @ R_init + t_init
        # i.e. world->init transform is x' = R_init^T (x - t_init) + t_init.
        # The SQ recovered is in init-frame; transform it back to world:
        # world coords of a body point p_b are
        #   x = R_world @ p_b + t_world
        # where R_world = R_init @ R_fit and t_world chosen so that
        # the recovered shape coincides with the original points.
        # Easier path: fit-frame coords y = R_init^T (x - t_init) + t_init,
        # so x = R_init (y - t_init) + t_init = R_init @ y + (t_init - R_init @ t_init).
        # Apply this outer transform to the fit SQ:
        outer_R = scratch_R0
        outer_t = t_init - scratch_R0 @ t_init
        sq = sq_in_init_frame.transformed(outer_R, outer_t)
        # Preserve id + attributes after pose update.
        sq.id = sq_in_init_frame.id
        sq.parent_id = sq_in_init_frame.parent_id
        sq.attributes = dict(sq_in_init_frame.attributes)
    else:
        sq = sq_in_init_frame

    # Diagnostics: evaluate residuals on full (subsampled) point set
    # using the *world-frame* SQ and the *world* points.
    F = sq.inside_outside(pts)
    F_eps1 = np.power(np.maximum(F, _TINY), float(sq.epsilon[0]))
    surface_resid = np.abs(F_eps1 - 1.0)
    inlier_fraction = float(np.mean(surface_resid <= inlier_threshold))
    rms = float(np.sqrt(np.mean(res.fun ** 2)))

    # Covariance: sigma^2 (J^T J)^{-1} where J is the Jacobian at the
    # optimum and sigma^2 = SSR / (N - dof).  Falls back to ``None`` if
    # the Jacobian is rank-deficient or scipy didn't return one.
    cov: Optional[np.ndarray] = None
    if return_covariance and getattr(res, "jac", None) is not None:
        J = np.asarray(res.jac, dtype=float)
        n = J.shape[0]
        dof = J.shape[1]
        if n > dof:
            try:
                JTJ = J.T @ J
                # Tiny ridge for numerical stability.
                JTJ_reg = JTJ + 1e-9 * np.eye(JTJ.shape[0])
                JTJ_inv = np.linalg.inv(JTJ_reg)
                ssr = float(np.sum(res.fun ** 2))
                sigma2 = ssr / max(n - dof, 1)
                cov = sigma2 * JTJ_inv
            except np.linalg.LinAlgError:
                cov = None
    if cov is not None:
        sq.covariance = cov

    diag = FitDiagnostics(
        n_points=int(pts.shape[0]),
        rms_residual=rms,
        inlier_fraction=inlier_fraction,
        converged=bool(res.success),
        n_iter=int(getattr(res, "nfev", 0)),
        cost=float(res.cost),
        method="trf",
    )
    return SuperquadricFit(superquadric=sq, diagnostics=diag)


# ---------------------------------------------------------------------------
# Tree-of-superquadrics: trunk + crown decomposition
# ---------------------------------------------------------------------------


@dataclass
class TreeFit:
    """Output of :func:`fit_tree`: linked ``trunk`` and ``crown`` SQs."""

    trunk: Superquadric
    crown: Superquadric
    trunk_diagnostics: FitDiagnostics
    crown_diagnostics: FitDiagnostics


def _vertical_trunk_ransac(
    points: np.ndarray,
    *,
    vertical_axis: np.ndarray,
    radius_max: float,
    iterations: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust vertical-line fit: returns ``(centerline_xy, inlier_idx, outlier_idx)``.

    Uses a simple "best radius around a horizontal centroid candidate"
    voting scheme rather than RANSAC over 3-point line samples — for an
    a priori vertical axis this is far more stable than fitting a free
    line in 3D, and equivalent to projecting points to the horizontal
    plane and clustering tightly around a 2D centre.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    z_hat = np.asarray(vertical_axis, dtype=float).reshape(3)
    z_hat = z_hat / max(np.linalg.norm(z_hat), _TINY)

    # Project points to the horizontal plane perpendicular to z_hat.
    proj = pts - np.outer(pts @ z_hat, z_hat)  # (N, 3) in 3D, but in plane

    # Find the best horizontal centre by candidate sampling + tightness
    # vote.  We're approximating: trunk points cluster tightly in 2D
    # around a single centre; crown points are spread over a wider
    # radius and biased upward.
    n = pts.shape[0]
    best_inliers: Optional[np.ndarray] = None
    best_centre: Optional[np.ndarray] = None
    best_score: float = -np.inf
    sample_size = min(64, n)
    for _ in range(iterations):
        idx = rng.choice(n, size=sample_size, replace=False)
        cand_centre = proj[idx].mean(axis=0)
        d = np.linalg.norm(proj - cand_centre, axis=1)
        inliers = d <= radius_max
        score = float(np.sum(inliers))
        # Prefer candidates whose inliers also have *low z-spread relative
        # to their height range* — a real trunk extends vertically.
        if score > 0:
            inlier_pts = pts[inliers]
            z = inlier_pts @ z_hat
            z_range = z.max() - z.min()
            score += 0.1 * z_range  # weak vertical-extent bonus
        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_centre = cand_centre

    if best_inliers is None:
        # Degenerate: declare everything an outlier.
        return (
            np.zeros(3, dtype=float),
            np.zeros(0, dtype=np.int64),
            np.arange(n, dtype=np.int64),
        )

    in_idx = np.flatnonzero(best_inliers)
    out_idx = np.flatnonzero(~best_inliers)
    return best_centre, in_idx, out_idx


def fit_tree(
    points: np.ndarray,
    *,
    vertical_axis: Optional[np.ndarray] = None,
    trunk_radius_max: float = 0.4,
    trunk_radius_min: float = 0.04,
    ransac_iters: int = 64,
    crown_min_points: int = 16,
    rng: Optional[np.random.Generator] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> TreeFit:
    """Fit a 2-part superquadric tree (trunk cylinder + crown ellipsoid).

    The pipeline is: vertical-trunk RANSAC -> trunk SQ fit (cylinder
    template, ``eps = (1.0, 0.1)``, axis-aligned with ``vertical_axis``)
    -> remaining points fit as a crown ellipsoid (free pose,
    ``eps = (1, 1)``) with ``parent_id`` linking back to the trunk.

    Parameters
    ----------
    points
        ``(N, 3)`` cloud belonging to a single tree (caller should pre-
        cluster vegetation by instance — e.g. via Mask2Former or
        OpenMask3D — before calling this).
    vertical_axis
        World-frame "up" direction.  Defaults to ``[0, 0, 1]`` (ENU /
        Cesium-style).  Pass the local gravity vector if your trike's
        ``map`` frame is tilted.
    trunk_radius_max / trunk_radius_min
        Allowed trunk radius range in meters.
    ransac_iters
        Number of RANSAC voting iterations for the trunk centerline.
    crown_min_points
        Minimum points required to attempt a crown fit; if fewer
        remain after trunk extraction, the crown is fit on the full
        point cloud (with ``parent_id = trunk.id`` still set).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 32:
        raise ValueError(
            f"fit_tree: need >= 32 points to split into trunk+crown, got {pts.shape[0]}."
        )
    rng = rng or np.random.default_rng(0)
    z_hat = np.array([0.0, 0.0, 1.0]) if vertical_axis is None else np.asarray(
        vertical_axis, dtype=float
    ).reshape(3)
    z_hat = z_hat / max(np.linalg.norm(z_hat), _TINY)

    # Find trunk inliers via horizontal-cluster RANSAC.
    centre_xyz, in_idx, out_idx = _vertical_trunk_ransac(
        pts,
        vertical_axis=z_hat,
        radius_max=trunk_radius_max,
        iterations=ransac_iters,
        rng=rng,
    )

    trunk_pts = pts[in_idx] if in_idx.size >= 16 else pts.copy()
    crown_pts = pts[out_idx] if out_idx.size >= crown_min_points else pts.copy()

    # Trunk SQ: vertical cylinder template.  Use measured trunk extents
    # for an initial guess and freeze epsilon to (1.0, 0.1).
    z_proj = trunk_pts @ z_hat
    z_lo = float(np.percentile(z_proj, 5.0))
    z_hi = float(np.percentile(z_proj, 95.0))
    height = max(z_hi - z_lo, 4.0 * A_MIN)
    horiz = trunk_pts - np.outer(z_proj, z_hat)
    cxy = horiz.mean(axis=0)
    radii = np.linalg.norm(horiz - cxy, axis=1)
    radius = float(np.clip(np.percentile(radii, 90.0), trunk_radius_min, trunk_radius_max))

    base = cxy + z_lo * z_hat
    trunk_template = superquadric_cylinder(
        base=base,
        axis=z_hat,
        radius=radius,
        height=height,
        eps2=0.1,
        attributes={**(attributes or {}), "part": "trunk", "class": "tree_trunk"},
    )
    trunk_fit = fit_superquadric(
        trunk_pts,
        init=trunk_template,
        fix_epsilon=np.array([1.0, 0.1]),
        robust=True,
        attributes={**(attributes or {}), "part": "trunk", "class": "tree_trunk"},
    )
    trunk_sq = trunk_fit.superquadric

    # Crown SQ: free ellipsoid, parented at trunk.
    if crown_pts.shape[0] < 11:
        # Not enough points after trunk extraction — fall back to whole cloud.
        crown_pts = pts
    crown_fit = fit_superquadric(
        crown_pts,
        init=None,
        fix_epsilon=None,
        robust=True,
        attributes={**(attributes or {}), "part": "crown", "class": "tree_crown"},
        parent_id=trunk_sq.id,
    )
    crown_sq = crown_fit.superquadric

    return TreeFit(
        trunk=trunk_sq,
        crown=crown_sq,
        trunk_diagnostics=trunk_fit.diagnostics,
        crown_diagnostics=crown_fit.diagnostics,
    )

"""
sigma_m_p8.py — Q19 numerical check: sigma_m = 1/2 on P_8
==========================================================

Field-note context
------------------
Note 62b §4 (LQR-LQE duality)     --- defines sigma_m as the per-mode GP
                                      posterior variance (Laplace approx.
                                      to the MaxCal posterior) at the
                                      pre-stress fixed point h*.
Note 62c §3 step 1 (Q12 barriers) --- lists this computation as the first
                                      step in the proof strategy; points at
                                      "existing kernelcal data" on P_8.
Note 49 (Q12/PI^2/diffusion)      --- primal Riccati conjecture p_m = 2;
                                      dual by Note 62b:  sigma_m = 1/2.

Canonical P_8 setup
-------------------
Same as route3_conservation_test.py / P1 Experiments 1-4 / arXiv:2604.09745:

    N      = 8                         (path graph P_8)
    sigma2 = 1.0                       (observation noise variance)
    mu2    = 2.0                       (Lagrange multiplier)
    w_l    = 1                         (eigenvalue-blind weights)
    h_0    = 1                         (flat reference measure)

What this script does
---------------------
For each of the two sources (generic Gaussian MI, and the calibrated
Cowan-Farquhar mirror) it:

  1. Runs the MaxCal fixed-point iteration on P_8  -->  h*.
  2. Forms the Hessian in both h-coordinates and log-coordinates
     (the log-coord Hessian is the OU generator A for the LQR-LQE
     linearization, see kernelcal.control.analyzer).
  3. Solves the primal LQR CARE at h* with the plant-phenotyping
     convention  (Q = 1/2 I,  R_ctrl = R_ctrl_scale I,  B = I),
     reports p_m = diag(P) and its deviation from the conjectured 2.
  4. Solves the dual LQE CARE with the transposed system
     (C = I,  W = 1/2 I,  V = R_ctrl_scale I),  reports sigma_m = diag(Sigma)
     and its deviation from the conjectured 1/2.
  5. Checks the LQR-LQE duality P * Sigma = I numerically by computing
     the Frobenius norm of  P * Sigma - I.

Duality caveat
--------------
P * Sigma = I holds only when the primal and dual control-cost scales
are chosen so the mode-wise Riccati products collapse to unity.  For
the self-dual choice R_ctrl_scale = 1 this does not hold identically
for arbitrary sources; we also scan R_ctrl_scale to locate the scale
at which || P Sigma - I ||_F is minimized, which is the operationally
meaningful Note-62b duality point (the LQR-LQE R / V pairing that
makes the primal and dual Riccati solutions inverse at the MaxCal
fixed point).

Expected numerical outcome (locked in by tests/test_sigma_m_p8.py)
------------------------------------------------------------------
On P_8 with the canonical arXiv parameters the Gaussian MI source
produces h* uniform at ~0.1547, diag(H_log) uniform at ~-0.1368,
p_m = sigma_m uniform at ~0.5834 under R = 1.  The conjecture
sigma_m = 1/2 therefore holds to ~17 % relative; the exact LQR-LQE
duality R* ~ 4.22 is the operational value where P * Sigma ~ I.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import solve_continuous_are

from kernelcal.spectral import (
    CowanFarquharSource,
    GaussianMISource,
    SpectralGraph,
    SpectralKernelDynamics,
    hessian_matrix,
)
from kernelcal.control import (
    fit_riccati_analytic,
    riccati_conjecture_test,
)


# ---------------------------------------------------------------------------
# Canonical P_8 parameters (see module docstring for provenance).
# ---------------------------------------------------------------------------

N_MODES_DEFAULT = 8
SIGMA2_DEFAULT = 1.0
MU2_DEFAULT = 2.0
Q_FISHER_RAO = 0.5
R_CTRL_SCALE_DEFAULT = 1.0
P_M_TARGET = 2.0
SIGMA_M_TARGET = 0.5
DEFAULT_TOLERANCE = 0.10   # 10 % relative, matches CAREAnalyzerConfig default


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SigmaMResult:
    """Numerical outcome of one (source, R_ctrl_scale) Q19 evaluation."""

    # Identity
    source_label: str
    N: int
    R_ctrl_scale: float

    # Fixed point
    h_star: np.ndarray
    fixed_point_converged: bool
    field_residual: float

    # Linearization
    eigenvalues: np.ndarray          # Laplacian eigenvalues lambda_l
    H_log_diag: np.ndarray            # diag(H_ell) (negative at stable max)
    A_log: np.ndarray                 # (N, N) OU generator in log-coords

    # Primal LQR
    P: np.ndarray
    p_m: np.ndarray
    p_m_max_abs_relative: float
    p_m_passes: bool

    # Dual LQE
    Sigma: np.ndarray
    sigma_m: np.ndarray
    sigma_m_max_abs_relative: float
    sigma_m_passes: bool

    # Duality
    p_times_sigma_diag: np.ndarray    # entrywise p_m * sigma_m (should be ~ 1 at the duality point)
    p_times_sigma_mean: float
    duality_residual_fro: float       # || P Sigma - I ||_F

    def summary(self) -> str:
        """Human-readable block suitable for printing inline."""
        lines = [
            f"----- Q19 :: {self.source_label}  (R_ctrl_scale = {self.R_ctrl_scale:g}) -----",
            f"  N                     : {self.N}",
            f"  fixed point converged : {self.fixed_point_converged}",
            f"  field residual        : {self.field_residual:.3e}",
            f"  h*                    : {np.array2string(self.h_star, precision=4)}",
            f"  diag(H_log)           : {np.array2string(self.H_log_diag, precision=4)}",
            "",
            "  primal  p_m           :  " + np.array2string(self.p_m, precision=4),
            f"  target                :  p_m = {P_M_TARGET:g}",
            f"  max |p_m - 2| / 2     :  {self.p_m_max_abs_relative:.4f}  "
            f"({'PASS' if self.p_m_passes else 'FAIL'} at tol={DEFAULT_TOLERANCE:g})",
            "",
            "  dual    sigma_m       :  " + np.array2string(self.sigma_m, precision=4),
            f"  target                :  sigma_m = {SIGMA_M_TARGET:g}",
            f"  max |sigma_m-1/2|/(1/2):  {self.sigma_m_max_abs_relative:.4f}  "
            f"({'PASS' if self.sigma_m_passes else 'FAIL'} at tol={DEFAULT_TOLERANCE:g})",
            "",
            "  duality p_m * sigma_m :  " + np.array2string(self.p_times_sigma_diag, precision=4),
            f"  mean(p_m * sigma_m)   :  {self.p_times_sigma_mean:.4f}  (target 1)",
            f"  || P Sigma - I ||_F   :  {self.duality_residual_fro:.3e}",
            "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------


def _log_coord_hessian(h_star: np.ndarray, source_jacobian: np.ndarray) -> np.ndarray:
    """Hessian of the MaxCal functional in log-coordinates.

    H_{ell,lm}  =  h*_l * h*_m * H_{h,lm}
    with        H_{h,lm}  =  -delta_{lm} / h*_l  -  dT_l / dh_m | h*

    At a critical point the off-diagonal term from the chain rule vanishes
    (because dJ/dh = 0 there), so the transformation is exactly the outer
    product h* h*^T * H_h elementwise.
    """
    H_h = hessian_matrix(h_star, source_jacobian)
    return (np.outer(h_star, h_star)) * H_h


def _ou_drift_from_hessian(H_log: np.ndarray) -> np.ndarray:
    """OU mean-reversion generator A from the log-coord Hessian.

    At a strict local max of the MaxCal action the gradient flow in log-
    coordinates is  d ell/dt = H_ell * ell,  with H_ell negative definite.
    This negative-definite matrix is exactly the A fed into the CARE so
    that (A, I) is stabilizable and (A, sqrt(Q)) is detectable.
    """
    A = np.asarray(H_log, dtype=float)
    A = 0.5 * (A + A.T)
    # Guard: if any diagonal is non-negative we are not at a stable max and
    # the CARE will not behave; surface this as an error rather than silently
    # returning a garbage P.
    if not np.all(np.diag(A) < 0):
        raise RuntimeError(
            "Log-coord Hessian has non-negative diagonal; this is not a "
            "stable MaxCal fixed point. Check the source parameters."
        )
    return A


def _solve_primal_dual(
    A: np.ndarray,
    R_ctrl_scale: float,
    q_scale: float = Q_FISHER_RAO,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the primal LQR and dual LQE CAREs with B = C = I."""
    n = A.shape[0]
    B = np.eye(n)
    Q = q_scale * np.eye(n)
    R = R_ctrl_scale * np.eye(n)

    P = fit_riccati_analytic(A, B, Q, R)

    # Dual LQE: A Sigma + Sigma A^T - Sigma C^T V^-1 C Sigma + W = 0
    # scipy.solve_continuous_are solves A^T X + X A - X B R^-1 B^T X + Q = 0
    # so we feed (A = A^T, B = C^T = I, Q = W = q_scale I, R = V = R_ctrl_scale I)
    # and obtain Sigma.
    Sigma = solve_continuous_are(A.T, np.eye(n), Q, R)
    Sigma = 0.5 * (Sigma + Sigma.T)
    return P, Sigma


# ---------------------------------------------------------------------------
# Source builders
# ---------------------------------------------------------------------------


def _build_gaussian_mi_source(
    graph: SpectralGraph,
    sigma2: float = SIGMA2_DEFAULT,
    mu2: float = MU2_DEFAULT,
    eigenvalue_weighted: bool = False,
) -> GaussianMISource:
    """The arXiv P1 / P_8 canonical source."""
    eig = graph.eigenvalues if eigenvalue_weighted else None
    return GaussianMISource(sigma2=sigma2, mu2=mu2, eigenvalues=eig)


def _build_cowan_farquhar_calibrated(
    graph: SpectralGraph, h_star_target: np.ndarray
) -> CowanFarquharSource:
    """Calibrated instrumentation source.

    Built so that T_l(h_star_target) = 1/8 - lambda_l, which is the
    Section IV-J condition under which the p_m = 2 Riccati conjecture
    holds (by construction, modulo least-squares residual).
    """
    return CowanFarquharSource.calibrated(
        eigenvalues=graph.eigenvalues,
        h_star_target=h_star_target,
        eigenvalue_weighted=False,
    )


# ---------------------------------------------------------------------------
# Q19 evaluator
# ---------------------------------------------------------------------------


def evaluate_sigma_m(
    graph: SpectralGraph,
    source,
    *,
    source_label: str,
    R_ctrl_scale: float = R_CTRL_SCALE_DEFAULT,
    tolerance: float = DEFAULT_TOLERANCE,
    h_init: Optional[np.ndarray] = None,
) -> SigmaMResult:
    """Run the full Q19 numerical check on (graph, source)."""

    dyn = SpectralKernelDynamics(graph, source)
    fp = dyn.fixed_point_iteration(h_init=h_init)
    h_star = fp.h_star

    J = source.jacobian(h_star)
    H_log = _log_coord_hessian(h_star, J)
    A = _ou_drift_from_hessian(H_log)

    P, Sigma = _solve_primal_dual(A, R_ctrl_scale)

    p_m_test = riccati_conjecture_test(P, p_m_target=P_M_TARGET, tolerance=tolerance)
    sigma_m = np.diag(Sigma).copy()
    sigma_dev = (sigma_m - SIGMA_M_TARGET) / SIGMA_M_TARGET
    sigma_max_rel = float(np.max(np.abs(sigma_dev)))
    sigma_passes = bool(sigma_max_rel <= tolerance)

    p_times_sigma = p_m_test.p_m * sigma_m
    duality_err = float(np.linalg.norm(P @ Sigma - np.eye(graph.N), ord="fro"))

    field_res = 0.0
    if fp.residual_history:
        field_res = float(fp.residual_history[-1])

    return SigmaMResult(
        source_label=source_label,
        N=graph.N,
        R_ctrl_scale=float(R_ctrl_scale),
        h_star=h_star,
        fixed_point_converged=bool(fp.converged),
        field_residual=field_res,
        eigenvalues=graph.eigenvalues.copy(),
        H_log_diag=np.diag(H_log).copy(),
        A_log=A,
        P=P,
        p_m=p_m_test.p_m,
        p_m_max_abs_relative=float(p_m_test.max_abs_relative),
        p_m_passes=bool(p_m_test.passes),
        Sigma=Sigma,
        sigma_m=sigma_m,
        sigma_m_max_abs_relative=sigma_max_rel,
        sigma_m_passes=sigma_passes,
        p_times_sigma_diag=p_times_sigma,
        p_times_sigma_mean=float(np.mean(p_times_sigma)),
        duality_residual_fro=duality_err,
    )


# ---------------------------------------------------------------------------
# Duality scale sweep
# ---------------------------------------------------------------------------


def find_duality_scale(
    graph: SpectralGraph,
    source,
    *,
    r_grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Locate R_ctrl_scale minimizing || P Sigma - I ||_F.

    The LQR-LQE duality of Note 62b is a claim about a specific
    parameter pairing, not about an arbitrary (Q, R, W, V) tuple. The
    duality point is the scale at which mode-wise p_m * sigma_m collapses
    to 1. This helper locates that point by 1-D search.
    """
    if r_grid is None:
        r_grid = np.geomspace(1e-3, 1e3, 49)
    best_r = float(r_grid[0])
    best_err = float("inf")
    for r in r_grid:
        try:
            res = evaluate_sigma_m(graph, source, source_label="_scan",
                                   R_ctrl_scale=float(r))
        except Exception:
            continue
        if res.duality_residual_fro < best_err:
            best_err = res.duality_residual_fro
            best_r = float(r)
    return best_r, best_err


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_q19_report(verbose: bool = True) -> Dict[str, SigmaMResult]:
    """Run the full Q19 report on P_8 with both sources.

    Returns a dict keyed by source label, each mapped to its
    SigmaMResult for the canonical R_ctrl_scale = 1 run.
    """
    graph = SpectralGraph.path_graph(N_MODES_DEFAULT)
    src_gmi = _build_gaussian_mi_source(graph)

    # First pass: pin down h* on GMI to calibrate CF against it.
    gmi_preview = SpectralKernelDynamics(graph, src_gmi).fixed_point_iteration()
    src_cf = _build_cowan_farquhar_calibrated(graph, gmi_preview.h_star)

    results: Dict[str, SigmaMResult] = {}

    for label, source in (
        ("Gaussian MI (sigma^2=1, mu_2=2, w_l=1)", src_gmi),
        ("Cowan-Farquhar (calibrated to GMI fixed point)", src_cf),
    ):
        res = evaluate_sigma_m(
            graph, source,
            source_label=label,
            R_ctrl_scale=R_CTRL_SCALE_DEFAULT,
        )
        results[label] = res
        if verbose:
            print(res.summary())

    # Duality scan to surface the operational p * sigma = 1 point.
    if verbose:
        print("----- duality scale scan (min || P Sigma - I ||_F) -----")
        for label, source in (
            ("Gaussian MI", src_gmi),
            ("Cowan-Farquhar", src_cf),
        ):
            r_best, err_best = find_duality_scale(graph, source)
            print(
                f"  {label:<32s}  R_ctrl_scale* = {r_best:8.4f}   "
                f"|| P Sigma - I ||_F  =  {err_best:.3e}"
            )
        print("")

    return results


def main() -> None:
    """Console-script entry point for the Q19 numerical check.

    Wired to ``kernelcal-sigma-m-p8`` in ``pyproject.toml``.
    """
    print("Q19 :: sigma_m = 1/2 numerical check on P_8")
    print("=" * 60)
    run_q19_report(verbose=True)


if __name__ == "__main__":
    main()

"""
kernelcal.control.care
======================
Core CARE solvers, OU identification, and Riccati-conjecture diagnostics.

All public functions are pure functions of numpy arrays; no mutable state.
The high-level per-campaign coordinator lives in `analyzer.py`.

Notation
--------
A, B        : drift and actuation matrices of the linearized kernel OU
Q, R_ctrl   : CARE state cost and control cost
P           : Riccati gain matrix (the solution)
h*          : MaxCal fixed-point spectral weights
delta_ell   : log-coordinate perturbation log h - log h*
C_obs       : observation matrix (from the GP's ARD trajectory)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import least_squares

K_BOLTZMANN = 1.380649e-23


# ===========================================================================
# Landauer lower bound on R_ctrl
# ===========================================================================

def landauer_R_lower_bound(
    delta_I_k: float,
    temperature_kelvin: float = 300.0,
) -> float:
    """Thermodynamic lower bound on the control cost R_ctrl.

    The MaxCal-governed kernel change has an irreducible information cost
    W >= k_B T * delta_I_k (Landauer bound, Section IV-J).  Used as a floor
    for the CARE control cost so the resulting P is physically realizable.

    Parameters
    ----------
    delta_I_k : float
        Information change delta I_k induced by the control action
        (nats of kernel MI change per step).
    temperature_kelvin : float
        Ambient temperature (default 300 K).

    Returns
    -------
    Scalar lower bound on R_ctrl in joules.
    """
    if delta_I_k < 0.0:
        raise ValueError("delta_I_k must be non-negative.")
    return float(K_BOLTZMANN * temperature_kelvin * delta_I_k)


# ===========================================================================
# CARE residual (objective) and closed-form solver
# ===========================================================================

def care_residual(
    P: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R_ctrl: np.ndarray,
) -> np.ndarray:
    """Evaluate the CARE residual matrix
    A^T P + P A - P B R_ctrl^{-1} B^T P + Q.

    Zero at the exact CARE solution.  Frobenius norm of this residual is
    the objective minimized by `fit_riccati_residual`.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R_ctrl = np.asarray(R_ctrl, dtype=float)
    P = np.asarray(P, dtype=float)
    R_inv = np.linalg.inv(R_ctrl)
    return A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + Q


def fit_riccati_analytic(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R_ctrl: np.ndarray,
) -> np.ndarray:
    """Structural CARE solve when (A, B, Q, R_ctrl) are trusted.

    Wraps `scipy.linalg.solve_continuous_are`.  Requires (A, B) stabilizable
    and (A, Q^{1/2}) detectable.  Returns the unique symmetric PSD
    stabilizing solution.

    Use this path when A has been estimated cleanly from a long pre-stress
    trajectory.  For noisy A (e.g. during stress onset), prefer
    `fit_riccati_residual` with a warm start.
    """
    P = solve_continuous_are(np.asarray(A, dtype=float),
                             np.asarray(B, dtype=float),
                             np.asarray(Q, dtype=float),
                             np.asarray(R_ctrl, dtype=float))
    P = 0.5 * (P + P.T)
    return np.asarray(P, dtype=float)


def _symm_from_lower(n: int, vec: np.ndarray) -> np.ndarray:
    """Reconstruct a symmetric matrix from its lower-triangular packing."""
    P = np.zeros((n, n), dtype=float)
    idx = np.tril_indices(n)
    P[idx] = vec
    P = P + P.T - np.diag(np.diag(P))
    return P


def _lower_pack(P: np.ndarray) -> np.ndarray:
    idx = np.tril_indices(P.shape[0])
    return P[idx]


@dataclass
class RiccatiAnalysisResult:
    """Outcome of a Riccati estimation."""
    P: np.ndarray                 # (N, N) symmetric Riccati gain
    residual_frobenius: float     # ||A^T P + P A - P B R^-1 B^T P + Q||_F
    method: str                   # 'analytic' | 'residual'
    converged: bool
    diagonal: np.ndarray          # diag(P), per-mode leakage cost
    off_diagonal_mass: float      # ||P - diag(P)||_F
    coupling_entropy: float       # S_coup(P)
    eigvals: np.ndarray           # eigvals of P (all >= 0 if PSD)


def fit_riccati_residual(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R_ctrl: np.ndarray,
    P_init: Optional[np.ndarray] = None,
    enforce_psd: bool = True,
    max_iter: int = 2000,
    tol: float = 1e-10,
) -> RiccatiAnalysisResult:
    """CARE-residual minimization estimator of P.

    Direct implementation of Eq. (eq:care_fit) in the plant-phenotyping
    paper.  Solves Levenberg-Marquardt least-squares on the vectorized
    residual of A^T P + P A - P B R^-1 B^T P + Q over symmetric P,
    optionally projected to PSD via eigen-truncation.

    Uses `scipy.optimize.least_squares` with method 'lm', which handles the
    rank-deficient Gauss-Newton Hessian at the CARE solution far better
    than quasi-Newton on the scalarized loss.  When a reliable warm start
    (e.g. the analytic solution from the previous rotation) is not
    provided, the function bootstraps by trying the analytic solve first
    and falling back to Q as a last resort.

    This is the estimator of choice for rotation-by-rotation analysis of
    the kernelcal trajectory when A is estimated with noise.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R_ctrl = np.asarray(R_ctrl, dtype=float)
    n = A.shape[0]
    R_inv = np.linalg.inv(R_ctrl)

    if P_init is None:
        try:
            P_init = solve_continuous_are(A, B, Q, R_ctrl)
        except Exception:
            P_init = 0.5 * (Q + Q.T)
    P_init = 0.5 * (P_init + P_init.T)
    x0 = _lower_pack(P_init)

    def residual_vec(x: np.ndarray) -> np.ndarray:
        P = _symm_from_lower(n, x)
        res = A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + Q
        return res.ravel()

    result = least_squares(
        residual_vec, x0, method="lm",
        max_nfev=max_iter, xtol=tol, ftol=tol, gtol=tol,
    )
    P = _symm_from_lower(n, result.x)

    if enforce_psd:
        w, V = np.linalg.eigh(P)
        w = np.clip(w, 0.0, None)
        P = (V * w) @ V.T
        P = 0.5 * (P + P.T)

    residual_mat = care_residual(P, A, B, Q, R_ctrl)
    residual_fro = float(np.linalg.norm(residual_mat, ord="fro"))
    converged = residual_fro < 1e-6 or (
        bool(result.success) and residual_fro < 1e-3
    )

    return RiccatiAnalysisResult(
        P=P,
        residual_frobenius=residual_fro,
        method="residual",
        converged=converged,
        diagonal=np.diag(P).copy(),
        off_diagonal_mass=off_diagonal_frobenius(P),
        coupling_entropy=coupling_entropy_off_diagonal(P),
        eigvals=np.linalg.eigvalsh(P),
    )


# ===========================================================================
# Off-diagonal biosignature summaries
# ===========================================================================

def off_diagonal_frobenius(P: np.ndarray) -> float:
    """||P - diag(P)||_F  --- total off-diagonal coupling mass of P.

    Grows under stress-onset as the controller's coupling-maintenance cost
    rises (Section IV-J, Table: Predicted Riccati-gain signatures).
    """
    P = np.asarray(P, dtype=float)
    P_off = P - np.diag(np.diag(P))
    return float(np.linalg.norm(P_off, ord="fro"))


def coupling_entropy_off_diagonal(P: np.ndarray) -> float:
    """Coupling entropy S_coup of the off-diagonal block of P.

    Normalizes |P_{lm}| (l != m) row-wise to a probability distribution and
    returns the mean row entropy.  Zero when P is purely diagonal;
    log(N-1) when couplings are uniformly distributed.

    This is the CARE-side analogue of the source-Jacobian coupling entropy
    already provided by kernelcal.spectral.dynamics.coupling_entropy,
    applied here to the estimated Riccati gain P rather than to dT/dh.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if n < 2:
        return 0.0
    entropies: List[float] = []
    for l in range(n):
        row = np.abs(P[l].copy())
        row[l] = 0.0
        s = row.sum()
        if s < 1e-15:
            entropies.append(float(np.log(n - 1)))
        else:
            p = row / s
            p = np.where(p > 0, p, 1.0)
            entropies.append(float(-np.sum(p * np.log(p))))
    return float(np.mean(entropies))


# ===========================================================================
# Riccati conjecture p_m = 2 test (Eq. riccati_conjecture in the paper)
# ===========================================================================

@dataclass
class RiccatiConjectureTest:
    """Per-mode test of the p_m = 2 conjecture under a Cowan-Farquhar
    source (Eq. eq:riccati_conjecture of the plant-phenotyping paper).
    """
    p_m: np.ndarray                # diag(P), per-mode Riccati gain
    p_m_target: float              # 2.0 by default
    deviation: np.ndarray          # p_m - p_m_target
    relative_deviation: np.ndarray # (p_m - target) / target
    max_abs_relative: float
    passes: bool                   # True if all modes within tolerance
    tolerance: float


def riccati_conjecture_test(
    P: np.ndarray,
    p_m_target: float = 2.0,
    tolerance: float = 0.10,
) -> RiccatiConjectureTest:
    """Check whether diagonal Riccati gains match the p_m = 2 prediction
    of the mode-separable Cowan-Farquhar regime.

    Under the pre-stress phase (G = G_opt) the conjecture predicts
    p_m = g_{mm} * h_m^* = 2 for every mode m.  This test reports per-mode
    deviations and a pass/fail flag at the given fractional tolerance.
    Any claim that "p_m = 2" is satisfied on a rotation should cite
    `passes` together with `max_abs_relative`.
    """
    P = np.asarray(P, dtype=float)
    p_m = np.diag(P).copy()
    dev = p_m - p_m_target
    rel = dev / p_m_target if p_m_target != 0.0 else dev
    max_abs_rel = float(np.max(np.abs(rel)))
    return RiccatiConjectureTest(
        p_m=p_m,
        p_m_target=p_m_target,
        deviation=dev,
        relative_deviation=rel,
        max_abs_relative=max_abs_rel,
        passes=bool(max_abs_rel <= tolerance),
        tolerance=tolerance,
    )


# ===========================================================================
# OU identification in log-coordinates
# ===========================================================================

@dataclass
class OUIdentificationResult:
    """Result of OU mean-reversion identification.

    Fits log-coordinate kernel perturbations delta_ell to
        d(delta_ell)/dt = A * delta_ell + B * u + w
    by least squares across the provided trajectory.
    """
    A: np.ndarray                  # (N, N) estimated mean-reversion
    B: Optional[np.ndarray]        # (N, K) estimated actuation (if u provided)
    residual_covariance: np.ndarray  # empirical process noise cov W_hat
    fit_rms: float                 # RMS regression residual
    n_samples: int


def estimate_A_log_OU(
    log_h_trajectory: np.ndarray,
    dt: float,
    control: Optional[np.ndarray] = None,
    diagonal_only: bool = False,
    regularization: float = 1e-8,
) -> OUIdentificationResult:
    """Identify OU mean-reversion A (and optionally B) from a log-coordinate
    kernel trajectory produced over many rotations.

    Parameters
    ----------
    log_h_trajectory : (T, N) array of log h_r^* perturbations about h^*.
        Rows must be zero-mean; subtract the pre-stress mean before calling.
    dt : float
        Time between rows (seconds).
    control : (T, K) array of control inputs aligned with the rows
        (optional; if provided, B is estimated jointly).
    diagonal_only : if True, constrain A to be diagonal (mode-separable
        assumption from Section IV-J of the paper).
    regularization : Tikhonov ridge on the regression.

    Returns
    -------
    OUIdentificationResult
    """
    X = np.asarray(log_h_trajectory, dtype=float)
    if X.ndim != 2:
        raise ValueError("log_h_trajectory must be 2-D (T, N).")
    T, N = X.shape
    if T < 3:
        raise ValueError("Need at least 3 samples to identify A.")

    # Discrete-time fit: X_{t+1} = M X_t + G U_t + noise, then A = (M - I)/dt.
    # This is far more robust to measurement noise than a finite-difference
    # fit on dX/dt, because the integrated signal-to-noise ratio over a
    # single step stays bounded as dt -> 0.
    X0 = X[:-1]
    X1 = X[1:]
    if control is not None:
        U = np.asarray(control, dtype=float)
        if U.shape[0] != T:
            raise ValueError("control must align with log_h_trajectory rows.")
        K = U.shape[1]
        U0 = U[:-1]
        Phi = np.hstack([X0, U0])
    else:
        U = None
        K = 0
        Phi = X0

    n_feat = Phi.shape[1]

    if diagonal_only:
        A = np.zeros((N, N), dtype=float)
        B = np.zeros((N, K), dtype=float) if K > 0 else None
        residuals = np.zeros_like(X1)
        for m in range(N):
            if K > 0:
                phi_m = np.hstack([X0[:, m:m + 1], U0])
            else:
                phi_m = X0[:, m:m + 1]
            target_m = X1[:, m]
            G_m = phi_m.T @ phi_m + regularization * np.eye(phi_m.shape[1])
            theta = np.linalg.solve(G_m, phi_m.T @ target_m)
            M_mm = theta[0]
            A[m, m] = (M_mm - 1.0) / dt
            if K > 0:
                B[m, :] = theta[1:] / dt
            residuals[:, m] = target_m - phi_m @ theta
    else:
        G_mat = Phi.T @ Phi + regularization * np.eye(n_feat)
        Theta = np.linalg.solve(G_mat, Phi.T @ X1).T  # (N, N+K)
        M = Theta[:, :N]
        A = (M - np.eye(N)) / dt
        B = Theta[:, N:] / dt if K > 0 else None
        residuals = X1 - Phi @ Theta.T

    W_hat = (residuals.T @ residuals) / max(T - 2, 1)
    fit_rms = float(np.sqrt(np.mean(residuals * residuals)))

    return OUIdentificationResult(
        A=A,
        B=B,
        residual_covariance=W_hat,
        fit_rms=fit_rms,
        n_samples=T,
    )


# ===========================================================================
# ARD trajectory -> empirical observation matrix C_obs
# ===========================================================================

def ard_to_observation_matrix(
    ard_lengthscales: np.ndarray,
    mode_basis: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Convert GP ARD length-scales into an empirical observation matrix.

    Implements the structural identification of Section IV-J of the
    plant-phenotyping paper: "The ARD weight trajectory discovered by the
    GP IS the empirical estimate of C_obs across the campaign."

    Two modes of operation:
    * If `mode_basis` is None, return a diagonal M x M matrix
      diag(sigma_d^{-2}) --- each band d contributes independently.
    * If `mode_basis` is a (M, N) projector from input bands to Laplacian
      eigenmodes of the population graph, return the M x N matrix
      diag(sigma_d^{-2}) @ mode_basis.  The rows express each band's
      projection onto spectral modes, weighted by relevance.

    Parameters
    ----------
    ard_lengthscales : (M,) array of per-band GP length-scales ell_d.
        The relevance weight is sigma_d^{-2} = 1 / ell_d^2 by convention.
    mode_basis : optional (M, N) projector onto population spectral modes.
    normalize : if True, scale rows to unit L1 so rows are band-relevance
        probability distributions over modes.

    Returns
    -------
    C_obs : (M, M) if mode_basis is None, else (M, N).
    """
    ell = np.asarray(ard_lengthscales, dtype=float)
    if ell.ndim != 1:
        raise ValueError("ard_lengthscales must be 1-D of length M.")
    w = 1.0 / np.maximum(ell * ell, 1e-12)  # relevance per band

    if mode_basis is None:
        C_obs = np.diag(w)
    else:
        B = np.asarray(mode_basis, dtype=float)
        if B.shape[0] != w.shape[0]:
            raise ValueError(
                "mode_basis must have shape (M, N); first dim must match "
                "the number of ARD length-scales."
            )
        C_obs = np.diag(w) @ B

    if normalize:
        row_sums = np.sum(np.abs(C_obs), axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        C_obs = C_obs / row_sums
    return np.asarray(C_obs, dtype=float)

"""
Core MaxCal functional: path entropy and Lagrange multiplier fitting.

Maps to Equation (1) of the paper:

    S[p] = −Σ_γ p[γ] ln(p[γ] / q[γ])

subject to  ⟨f_i[γ]⟩_p = F_i.

The MaxCal solution is

    p[γ] ∝ q[γ] exp(−Σ_i λ_i f_i[γ])

where the Lagrange multipliers λ are found by minimising the dual:

    L(λ) = log Z(λ) − λ · F,    Z(λ) = Σ_γ q[γ] exp(−λ · f(γ))

In the discrete, finite-state approximation used here, "trajectories" are
individual locations (or kernel snapshots), and p[γ] is a categorical
distribution over those states.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


# ---------------------------------------------------------------------------
# Path entropy (relative entropy / negative KL from q to p)
# ---------------------------------------------------------------------------

def path_entropy(log_p: np.ndarray, log_q: np.ndarray) -> float:
    """Relative path entropy  S[p] = −Σ p[γ] ln(p[γ]/q[γ]) = −KL(p‖q).

    Accepts log-probability vectors for numerical stability.

    Parameters
    ----------
    log_p : array of shape (N,)
        Log-probabilities of the current distribution (need not be normalised).
    log_q : array of shape (N,)
        Log-probabilities of the reference distribution.

    Returns
    -------
    float
        Path entropy value (≥ 0, maximised when p = q).
    """
    log_p = np.asarray(log_p, dtype=float)
    log_q = np.asarray(log_q, dtype=float)
    log_p = log_p - logsumexp(log_p)
    log_q = log_q - logsumexp(log_q)
    p = np.exp(log_p)
    return float(-np.sum(p * (log_p - log_q)))


# ---------------------------------------------------------------------------
# Partition function and dual objective
# ---------------------------------------------------------------------------

def _log_partition(lambdas: np.ndarray,
                   log_q: np.ndarray,
                   F_matrix: np.ndarray) -> float:
    """log Z(λ) = log Σ_γ q[γ] exp(−λ · f(γ)).

    Parameters
    ----------
    lambdas : (C,) Lagrange multiplier vector.
    log_q   : (N,) log reference weights.
    F_matrix: (N, C) constraint feature matrix  f_i(γ_j).
    """
    log_unnorm = log_q - F_matrix @ lambdas
    return float(logsumexp(log_unnorm))


def lagrange_dual(lambdas: np.ndarray,
                  log_q: np.ndarray,
                  F_matrix: np.ndarray,
                  constraint_values: np.ndarray) -> float:
    """Dual objective L(λ) = log Z(λ) − λ · F.

    Minimising L(λ) over λ recovers the MaxCal distribution.
    """
    logZ = _log_partition(lambdas, log_q, F_matrix)
    return logZ - float(lambdas @ constraint_values)


def _lagrange_dual_grad(lambdas: np.ndarray,
                        log_q: np.ndarray,
                        F_matrix: np.ndarray,
                        constraint_values: np.ndarray) -> np.ndarray:
    """Gradient of L(λ) w.r.t. λ.

    ∂L/∂λ_i = ⟨f_i⟩_{p(λ)} − F_i
    """
    log_unnorm = log_q - F_matrix @ lambdas
    log_p = log_unnorm - logsumexp(log_unnorm)
    p = np.exp(log_p)
    expected_f = F_matrix.T @ p  # (C,)
    return expected_f - constraint_values


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_lagrange_multipliers(
    log_q: np.ndarray,
    F_matrix: np.ndarray,
    constraint_values: np.ndarray,
    lambda0: Optional[np.ndarray] = None,
    method: str = "L-BFGS-B",
    tol: float = 1e-9,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, bool]:
    """Find Lagrange multipliers that satisfy the MaxCal constraints.

    Minimises the dual L(λ) = log Z(λ) − λ · F via scipy.optimize.

    Parameters
    ----------
    log_q : (N,)
        Log of reference weights q[γ].
    F_matrix : (N, C)
        Feature matrix; F_matrix[i, j] = f_j(γ_i).
    constraint_values : (C,)
        Target expected values F_j = ⟨f_j⟩_p.
    lambda0 : (C,) optional
        Initial Lagrange multipliers (default: zeros).
    method : str
        scipy optimizer (default L-BFGS-B).
    tol : float
        Optimiser convergence tolerance.
    max_iter : int
        Maximum optimiser iterations.

    Returns
    -------
    lambdas : (C,) fitted Lagrange multipliers.
    success : bool  — True if optimisation converged.
    """
    log_q = np.asarray(log_q, dtype=float)
    F_matrix = np.asarray(F_matrix, dtype=float)
    constraint_values = np.asarray(constraint_values, dtype=float)

    n_constraints = F_matrix.shape[1]
    if lambda0 is None:
        lambda0 = np.zeros(n_constraints)

    result = minimize(
        fun=lagrange_dual,
        x0=lambda0,
        args=(log_q, F_matrix, constraint_values),
        jac=_lagrange_dual_grad,
        method=method,
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )
    return result.x, result.success


# ---------------------------------------------------------------------------
# MaxCal log-weights
# ---------------------------------------------------------------------------

def maxcal_log_weights(
    log_q: np.ndarray,
    lambdas: np.ndarray,
    F_matrix: np.ndarray,
) -> np.ndarray:
    """Return normalised log-probabilities of the MaxCal distribution.

    log p[γ] = log q[γ] − λ · f(γ) − log Z(λ)

    Parameters
    ----------
    log_q    : (N,) log reference weights.
    lambdas  : (C,) fitted Lagrange multipliers.
    F_matrix : (N, C) feature matrix.

    Returns
    -------
    log_p : (N,) normalised log-probabilities.
    """
    log_q = np.asarray(log_q, dtype=float)
    log_unnorm = log_q - F_matrix @ lambdas
    return log_unnorm - logsumexp(log_unnorm)


# ---------------------------------------------------------------------------
# Constraint diagnostics
# ---------------------------------------------------------------------------

def constraint_residuals(
    log_p: np.ndarray,
    F_matrix: np.ndarray,
    constraint_values: np.ndarray,
) -> np.ndarray:
    """⟨f_i⟩_p − F_i for each constraint.  Should be near zero after fitting."""
    p = np.exp(log_p - logsumexp(log_p))
    expected = F_matrix.T @ p
    return expected - np.asarray(constraint_values)

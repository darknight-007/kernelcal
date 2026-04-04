"""
Hellinger kernel and the NTK–Hellinger comparison (Conjecture 3).

The Hellinger kernel is the canonical kernel whose geometry respects
sufficient statistics (Chentsov 1982):

    k_H(p, q) = ∫ √(p(x) q(x)) dν(x)  =  1 − H²(p,q)/2

where H²(p,q) = ∫ (√p − √q)² dν is the squared Hellinger distance.

Conjecture 3 of the paper proposes that the NTK of wide networks converges
during training to the Hellinger kernel on the induced distribution over the
representation space.  This module provides:

  - Hellinger kernel matrices from discrete probability vectors (softmax outputs)
  - The HS distance between an empirical NTK and the Hellinger kernel baseline
  - A convergence test (does the NTK approach the Hellinger kernel over time?)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..kernel.space import hilbert_schmidt_distance, normalize_kernel, project_to_psd


# ---------------------------------------------------------------------------
# Hellinger kernel from probability distributions
# ---------------------------------------------------------------------------

def hellinger_distance(p: np.ndarray, q: np.ndarray,
                       eps: float = 1e-12) -> float:
    """Hellinger distance H(p, q) = (1/√2) ‖√p − √q‖_2.

    Parameters
    ----------
    p, q : (D,) discrete probability vectors (need not be normalised).
    eps  : small value to avoid sqrt(0).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (np.sum(p) + eps)
    q = q / (np.sum(q) + eps)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p + eps) - np.sqrt(q + eps)) ** 2)))


def hellinger_kernel_value(p: np.ndarray, q: np.ndarray,
                           eps: float = 1e-12) -> float:
    """Scalar Hellinger kernel k_H(p,q) = ∫√(pq)dν  (Bhattacharyya coefficient).

    For discrete distributions: k_H(p,q) = Σ_x √(p(x)·q(x)).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (np.sum(p) + eps)
    q = q / (np.sum(q) + eps)
    return float(np.sum(np.sqrt(np.maximum(p, 0) * np.maximum(q, 0))))


def hellinger_kernel_matrix(
    distributions: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute the Hellinger kernel matrix from a set of probability vectors.

    Parameters
    ----------
    distributions : (N, D) array where each row is a probability distribution
        (e.g., softmax outputs from a neural network over D classes).
    eps : small constant for numerical stability.

    Returns
    -------
    K_H : (N, N) PSD Hellinger kernel matrix.
        K_H[i,j] = Σ_d √(p_i(d) · p_j(d))
    """
    D = np.asarray(distributions, dtype=float)
    row_sums = D.sum(axis=1, keepdims=True)
    D = D / np.where(row_sums == 0, 1.0, row_sums)  # normalise rows
    sqrt_D = np.sqrt(np.maximum(D, 0))
    K_H = sqrt_D @ sqrt_D.T
    return project_to_psd(K_H)


# ---------------------------------------------------------------------------
# NTK–Hellinger comparison (Conjecture 3)
# ---------------------------------------------------------------------------

def compare_ntk_to_hellinger(
    ntk_matrix: np.ndarray,
    distributions: np.ndarray,
    normalise: bool = True,
) -> dict:
    """Compute the HS distance between an empirical NTK and the Hellinger kernel.

    This operationalises Conjecture 3: as training progresses and width → ∞,
    this distance should decrease toward zero.

    Parameters
    ----------
    ntk_matrix : (N, N) empirical NTK at the current training step.
    distributions : (N, D) softmax / probability outputs of the model
        on the same N probe inputs.
    normalise : bool
        If True, normalise both matrices to unit trace before comparison
        (removes scale ambiguity).

    Returns
    -------
    dict with keys:
      'hs_distance'     — HS distance between NTK and Hellinger kernel.
      'ntk_trace'       — trace of the NTK matrix.
      'hellinger_trace' — trace of the Hellinger kernel matrix.
      'correlation'     — Frobenius inner product after normalisation.
    """
    K_ntk = np.asarray(ntk_matrix, dtype=float)
    K_H = hellinger_kernel_matrix(distributions)

    if normalise:
        tr_ntk = np.trace(K_ntk)
        tr_H = np.trace(K_H)
        K_ntk_n = K_ntk / (tr_ntk + 1e-12)
        K_H_n = K_H / (tr_H + 1e-12)
    else:
        K_ntk_n, K_H_n = K_ntk, K_H

    d_hs = hilbert_schmidt_distance(K_ntk_n, K_H_n)
    correlation = float(np.sum(K_ntk_n * K_H_n))

    return {
        "hs_distance": d_hs,
        "ntk_trace": float(np.trace(K_ntk)),
        "hellinger_trace": float(np.trace(K_H)),
        "correlation": correlation,
        "ntk_matrix": K_ntk,
        "hellinger_matrix": K_H,
    }


def ntk_hellinger_convergence_series(
    ntk_matrices: List[np.ndarray],
    distributions_series: List[np.ndarray],
    normalise: bool = True,
) -> np.ndarray:
    """HS distance between NTK and Hellinger kernel at each training step.

    Parameters
    ----------
    ntk_matrices : list of (N, N) NTK snapshots over training.
    distributions_series : list of (N, D) softmax outputs at each snapshot.

    Returns
    -------
    distances : (T,) array of HS distances, one per snapshot.
    """
    return np.array([
        compare_ntk_to_hellinger(K, D, normalise=normalise)["hs_distance"]
        for K, D in zip(ntk_matrices, distributions_series)
    ])


# ---------------------------------------------------------------------------
# Fisher-Rao metric from Hellinger geometry
# ---------------------------------------------------------------------------

def fisher_rao_distance(p: np.ndarray, q: np.ndarray,
                        eps: float = 1e-12) -> float:
    """Geodesic distance on the statistical manifold via Hellinger embedding.

    d_FR(p,q) = 2 · arccos(k_H(p,q)) = 2 · arccos(Σ √(p_i q_i))
    """
    bc = hellinger_kernel_value(p, q, eps=eps)
    bc = np.clip(bc, -1.0, 1.0)
    return float(2 * np.arccos(bc))

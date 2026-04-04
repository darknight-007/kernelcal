"""
RKHS complexity and the assembly theory interface.

Maps to Section 6 of the paper ("Relation to Assembly Theory") and the bound:

    a(x) ≥ c · ‖k_x‖_H + O(1)

where a(x) is the assembly index of object x and ‖k_x‖_H is the RKHS norm
of the feature associated with x.

In the discrete, finite-sample setting:
  - The "kernel" for a set of N objects is the N×N Gram matrix K.
  - The RKHS norm of a single object x_i is √K[i,i]  (self-similarity).
  - The spectral complexity of K measures the full representational cost of
    the ensemble.

For DeepGIS, "objects" are image tiles or detected features (SAM masks,
Grounding DINO bounding boxes).  Their embeddings define a kernel matrix, and
RKHS norms provide per-tile complexity scores that feed into the World Sampler
as an assembly-theory-motivated reward signal.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..kernel.space import (
    kernel_from_embeddings,
    project_to_psd,
    normalize_kernel,
    hilbert_schmidt_norm,
)


# ---------------------------------------------------------------------------
# RKHS norm of individual objects
# ---------------------------------------------------------------------------

def rkhs_norm(K: np.ndarray) -> np.ndarray:
    """Per-object RKHS norms from a Gram matrix.

    ‖φ(x_i)‖_H = √K[i,i]

    Parameters
    ----------
    K : (N, N) PSD kernel (Gram) matrix.

    Returns
    -------
    norms : (N,) array of non-negative RKHS norms, one per object.
    """
    K = np.asarray(K, dtype=float)
    return np.sqrt(np.maximum(np.diag(K), 0.0))


# ---------------------------------------------------------------------------
# Spectral complexity measures of the full kernel
# ---------------------------------------------------------------------------

def spectral_complexity(K: np.ndarray, eps: float = 1e-10) -> float:
    """Entropy of the normalised eigenvalue spectrum of K.

    S = −Σ_i (λ_i / tr(K)) ln(λ_i / tr(K))

    A uniform spectrum (all eigenvalues equal) gives maximum complexity.
    A rank-1 kernel gives zero complexity.  This is the von Neumann entropy
    of the density matrix K / tr(K).

    Parameters
    ----------
    K : (N, N) PSD kernel matrix.
    eps : floor for eigenvalues to avoid log(0).

    Returns
    -------
    float in [0, ln N].
    """
    K = np.asarray(K, dtype=float)
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.maximum(eigvals, 0.0)
    total = np.sum(eigvals)
    if total < eps:
        return 0.0
    p = eigvals / total
    p = np.where(p > eps, p, eps)
    return float(-np.sum(p * np.log(p)))


def effective_dimension(K: np.ndarray, threshold: float = 0.99) -> int:
    """Number of eigenvalues needed to explain *threshold* of total variance.

    This is the 'statistical dimension' of the RKHS, which lower-bounds the
    assembly complexity of the ensemble.
    """
    K = np.asarray(K, dtype=float)
    eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
    eigvals = np.maximum(eigvals, 0.0)
    cumvar = np.cumsum(eigvals) / (np.sum(eigvals) + 1e-12)
    return int(np.searchsorted(cumvar, threshold)) + 1


def nuclear_norm(K: np.ndarray) -> float:
    """Nuclear (trace) norm = sum of singular values.

    For PSD matrices this equals tr(K).  Used as a proxy for RKHS complexity
    of the whole ensemble.
    """
    return float(np.trace(np.asarray(K, dtype=float)))


# ---------------------------------------------------------------------------
# Per-tile complexity map for DeepGIS
# ---------------------------------------------------------------------------

def complexity_map(
    embeddings_per_tile: List[np.ndarray],
    kernel_fn=None,
    normalise: bool = True,
) -> np.ndarray:
    """Compute a scalar complexity score for each geospatial tile.

    Parameters
    ----------
    embeddings_per_tile : list of (M_i, D) arrays.
        Each element contains the feature embeddings of all detected objects
        (SAM masks, Grounding DINO boxes, etc.) within one tile.
    kernel_fn : callable (M, D) → (M, M) or None.
        If None, the normalised cosine kernel is used.
    normalise : bool
        If True, scores are linearly rescaled to [0, 1].

    Returns
    -------
    scores : (T,) array of complexity scores, one per tile.
        Higher score = more complex representational structure = higher
        assembly index lower bound.
    """
    scores = []
    for emb in embeddings_per_tile:
        emb = np.asarray(emb, dtype=float)
        if len(emb) == 0:
            scores.append(0.0)
            continue
        if emb.ndim == 1:
            emb = emb[None, :]
        K = kernel_from_embeddings(emb, kernel_fn)
        scores.append(spectral_complexity(K))

    scores = np.array(scores, dtype=float)
    if normalise and scores.max() > 0:
        scores = scores / scores.max()
    return scores


# ---------------------------------------------------------------------------
# Assembly theory lower bound
# ---------------------------------------------------------------------------

def assembly_index_lower_bound(
    K: np.ndarray,
    c: float = 1.0,
) -> np.ndarray:
    """Per-object lower bound on assembly index:  a(x_i) ≥ c · ‖k_{x_i}‖_H.

    Parameters
    ----------
    K : (N, N) Gram matrix.
    c : scaling constant (theory provides existence of c > 0;
        empirically calibrated from domain data).

    Returns
    -------
    bounds : (N,) array.
    """
    return c * rkhs_norm(K)


# ---------------------------------------------------------------------------
# Reward signal for the MaxCal World Sampler
# ---------------------------------------------------------------------------

def assembly_reward_signal(
    complexity_scores: np.ndarray,
    coverage_counts: Optional[np.ndarray] = None,
    coverage_weight: float = 0.3,
    complexity_weight: float = 0.7,
    eps: float = 1e-8,
) -> np.ndarray:
    """Combine complexity and coverage into a per-location reward for MaxCalSampler.

    Locations with high assembly complexity and low coverage are rewarded most,
    directing the sampler toward information-rich, undersampled regions.

    Parameters
    ----------
    complexity_scores : (N,) RKHS complexity scores per location.
    coverage_counts   : (N,) number of times each location has been visited.
                        If None, no coverage correction is applied.
    coverage_weight   : weight on inverse-coverage term (0 to 1).
    complexity_weight : weight on complexity term (0 to 1).

    Returns
    -------
    reward : (N,) non-negative reward vector, normalised to [0, 1].
    """
    c = np.asarray(complexity_scores, dtype=float)
    c = c / (c.max() + eps)

    if coverage_counts is not None:
        cov = np.asarray(coverage_counts, dtype=float)
        inv_cov = 1.0 / (1.0 + cov)
        inv_cov = inv_cov / (inv_cov.max() + eps)
        reward = complexity_weight * c + coverage_weight * inv_cov
    else:
        reward = c

    reward = reward / (reward.max() + eps)
    return reward

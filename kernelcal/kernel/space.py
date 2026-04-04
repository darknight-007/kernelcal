"""
Kernel space operations: Hilbert-Schmidt geometry, PSD cone, and kernel algebra.

Maps to Section 3 of the paper ("The Space of Kernels").  The Hilbert-Schmidt
norm on integral operators T_k : L²(ν) → L²(ν) is approximated by the
Frobenius norm of the n×n Gram matrix, normalised by √n so that the metric is
independent of sample size.

    d_HS(k₁, k₂) ≈ ‖K₁ − K₂‖_F / √n
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def hilbert_schmidt_norm(K: ArrayLike) -> float:
    """Normalised Hilbert-Schmidt (Frobenius) norm of a kernel matrix.

    Approximates ‖T_k‖_HS = (∫∫ k(x,x')² dν dν')^{1/2} from an n×n Gram
    matrix.  Division by √n makes the value scale-invariant in sample size.
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    return np.linalg.norm(K, "fro") / np.sqrt(n)


def hilbert_schmidt_distance(K1: ArrayLike, K2: ArrayLike) -> float:
    """Hilbert-Schmidt distance between two kernel matrices of the same size.

    d_HS(k₁, k₂) = ‖T_{k₁} − T_{k₂}‖_HS
    """
    K1 = np.asarray(K1, dtype=float)
    K2 = np.asarray(K2, dtype=float)
    if K1.shape != K2.shape:
        raise ValueError(
            f"Kernel matrices must have identical shape; got {K1.shape} and {K2.shape}"
        )
    return hilbert_schmidt_norm(K1 - K2)


# ---------------------------------------------------------------------------
# PSD cone utilities
# ---------------------------------------------------------------------------

def is_psd(K: ArrayLike, tol: float = 1e-8) -> bool:
    """Return True if K is symmetric and positive semi-definite up to tol."""
    K = np.asarray(K, dtype=float)
    if not np.allclose(K, K.T, atol=tol):
        return False
    eigvals = np.linalg.eigvalsh(K)
    return bool(np.all(eigvals >= -tol))


def project_to_psd(K: ArrayLike, epsilon: float = 0.0) -> np.ndarray:
    """Project K onto the PSD cone by clipping negative eigenvalues.

    Returns the nearest PSD matrix (in Frobenius norm) by setting all
    eigenvalues below *epsilon* to *epsilon*.  Set epsilon > 0 for strict
    positive definiteness.
    """
    K = np.asarray(K, dtype=float)
    K = (K + K.T) / 2.0  # enforce symmetry first
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.maximum(eigvals, epsilon)
    return (eigvecs * eigvals) @ eigvecs.T


# ---------------------------------------------------------------------------
# Kernel algebra  (K is a convex cone closed under + and pointwise ·)
# ---------------------------------------------------------------------------

def kernel_sum(K1: ArrayLike, K2: ArrayLike,
               alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    """Conic combination α·k₁ + β·k₂  (remains in K for α,β ≥ 0)."""
    K1, K2 = np.asarray(K1, dtype=float), np.asarray(K2, dtype=float)
    if alpha < 0 or beta < 0:
        raise ValueError("Coefficients must be non-negative to stay in the PSD cone.")
    return alpha * K1 + beta * K2


def kernel_product(K1: ArrayLike, K2: ArrayLike) -> np.ndarray:
    """Hadamard (elementwise) product k₁ · k₂  (remains in K by Schur's theorem)."""
    return np.asarray(K1, dtype=float) * np.asarray(K2, dtype=float)


def normalize_kernel(K: ArrayLike) -> np.ndarray:
    """Normalise so that K[i,i] = 1 for all i  (cosine-style normalisation)."""
    K = np.asarray(K, dtype=float)
    diag = np.sqrt(np.diag(K))
    diag = np.where(diag == 0, 1.0, diag)
    return K / np.outer(diag, diag)


# ---------------------------------------------------------------------------
# Kernel construction helpers
# ---------------------------------------------------------------------------

def rbf_kernel(X: ArrayLike, Y: ArrayLike | None = None,
               length_scale: float = 1.0) -> np.ndarray:
    """Radial basis function (Gaussian) kernel k(x,y) = exp(−‖x−y‖²/(2ℓ²))."""
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    sq_dists = (
        np.sum(X ** 2, axis=1, keepdims=True)
        + np.sum(Y ** 2, axis=1)
        - 2 * X @ Y.T
    )
    return np.exp(-sq_dists / (2 * length_scale ** 2))


def linear_kernel(X: ArrayLike, Y: ArrayLike | None = None) -> np.ndarray:
    """Linear kernel k(x,y) = xᵀy."""
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    return X @ Y.T


def polynomial_kernel(X: ArrayLike, Y: ArrayLike | None = None,
                      degree: int = 3, coef0: float = 1.0) -> np.ndarray:
    """Polynomial kernel k(x,y) = (xᵀy + c)^d."""
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    return (X @ Y.T + coef0) ** degree


def kernel_from_embeddings(embeddings: ArrayLike,
                           kernel_fn=None) -> np.ndarray:
    """Compute a kernel matrix from a set of feature embeddings.

    If kernel_fn is None, uses the normalised linear (cosine) kernel.
    """
    emb = np.asarray(embeddings, dtype=float)
    if kernel_fn is None:
        K = linear_kernel(emb)
        K = normalize_kernel(K)
    else:
        K = kernel_fn(emb)
    return project_to_psd(K)

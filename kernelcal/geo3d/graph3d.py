"""Build graph Laplacians from 3D points."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def subsample_points(points_xyz: np.ndarray, max_points: int, seed: int | None = 0) -> np.ndarray:
    """Return at most ``max_points`` rows, sampled uniformly without replacement."""
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3).")
    if max_points < 2:
        raise ValueError("max_points must be at least 2.")
    n = pts.shape[0]
    if n <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return pts[idx]


def knn_symmetric_adjacency(points_xyz: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """Symmetric k-NN Gaussian adjacency matrix."""
    pts = np.asarray(points_xyz, dtype=float)
    n = pts.shape[0]
    if n < 2:
        raise ValueError("Need at least two points.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    k_eff = min(k + 1, n)
    tree = cKDTree(pts)
    W = np.zeros((n, n), dtype=float)
    two_sigma2 = 2.0 * sigma**2

    for i in range(n):
        dists, idx = tree.query(pts[i], k=k_eff)
        if k_eff == 1:
            continue
        for d, j in zip(dists[1:], idx[1:]):
            j = int(j)
            w = float(np.exp(-(d * d) / two_sigma2))
            W[i, j] = max(W[i, j], w)
            W[j, i] = max(W[j, i], w)

    np.fill_diagonal(W, 0.0)
    return W


def adjacency_to_laplacian(W: np.ndarray) -> np.ndarray:
    """Combinatorial Laplacian ``L = D - W``."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square.")
    d = W.sum(axis=1)
    return np.diag(d) - W

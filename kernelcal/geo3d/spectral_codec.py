"""Spectral truncation codec for geometry-induced graph kernels."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np

from ..spectral import SpectralGraph
from .graph3d import adjacency_to_laplacian, knn_symmetric_adjacency, subsample_points


@dataclass
class CompressedSpectralKernel:
    """Low-rank representation of ``K_h = Phi diag(h) Phi^T``."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    h: np.ndarray
    meta: dict[str, Any]

    def to_bytes(self) -> bytes:
        """Serialize to compressed NPZ payload."""
        buf = BytesIO()
        np.savez_compressed(
            buf,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            h=self.h,
            meta=np.array(self.meta, dtype=object),
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "CompressedSpectralKernel":
        """Deserialize NPZ payload."""
        z = np.load(BytesIO(data), allow_pickle=True)
        raw = z["meta"].item() if "meta" in z.files else {}
        meta = raw if isinstance(raw, dict) else {}
        return cls(
            eigenvalues=np.asarray(z["eigenvalues"]),
            eigenvectors=np.asarray(z["eigenvectors"]),
            h=np.asarray(z["h"]),
            meta=meta,
        )


def compress_point_cloud(
    points_xyz: np.ndarray,
    *,
    max_points: int = 512,
    k_neighbors: int = 8,
    sigma: float = 1.0,
    n_modes: int = 32,
    heat_tau: float | None = 1.0,
    seed: int | None = 0,
) -> CompressedSpectralKernel:
    """Compress a point cloud into truncated graph spectral kernel coordinates."""
    pts = subsample_points(points_xyz, max_points=max_points, seed=seed)
    W = knn_symmetric_adjacency(pts, k=k_neighbors, sigma=sigma)
    L = adjacency_to_laplacian(W)
    sg = SpectralGraph(L)

    k = min(int(n_modes), sg.N)
    if k < 1:
        raise ValueError("n_modes must be at least 1.")
    lam = sg.eigenvalues[:k]
    Phi = sg.eigenvectors[:, :k]
    if heat_tau is not None and heat_tau > 0:
        h = np.exp(-lam * float(heat_tau))
    else:
        h = np.ones(k)
    h = np.maximum(h, 1e-12)

    meta = {
        "kind": "point_cloud",
        "max_points": int(max_points),
        "k_neighbors": int(k_neighbors),
        "sigma": float(sigma),
        "n_modes": int(k),
        "heat_tau": None if heat_tau is None else float(heat_tau),
    }
    return CompressedSpectralKernel(eigenvalues=lam, eigenvectors=Phi, h=h, meta=meta)


def decompress_to_kernel(c: CompressedSpectralKernel) -> np.ndarray:
    """Reconstruct dense kernel matrix from compressed representation."""
    return (c.eigenvectors * c.h) @ c.eigenvectors.T

"""
Spectral graph construction and Laplacian eigendecomposition.

Maps to Section 3.1 of the paper ("The Spectral Reduction").  Every
Laplacian-commuting (graph-filter) kernel on a finite connected graph is
parameterised by its spectral transfer function

    h_t = (h_t(λ_l))  ∈ ℝ_{>0}^N

via  K_h = Φ diag(h) Φᵀ,  where L = Φ Λ Φᵀ is the Laplacian eigendecomposition.

The primary class here, SpectralGraph, wraps that decomposition and exposes
the per-mode kernel matrix so that all downstream objects can work purely in
the N-dimensional spectral coordinate space.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SpectralGraph:
    """Finite connected graph with pre-computed Laplacian eigendecomposition.

    Parameters
    ----------
    laplacian : (N, N) array
        Symmetric graph Laplacian L.  Connectivity is not verified; caller
        is responsible for passing a valid Laplacian of a connected graph.

    Attributes
    ----------
    N : int
        Number of nodes.
    eigenvalues : (N,) ndarray
        Laplacian eigenvalues λ_0 ≤ λ_1 ≤ … ≤ λ_{N-1}, sorted ascending.
    eigenvectors : (N, N) ndarray
        Orthonormal eigenvectors Φ; column l is the l-th eigenvector.
    laplacian : (N, N) ndarray
        Original Laplacian matrix.
    """

    def __init__(self, laplacian: np.ndarray) -> None:
        L = np.asarray(laplacian, dtype=float)
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError("Laplacian must be a square 2-D array.")
        L = (L + L.T) / 2.0  # enforce symmetry
        eigvals, eigvecs = np.linalg.eigh(L)
        self.laplacian = L
        self.N: int = L.shape[0]
        self.eigenvalues: np.ndarray = eigvals          # (N,) ascending
        self.eigenvectors: np.ndarray = eigvecs         # (N, N)

    # ------------------------------------------------------------------
    # Kernel construction
    # ------------------------------------------------------------------

    def kernel_matrix(self, h: np.ndarray) -> np.ndarray:
        """Build the kernel matrix K_h = Φ diag(h) Φᵀ.

        Parameters
        ----------
        h : (N,) array
            Spectral transfer function values h(λ_l) > 0.

        Returns
        -------
        (N, N) kernel matrix.
        """
        h = np.asarray(h, dtype=float)
        if h.shape != (self.N,):
            raise ValueError(f"h must have shape ({self.N},); got {h.shape}.")
        if np.any(h <= 0):
            raise ValueError("All spectral weights h(λ_l) must be strictly positive.")
        Phi = self.eigenvectors
        return (Phi * h) @ Phi.T

    def heat_kernel_weights(self, tau: float) -> np.ndarray:
        """Spectral weights of the heat kernel h_τ(λ_l) = exp(−λ_l τ).

        The l=0 mode (λ_0 = 0) gives weight 1; larger eigenvalues decay
        faster with diffusion time τ > 0.
        """
        if tau <= 0:
            raise ValueError("Diffusion time tau must be positive.")
        return np.exp(-self.eigenvalues * tau)

    def flat_weights(self) -> np.ndarray:
        """Reference (flat) spectral weights h_0(λ_l) = 1 for all l."""
        return np.ones(self.N)

    # ------------------------------------------------------------------
    # Fiedler value
    # ------------------------------------------------------------------

    @property
    def fiedler_value(self) -> float:
        """Second-smallest Laplacian eigenvalue λ_1 (algebraic connectivity)."""
        return float(self.eigenvalues[1])

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def path_graph(cls, N: int) -> "SpectralGraph":
        """Construct the path graph P_N and return its SpectralGraph.

        Laplacian eigenvalues are  λ_l = 2(1 − cos(πl/N)),  l = 0, …, N−1.
        """
        if N < 2:
            raise ValueError("Path graph requires at least 2 nodes.")
        L = np.zeros((N, N))
        for i in range(N - 1):
            L[i, i] += 1
            L[i + 1, i + 1] += 1
            L[i, i + 1] -= 1
            L[i + 1, i] -= 1
        return cls(L)

    @classmethod
    def cycle_graph(cls, N: int) -> "SpectralGraph":
        """Construct the cycle graph C_N and return its SpectralGraph."""
        if N < 3:
            raise ValueError("Cycle graph requires at least 3 nodes.")
        L = np.zeros((N, N))
        for i in range(N):
            j = (i + 1) % N
            L[i, i] += 1
            L[j, j] += 1
            L[i, j] -= 1
            L[j, i] -= 1
        return cls(L)

    @classmethod
    def path_graph_with_weak_edge(
        cls, N: int, i: int, j: int, epsilon: float
    ) -> "SpectralGraph":
        """Path graph P_N with edge (i, j) replaced by a weighted edge ε ∈ [0, 1].

        Setting ε → 0 progressively disconnects the graph, driving λ_1 → 0.
        Used for the phase-transition experiment (Remark 8 / Q6).
        """
        g = cls.path_graph(N)
        L = g.laplacian.copy()
        # remove the full-weight contribution of (i, j)
        L[i, i] -= 1
        L[j, j] -= 1
        L[i, j] += 1
        L[j, i] += 1
        # add back the ε-weighted contribution
        L[i, i] += epsilon
        L[j, j] += epsilon
        L[i, j] -= epsilon
        L[j, i] -= epsilon
        return cls(L)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SpectralGraph(N={self.N}, "
            f"λ_1={self.fiedler_value:.4f}, "
            f"λ_max={self.eigenvalues[-1]:.4f})"
        )

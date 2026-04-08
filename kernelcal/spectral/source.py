"""
Source functional T_l[h] for spectral kernel dynamics.

Maps to Definition 1 and Remark 4 of the companion paper.  Provides a
concrete, fully specified instantiation of the abstract source functional
𝒯_l[h] used throughout Section 3.

The Gaussian mutual-information source
---------------------------------------
When K_h = Φ diag(h) Φᵀ is the covariance of a zero-mean Gaussian
GP with observation noise σ², the per-mode mutual information is

    I_k^(l) = ½ w_l log(1 + h(λ_l)/σ²)

where the spectral weight w_l is either:

  * w_l = 1  (eigenvalue-blind, default when eigenvalues=None): all modes
    contribute equally.  Satisfies A1–A3; used for Exps 1–5.

  * w_l = λ_l  (eigenvalue-aware, when eigenvalues are supplied): the
    information contributed by mode l scales with its frequency.  The Fiedler
    mode λ_1 → 0 contributes vanishingly little as the graph approaches
    disconnection, making h* non-uniform and enabling H[h*], Δ, and Sc to
    vary during the phase-transition sweep (Exp 6 / Q6).

Taking μ₁ = μ₃ = 0 (Landauer and KL terms absent) and μ₂ > 0, the
source functional reduces to

    𝒯_l[h] = μ₂ · w_l / (2(σ² + h(λ_l)))

Assumptions A1–A3 are preserved in both modes:
  (A1) Mode-separable: 𝒯_l depends only on h(λ_l) (and the fixed w_l).
  (A2) μ₁ = μ₃ = 0.
  (A3) ∂²I/∂h² = −w_l/(2(σ²+h)²) < 0 for w_l > 0  (MI concave in h).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class GaussianMISource:
    """Concrete source functional derived from Gaussian mutual information.

    Parameters
    ----------
    sigma2 : float
        Observation noise variance σ² > 0.
    mu2 : float
        Lagrange multiplier μ₂ for the mutual-information constraint.
    eigenvalues : (N,) array-like or None
        Graph Laplacian eigenvalues λ_0 ≤ λ_1 ≤ … ≤ λ_{N-1}.
        When supplied, the per-mode MI is scaled by λ_l so that the
        source "feels" the graph spectrum.  Leave as None (default) for
        the eigenvalue-blind version used in Exps 1–5.

    All methods operate on the spectral weight vector
    h = (h(λ_0), …, h(λ_{N-1})) ∈ ℝ_{>0}^N and return per-mode arrays.
    """

    def __init__(
        self,
        sigma2: float = 1.0,
        mu2: float = 1.0,
        eigenvalues: Optional[np.ndarray] = None,
    ) -> None:
        if sigma2 <= 0:
            raise ValueError("sigma2 must be strictly positive.")
        self.sigma2 = float(sigma2)
        self.mu2 = float(mu2)
        if eigenvalues is not None:
            self._w = np.asarray(eigenvalues, dtype=float)
        else:
            self._w = None  # weights resolved lazily per-call as ones(N)

    def _weights(self, N: int) -> np.ndarray:
        """Return the per-mode weights w_l (shape (N,))."""
        if self._w is not None:
            if self._w.shape != (N,):
                raise ValueError(
                    "eigenvalues length does not match h length: "
                    f"got {self._w.shape[0]} vs {N}."
                )
            return self._w
        return np.ones(N, dtype=float)

    # ------------------------------------------------------------------
    # Per-mode MI and its derivatives
    # ------------------------------------------------------------------

    def mutual_information(self, h: np.ndarray) -> np.ndarray:
        """Per-mode MI:  I_k^(l) = ½ w_l log(1 + h(λ_l)/σ²).

        Parameters
        ----------
        h : (N,) spectral weights.

        Returns
        -------
        (N,) array of per-mode MI values.
        """
        h = np.asarray(h, dtype=float)
        return 0.5 * self._weights(h.size) * np.log1p(h / self.sigma2)

    def dI_dh(self, h: np.ndarray) -> np.ndarray:
        """First derivative: ∂I_k^(l)/∂h(λ_l) = w_l/(2(σ²+h(λ_l)))."""
        h = np.asarray(h, dtype=float)
        return self._weights(h.size) / (2.0 * (self.sigma2 + h))

    def d2I_dh2(self, h: np.ndarray) -> np.ndarray:
        """Second derivative: ∂²I_k^(l)/∂h(λ_l)² = −w_l/(2(σ²+h(λ_l))²).

        Strictly negative whenever w_l > 0, confirming assumption (A3):
        MI is concave in spectral weight.
        """
        h = np.asarray(h, dtype=float)
        return -self._weights(h.size) / (2.0 * (self.sigma2 + h) ** 2)

    # ------------------------------------------------------------------
    # Source functional and its derivative (used by SpectralKernelDynamics)
    # ------------------------------------------------------------------

    def T(self, h: np.ndarray) -> np.ndarray:
        """Source functional 𝒯_l[h] = μ₂ · w_l / (2(σ² + h(λ_l))).

        Parameters
        ----------
        h : (N,) spectral weights.

        Returns
        -------
        (N,) array of per-mode source values.
        """
        return self.mu2 * self.dI_dh(h)

    def dT_dh(self, h: np.ndarray) -> np.ndarray:
        """Diagonal of the source Jacobian: ∂𝒯_l/∂h(λ_l) = μ₂ · d²I/dh².

        The off-diagonal entries ∂𝒯_l/∂h(λ_m) for m ≠ l are identically zero
        because this source is mode-separable (assumption A1 is exactly
        satisfied).

        Returns
        -------
        (N,) array of ∂𝒯_l/∂h(λ_l) values (all ≤ 0 by A3).
        """
        return self.mu2 * self.d2I_dh2(h)

    def jacobian(self, h: np.ndarray) -> np.ndarray:
        """Full N×N source Jacobian ∂𝒯_l/∂h(λ_m).

        For this mode-separable source the Jacobian is diagonal.  Provided
        for compatibility with the full Hessian calculation in
        SpectralKernelDynamics and for future use with coupled sources.

        Returns
        -------
        (N, N) diagonal matrix with entries dT_dh(h) on the diagonal.
        """
        return np.diag(self.dT_dh(h))

    # ------------------------------------------------------------------
    # Stability check (Corollary 3 / condition (12) in the paper)
    # ------------------------------------------------------------------

    def stability_margin(self, h_star: np.ndarray) -> np.ndarray:
        """Per-mode stability margin at critical point h*.

        Stability requires  ∂𝒯_l/∂h(λ_l)|_{h*} > −1/h*(λ_l)  for all l.
        Returns the signed margin  ∂𝒯_l/∂h(λ_l) − (−1/h*(λ_l)),
        which must be positive everywhere for local stability.

        Returns
        -------
        (N,) array of stability margins (positive = stable).
        """
        h_star = np.asarray(h_star, dtype=float)
        return self.dT_dh(h_star) - (-1.0 / h_star)

    def is_stable(self, h_star: np.ndarray) -> bool:
        """True if all per-mode stability margins are positive."""
        return bool(np.all(self.stability_margin(h_star) > 0))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        spectral = "spectral" if self._w is not None else "flat"
        return f"GaussianMISource(sigma2={self.sigma2}, mu2={self.mu2}, weights={spectral})"


class CoupledGaussianMISource(GaussianMISource):
    """Gaussian MI source with explicit inter-modal coupling.

    Extends GaussianMISource by adding a linear coupling term

        eta * (C h)_l

    to the source, where C is an N x N coupling matrix in spectral
    coordinates. This yields a non-diagonal source Jacobian and enables
    explicit tests of Q6's coupling-aware diagnostics.
    """

    def __init__(
        self,
        sigma2: float = 1.0,
        mu2: float = 1.0,
        eigenvalues: Optional[np.ndarray] = None,
        coupling_matrix: Optional[np.ndarray] = None,
        eta: float = 0.02,
    ) -> None:
        super().__init__(sigma2=sigma2, mu2=mu2, eigenvalues=eigenvalues)
        self.eta = float(eta)
        if coupling_matrix is None:
            self._C = None
        else:
            C = np.asarray(coupling_matrix, dtype=float)
            if C.ndim != 2 or C.shape[0] != C.shape[1]:
                raise ValueError("coupling_matrix must be a square 2-D array.")
            C = C.copy()
            np.fill_diagonal(C, 0.0)  # keep diagonal controlled by MI term
            self._C = C

    def _coupling(self, N: int) -> np.ndarray:
        if self._C is None:
            return np.zeros((N, N), dtype=float)
        if self._C.shape != (N, N):
            raise ValueError(
                "coupling_matrix shape does not match h length: "
                f"got {self._C.shape} for N={N}."
            )
        return self._C

    def T(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        return super().T(h) + self.eta * (self._coupling(h.size) @ h)

    def dT_dh(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        C = self._coupling(h.size)
        return super().dT_dh(h) + self.eta * np.diag(C)

    def jacobian(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        return np.diag(super().dT_dh(h)) + self.eta * self._coupling(h.size)

    def __repr__(self) -> str:
        spectral = "spectral" if self._w is not None else "flat"
        coupled = self._C is not None
        return (
            "CoupledGaussianMISource("
            f"sigma2={self.sigma2}, mu2={self.mu2}, weights={spectral}, "
            f"eta={self.eta}, coupled={coupled})"
        )

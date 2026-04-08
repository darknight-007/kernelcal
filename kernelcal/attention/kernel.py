"""
AttentionKernel: spectral MaxCal analysis of a single attention matrix.

Given an attention weight matrix A ∈ R^{N×N} (from one head of one layer),
this class:
  1. Symmetrizes to get a proper kernel: K = (A + A.T) / 2
  2. Builds a SpectralGraph from K (using K as the Laplacian-equivalent)
  3. Runs MaxCal spectral diagnostics via SpectralKernelDynamics
  4. Reports: spectral entropy H[h_t], Fiedler gap Δ', fixed-point residual
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AttentionKernelResult:
    """Spectral MaxCal diagnostics for one attention head at one step."""
    layer: int
    head: int
    step: int
    seq_len: int
    # Spectral
    fiedler_value: float
    spectral_entropy: float
    hessian_gap: float
    fiedler_gap: float
    coupling_entropy: float
    residual_inf_norm: float
    converged: bool
    # Raw kernel (CPU numpy, symmetrized)
    kernel_matrix: np.ndarray  # (N, N)
    h_star: np.ndarray          # (N,) spectral weights at fixed point

    def summary(self) -> str:
        lines = [
            f"AttentionKernel  layer={self.layer}  head={self.head}  step={self.step}",
            f"  seq_len={self.seq_len}  fiedler_value={self.fiedler_value:.5f}",
            f"  spectral_entropy={self.spectral_entropy:.4f}",
            f"  hessian_gap={self.hessian_gap:.4f}  fiedler_gap={self.fiedler_gap:.4f}",
            f"  coupling_entropy={self.coupling_entropy:.4f}",
            f"  residual_inf_norm={self.residual_inf_norm:.3e}  converged={self.converged}",
        ]
        return "\n".join(lines)


class AttentionKernel:
    """
    Spectral MaxCal analysis of a transformer attention matrix.

    Parameters
    ----------
    attn_weights : (N, N) array-like
        Attention weight matrix for one head (row-stochastic or raw logits).
        Will be symmetrized: K = (A + A.T) / 2 to obtain a valid kernel.
    layer : int
        Layer index (for bookkeeping).
    head : int
        Head index (for bookkeeping).
    step : int
        Training step (for bookkeeping).
    sigma2 : float
        Observation noise for the Gaussian MI source.
    mu2 : float
        Lagrange multiplier for mutual information constraint.
    eigenvalue_aware : bool
        If True, source weights w_l = λ_l (eigenvalue-aware).
    """

    def __init__(
        self,
        attn_weights: np.ndarray,
        layer: int = 0,
        head: int = 0,
        step: int = 0,
        sigma2: float = 1.0,
        mu2: float = 2.0,
        eigenvalue_aware: bool = True,
    ) -> None:
        self.layer = layer
        self.head = head
        self.step = step
        self.sigma2 = sigma2
        self.mu2 = mu2
        self.eigenvalue_aware = eigenvalue_aware

        A = np.asarray(attn_weights, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"attn_weights must be square 2-D; got {A.shape}")
        self._K = self._symmetrize(A)

    # ------------------------------------------------------------------

    @staticmethod
    def _symmetrize(A: np.ndarray) -> np.ndarray:
        """Symmetrize and make non-negative: K = (A + A.T) / 2, then clip."""
        K = (A + A.T) / 2.0
        # Attention weights are in [0,1]; after symmetrizing they stay ≥ 0.
        # For raw logits: shift so minimum off-diagonal is 0.
        if K.min() < 0:
            K = K - K.min()
        np.fill_diagonal(K, 0.0)
        return K

    @staticmethod
    def _laplacian_from_kernel(K: np.ndarray) -> np.ndarray:
        """Build combinatorial Laplacian L = D - K from kernel/adjacency K."""
        D = np.diag(K.sum(axis=1))
        return D - K

    def analyse(self, fp_max_iter: int = 300, fp_tol: float = 1e-9) -> AttentionKernelResult:
        """
        Run MaxCal spectral diagnostics on this attention kernel.

        Returns
        -------
        AttentionKernelResult
        """
        from ..spectral.graph import SpectralGraph
        from ..spectral.source import GaussianMISource
        from ..spectral.dynamics import (
            SpectralKernelDynamics,
            spectral_entropy,
            field_equation_residual,
        )

        L = self._laplacian_from_kernel(self._K)
        graph = SpectralGraph(L)

        src = GaussianMISource(
            sigma2=self.sigma2,
            mu2=self.mu2,
            eigenvalues=(graph.eigenvalues if self.eigenvalue_aware else None),
        )
        dyn = SpectralKernelDynamics(graph=graph, source=src)
        fp = dyn.fixed_point_iteration(max_iter=fp_max_iter, tol=fp_tol)
        stab = dyn.stability_analysis(fp.h_star)

        T_vals = src.T(fp.h_star)
        res = field_equation_residual(fp.h_star, dyn.h0, T_vals)
        residual_inf = float(np.max(np.abs(res)))

        return AttentionKernelResult(
            layer=self.layer,
            head=self.head,
            step=self.step,
            seq_len=self._K.shape[0],
            fiedler_value=float(graph.fiedler_value),
            spectral_entropy=float(spectral_entropy(fp.h_star)),
            hessian_gap=float(stab.gap),
            fiedler_gap=float(stab.fiedler_gap),
            coupling_entropy=float(stab.coupling_entropy_value),
            residual_inf_norm=residual_inf,
            converged=fp.converged,
            kernel_matrix=self._K.copy(),
            h_star=fp.h_star.copy(),
        )

    # ------------------------------------------------------------------
    # Factory methods

    @classmethod
    def from_numpy(
        cls,
        attn_weights: np.ndarray,
        layer: int = 0,
        head: int = 0,
        step: int = 0,
        **kwargs,
    ) -> "AttentionKernel":
        """Create from a (N, N) numpy array."""
        return cls(attn_weights, layer=layer, head=head, step=step, **kwargs)

    @classmethod
    def from_torch(
        cls,
        attn_weights,  # torch.Tensor (N, N) or (B, N, N) — first batch taken
        layer: int = 0,
        head: int = 0,
        step: int = 0,
        **kwargs,
    ) -> "AttentionKernel":
        """Create from a PyTorch tensor (moved to CPU automatically)."""
        arr = attn_weights.detach().float().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]  # take first batch
        return cls(arr, layer=layer, head=head, step=step, **kwargs)

    @classmethod
    def synthetic(
        cls,
        seq_len: int = 32,
        temperature: float = 1.0,
        seed: int = 0,
        **kwargs,
    ) -> "AttentionKernel":
        """
        Generate synthetic softmax attention weights for testing.
        No PyTorch or GPU required.
        """
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal((seq_len, seq_len)) / temperature
        # Row-wise softmax
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        A = exp_l / exp_l.sum(axis=1, keepdims=True)
        return cls(A, **kwargs)

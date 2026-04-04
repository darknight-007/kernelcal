"""
Fixed-point detection and stability analysis for kernel trajectories.

Maps to Section 4 of the paper ("Fixed-Point Conditions and Stability").
A kernel k* is a fixed point when δS/δγ|_{γ≡k*} = 0.  In the discrete,
finite-sample setting we operationalise this as convergence of the HS
distance sequence to zero, and measure stability via the spectral gap of
the Hessian approximation at the candidate fixed point.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .space import hilbert_schmidt_distance, hilbert_schmidt_norm, is_psd


class FixedPointDetector:
    """Streaming detector for kernel fixed points.

    Feed kernel matrices one at a time; the detector tracks HS-distance
    history and reports whether the sequence has converged to a fixed point,
    together with a stability score derived from the local curvature of the
    distance sequence.

    Parameters
    ----------
    tol : float
        HS distance threshold below which consecutive kernels are considered
        equal (fixed-point condition).
    window : int
        Number of consecutive small-distance steps required to declare
        convergence.
    """

    def __init__(self, tol: float = 1e-3, window: int = 5):
        self.tol = tol
        self.window = window
        self._kernels: List[np.ndarray] = []
        self._distances: List[float] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def update(self, K: np.ndarray) -> "FixedPointDetector":
        """Record the next kernel snapshot.  Returns self for chaining."""
        K = np.asarray(K, dtype=float)
        if self._kernels:
            d = hilbert_schmidt_distance(self._kernels[-1], K)
            self._distances.append(d)
        self._kernels.append(K.copy())
        return self

    # ------------------------------------------------------------------
    # Fixed-point tests
    # ------------------------------------------------------------------

    def is_fixed_point(self) -> bool:
        """True if the last *window* HS distances are all below tol."""
        if len(self._distances) < self.window:
            return False
        return bool(np.all(np.array(self._distances[-self.window:]) < self.tol))

    def candidate_fixed_point(self) -> Optional[np.ndarray]:
        """Return the most recent kernel if convergence has been declared."""
        if self.is_fixed_point():
            return self._kernels[-1].copy()
        return None

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    def stability_score(self) -> float:
        """Scalar stability score in [0, 1].

        Computed as 1 − (mean of last *window* distances) / tol.
        Returns 0 if not enough history, 1 if perfectly at fixed point.
        """
        if len(self._distances) < self.window:
            return 0.0
        recent = np.array(self._distances[-self.window:])
        mean_d = float(np.mean(recent))
        return float(max(0.0, 1.0 - mean_d / self.tol))

    def spectral_stability(self, K_star: Optional[np.ndarray] = None) -> float:
        """Estimate stability via the spectral gap of the candidate kernel.

        The spectral gap of the Gram matrix at a fixed point is related to
        the curvature of the MaxCal functional.  A larger gap implies a
        more stable fixed point (the basin of attraction is deeper).

        Returns the normalised spectral gap:
            (λ₁ − λ₂) / λ₁
        where λ₁ ≥ λ₂ ≥ ... are the sorted eigenvalues.
        """
        K = K_star if K_star is not None else self.candidate_fixed_point()
        if K is None:
            return 0.0
        eigvals = np.linalg.eigvalsh(K)
        eigvals = np.sort(eigvals)[::-1]
        if eigvals[0] <= 0:
            return 0.0
        if len(eigvals) < 2:
            return 1.0
        return float((eigvals[0] - eigvals[1]) / eigvals[0])

    # ------------------------------------------------------------------
    # Convergence time
    # ------------------------------------------------------------------

    def steps_to_convergence(self) -> Optional[int]:
        """Number of update steps until the trajectory first converged."""
        dists = np.array(self._distances)
        for i in range(len(dists) - self.window + 1):
            if np.all(dists[i: i + self.window] < self.tol):
                return i + self.window
        return None

    # ------------------------------------------------------------------
    # Landscape classification
    # ------------------------------------------------------------------

    def classify(self) -> str:
        """Classify the current trajectory phase.

        Returns one of:
          'transient'   — distances large and decreasing (approaching FP)
          'stable_fp'   — at a stable fixed point
          'oscillating' — distances large and not monotone (no convergence)
          'insufficient_data' — fewer than window steps recorded
        """
        if len(self._distances) < self.window:
            return "insufficient_data"
        if self.is_fixed_point():
            return "stable_fp"
        recent = np.array(self._distances[-self.window:])
        diffs = np.diff(recent)
        if np.all(diffs <= 0):
            return "transient"
        return "oscillating"

    # ------------------------------------------------------------------
    # Landscape scan: detect all fixed points in an existing trajectory
    # ------------------------------------------------------------------

    @staticmethod
    def scan_trajectory(kernels: List[np.ndarray],
                        tol: float = 1e-3,
                        window: int = 3) -> List[Tuple[int, float]]:
        """Scan a list of kernels for convergence events.

        Returns a list of (index, stability_score) for every window where
        all distances fall below tol.
        """
        if len(kernels) < 2:
            return []
        dists = np.array([
            hilbert_schmidt_distance(kernels[i], kernels[i + 1])
            for i in range(len(kernels) - 1)
        ])
        results = []
        for i in range(len(dists) - window + 1):
            if np.all(dists[i: i + window] < tol):
                mean_d = float(np.mean(dists[i: i + window]))
                score = max(0.0, 1.0 - mean_d / tol)
                results.append((i + window, score))
        return results

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FixedPointDetector("
            f"steps={len(self._kernels)}, "
            f"fixed={self.is_fixed_point()}, "
            f"stability={self.stability_score():.3f})"
        )

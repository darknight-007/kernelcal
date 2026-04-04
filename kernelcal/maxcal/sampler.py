"""
MaxCal World Sampler: drop-in replacement for the DeepGIS World Sampler's
distribution update step.

The World Sampler in deepgis-xr/deepgis_xr/apps/web/world_sampler_api.py  # noqa
maintains a spatial probability distribution that is updated from reward
feedback.  This module replaces that heuristic update with a MaxCal rule:

    p(x) ∝ q(x) · exp(−Σ_i λ_i f_i(x))

where:
  q(x)    — reference distribution (prior geospatial knowledge)
  f_i(x)  — observable feature at location x (coverage, reward, entropy)
  λ_i     — Lagrange multipliers fitted to satisfy ⟨f_i⟩_p = F_i

Fixed-point detection is built in: when the distribution stabilises across
successive updates, the sampler declares convergence — interpreted as
arriving at a self-consistent sampling kernel.

DeepGIS API compatibility
-------------------------
The MaxCalSampler exposes the same surface as the existing World Sampler:
    .initialize(locations, ...)
    .sample(n) → list of (lon, lat)
    .update(feedback_values)
    .statistics() → dict
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import logsumexp

from .functional import (
    fit_lagrange_multipliers,
    maxcal_log_weights,
    path_entropy,
    constraint_residuals,
)
from ..kernel.space import kernel_from_embeddings, hilbert_schmidt_distance
from ..kernel.fixed_points import FixedPointDetector


class MaxCalSampler:
    """Adaptive geospatial sampler governed by Maximum Caliber.

    Parameters
    ----------
    locations : (N, 2) array of (lon, lat) coordinates.
    reference_weights : (N,) prior weights q(x).  Uniform if None.
    constraint_fns : list of callables (N,2)→(N,) computing f_i over all locs.
    constraint_targets : list of target expected values F_i = ⟨f_i⟩_p.
    kernel_fn : callable mapping (N,2) features to (N,N) kernel matrix.
                If None, the RBF kernel over lon/lat is used internally.
    fixed_point_tol : HS-distance tolerance for fixed-point detection.
    fixed_point_window : consecutive near-zero-distance steps for convergence.
    """

    def __init__(
        self,
        locations: np.ndarray,
        reference_weights: Optional[np.ndarray] = None,
        constraint_fns: Optional[List[Callable]] = None,
        constraint_targets: Optional[List[float]] = None,
        kernel_fn: Optional[Callable] = None,
        fixed_point_tol: float = 1e-3,
        fixed_point_window: int = 5,
    ):
        self.locations = np.asarray(locations, dtype=float)
        n = len(self.locations)

        if reference_weights is not None:
            w = np.asarray(reference_weights, dtype=float)
            w = np.clip(w, 1e-300, None)
            self._log_q = np.log(w) - logsumexp(np.log(w))
        else:
            self._log_q = -np.log(n) * np.ones(n)

        self._log_p = self._log_q.copy()
        self._lambdas = np.zeros(len(constraint_fns) if constraint_fns else 0)

        self.constraint_fns = constraint_fns or []
        self.constraint_targets = (
            list(constraint_targets) if constraint_targets else []
        )
        self._kernel_fn = kernel_fn

        self._fp_detector = FixedPointDetector(
            tol=fixed_point_tol, window=fixed_point_window
        )
        self._update_count = 0
        self._entropy_history: List[float] = []
        self._kernel_history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int, replace: bool = True) -> np.ndarray:
        """Draw n locations according to the current MaxCal distribution.

        Returns
        -------
        (n, 2) array of (lon, lat) sampled locations.
        """
        p = self.distribution()
        idx = np.random.choice(len(self.locations), size=n,
                               replace=replace, p=p)
        return self.locations[idx]

    def distribution(self) -> np.ndarray:
        """Current normalised probability over all locations."""
        return np.exp(self._log_p)

    def log_distribution(self) -> np.ndarray:
        return self._log_p.copy()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        feedback: Optional[np.ndarray] = None,
        constraint_overrides: Optional[Dict[int, float]] = None,
    ) -> "MaxCalSampler":
        """Re-fit Lagrange multipliers given new feedback.

        Parameters
        ----------
        feedback : (N,) per-location reward / observation values.
            If provided and no explicit constraint_fns were set, this is
            used as the single constraint feature with target = mean(feedback).
        constraint_overrides : dict mapping constraint index → new target value.
            Use to update F_i values between calls.
        """
        fns = list(self.constraint_fns)
        targets = list(self.constraint_targets)

        if feedback is not None and not fns:
            fb = np.asarray(feedback, dtype=float)
            fns = [lambda locs, _fb=fb: _fb]
            targets = [float(np.mean(fb))]

        if constraint_overrides:
            for idx, val in constraint_overrides.items():
                if idx < len(targets):
                    targets[idx] = val

        if not fns:
            # No constraints → maximum entropy = reference distribution
            self._log_p = self._log_q.copy()
        else:
            F_matrix = np.column_stack([fn(self.locations) for fn in fns])
            F_targets = np.array(targets, dtype=float)

            n_c = F_matrix.shape[1]
            lambda0 = (self._lambdas if len(self._lambdas) == n_c
                       else np.zeros(n_c))
            lambdas, _ = fit_lagrange_multipliers(
                self._log_q, F_matrix, F_targets, lambda0=lambda0
            )
            self._lambdas = lambdas
            self._log_p = maxcal_log_weights(self._log_q, lambdas, F_matrix)

        # Track entropy and kernel evolution for fixed-point detection
        self._entropy_history.append(self.entropy())
        K = self._current_kernel()
        self._kernel_history.append(K)
        self._fp_detector.update(K)
        self._update_count += 1
        return self

    def _current_kernel(self) -> np.ndarray:
        """Probability-weighted kernel matrix at the current distribution."""
        p = self.distribution()
        if self._kernel_fn is not None:
            K_base = self._kernel_fn(self.locations)
        else:
            from ..kernel.space import rbf_kernel
            scale = np.std(self.locations) if np.std(self.locations) > 0 else 1.0
            K_base = rbf_kernel(self.locations, length_scale=scale)
        # Weight the kernel by the marginal distributions
        K_weighted = np.outer(p, p) * K_base
        return K_weighted

    # ------------------------------------------------------------------
    # Fixed-point / convergence
    # ------------------------------------------------------------------

    def is_at_fixed_point(self) -> bool:
        """True if the sampling distribution has converged."""
        return self._fp_detector.is_fixed_point()

    def stability_score(self) -> float:
        """Scalar stability score of the current distribution (0 → 1)."""
        return self._fp_detector.stability_score()

    def classify(self) -> str:
        """'stable_fp', 'transient', 'oscillating', or 'insufficient_data'."""
        return self._fp_detector.classify()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def entropy(self) -> float:
        """Shannon entropy (nats) of the current distribution."""
        p = self.distribution()
        return float(-np.sum(p * np.log(np.clip(p, 1e-300, None))))

    def path_entropy_from_reference(self) -> float:
        """MaxCal path entropy S[p] = −KL(p‖q)."""
        return path_entropy(self._log_p, self._log_q)

    def constraint_residuals(self) -> Optional[np.ndarray]:
        """⟨f_i⟩_p − F_i.  Near-zero means constraints are satisfied."""
        if not self.constraint_fns:
            return None
        F_matrix = np.column_stack([fn(self.locations) for fn in self.constraint_fns])
        F_targets = np.array(self.constraint_targets)
        return constraint_residuals(self._log_p, F_matrix, F_targets)

    def statistics(self) -> Dict:
        """Summary dict compatible with the DeepGIS world sampler stats API."""
        p = self.distribution()
        top_idx = int(np.argmax(p))
        return {
            "n_locations": len(self.locations),
            "update_count": self._update_count,
            "entropy_nats": self.entropy(),
            "path_entropy_from_reference": self.path_entropy_from_reference(),
            "peak_probability": float(p[top_idx]),
            "peak_location": self.locations[top_idx].tolist(),
            "stability_score": self.stability_score(),
            "is_fixed_point": self.is_at_fixed_point(),
            "classification": self.classify(),
            "lagrange_multipliers": self._lambdas.tolist(),
        }

    def __repr__(self) -> str:
        return (
            f"MaxCalSampler("
            f"n={len(self.locations)}, "
            f"updates={self._update_count}, "
            f"H={self.entropy():.3f} nats, "
            f"fixed_point={self.is_at_fixed_point()})"
        )

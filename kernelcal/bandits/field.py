"""
Spatiotemporal 2-D field with structurally distinct kernel requirements.

Motivation — the "different lens" problem
-----------------------------------------
Arms are (x, t) pairs where x is a spatial coordinate and t is time
(normalised to [0,1]).  The field has two structurally distinct regions:

  * Left half  (x < 0.5): f(x,t) has PERIODIC temporal structure.
      True kernel: SE(x) × Periodic(t).
      An SE(x,t) kernel with any lengthscale cannot represent this;
      refocusing the same SE lens never produces the right model.

  * Right half (x ≥ 0.5): f(x,t) is spatiotemporally smooth.
      True kernel: Anisotropic SE(x,t).
      A periodic kernel would overfit here.

This is the "different product kernel" problem:
  - Changing ℓ_x or ℓ_t within an SE family = refocusing the same lens.
  - Switching to SE(x) × Periodic(t)         = a structurally different lens
    that captures periodicity the SE family cannot represent at any scale.

Classes
-------
SpatiotemporalField
    Generates the ground-truth spatiotemporal reward function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Ground-truth field functions
# ---------------------------------------------------------------------------

def _f_periodic_region(x: float, t: float,
                        amplitude: float = 1.0,
                        ell_x: float = 0.35,
                        period: float = 0.33,
                        noise_rng: np.random.Generator = None) -> float:
    """Left-region reward: spatially smooth, temporally periodic.

    f(x,t) = A · exp(-x²/2ℓ_x²) · cos(2π t / p)

    This requires SE(x) × Periodic(t).  No SE(x,t) kernel recovers this
    structure regardless of lengthscale; the cosine modulation is
    structurally absent from any SE product.
    """
    spatial  = np.exp(-0.5 * (x - 0.25)**2 / ell_x**2)
    temporal = np.cos(2.0 * np.pi * t / period)
    return float(amplitude * spatial * temporal)


def _f_smooth_region(x: float, t: float,
                      amplitude: float = 1.0,
                      ell_x: float = 0.20,
                      ell_t: float = 0.30) -> float:
    """Right-region reward: spatiotemporally smooth Gaussian bump.

    f(x,t) = A · exp(-(x-0.75)²/2ℓ_x² - (t-0.5)²/2ℓ_t²)

    Correctly modelled by Anisotropic SE(x,t).
    A periodic kernel would fit spurious periodicity here.
    """
    dx = (x - 0.75)**2 / ell_x**2
    dt = (t - 0.50)**2 / ell_t**2
    return float(amplitude * np.exp(-0.5 * (dx + dt)))


# ---------------------------------------------------------------------------
# Main field class
# ---------------------------------------------------------------------------

@dataclass
class SpatiotemporalField:
    """Spatiotemporal 2-D field requiring structurally different kernels per region.

    Arms are (x, t) pairs:
      x ∈ [0, 1]  — spatial coordinate (e.g. along-shore position)
      t ∈ [0, 1]  — normalised time (e.g. within a tidal cycle)

    Left half  (x < 0.5): SE(x) × Periodic(t) is the correct kernel class.
    Right half (x ≥ 0.5): Anisotropic SE(x,t) is the correct kernel class.

    Parameters
    ----------
    n_arms_x, n_arms_t : grid dimensions
    sigma_n            : observation noise std
    period             : true period of the left-region temporal oscillation
    seed               : random seed
    """

    n_arms_x: int   = 5
    n_arms_t: int   = 6       # more temporal samples to reveal periodicity
    sigma_n: float  = 0.08
    period: float   = 0.33    # ~3 full cycles over t ∈ [0,1]
    amplitude: float = 1.2
    seed: int        = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.arm_locations = self._build_grid()
        self.K_arms = len(self.arm_locations)
        self._f_true = self._evaluate_field()

    def _build_grid(self) -> np.ndarray:
        xs = np.linspace(0.0, 1.0, self.n_arms_x)
        ts = np.linspace(0.0, 1.0, self.n_arms_t)
        return np.array([[x, t] for x in xs for t in ts])

    def _evaluate_field(self) -> np.ndarray:
        f = np.zeros(self.K_arms)
        for k, (x, t) in enumerate(self.arm_locations):
            if x < 0.5:   # periodic region
                f[k] = _f_periodic_region(
                    x, t, amplitude=self.amplitude, period=self.period)
            else:         # smooth region
                f[k] = _f_smooth_region(x, t, amplitude=self.amplitude)
        return f

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def f_true(self) -> np.ndarray:
        return self._f_true

    @property
    def best_arm(self) -> int:
        return int(np.argmax(self._f_true))

    @property
    def best_reward(self) -> float:
        return float(self._f_true[self.best_arm])

    def pull(self, arm: int) -> float:
        return float(self._f_true[arm] + self._rng.normal(0.0, self.sigma_n))

    def suboptimality_gap(self, arm: int) -> float:
        return self.best_reward - float(self._f_true[arm])

    def arm_region(self, arm: int) -> str:
        """'periodic' (left) or 'smooth' (right)."""
        return "periodic" if self.arm_locations[arm, 0] < 0.5 else "smooth"

    def required_kernel_class(self, arm: int) -> str:
        """What kernel class is structurally necessary for this arm's region."""
        return "SE×Periodic" if self.arm_region(arm) == "periodic" else "SE"

    # Keep old name as alias for backward compat with experiment.py
    @property
    def sigma_f(self) -> float:
        return self.amplitude


# Backward-compatible alias used by existing experiment.py
AnisotropicField = SpatiotemporalField

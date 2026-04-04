"""
Informative path planner for the Earth Rover.

Replaces static GPS waypoint missions with a MaxCal-governed distribution over
candidate locations, subject to:
  - Energy constraint:   ⟨E_i⟩_p ≤ E_budget   (battery remaining)
  - Novelty constraint:  ⟨I_i⟩_p ≥ I_target    (SLAM novelty / semantic reward)
  - Coverage constraint: ⟨visits_i⟩_p ≥ min_visits (no region neglected)

The planner wraps MaxCalSampler with rover-specific constraint functions and
exposes a simple interface: next_waypoint() returns the (lon, lat) of the most
probable next location to visit.

Maps to Thread 2 of NAVIGATION.md.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from ..maxcal.sampler import MaxCalSampler
from ..kernel.fixed_points import FixedPointDetector
from ..thermodynamics.bounds import landauer_bound, kernel_mutual_information_change


# ---------------------------------------------------------------------------
# Energy model helpers
# ---------------------------------------------------------------------------

def euclidean_energy_estimate(
    locations: np.ndarray,
    current_position: np.ndarray,
    joules_per_metre: float = 50.0,
) -> np.ndarray:
    """Estimate energy (joules) to travel from current_position to each candidate.

    Parameters
    ----------
    locations : (N, 2) candidate (lon, lat) array.
    current_position : (2,) current (lon, lat).
    joules_per_metre : float — empirical energy cost per metre for the trike.
        Default 50 J/m ≈ 300W motor at 6 m/s.

    Returns
    -------
    energies : (N,) joule estimates.
    """
    # Approximate degree-to-metre conversion (valid for mid-latitudes)
    DEG_TO_M = 111_320.0
    delta = (locations - current_position) * DEG_TO_M
    distances = np.linalg.norm(delta, axis=1)
    return distances * joules_per_metre


# ---------------------------------------------------------------------------
# InformativePathPlanner
# ---------------------------------------------------------------------------

class InformativePathPlanner:
    """MaxCal-based informative path planner for the Earth Rover.

    Parameters
    ----------
    candidate_waypoints : (N, 2) array of (lon, lat) candidate locations.
    energy_budget_joules : float — remaining battery energy budget.
    joules_per_metre : float — empirical drive energy cost per metre.
    novelty_weight : float — weight on semantic novelty vs. coverage.
    fixed_point_tol : float — HS threshold for stable-patrol detection.
    """

    def __init__(
        self,
        candidate_waypoints: np.ndarray,
        energy_budget_joules: float = 100_000.0,
        joules_per_metre: float = 50.0,
        novelty_weight: float = 0.7,
        coverage_weight: float = 0.3,
        fixed_point_tol: float = 1e-2,
        fixed_point_window: int = 5,
    ):
        self.waypoints = np.asarray(candidate_waypoints, dtype=float)
        self.energy_budget = energy_budget_joules
        self.joules_per_metre = joules_per_metre
        self.novelty_weight = novelty_weight
        self.coverage_weight = coverage_weight

        n = len(self.waypoints)
        self._visit_counts = np.zeros(n)
        self._novelty_scores = np.zeros(n)
        self._current_position = self.waypoints[0].copy()

        # Build the MaxCal sampler — constraints added dynamically
        self._sampler = MaxCalSampler(
            self.waypoints,
            fixed_point_tol=fixed_point_tol,
            fixed_point_window=fixed_point_window,
        )
        self._fp_detector = FixedPointDetector(
            tol=fixed_point_tol, window=fixed_point_window
        )
        self._update_count = 0

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update(
        self,
        current_position: Optional[np.ndarray] = None,
        battery_joules_remaining: Optional[float] = None,
        semantic_scores: Optional[np.ndarray] = None,
    ) -> "InformativePathPlanner":
        """Re-compute the MaxCal distribution given current field state.

        Parameters
        ----------
        current_position : (2,) current (lon, lat).
        battery_joules_remaining : float — remaining battery (joules).
        semantic_scores : (N,) per-waypoint novelty / information scores.
            Higher = more information expected at that location.
        """
        if current_position is not None:
            self._current_position = np.asarray(current_position, dtype=float)
        if battery_joules_remaining is not None:
            self.energy_budget = battery_joules_remaining
        if semantic_scores is not None:
            self._novelty_scores = np.asarray(semantic_scores, dtype=float)

        # Build combined reward signal
        reward = self._combined_reward()

        # Update sampler — the energy cost is encoded as inverse reward
        # (visiting expensive locations is penalised)
        energies = euclidean_energy_estimate(
            self.waypoints, self._current_position, self.joules_per_metre
        )
        energy_fraction = energies / (self.energy_budget + 1e-9)
        energy_penalty = np.exp(-energy_fraction)  # high penalty near budget limit

        combined_feedback = reward * energy_penalty
        self._sampler.update(feedback=combined_feedback)
        self._update_count += 1
        return self

    def _combined_reward(self) -> np.ndarray:
        """Combine novelty scores and inverse coverage into a scalar reward."""
        eps = 1e-8
        novelty = self._novelty_scores
        if novelty.max() > 0:
            novelty = novelty / (novelty.max() + eps)

        inv_cov = 1.0 / (1.0 + self._visit_counts)
        inv_cov = inv_cov / (inv_cov.max() + eps)

        return self.novelty_weight * novelty + self.coverage_weight * inv_cov

    # ------------------------------------------------------------------
    # Waypoint selection
    # ------------------------------------------------------------------

    def next_waypoint(self) -> np.ndarray:
        """Return the (lon, lat) of the highest-probability next waypoint."""
        p = self._sampler.distribution()
        idx = int(np.argmax(p))
        self._visit_counts[idx] += 1
        return self.waypoints[idx].copy()

    def sample_waypoint(self) -> np.ndarray:
        """Sample a waypoint from the current MaxCal distribution."""
        wp = self._sampler.sample(n=1)[0]
        idx = np.argmin(np.linalg.norm(self.waypoints - wp, axis=1))
        self._visit_counts[idx] += 1
        return wp

    def top_k_waypoints(self, k: int = 5) -> np.ndarray:
        """Return the top-k highest-probability waypoints."""
        p = self._sampler.distribution()
        indices = np.argsort(p)[::-1][:k]
        return self.waypoints[indices]

    # ------------------------------------------------------------------
    # Fixed-point / patrol stability
    # ------------------------------------------------------------------

    def is_at_fixed_point(self) -> bool:
        """True when the planner has converged to an optimal patrol loop."""
        return self._sampler.is_at_fixed_point()

    def patrol_stability_score(self) -> float:
        return self._sampler.stability_score()

    def classify(self) -> str:
        return self._sampler.classify()

    # ------------------------------------------------------------------
    # Thermodynamic efficiency
    # ------------------------------------------------------------------

    def thermodynamic_efficiency(
        self,
        energy_spent_joules: float,
        K_before: np.ndarray,
        K_after: np.ndarray,
        temperature_kelvin: float = 298.15,
    ) -> dict:
        """Measure information gained per joule for a completed traverse.

        Parameters
        ----------
        energy_spent_joules : float — battery energy consumed on the leg.
        K_before, K_after : (N, N) SLAM kernel before and after the leg.
        """
        delta_I = kernel_mutual_information_change(K_before, K_after)
        bound = landauer_bound(delta_I, temperature_kelvin)
        return {
            "delta_I_nats": delta_I,
            "landauer_bound_joules": bound,
            "energy_spent_joules": energy_spent_joules,
            "efficiency": bound / (energy_spent_joules + 1e-300),
            "nats_per_joule": delta_I / (energy_spent_joules + 1e-300),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def distribution(self) -> np.ndarray:
        return self._sampler.distribution()

    def coverage_map(self) -> np.ndarray:
        return self._visit_counts.copy()

    def statistics(self) -> dict:
        return {
            **self._sampler.statistics(),
            "update_count": self._update_count,
            "total_visits": int(self._visit_counts.sum()),
            "unvisited_fraction": float(np.mean(self._visit_counts == 0)),
            "patrol_stability": self.patrol_stability_score(),
            "patrol_classification": self.classify(),
        }

    def __repr__(self) -> str:
        return (
            f"InformativePathPlanner("
            f"n_waypoints={len(self.waypoints)}, "
            f"budget_J={self.energy_budget:.0f}, "
            f"fixed_point={self.is_at_fixed_point()}, "
            f"stability={self.patrol_stability_score():.3f})"
        )

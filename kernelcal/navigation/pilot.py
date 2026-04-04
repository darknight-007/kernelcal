"""
Inverse MaxCal from human-pilot demonstrations.

The human pilot's recorded paths are observations from an implicit MaxCal distribution.
This module recovers the Lagrange multipliers λ that make the MaxCal distribution
most consistent with the demonstrated trajectories — effectively learning the
pilot's implicit constraints (energy preference, novelty-seeking, obstacle avoidance).

Once learned, λ can be transferred to new terrain: the rover generates
human-pilot-consistent paths in places the pilot has never visited.

Maps to Thread 3 of NAVIGATION.md.

Inverse MaxCal / maximum-entropy IRL reference:
  Ziebart et al. (2008) "Maximum Entropy Inverse Reinforcement Learning"
  Here adapted to the MaxCal (path entropy) setting.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from ..maxcal.functional import maxcal_log_weights
from ..navigation.planner import InformativePathPlanner


# ---------------------------------------------------------------------------
# Feature extraction from paths
# ---------------------------------------------------------------------------

def path_feature_vector(
    path: np.ndarray,
    feature_fns: List[Callable],
    waypoints: np.ndarray,
) -> np.ndarray:
    """Compute the aggregate feature vector for a demonstrated path.

    For each waypoint index visited in *path*, computes feature_fn(waypoints[idx])
    and sums over the path.  This is the "observed constraint value" F_j for
    inverse MaxCal.

    Parameters
    ----------
    path : (T,) integer indices into waypoints visited in order.
    feature_fns : list of (N,) → scalar or (N,) array callables.
    waypoints : (N, 2) candidate location array.

    Returns
    -------
    features : (C,) summed feature vector for this path.
    """
    C = len(feature_fns)
    features = np.zeros(C)
    for idx in path:
        for c, fn in enumerate(feature_fns):
            val = fn(waypoints)
            features[c] += float(val[idx]) if hasattr(val, "__len__") else float(fn(waypoints[idx]))
    return features / max(len(path), 1)


# ---------------------------------------------------------------------------
# Inverse MaxCal learner
# ---------------------------------------------------------------------------

class HumanPilotDemonstrationLearner:
    """Recovers Lagrange multipliers from human-pilot-demonstrated paths.

    The inverse MaxCal objective:
        max_λ  Σ_{demo} log p_λ[γ_demo]
             = Σ_{demo} [−λ·F(γ_demo) − log Z(λ)]

    Equivalently: minimise the negative log-likelihood of the demonstrations
    under the MaxCal distribution p_λ[γ] ∝ q[γ] exp(−λ·f(γ)).

    Parameters
    ----------
    waypoints : (N, 2) candidate locations.
    feature_fns : list of callables.  Each maps waypoints (N,2) → (N,) scores.
        Examples: energy_cost_fn, novelty_fn, obstacle_proximity_fn.
    reference_weights : (N,) prior q over locations.  Uniform if None.
    """

    def __init__(
        self,
        waypoints: np.ndarray,
        feature_fns: List[Callable],
        reference_weights: Optional[np.ndarray] = None,
    ):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.feature_fns = feature_fns
        n = len(self.waypoints)

        if reference_weights is not None:
            w = np.asarray(reference_weights, dtype=float)
            self._log_q = np.log(w / (w.sum() + 1e-300))
        else:
            self._log_q = -np.log(n) * np.ones(n)

        # Pre-compute feature matrix (N, C)
        self._F_matrix = np.column_stack([
            fn(self.waypoints) for fn in feature_fns
        ])

        self._demonstrations: List[np.ndarray] = []
        self._lambdas: Optional[np.ndarray] = None
        self._fit_result = None

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_demonstration(self, path_indices: np.ndarray) -> "HumanPilotDemonstrationLearner":
        """Add one demonstrated path as a sequence of waypoint indices.

        Parameters
        ----------
        path_indices : (T,) integer indices into self.waypoints.
        """
        self._demonstrations.append(np.asarray(path_indices, dtype=int))
        return self

    def add_demonstration_from_positions(
        self,
        positions: np.ndarray,
    ) -> "HumanPilotDemonstrationLearner":
        """Add a demonstration given (T, 2) lon/lat positions.

        Maps each position to the nearest candidate waypoint.
        """
        indices = np.array([
            int(np.argmin(np.linalg.norm(self.waypoints - pos, axis=1)))
            for pos in positions
        ])
        return self.add_demonstration(indices)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        lambda0: Optional[np.ndarray] = None,
        method: str = "L-BFGS-B",
        tol: float = 1e-8,
        max_iter: int = 1000,
    ) -> np.ndarray:
        """Recover Lagrange multipliers from all added demonstrations.

        Minimises the negative log-likelihood:
            L(λ) = Σ_demo [λ·F(demo) + log Z(λ)]

        Returns
        -------
        lambdas : (C,) fitted Lagrange multipliers.
        """
        if not self._demonstrations:
            raise ValueError("No demonstrations added.  Call add_demonstration() first.")

        C = self._F_matrix.shape[1]
        if lambda0 is None:
            lambda0 = np.zeros(C)

        # Average observed feature vector across demonstrations
        demo_features = np.array([
            path_feature_vector(demo, self.feature_fns, self.waypoints)
            for demo in self._demonstrations
        ])
        mean_demo_features = demo_features.mean(axis=0)

        def neg_log_likelihood(lambdas):
            log_unnorm = self._log_q - self._F_matrix @ lambdas
            log_Z = logsumexp(log_unnorm)
            # Expected features under p_λ
            log_p = log_unnorm - log_Z
            p = np.exp(log_p)
            expected_f = self._F_matrix.T @ p
            # NLL = λ·F_demo + log Z
            nll = float(lambdas @ mean_demo_features + log_Z)
            grad = mean_demo_features - expected_f
            return nll, grad

        result = minimize(
            neg_log_likelihood,
            x0=lambda0,
            jac=True,
            method=method,
            options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
        )
        self._lambdas = result.x
        self._fit_result = result
        return self._lambdas

    # ------------------------------------------------------------------
    # Transfer to new terrain
    # ------------------------------------------------------------------

    def make_planner(
        self,
        candidate_waypoints: np.ndarray,
        energy_budget_joules: float = 100_000.0,
        joules_per_metre: float = 50.0,
    ) -> InformativePathPlanner:
        """Create an InformativePathPlanner that uses the learned human-pilot preferences.

        The learned λ values are used as the initial Lagrange multipliers for the
        new planner's MaxCalSampler, biasing it toward human-pilot-consistent paths
        in the new terrain.

        Parameters
        ----------
        candidate_waypoints : (M, 2) new terrain waypoints.
        """
        if self._lambdas is None:
            raise RuntimeError("Call fit() before make_planner().")

        planner = InformativePathPlanner(
            candidate_waypoints=candidate_waypoints,
            energy_budget_joules=energy_budget_joules,
            joules_per_metre=joules_per_metre,
        )

        # Inject learned feature scores as initial novelty signal
        # (approximate transfer: apply learned λ to new terrain features)
        new_F = np.column_stack([
            fn(candidate_waypoints) for fn in self.feature_fns
        ])
        learned_scores = -new_F @ self._lambdas  # higher = pilot-preferred
        learned_scores -= learned_scores.min()

        planner.update(semantic_scores=learned_scores)
        return planner

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def learned_preferences(self) -> dict:
        """Return a description of the learned pilot Lagrange multipliers."""
        if self._lambdas is None:
            return {"status": "not fitted"}
        fn_names = [getattr(fn, "__name__", f"feature_{i}")
                    for i, fn in enumerate(self.feature_fns)]
        return {
            name: float(lam)
            for name, lam in zip(fn_names, self._lambdas)
        }

    def log_likelihood(self) -> Optional[float]:
        """Log-likelihood of the demonstrations under the fitted distribution."""
        if self._fit_result is None:
            return None
        return float(-self._fit_result.fun)

    def distribution(self) -> Optional[np.ndarray]:
        """Current MaxCal distribution over waypoints given learned λ."""
        if self._lambdas is None:
            return None
        log_p = maxcal_log_weights(self._log_q, self._lambdas, self._F_matrix)
        return np.exp(log_p)

    def __repr__(self) -> str:
        return (
            f"HumanPilotDemonstrationLearner("
            f"n_demos={len(self._demonstrations)}, "
            f"n_features={len(self.feature_fns)}, "
            f"fitted={self._lambdas is not None})"
        )

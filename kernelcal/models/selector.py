"""
MaxCal multi-model kernel selector for DeepGIS-XR.

Each AI model in DeepGIS (SAM, YOLOv8, Grounding DINO, Zero-Shot,
Mask2Former) corresponds to a point in kernel space K.  This module
maintains a probability distribution over models and updates it via
MaxCal: the model most likely to maximise representational gain given
current scene statistics and compute budget.

The MaxCal update rule:
    p(model_i) ∝ q(model_i) · exp(−λ₁·cost_i − λ₂·(−MI_gain_i))

where:
  q(model_i)  — reference prior (e.g., uniform, or speed-proportional)
  cost_i      — compute cost of model i (seconds or GPU joules)
  MI_gain_i   — expected mutual information gain from using model i
  λ₁, λ₂     — Lagrange multipliers fitted to cost and gain constraints

The kernel trajectory of the selector (which model is chosen over time)
feeds into a FixedPointDetector to identify stable model preferences —
the model selection equivalent of a self-consistent kernel fixed point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import logsumexp

from ..kernel.space import hilbert_schmidt_distance, kernel_from_embeddings
from ..kernel.fixed_points import FixedPointDetector
from ..maxcal.functional import fit_lagrange_multipliers, maxcal_log_weights


# ---------------------------------------------------------------------------
# Model record
# ---------------------------------------------------------------------------

DEEPGIS_MODELS = ["sam", "yolov8", "grounding_dino", "zero_shot", "mask2former"]


@dataclass
class ModelRecord:
    """Accumulated statistics for a single model."""
    name: str
    total_uses: int = 0
    total_cost_seconds: float = 0.0
    total_mi_gain_nats: float = 0.0
    last_kernel_matrix: Optional[np.ndarray] = None

    @property
    def mean_cost(self) -> float:
        return self.total_cost_seconds / max(self.total_uses, 1)

    @property
    def mean_mi_gain(self) -> float:
        return self.total_mi_gain_nats / max(self.total_uses, 1)

    @property
    def efficiency(self) -> float:
        """Nats per second."""
        return self.mean_mi_gain / max(self.mean_cost, 1e-9)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class ModelKernelSelector:
    """MaxCal-governed selector over DeepGIS AI models.

    Parameters
    ----------
    model_names : list of str
        Model identifiers.  Defaults to the five DeepGIS models.
    reference_weights : (M,) array or None
        Prior over models q(m).  Uniform if None.
    cost_budget_seconds : float
        Target mean compute cost per inference call (constraint on ⟨cost⟩).
    mi_target_nats : float or None
        Target mean mutual information gain.  If None no MI constraint.
    temperature : float
        Softmax temperature for exploration vs exploitation (0 → greedy).
    """

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        reference_weights: Optional[np.ndarray] = None,
        cost_budget_seconds: float = 5.0,
        mi_target_nats: Optional[float] = None,
        temperature: float = 1.0,
    ):
        self.model_names = model_names or list(DEEPGIS_MODELS)
        m = len(self.model_names)

        if reference_weights is not None:
            w = np.asarray(reference_weights, dtype=float)
            self._log_q = np.log(w / w.sum())
        else:
            self._log_q = -np.log(m) * np.ones(m)

        self._log_p = self._log_q.copy()
        self._lambdas = np.zeros(2)
        self.cost_budget = cost_budget_seconds
        self.mi_target = mi_target_nats
        self.temperature = temperature

        self._records: Dict[str, ModelRecord] = {
            name: ModelRecord(name=name) for name in self.model_names
        }
        self._selection_history: List[str] = []
        self._fp_detector = FixedPointDetector(tol=1e-2, window=5)
        self._kernel_snapshots: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Registration of outcomes
    # ------------------------------------------------------------------

    def register_outcome(
        self,
        model_name: str,
        cost_seconds: float,
        mi_gain_nats: float,
        kernel_matrix: Optional[np.ndarray] = None,
    ) -> "ModelKernelSelector":
        """Record the outcome of using a model on a scene.

        Parameters
        ----------
        model_name : str
        cost_seconds : float  Wall time (or GPU time) used.
        mi_gain_nats : float  Mutual information gained (nats).
        kernel_matrix : (N,N) or None  Kernel from model's embeddings.
        """
        rec = self._records.get(model_name)
        if rec is None:
            raise KeyError(f"Unknown model '{model_name}'.")
        rec.total_uses += 1
        rec.total_cost_seconds += cost_seconds
        rec.total_mi_gain_nats += mi_gain_nats
        if kernel_matrix is not None:
            rec.last_kernel_matrix = np.asarray(kernel_matrix, dtype=float)
        return self

    # ------------------------------------------------------------------
    # MaxCal distribution update
    # ------------------------------------------------------------------

    def update(self) -> "ModelKernelSelector":
        """Re-fit Lagrange multipliers from accumulated model statistics.

        Uses per-model mean cost and mean MI gain as constraint features.
        """
        m = len(self.model_names)
        costs = np.array([self._records[n].mean_cost for n in self.model_names])
        mi_gains = np.array([self._records[n].mean_mi_gain for n in self.model_names])

        # Build constraint feature matrix: [cost, -MI_gain] per model
        F_matrix = np.column_stack([costs, -mi_gains])
        targets = np.array([self.cost_budget,
                            -(self.mi_target if self.mi_target is not None
                              else float(np.mean(mi_gains)))])

        if np.all(costs == 0) and np.all(mi_gains == 0):
            self._log_p = self._log_q.copy()
        else:
            lambdas, _ = fit_lagrange_multipliers(
                self._log_q, F_matrix, targets, lambda0=self._lambdas
            )
            self._lambdas = lambdas
            self._log_p = maxcal_log_weights(self._log_q, lambdas, F_matrix)

        # Apply temperature scaling for exploration
        if self.temperature != 1.0 and self.temperature > 0:
            log_p_scaled = self._log_p / self.temperature
            self._log_p = log_p_scaled - logsumexp(log_p_scaled)

        # Track for fixed-point detection
        K_sel = self._selection_kernel()
        self._kernel_snapshots.append(K_sel)
        self._fp_detector.update(K_sel)
        return self

    def _selection_kernel(self) -> np.ndarray:
        """Build a kernel matrix over models from their embedding similarity."""
        p = self.distribution()
        matrices = [
            self._records[n].last_kernel_matrix
            for n in self.model_names
            if self._records[n].last_kernel_matrix is not None
        ]
        if not matrices:
            # Fallback: cost-MI efficiency as 1-D feature
            feats = np.array([[self._records[n].efficiency]
                              for n in self.model_names])
            K = feats @ feats.T
        else:
            # Use the selection probability as a diagonal kernel
            K = np.diag(p)
        return K

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self) -> str:
        """Return the model name with the highest current probability."""
        idx = int(np.argmax(self._log_p))
        name = self.model_names[idx]
        self._selection_history.append(name)
        return name

    def sample_model(self) -> str:
        """Sample a model according to the current MaxCal distribution."""
        p = self.distribution()
        idx = int(np.random.choice(len(self.model_names), p=p))
        name = self.model_names[idx]
        self._selection_history.append(name)
        return name

    def distribution(self) -> np.ndarray:
        """Current probability distribution over models."""
        return np.exp(self._log_p - logsumexp(self._log_p))

    # ------------------------------------------------------------------
    # Fixed-point / stability
    # ------------------------------------------------------------------

    def is_at_fixed_point(self) -> bool:
        return self._fp_detector.is_fixed_point()

    def stability_score(self) -> float:
        return self._fp_detector.stability_score()

    # ------------------------------------------------------------------
    # Kernel distances between models
    # ------------------------------------------------------------------

    def pairwise_kernel_distances(self) -> Optional[np.ndarray]:
        """(M, M) HS-distance matrix between models that have kernel matrices."""
        names_with_K = [
            n for n in self.model_names
            if self._records[n].last_kernel_matrix is not None
        ]
        if len(names_with_K) < 2:
            return None
        m = len(names_with_K)
        D = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                Ki = self._records[names_with_K[i]].last_kernel_matrix
                Kj = self._records[names_with_K[j]].last_kernel_matrix
                if Ki.shape == Kj.shape:
                    d = hilbert_schmidt_distance(Ki, Kj)
                    D[i, j] = D[j, i] = d
        return D

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        dist = self.distribution()
        return {
            "models": self.model_names,
            "probabilities": dist.tolist(),
            "selected": self.model_names[int(np.argmax(dist))],
            "lagrange_multipliers": self._lambdas.tolist(),
            "is_fixed_point": self.is_at_fixed_point(),
            "stability_score": self.stability_score(),
            "model_stats": {
                n: {
                    "uses": self._records[n].total_uses,
                    "mean_cost_s": self._records[n].mean_cost,
                    "mean_mi_gain_nats": self._records[n].mean_mi_gain,
                    "efficiency_nats_per_s": self._records[n].efficiency,
                }
                for n in self.model_names
            },
        }

    def __repr__(self) -> str:
        dist = self.distribution()
        best = self.model_names[int(np.argmax(dist))]
        return (f"ModelKernelSelector("
                f"models={self.model_names}, "
                f"best='{best}' ({dist.max():.2f}), "
                f"fixed_point={self.is_at_fixed_point()})")

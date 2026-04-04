"""
Semantic SLAM kernel tracker for the Earth Rover.

Tracks how the feature representation of the environment evolves as the rover
explores, using HS distance between consecutive SLAM keyframe kernels as a
continuous novelty and loop-closure confidence measure.

Maps to Thread 1 of NAVIGATION.md:
  - ‖K(t) − K(t−1)‖_HS  → novelty score (large = novel terrain)
  - 1/(1 + d_HS(K_now, K_stored)) → loop closure confidence
  - FixedPointDetector → "rover knows where it is"

Designed to consume ORB-SLAM3 descriptor outputs (binary or float vectors)
via a simple Python callback.  ROS2 wrappers are in ros_bridge.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..kernel.space import (
    hilbert_schmidt_distance,
    kernel_from_embeddings,
    rbf_kernel,
    project_to_psd,
)
from ..kernel.trajectory import KernelTrajectory
from ..kernel.fixed_points import FixedPointDetector
from ..assembly.complexity import spectral_complexity, rkhs_norm


# ---------------------------------------------------------------------------
# Descriptor → kernel helpers
# ---------------------------------------------------------------------------

def descriptors_to_kernel(
    descriptors: np.ndarray,
    mode: str = "cosine",
    rbf_scale: float = 1.0,
) -> np.ndarray:
    """Build a kernel matrix from a set of feature descriptors.

    Parameters
    ----------
    descriptors : (M, D) array of feature vectors (float or binary).
    mode : 'cosine' | 'rbf' | 'hamming_rbf'
        'cosine'      — normalised inner product (default, works for ORB floats)
        'rbf'         — Gaussian RBF over L2 distance
        'hamming_rbf' — Gaussian RBF over Hamming distance (for binary ORB)
    rbf_scale : float — length scale for RBF variants.

    Returns
    -------
    K : (M, M) PSD kernel matrix.
    """
    D = np.asarray(descriptors, dtype=float)
    if D.ndim == 1:
        D = D[None, :]
    if len(D) == 0:
        return np.zeros((0, 0))

    if mode == "cosine":
        return kernel_from_embeddings(D)
    elif mode == "rbf":
        return rbf_kernel(D, length_scale=rbf_scale)
    elif mode == "hamming_rbf":
        # Approximate Hamming distance via L1 on binarised descriptors
        D_bin = (D > 0.5).astype(float)
        n = D_bin.shape[0]
        hamming = np.zeros((n, n))
        for i in range(n):
            hamming[i] = np.sum(D_bin != D_bin[i], axis=1)
        return np.exp(-hamming / (2 * rbf_scale ** 2))
    else:
        raise ValueError(f"Unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# SLAM Kernel Tracker
# ---------------------------------------------------------------------------

@dataclass
class KeyframeRecord:
    keyframe_id: int
    kernel: np.ndarray
    novelty_score: float
    n_descriptors: int
    complexity: float


class SemanticSLAMKernelTracker:
    """Tracks kernel evolution of a SLAM map over time.

    Feed ORB-SLAM3 (or any SLAM system) descriptor batches frame-by-frame.
    The tracker maintains a KernelTrajectory, detects fixed points (well-mapped
    regions), and computes loop-closure confidence against stored keyframes.

    Parameters
    ----------
    descriptor_dim : int
        Dimensionality of feature descriptors (e.g. 32 for ORB).
    descriptor_mode : str
        Kernel construction mode ('cosine', 'rbf', 'hamming_rbf').
    rbf_scale : float
        Length scale for RBF kernel (in descriptor distance units).
    fixed_point_tol : float
        HS distance below which the map is considered stable.
    fixed_point_window : int
        Consecutive stable frames required.
    max_stored_keyframes : int
        Maximum number of keyframes to store for loop-closure queries.
    """

    def __init__(
        self,
        descriptor_dim: int = 32,
        descriptor_mode: str = "cosine",
        rbf_scale: float = 1.0,
        fixed_point_tol: float = 1e-2,
        fixed_point_window: int = 5,
        max_stored_keyframes: int = 500,
    ):
        self.descriptor_dim = descriptor_dim
        self.descriptor_mode = descriptor_mode
        self.rbf_scale = rbf_scale
        self.max_stored_keyframes = max_stored_keyframes

        self._trajectory = KernelTrajectory(name="SLAM kernel")
        self._fp_detector = FixedPointDetector(
            tol=fixed_point_tol, window=fixed_point_window
        )
        self._keyframes: Dict[int, KeyframeRecord] = {}
        self._step: int = 0
        self._prev_kernel: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        descriptors: np.ndarray,
        keyframe_id: Optional[int] = None,
        store_keyframe: bool = True,
    ) -> float:
        """Process a new batch of descriptors from a SLAM keyframe.

        Parameters
        ----------
        descriptors : (M, D) descriptor matrix from the current frame.
        keyframe_id : int or None — if None, auto-incremented.
        store_keyframe : bool — whether to store for loop-closure queries.

        Returns
        -------
        novelty_score : float — HS distance from the previous frame's kernel.
                        Large = novel environment.  Zero = first frame.
        """
        descriptors = np.asarray(descriptors, dtype=float)
        if keyframe_id is None:
            keyframe_id = self._step

        K = descriptors_to_kernel(
            descriptors,
            mode=self.descriptor_mode,
            rbf_scale=self.rbf_scale,
        )

        novelty = (
            hilbert_schmidt_distance(self._prev_kernel, K)
            if self._prev_kernel is not None
            and self._prev_kernel.shape == K.shape
            else 0.0
        )

        self._trajectory.add(t=float(self._step), K=K)
        self._fp_detector.update(K)

        record = KeyframeRecord(
            keyframe_id=keyframe_id,
            kernel=K,
            novelty_score=novelty,
            n_descriptors=len(descriptors),
            complexity=spectral_complexity(K),
        )

        if store_keyframe:
            self._keyframes[keyframe_id] = record
            # Evict oldest if over capacity
            if len(self._keyframes) > self.max_stored_keyframes:
                oldest = min(self._keyframes)
                del self._keyframes[oldest]

        self._prev_kernel = K
        self._step += 1
        return novelty

    # ------------------------------------------------------------------
    # Novelty
    # ------------------------------------------------------------------

    def novelty_score(self) -> float:
        """HS distance between the current and previous keyframe kernels."""
        dists = self._trajectory.segment_distances()
        return float(dists[-1]) if len(dists) > 0 else 0.0

    def novelty_map(self) -> np.ndarray:
        """Full history of per-step novelty scores."""
        return self._trajectory.segment_distances()

    def is_novel_terrain(self, threshold: Optional[float] = None) -> bool:
        """True if the current frame is significantly different from the last."""
        thr = threshold if threshold is not None else self._fp_detector.tol * 5
        return self.novelty_score() > thr

    # ------------------------------------------------------------------
    # Loop closure
    # ------------------------------------------------------------------

    def loop_closure_confidence(
        self,
        query_descriptors: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[float, Optional[int]]:
        """Estimate loop-closure confidence against stored keyframes.

        Parameters
        ----------
        query_descriptors : (M, D) descriptors from current frame.
        top_k : int — consider only the top-k most similar keyframes.

        Returns
        -------
        confidence : float in [0, 1].  Higher = more likely revisiting known place.
        best_keyframe_id : int or None — ID of the most similar stored keyframe.
        """
        if not self._keyframes:
            return 0.0, None

        K_query = descriptors_to_kernel(
            query_descriptors,
            mode=self.descriptor_mode,
            rbf_scale=self.rbf_scale,
        )

        best_conf = 0.0
        best_id = None

        for kf_id, record in self._keyframes.items():
            if record.kernel.shape != K_query.shape:
                continue
            d = hilbert_schmidt_distance(K_query, record.kernel)
            conf = 1.0 / (1.0 + d)
            if conf > best_conf:
                best_conf = conf
                best_id = kf_id

        return float(best_conf), best_id

    def top_k_loop_closure_candidates(
        self, query_descriptors: np.ndarray, k: int = 5
    ) -> List[Tuple[int, float]]:
        """Return the k keyframes with highest loop-closure confidence.

        Returns
        -------
        list of (keyframe_id, confidence) sorted descending.
        """
        K_query = descriptors_to_kernel(
            query_descriptors,
            mode=self.descriptor_mode,
            rbf_scale=self.rbf_scale,
        )
        results = []
        for kf_id, record in self._keyframes.items():
            if record.kernel.shape != K_query.shape:
                continue
            d = hilbert_schmidt_distance(K_query, record.kernel)
            results.append((kf_id, 1.0 / (1.0 + d)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    # ------------------------------------------------------------------
    # Fixed-point / map stability
    # ------------------------------------------------------------------

    def is_well_mapped(self) -> bool:
        """True if the SLAM kernel has converged — rover knows where it is."""
        return self._fp_detector.is_fixed_point()

    def map_stability_score(self) -> float:
        """Scalar stability score in [0,1].  1 = perfectly stable map."""
        return self._fp_detector.stability_score()

    def classify_map_state(self) -> str:
        """'stable_fp' | 'transient' | 'oscillating' | 'insufficient_data'."""
        return self._fp_detector.classify()

    # ------------------------------------------------------------------
    # Semantic complexity (assembly index proxy)
    # ------------------------------------------------------------------

    def current_complexity(self) -> float:
        """Von Neumann entropy of the current frame's kernel."""
        if self._prev_kernel is None:
            return 0.0
        return spectral_complexity(self._prev_kernel)

    def complexity_history(self) -> np.ndarray:
        """Complexity at each stored keyframe."""
        kfs = sorted(self._keyframes.values(), key=lambda r: r.keyframe_id)
        return np.array([r.complexity for r in kfs])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        dists = self.novelty_map()
        return {
            "n_keyframes_processed": self._step,
            "n_keyframes_stored": len(self._keyframes),
            "current_novelty": self.novelty_score(),
            "mean_novelty": float(np.mean(dists)) if len(dists) > 0 else 0.0,
            "map_stability": self.map_stability_score(),
            "map_state": self.classify_map_state(),
            "is_well_mapped": self.is_well_mapped(),
            "current_complexity": self.current_complexity(),
        }

    def __repr__(self) -> str:
        return (
            f"SemanticSLAMKernelTracker("
            f"frames={self._step}, "
            f"stored_kf={len(self._keyframes)}, "
            f"novelty={self.novelty_score():.4f}, "
            f"state={self.classify_map_state()})"
        )

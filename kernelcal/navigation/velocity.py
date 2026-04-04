"""
Terrain-aware velocity control via kernel dynamics.

Maps the RKHS state of the SLAM kernel — novelty, stability, spectral complexity —
to a scalar velocity command in [v_min, v_max].

Design principle
----------------
Speed is governed by two complementary kernel signals:

  1. **Novelty** (‖K(t) − K(t−1)‖_HS):
       High novelty  → unknown terrain   → slow down (explore carefully)
       Low novelty   → familiar terrain  → speed up

  2. **Stability** (FixedPointDetector score ∈ [0,1]):
       Low stability → transient phase   → cautious speed
       High stability → fixed point      → confident speed

Combined via a multiplicative gate:

    v(t) = v_max × σ_novelty(t) × σ_stability(t) × σ_complexity(t)

where each σ is a monotone mapping to [σ_min, 1].

Additionally, a *look-ahead* kernel check can compare the descriptor kernel at the
candidate next waypoint against the SLAM fixed-point kernel — high distance → preemptive
slow-down before entering unknown terrain, analogous to a preview controller.

ROS 2 integration is in ros_bridge.py (KernelVelocityNode).

ros2_ws packages consumed
--------------------------
  /orb_slam3/tracking_state   (std_msgs/Int32: 0=no img, 1=not init, 2=OK, 3=lost)
  /orb_slam3/map_points       (sensor_msgs/PointCloud2 — sparse 3-D SLAM points)
  /orb_slam3/camera_pose      (geometry_msgs/PoseStamped)
  /fmu/in/trajectory_setpoint (px4_msgs/TrajectorySetpoint — velocity field)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..kernel.space import hilbert_schmidt_distance
from ..kernel.fixed_points import FixedPointDetector
from ..assembly.complexity import spectral_complexity


# ---------------------------------------------------------------------------
# ORB-SLAM3 tracking state codes (mirrors orb_slam3 ROS wrapper convention)
# ---------------------------------------------------------------------------

TRACKING_NO_IMAGE       = 0
TRACKING_NOT_INITIALISED = 1
TRACKING_OK             = 2
TRACKING_LOST           = 3


# ---------------------------------------------------------------------------
# Velocity gate functions
# ---------------------------------------------------------------------------

def _sigmoid_gate(x: float, centre: float, sharpness: float = 8.0) -> float:
    """Smooth monotone gate: 0 → near 0 as x→∞, 1 → near 1 as x→−∞."""
    return float(1.0 / (1.0 + np.exp(sharpness * (x - centre))))


def novelty_to_speed_factor(
    novelty: float,
    novelty_safe: float = 0.05,
    novelty_danger: float = 0.8,
    min_factor: float = 0.10,
) -> float:
    """Map HS novelty score → speed factor ∈ [min_factor, 1.0].

    Below novelty_safe  → factor = 1.0 (familiar ground, full speed)
    Above novelty_danger → factor = min_factor (unknown terrain, crawl)
    """
    if novelty <= novelty_safe:
        return 1.0
    if novelty >= novelty_danger:
        return min_factor
    t = (novelty - novelty_safe) / (novelty_danger - novelty_safe)
    return float(min_factor + (1.0 - min_factor) * (1.0 - t))


def stability_to_speed_factor(
    stability: float,
    min_factor: float = 0.25,
) -> float:
    """Map stability score [0,1] → speed factor [min_factor, 1.0]."""
    return float(min_factor + (1.0 - min_factor) * stability)


def complexity_to_speed_factor(
    complexity: float,
    complexity_ref: float = 2.0,
    sharpness: float = 3.0,
    min_factor: float = 0.15,
) -> float:
    """Map spectral complexity → speed factor.

    High complexity (rough, feature-rich terrain) → reduce speed.
    complexity_ref: expected complexity on familiar, flat terrain.
    """
    excess = max(complexity - complexity_ref, 0.0)
    factor = np.exp(-sharpness * excess / (complexity_ref + 1e-9))
    return float(min_factor + (1.0 - min_factor) * factor)


def tracking_state_to_speed_factor(state: int) -> float:
    """Map ORB-SLAM3 tracking state → emergency speed factor."""
    if state == TRACKING_OK:
        return 1.0
    elif state == TRACKING_NOT_INITIALISED:
        return 0.30
    elif state == TRACKING_LOST:
        return 0.0   # full stop — no localisation
    else:
        return 0.10  # no image


# ---------------------------------------------------------------------------
# Point cloud → kernel (for ORB-SLAM3 map points)
# ---------------------------------------------------------------------------

def map_points_to_kernel(
    points_xyz: np.ndarray,
    fov_radius: float = 3.0,
    n_sample: int = 40,
    length_scale: float = 1.0,
) -> Optional[np.ndarray]:
    """Build a local kernel matrix from nearby ORB-SLAM3 map points.

    Selects up to n_sample points within fov_radius metres, then computes
    an RBF kernel over their 3-D coordinates.  Returns None if fewer than
    2 points are in range.
    """
    if len(points_xyz) == 0:
        return None
    pts = np.asarray(points_xyz, dtype=float)
    dists = np.linalg.norm(pts, axis=1)
    nearby = pts[dists <= fov_radius]
    if len(nearby) < 2:
        return None
    if len(nearby) > n_sample:
        idx = np.random.choice(len(nearby), n_sample, replace=False)
        nearby = nearby[idx]
    from ..kernel.space import rbf_kernel
    return rbf_kernel(nearby, length_scale=length_scale)


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

@dataclass
class VelocityBand:
    """Speed band definition."""
    v_min: float = 0.0
    v_nominal: float = 1.5
    v_max: float = 3.0       # m/s (trike field speed)
    v_crawl: float = 0.3     # m/s used when tracking is lost


@dataclass
class VelocityRecord:
    step: int
    v_cmd: float
    novelty: float
    stability: float
    complexity: float
    tracking_state: int
    f_novelty: float
    f_stability: float
    f_complexity: float
    f_tracking: float


class TerrainKernelVelocityController:
    """Maps live kernel dynamics to a scalar forward-velocity command.

    Parameters
    ----------
    band : VelocityBand
        Speed limits for the platform.
    novelty_safe : float
        HS novelty below which the terrain is considered fully familiar.
    novelty_danger : float
        HS novelty above which the rover crawls.
    min_novelty_factor : float
        Minimum speed fraction when novelty is maximal.
    min_stability_factor : float
        Minimum speed fraction when stability is zero (transient phase).
    complexity_ref : float
        Baseline spectral complexity of flat, well-mapped terrain.
    use_look_ahead : bool
        If True, additionally gates speed on the HS distance between the
        next-waypoint kernel and the current SLAM fixed-point kernel.
    smoothing_alpha : float
        Exponential smoothing coefficient on the output velocity (0 = no
        smoothing, 1 = never updates).
    fixed_point_tol : float
        Tolerance for map-stability fixed-point detection.
    """

    def __init__(
        self,
        band: Optional[VelocityBand] = None,
        novelty_safe: float = 0.05,
        novelty_danger: float = 0.8,
        min_novelty_factor: float = 0.10,
        min_stability_factor: float = 0.25,
        complexity_ref: float = 2.0,
        use_look_ahead: bool = True,
        smoothing_alpha: float = 0.25,
        fixed_point_tol: float = 0.05,
        fixed_point_window: int = 4,
    ):
        self.band = band or VelocityBand()
        self.novelty_safe = novelty_safe
        self.novelty_danger = novelty_danger
        self.min_novelty_factor = min_novelty_factor
        self.min_stability_factor = min_stability_factor
        self.complexity_ref = complexity_ref
        self.use_look_ahead = use_look_ahead
        self.smoothing_alpha = smoothing_alpha

        self._fp_detector = FixedPointDetector(
            tol=fixed_point_tol, window=fixed_point_window
        )
        self._fixed_point_kernel: Optional[np.ndarray] = None
        self._v_smooth: float = self.band.v_nominal
        self._step: int = 0
        self._history: List[VelocityRecord] = []

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        novelty: float,
        stability: float,
        current_kernel: Optional[np.ndarray] = None,
        complexity: Optional[float] = None,
        tracking_state: int = TRACKING_OK,
        next_waypoint_kernel: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the next forward velocity command.

        Parameters
        ----------
        novelty : float — HS distance from previous kernel (from SemanticSLAMKernelTracker)
        stability : float in [0,1] — map stability score
        current_kernel : (N,N) current SLAM kernel matrix (optional, for complexity)
        complexity : float — spectral complexity override (computed from kernel if None)
        tracking_state : int — ORB-SLAM3 tracking state code
        next_waypoint_kernel : (M,M) kernel for the terrain ahead (look-ahead gate)

        Returns
        -------
        v_cmd : float — forward velocity in m/s
        """
        # Complexity
        if complexity is None and current_kernel is not None:
            complexity = spectral_complexity(current_kernel)
        elif complexity is None:
            complexity = self.complexity_ref

        # Update fixed-point kernel (stored for look-ahead)
        if current_kernel is not None:
            self._fp_detector.update(current_kernel)
            if self._fp_detector.is_fixed_point():
                self._fixed_point_kernel = current_kernel.copy()

        # Individual speed factors
        f_nov   = novelty_to_speed_factor(
            novelty, self.novelty_safe, self.novelty_danger, self.min_novelty_factor
        )
        f_stab  = stability_to_speed_factor(stability, self.min_stability_factor)
        f_cplx  = complexity_to_speed_factor(
            complexity, self.complexity_ref, min_factor=0.15
        )
        f_track = tracking_state_to_speed_factor(tracking_state)

        # Look-ahead gate: preemptive slow-down before entering novel terrain
        f_look = 1.0
        if (self.use_look_ahead
                and next_waypoint_kernel is not None
                and self._fixed_point_kernel is not None
                and next_waypoint_kernel.shape == self._fixed_point_kernel.shape):
            d_ahead = hilbert_schmidt_distance(
                next_waypoint_kernel, self._fixed_point_kernel
            )
            f_look = novelty_to_speed_factor(
                d_ahead, self.novelty_safe, self.novelty_danger,
                self.min_novelty_factor
            )

        # Combined factor
        combined = f_nov * f_stab * f_cplx * f_track * f_look

        # Map to velocity range
        v_target = self.band.v_min + combined * (self.band.v_max - self.band.v_min)

        # Hard floor: when tracking is lost, crawl or stop
        if tracking_state == TRACKING_LOST:
            v_target = 0.0
        elif tracking_state != TRACKING_OK:
            v_target = min(v_target, self.band.v_crawl)

        # Exponential smoothing
        self._v_smooth = (
            (1 - self.smoothing_alpha) * self._v_smooth
            + self.smoothing_alpha * v_target
        )
        v_cmd = float(np.clip(self._v_smooth, self.band.v_min, self.band.v_max))

        self._history.append(VelocityRecord(
            step=self._step,
            v_cmd=v_cmd,
            novelty=novelty,
            stability=stability,
            complexity=complexity,
            tracking_state=tracking_state,
            f_novelty=f_nov,
            f_stability=f_stab,
            f_complexity=f_cplx,
            f_tracking=f_track,
        ))
        self._step += 1
        return v_cmd

    # ------------------------------------------------------------------
    # Batch update from SemanticSLAMKernelTracker history
    # ------------------------------------------------------------------

    def replay_from_tracker(
        self,
        novelty_series: np.ndarray,
        stability_series: np.ndarray,
        complexity_series: Optional[np.ndarray] = None,
        tracking_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute velocity profile from pre-recorded tracker series.

        Useful for offline evaluation and plotting without a live ROS session.
        """
        n = len(novelty_series)
        v_profile = np.zeros(n)
        if complexity_series is None:
            complexity_series = np.full(n, self.complexity_ref)
        if tracking_states is None:
            tracking_states = np.full(n, TRACKING_OK, dtype=int)

        for i in range(n):
            v_profile[i] = self.update(
                novelty=float(novelty_series[i]),
                stability=float(stability_series[i]),
                complexity=float(complexity_series[i]),
                tracking_state=int(tracking_states[i]),
            )
        return v_profile

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def velocity_history(self) -> np.ndarray:
        return np.array([r.v_cmd for r in self._history])

    def factor_histories(self) -> dict:
        return {
            "novelty_factor":     np.array([r.f_novelty    for r in self._history]),
            "stability_factor":   np.array([r.f_stability  for r in self._history]),
            "complexity_factor":  np.array([r.f_complexity for r in self._history]),
            "tracking_factor":    np.array([r.f_tracking   for r in self._history]),
            "novelty":            np.array([r.novelty      for r in self._history]),
            "stability":          np.array([r.stability    for r in self._history]),
            "complexity":         np.array([r.complexity   for r in self._history]),
        }

    def is_stopped(self) -> bool:
        return self._v_smooth < 1e-3

    def current_speed(self) -> float:
        return self._v_smooth

    def summary(self) -> dict:
        h = self.velocity_history()
        return {
            "n_steps":       self._step,
            "mean_v":        float(np.mean(h)) if len(h) else 0.0,
            "min_v":         float(np.min(h))  if len(h) else 0.0,
            "max_v":         float(np.max(h))  if len(h) else 0.0,
            "stops":         int(np.sum(h < 0.05)),
            "crawl_steps":   int(np.sum(h < self.band.v_crawl)),
            "full_speed_pct": float(np.mean(h > 0.9 * self.band.v_max)) * 100,
            "fixed_point_reached": self._fp_detector.is_fixed_point(),
        }

    def __repr__(self) -> str:
        return (
            f"TerrainKernelVelocityController("
            f"v=[{self.band.v_min:.1f},{self.band.v_max:.1f}] m/s, "
            f"v_now={self.current_speed():.2f} m/s, "
            f"steps={self._step})"
        )

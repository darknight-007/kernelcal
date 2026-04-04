"""
Kernel trajectories: paths γ : [0,T] → K through kernel space.

A KernelTrajectory stores a sequence of (time, kernel_matrix) snapshots and
exposes the path geometry: cumulative HS distance, velocity, linear
interpolation, and convergence checks.  This is the discrete analogue of a
Bochner-integrable trajectory in (K, d_HS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from .space import hilbert_schmidt_distance, project_to_psd


@dataclass
class _Snapshot:
    t: float
    K: np.ndarray


class KernelTrajectory:
    """Ordered sequence of kernel snapshots with path-geometric utilities.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "NTK during fine-tuning", "sampler kernel").
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._snapshots: List[_Snapshot] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add(self, t: float, K: np.ndarray) -> "KernelTrajectory":
        """Append a snapshot at time t.  Returns self for chaining."""
        K = np.asarray(K, dtype=float)
        if self._snapshots and K.shape != self._snapshots[-1].K.shape:
            raise ValueError(
                f"All kernel matrices must share the same shape; "
                f"got {K.shape}, expected {self._snapshots[-1].K.shape}"
            )
        self._snapshots.append(_Snapshot(t=float(t), K=K.copy()))
        return self

    @classmethod
    def from_sequence(cls, kernels: List[np.ndarray],
                      times: Optional[List[float]] = None,
                      name: str = "") -> "KernelTrajectory":
        """Build a trajectory from a list of kernel matrices."""
        traj = cls(name=name)
        if times is None:
            times = list(range(len(kernels)))
        for t, K in zip(times, kernels):
            traj.add(t, K)
        return traj

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._snapshots)

    def times(self) -> np.ndarray:
        return np.array([s.t for s in self._snapshots])

    def kernels(self) -> List[np.ndarray]:
        return [s.K for s in self._snapshots]

    def __getitem__(self, idx: int) -> Tuple[float, np.ndarray]:
        s = self._snapshots[idx]
        return s.t, s.K

    # ------------------------------------------------------------------
    # Path geometry
    # ------------------------------------------------------------------

    def segment_distances(self) -> np.ndarray:
        """HS distance between each consecutive pair of snapshots."""
        if len(self) < 2:
            return np.array([])
        return np.array([
            hilbert_schmidt_distance(self._snapshots[i].K,
                                     self._snapshots[i + 1].K)
            for i in range(len(self) - 1)
        ])

    def path_length(self) -> float:
        """Total cumulative HS distance along the trajectory."""
        return float(np.sum(self.segment_distances()))

    def cumulative_length(self) -> np.ndarray:
        """Cumulative path length at each snapshot (length n, starts at 0)."""
        segs = self.segment_distances()
        return np.concatenate([[0.0], np.cumsum(segs)])

    def velocities(self) -> np.ndarray:
        """Instantaneous speed ‖dK/dt‖_HS at each interior segment."""
        if len(self) < 2:
            return np.array([])
        dists = self.segment_distances()
        dt = np.diff(self.times())
        dt = np.where(dt == 0, 1e-12, dt)
        return dists / dt

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate(self, t: float) -> np.ndarray:
        """Linearly interpolate the kernel matrix at an arbitrary time t.

        For t outside [t_0, t_N], returns the nearest endpoint.
        """
        times = self.times()
        if t <= times[0]:
            return self._snapshots[0].K.copy()
        if t >= times[-1]:
            return self._snapshots[-1].K.copy()
        idx = int(np.searchsorted(times, t)) - 1
        t0, K0 = self._snapshots[idx].t, self._snapshots[idx].K
        t1, K1 = self._snapshots[idx + 1].t, self._snapshots[idx + 1].K
        alpha = (t - t0) / (t1 - t0)
        K_interp = (1 - alpha) * K0 + alpha * K1
        return project_to_psd(K_interp)

    # ------------------------------------------------------------------
    # Convergence analysis
    # ------------------------------------------------------------------

    def is_convergent(self, tol: float = 1e-3, window: int = 5) -> bool:
        """Return True if the last *window* segment distances are all < tol."""
        dists = self.segment_distances()
        if len(dists) < window:
            return False
        return bool(np.all(dists[-window:] < tol))

    def convergence_time(self, tol: float = 1e-3, window: int = 3) -> Optional[float]:
        """Return the time at which the trajectory first enters the tol-ball.

        Returns None if not yet converged.
        """
        dists = self.segment_distances()
        for i in range(len(dists) - window + 1):
            if np.all(dists[i: i + window] < tol):
                return float(self._snapshots[i].t)
        return None

    def decay_rate(self) -> float:
        """Fit an exponential d(t) = A·exp(−λt) to segment distances.

        Returns the decay exponent λ (positive = converging).
        Uses least-squares fit in log space; returns 0 if not enough data.
        """
        dists = self.segment_distances()
        if len(dists) < 3:
            return 0.0
        times = 0.5 * (self.times()[:-1] + self.times()[1:])
        pos_mask = dists > 0
        if np.sum(pos_mask) < 2:
            return 0.0
        log_d = np.log(dists[pos_mask])
        t_fit = times[pos_mask]
        # least-squares: log_d = log_A - λ·t  →  λ = −slope
        A = np.vstack([t_fit, np.ones_like(t_fit)]).T
        result = np.linalg.lstsq(A, log_d, rcond=None)
        slope = result[0][0]
        return float(-slope)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        dists = self.segment_distances()
        lines = [
            f"KernelTrajectory '{self.name}'",
            f"  snapshots : {len(self)}",
            f"  time span : [{self.times()[0]:.3g}, {self.times()[-1]:.3g}]" if len(self) > 0 else "",
            f"  path length : {self.path_length():.4f}",
            f"  mean speed  : {np.mean(self.velocities()):.4f}" if len(dists) > 0 else "",
            f"  decay rate  : {self.decay_rate():.4f}",
            f"  converged   : {self.is_convergent()}",
        ]
        return "\n".join(l for l in lines if l)

    def __repr__(self) -> str:
        return (f"KernelTrajectory(name={self.name!r}, "
                f"snapshots={len(self)}, length={self.path_length():.4f})")

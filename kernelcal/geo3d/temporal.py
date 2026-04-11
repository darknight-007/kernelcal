"""Temporal spectral compression for point-cloud or LiDAR sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..kernel import KernelTrajectory, hilbert_schmidt_distance
from .spectral_codec import CompressedSpectralKernel, compress_point_cloud


@dataclass
class TemporalKernelSummary:
    """Compressed per-frame kernels and Hilbert-Schmidt path diagnostics."""

    times: np.ndarray
    compressed_frames: list[CompressedSpectralKernel]
    hs_distances: np.ndarray
    meta: dict = field(default_factory=dict)
    trajectory: KernelTrajectory | None = None

    def total_path_length(self) -> float:
        return float(np.sum(self.hs_distances))


def compress_temporal_clouds(
    clouds: Sequence[np.ndarray],
    times: Sequence[float] | None = None,
    *,
    max_points: int = 256,
    k_neighbors: int = 8,
    sigma: float = 1.0,
    n_modes: int = 24,
    heat_tau: float | None = 1.0,
    seed: int | None = 0,
    stable_subsample: bool = True,
) -> TemporalKernelSummary:
    """Compress each frame, then compute consecutive HS distances.

    Parameters
    ----------
    stable_subsample : bool, default True
        When True (recommended for temporal sequences), every frame draws
        its subsample using the same RNG seed.  This ensures that
        frame-to-frame HS distances reflect genuine scene change rather than
        sampling variation.  Set to False only when frames have very
        different point densities that make a fixed subsample unrepresentative.
    """
    if len(clouds) == 0:
        raise ValueError("clouds must be non-empty.")
    tlist = list(times) if times is not None else [float(i) for i in range(len(clouds))]
    if len(tlist) != len(clouds):
        raise ValueError("times length must match clouds.")

    min_count = min(np.asarray(pc).shape[0] for pc in clouds)
    if min_count < 2:
        raise ValueError("Each frame must contain at least 2 points.")
    target_points = min(int(max_points), int(min_count))

    frames: list[CompressedSpectralKernel] = []
    kernels: list[np.ndarray] = []
    for i, pc in enumerate(clouds):
        # stable_subsample=True: same seed for every frame so that HS distances
        # measure scene change, not sampling variation.
        frame_seed = seed if (stable_subsample or seed is None) else seed + i
        c = compress_point_cloud(
            pc,
            max_points=target_points,
            k_neighbors=k_neighbors,
            sigma=sigma,
            n_modes=n_modes,
            heat_tau=heat_tau,
            seed=frame_seed,
        )
        frames.append(c)
        kernels.append((c.eigenvectors * c.h) @ c.eigenvectors.T)

    hs = [
        hilbert_schmidt_distance(kernels[i + 1], kernels[i])
        for i in range(len(kernels) - 1)
    ]

    traj = KernelTrajectory(name="lidar_sequence")
    for t, K in zip(tlist, kernels):
        traj.add(t, K)

    return TemporalKernelSummary(
        times=np.asarray(tlist, dtype=float),
        compressed_frames=frames,
        hs_distances=np.asarray(hs, dtype=float),
        meta={
            "n_frames": len(clouds),
            "points_per_frame": target_points,
            "stable_subsample": stable_subsample,
        },
        trajectory=traj,
    )

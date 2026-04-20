"""Spectral compression of ROS2 depth camera / LiDAR point-cloud streams.

Architecture
------------
Each incoming frame (PointCloud2 or depth image → XYZ) is compressed via
``compress_point_cloud`` (spectral graph kernel, heat-kernel weights).
The codec maintains a rolling buffer of compressed frames and publishes:

  1. Compressed frame payload (NPZ bytes) on a latched topic.
  2. HS novelty score  ‖K_t − K_{t-1}‖_HS  — same role as the terrain
     kernel velocity signal in ``kernelcal.navigation.velocity``.
  3. A ``KernelTrajectory`` covering the full rolling window — enables
     retrospective distortion analysis and keyframe selection.

Keyframe policy (MaxCal-inspired)
----------------------------------
A frame becomes a keyframe when HS novelty exceeds ``novelty_keyframe``.
Keyframes are stored at full resolution (larger n_modes); delta frames use
fewer modes.  This mirrors the thermodynamic "plasticity budget": the kernel
only updates substantially when mutual information gain justifies the cost.

No-ROS path
-----------
``DepthStreamCodec`` and all helpers run without ROS2.  Only
``DepthStreamCodecNode`` in ros_bridge.py requires rclpy.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Callable, Deque, List, Optional

import numpy as np

from ..kernel.trajectory import KernelTrajectory
from ..kernel.space import hilbert_schmidt_distance

if TYPE_CHECKING:
    # Imported only for type-checkers; ``kernelcal.geo3d`` is heavy and is
    # resolved lazily at runtime inside the functions that need it so that
    # ``import kernelcal.video`` (and therefore ``import kernelcal``) does not
    # pay the geo3d / spectral-codec startup cost unless a frame is actually
    # compressed.
    from ..geo3d.spectral_codec import CompressedSpectralKernel


# ---------------------------------------------------------------------------
# PointCloud2 → XYZ numpy (no rclpy required)
# ---------------------------------------------------------------------------

def pointcloud2_to_xyz(msg) -> np.ndarray:
    """Convert a sensor_msgs/PointCloud2 message to an (N, 3) float64 array.

    Handles common field layouts produced by Intel RealSense, Velodyne,
    Ouster, and ORB-SLAM3's map_points publisher.

    Parameters
    ----------
    msg : sensor_msgs.msg.PointCloud2

    Returns
    -------
    (N, 3) array of XYZ coordinates (NaN rows removed).
    """
    import struct

    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in ("x", "y", "z")):
        raise ValueError("PointCloud2 must contain x, y, z fields.")

    x_off = fields["x"].offset
    y_off = fields["y"].offset
    z_off = fields["z"].offset
    step  = msg.point_step
    data  = bytes(msg.data)
    n     = msg.width * msg.height

    pts = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        base = i * step
        pts[i, 0] = struct.unpack_from("f", data, base + x_off)[0]
        pts[i, 1] = struct.unpack_from("f", data, base + y_off)[0]
        pts[i, 2] = struct.unpack_from("f", data, base + z_off)[0]

    pts = pts.astype(np.float64)
    valid = np.isfinite(pts).all(axis=1)
    return pts[valid]


def depth_image_to_xyz(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 0.001,
    max_depth: float = 10.0,
) -> np.ndarray:
    """Convert a uint16 depth image to an (N, 3) XYZ point cloud.

    Parameters
    ----------
    depth       : (H, W) uint16 depth image (millimetres by default)
    fx, fy      : focal lengths in pixels
    cx, cy      : principal point in pixels
    depth_scale : metres per depth unit (0.001 for mm → m)
    max_depth   : discard points beyond this distance (metres)
    """
    H, W = depth.shape
    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    z = depth.astype(np.float64) * depth_scale
    valid = (z > 0) & (z < max_depth)

    x = (uu[valid] - cx) * z[valid] / fx
    y = (vv[valid] - cy) * z[valid] / fy
    zv = z[valid]
    return np.stack([x, y, zv], axis=1)


# ---------------------------------------------------------------------------
# Compressed frame record
# ---------------------------------------------------------------------------

@dataclass
class CompressedDepthFrame:
    """One compressed depth frame."""

    timestamp: float
    frame_idx: int
    compressed: CompressedSpectralKernel
    is_keyframe: bool
    hs_novelty: float        # ‖K_t − K_{t-1}‖_HS  (0.0 for first frame)
    n_points_raw: int        # points before subsampling

    def to_bytes(self) -> bytes:
        return self.compressed.to_bytes()

    @property
    def n_modes(self) -> int:
        return self.compressed.meta.get("n_modes", len(self.compressed.h))


# ---------------------------------------------------------------------------
# Core codec (no ROS dependency)
# ---------------------------------------------------------------------------

@dataclass
class DepthStreamConfig:
    """Compression parameters for the depth stream codec."""

    # Point cloud subsampling
    max_points: int = 256          # points per frame after subsampling
    k_neighbors: int = 8           # k-NN graph connectivity
    sigma: float = 0.5             # Gaussian weight bandwidth (metres)

    # Spectral modes
    n_modes_keyframe: int = 32     # modes for keyframes (high quality)
    n_modes_delta: int = 16        # modes for delta frames (compact)
    heat_tau: float = 1.0

    # Keyframe policy
    novelty_keyframe: float = 0.15  # HS distance threshold for new keyframe
    force_keyframe_every: int = 30  # force keyframe at least every N frames

    # Rolling buffer
    window_size: int = 60          # frames kept in memory


class DepthStreamCodec:
    """Online spectral compressor for a depth point-cloud stream.

    Usage (standalone, no ROS):
    ::
        codec = DepthStreamCodec()
        for t, cloud in stream:
            result = codec.push(cloud, timestamp=t)
            print(result.hs_novelty, result.is_keyframe)

    Usage (ROS2): see ``kernelcal.video.ros_bridge.DepthStreamCodecNode``.
    """

    def __init__(self, config: DepthStreamConfig | None = None) -> None:
        self.cfg = config or DepthStreamConfig()
        self._frame_idx: int = 0
        self._prev_kernel: Optional[np.ndarray] = None
        self._frames_since_keyframe: int = 0
        self._buffer: Deque[CompressedDepthFrame] = deque(maxlen=self.cfg.window_size)
        self.trajectory = KernelTrajectory(name="depth_stream")
        self._novelty_callbacks: List[Callable[[float], None]] = []
        self._frame_callbacks: List[Callable[[CompressedDepthFrame], None]] = []

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_novelty(self, fn: Callable[[float], None]) -> None:
        """Register a callback fired with the HS novelty score each frame."""
        self._novelty_callbacks.append(fn)

    def on_frame(self, fn: Callable[[CompressedDepthFrame], None]) -> None:
        """Register a callback fired with each CompressedDepthFrame."""
        self._frame_callbacks.append(fn)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def push(
        self,
        points_xyz: np.ndarray,
        timestamp: float | None = None,
        seed: int | None = None,
    ) -> CompressedDepthFrame:
        """Compress one point-cloud frame, update trajectory, fire callbacks.

        Parameters
        ----------
        points_xyz : (N, 3) float array — raw point cloud for this frame.
        timestamp  : frame time (seconds); defaults to wall time.
        seed       : RNG seed for reproducible subsampling (None = random).

        Returns
        -------
        CompressedDepthFrame with novelty score and keyframe flag.
        """
        if timestamp is None:
            timestamp = time.time()

        from ..geo3d.spectral_codec import compress_point_cloud

        n_raw = len(points_xyz)

        is_keyframe = (
            self._frame_idx == 0
            or self._frames_since_keyframe >= self.cfg.force_keyframe_every
        )
        n_modes = self.cfg.n_modes_keyframe if is_keyframe else self.cfg.n_modes_delta

        c = compress_point_cloud(
            points_xyz,
            max_points=self.cfg.max_points,
            k_neighbors=self.cfg.k_neighbors,
            sigma=self.cfg.sigma,
            n_modes=n_modes,
            heat_tau=self.cfg.heat_tau,
            seed=seed if seed is not None else self._frame_idx,
        )

        # Reconstruct kernel for HS distance (small: max_points × max_points)
        K_now = (c.eigenvectors * c.h) @ c.eigenvectors.T

        # HS novelty
        if self._prev_kernel is None or self._prev_kernel.shape != K_now.shape:
            hs = 0.0
        else:
            hs = hilbert_schmidt_distance(K_now, self._prev_kernel)

        # Keyframe update based on novelty
        if not is_keyframe and hs >= self.cfg.novelty_keyframe:
            is_keyframe = True
            # Re-compress at keyframe quality
            c = compress_point_cloud(
                points_xyz,
                max_points=self.cfg.max_points,
                k_neighbors=self.cfg.k_neighbors,
                sigma=self.cfg.sigma,
                n_modes=self.cfg.n_modes_keyframe,
                heat_tau=self.cfg.heat_tau,
                seed=seed if seed is not None else self._frame_idx,
            )
            K_now = (c.eigenvectors * c.h) @ c.eigenvectors.T

        self._frames_since_keyframe = 0 if is_keyframe else self._frames_since_keyframe + 1
        self._prev_kernel = K_now
        self.trajectory.add(timestamp, K_now)

        frame = CompressedDepthFrame(
            timestamp=timestamp,
            frame_idx=self._frame_idx,
            compressed=c,
            is_keyframe=is_keyframe,
            hs_novelty=hs,
            n_points_raw=n_raw,
        )
        self._buffer.append(frame)
        self._frame_idx += 1

        for cb in self._novelty_callbacks:
            cb(hs)
        for cb in self._frame_callbacks:
            cb(frame)

        return frame

    # ------------------------------------------------------------------
    # Buffer / trajectory access
    # ------------------------------------------------------------------

    def novelty_history(self) -> np.ndarray:
        """HS novelty scores for all buffered frames."""
        return np.array([f.hs_novelty for f in self._buffer])

    def keyframe_indices(self) -> list[int]:
        return [f.frame_idx for f in self._buffer if f.is_keyframe]

    def payload_bytes_total(self) -> int:
        return sum(len(f.to_bytes()) for f in self._buffer)

    def compression_summary(self) -> dict:
        """Rate and distortion summary for the current buffer."""
        frames = list(self._buffer)
        if not frames:
            return {}
        novelties = np.array([f.hs_novelty for f in frames])
        kf = [f for f in frames if f.is_keyframe]
        df = [f for f in frames if not f.is_keyframe]
        return {
            "n_frames": len(frames),
            "n_keyframes": len(kf),
            "n_delta_frames": len(df),
            "novelty_mean": round(float(novelties.mean()), 4),
            "novelty_max": round(float(novelties.max()), 4),
            "keyframe_indices": self.keyframe_indices(),
            "payload_bytes_total": self.payload_bytes_total(),
            "bytes_per_frame_mean": round(self.payload_bytes_total() / len(frames)),
            "trajectory_path_length": round(float(self.trajectory.cumulative_length()[-1]), 4) if len(self.trajectory) > 0 else 0.0,
        }

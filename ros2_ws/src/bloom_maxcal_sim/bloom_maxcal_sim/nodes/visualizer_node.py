"""
ROS2 node: Live Matplotlib Visualizer.

Renders the bloom concentration field, double-gyre velocity vectors,
rover trajectory, current waypoint, and MaxCal diagnostics in a single
interactive matplotlib figure that updates at a fixed rate.

Subscribes
----------
/bloom/field           Float32MultiArray  — concentration grid
/bloom/velocity_field  Float32MultiArray  — (U,V) velocity grid
/bloom/params          Float32MultiArray  — [Lx, Ly, nx, ny, t]
/rover/path            nav_msgs/Path      — rover trajectory
/rover/pose            geometry_msgs/PoseStamped
/maxcal/waypoint       geometry_msgs/PointStamped
/maxcal/diagnostics    Float32MultiArray  — [H, hs_dist, λ_B, λ_G, λ_D, b_obs, g_mag, L_nJ]

Parameters
----------
update_rate   (float, 2.0)   — figure refresh Hz
quiver_skip   (int,  8)      — skip every N grid points in velocity quiver
cmap          (str, 'YlGn')  — matplotlib colormap for bloom
figsize_w     (float, 14.0)
figsize_h     (float, 7.0)
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional, List

import rclpy
from rclpy.node import Node
import rclpy.qos as rqos
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray

import numpy as np

# Non-interactive backend guard — switch before importing pyplot
import matplotlib
matplotlib.use('TkAgg')          # change to 'Qt5Agg' if TkAgg unavailable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


class VisualizerNode(Node):
    """Live matplotlib visualizer for the bloom + rover simulation."""

    def __init__(self) -> None:
        super().__init__('visualizer_node')

        self.declare_parameter('update_rate', 2.0)
        self.declare_parameter('quiver_skip', 8)
        self.declare_parameter('cmap', 'YlGn')
        self.declare_parameter('figsize_w', 14.0)
        self.declare_parameter('figsize_h', 7.0)

        self._update_rate = float(self.get_parameter('update_rate').value)
        self._skip = int(self.get_parameter('quiver_skip').value)
        self._cmap = str(self.get_parameter('cmap').value)

        # Data cache
        self._bloom_grid: Optional[np.ndarray] = None
        self._vel_U: Optional[np.ndarray] = None
        self._vel_V: Optional[np.ndarray] = None
        self._Lx: float = 100.0
        self._Ly: float = 100.0
        self._nx: int = 120
        self._ny: int = 120
        self._bloom_t: float = 0.0
        self._rover_xs: List[float] = []
        self._rover_ys: List[float] = []
        self._rover_theta: float = 0.0
        self._rover_x: float = 50.0
        self._rover_y: float = 50.0
        self._waypoint: Optional[PointStamped] = None
        self._diag: List[float] = []

        # Diagnostic history
        self._hist_bloom: List[float] = []
        self._hist_H: List[float] = []
        self._hist_hs: List[float] = []
        self._hist_landauer: List[float] = []

        # QoS
        be = rqos.QoSProfile(
            reliability=rqos.ReliabilityPolicy.BEST_EFFORT,
            history=rqos.HistoryPolicy.KEEP_LAST, depth=1,
        )
        rel = rqos.QoSProfile(
            reliability=rqos.ReliabilityPolicy.RELIABLE,
            history=rqos.HistoryPolicy.KEEP_LAST, depth=5,
        )

        # Subscribers
        self.create_subscription(Float32MultiArray, '/bloom/field', self._bloom_cb, be)
        self.create_subscription(Float32MultiArray, '/bloom/velocity_field', self._vel_cb, be)
        self.create_subscription(Float32MultiArray, '/bloom/params', self._params_cb, rel)
        self.create_subscription(PoseStamped, '/rover/pose', self._pose_cb, be)
        self.create_subscription(Path, '/rover/path', self._path_cb, rel)
        self.create_subscription(PointStamped, '/maxcal/waypoint', self._wp_cb, rel)
        self.create_subscription(Float32MultiArray, '/maxcal/diagnostics', self._diag_cb, rel)

        # Build figure
        fw = float(self.get_parameter('figsize_w').value)
        fh = float(self.get_parameter('figsize_h').value)
        self._setup_figure(fw, fh)

        # Draw timer
        self._timer = self.create_timer(1.0 / self._update_rate, self._draw_cb)
        self.get_logger().info(
            f"VisualizerNode started: rate={self._update_rate} Hz"
        )

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _bloom_cb(self, msg: Float32MultiArray) -> None:
        dims = msg.layout.dim
        if len(dims) >= 2:
            ny, nx = dims[0].size, dims[1].size
            self._bloom_grid = np.array(msg.data, dtype=np.float32).reshape(ny, nx)

    def _vel_cb(self, msg: Float32MultiArray) -> None:
        dims = msg.layout.dim
        if len(dims) >= 3:
            c, ny, nx = dims[0].size, dims[1].size, dims[2].size
            arr = np.array(msg.data, dtype=np.float32).reshape(c, ny, nx)
            if c >= 2:
                self._vel_U = arr[0]
                self._vel_V = arr[1]

    def _params_cb(self, msg: Float32MultiArray) -> None:
        d = msg.data
        if len(d) >= 5:
            self._Lx, self._Ly = float(d[0]), float(d[1])
            self._nx, self._ny = int(d[2]), int(d[3])
            self._bloom_t = float(d[4])

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._rover_x = msg.pose.position.x
        self._rover_y = msg.pose.position.y
        q = msg.pose.orientation
        self._rover_theta = math.atan2(
            2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z)
        )

    def _path_cb(self, msg: Path) -> None:
        self._rover_xs = [p.pose.position.x for p in msg.poses]
        self._rover_ys = [p.pose.position.y for p in msg.poses]

    def _wp_cb(self, msg: PointStamped) -> None:
        self._waypoint = msg

    def _diag_cb(self, msg: Float32MultiArray) -> None:
        self._diag = list(msg.data)
        if len(self._diag) >= 8:
            self._hist_H.append(self._diag[0])
            self._hist_hs.append(self._diag[1])
            self._hist_bloom.append(self._diag[5])
            self._hist_landauer.append(self._diag[7])
            # Trim to 200 points
            for lst in (self._hist_H, self._hist_hs, self._hist_bloom, self._hist_landauer):
                if len(lst) > 200:
                    del lst[0]

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------

    def _setup_figure(self, fw: float, fh: float) -> None:
        plt.ion()
        self._fig = plt.figure(figsize=(fw, fh))
        self._fig.suptitle("MaxCal Algae Bloom Follower — ROS2 Simulation", fontsize=13)
        gs = GridSpec(2, 3, figure=self._fig, hspace=0.4, wspace=0.35)

        self._ax_bloom = self._fig.add_subplot(gs[:, :2])
        self._ax_bloom.set_title("Bloom Field & Rover")
        self._ax_bloom.set_xlabel("x (m)")
        self._ax_bloom.set_ylabel("y (m)")

        self._ax_H = self._fig.add_subplot(gs[0, 2])
        self._ax_H.set_title("MaxCal Entropy  (nats)")
        self._ax_H.set_xlabel("step")

        self._ax_hs = self._fig.add_subplot(gs[1, 2])
        self._ax_hs.set_title("HS Kernel Distance")
        self._ax_hs.set_xlabel("step")

        # Bloom imshow placeholder
        dummy = np.zeros((10, 10))
        self._im = self._ax_bloom.imshow(
            dummy, origin='lower', cmap=self._cmap,
            vmin=0.0, vmax=1.0, aspect='auto',
            extent=[0, self._Lx, 0, self._Ly],
        )
        self._fig.colorbar(self._im, ax=self._ax_bloom, label='Bloom conc.')
        self._quiv = None

        self._traj_line, = self._ax_bloom.plot([], [], 'b-', lw=1.2, alpha=0.7, label='Trajectory')
        self._rover_dot, = self._ax_bloom.plot([], [], 'bs', ms=8, label='Rover')
        self._heading_arr = self._ax_bloom.annotate(
            '', xy=(0.5, 0.5), xytext=(0.5, 0.5),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
        )
        self._wp_dot, = self._ax_bloom.plot([], [], 'r^', ms=10, label='MaxCal WP')
        self._ax_bloom.legend(loc='upper right', fontsize=8)

        self._line_H, = self._ax_H.plot([], [], 'g-', lw=1.5)
        self._line_hs, = self._ax_hs.plot([], [], 'm-', lw=1.5)

        plt.pause(0.01)

    # ------------------------------------------------------------------
    # Draw callback
    # ------------------------------------------------------------------

    def _draw_cb(self) -> None:
        try:
            self._update_figure()
        except Exception as e:
            self.get_logger().warn(f"Visualizer draw error: {e}")

    def _update_figure(self) -> None:
        # Bloom field
        if self._bloom_grid is not None:
            self._im.set_data(self._bloom_grid)
            self._im.set_extent([0, self._Lx, 0, self._Ly])
            vmax = max(float(self._bloom_grid.max()), 0.1)
            self._im.set_clim(0.0, vmax)

        # Velocity quiver
        if self._vel_U is not None and self._vel_V is not None:
            ny, nx = self._vel_U.shape
            xs = np.linspace(0, self._Lx, nx)
            ys = np.linspace(0, self._Ly, ny)
            sk = self._skip
            XX, YY = np.meshgrid(xs[::sk], ys[::sk])
            U = self._vel_U[::sk, ::sk]
            V = self._vel_V[::sk, ::sk]
            if self._quiv is not None:
                self._quiv.remove()
            self._quiv = self._ax_bloom.quiver(
                XX, YY, U, V, scale=0.5, alpha=0.4, color='navy', width=0.002
            )

        # Rover trajectory
        if self._rover_xs:
            self._traj_line.set_data(self._rover_xs, self._rover_ys)
        self._rover_dot.set_data([self._rover_x], [self._rover_y])

        # Heading arrow
        arr_len = 4.0
        self._heading_arr.set_position((self._rover_x, self._rover_y))
        self._heading_arr.xy = (
            self._rover_x + arr_len * math.cos(self._rover_theta),
            self._rover_y + arr_len * math.sin(self._rover_theta),
        )

        # Waypoint
        if self._waypoint is not None:
            self._wp_dot.set_data([self._waypoint.point.x], [self._waypoint.point.y])

        # Axis limits
        self._ax_bloom.set_xlim(0, self._Lx)
        self._ax_bloom.set_ylim(0, self._Ly)
        self._ax_bloom.set_title(
            f"Bloom Field & Rover  (t={self._bloom_t:.0f} s)"
        )

        # Entropy history
        if self._hist_H:
            xs = list(range(len(self._hist_H)))
            self._line_H.set_data(xs, self._hist_H)
            self._ax_H.set_xlim(0, max(1, len(xs)))
            self._ax_H.set_ylim(0, max(max(self._hist_H) * 1.1, 0.1))

        # HS distance history
        if self._hist_hs:
            xs = list(range(len(self._hist_hs)))
            self._line_hs.set_data(xs, self._hist_hs)
            self._ax_hs.set_xlim(0, max(1, len(xs)))
            self._ax_hs.set_ylim(0, max(max(self._hist_hs) * 1.1, 1e-6))

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualizerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

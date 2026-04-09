"""
ROS2 node: MaxCal Bloom-Following Controller.

Subscribes to rover pose and bloom field; runs the MaxCal distribution over
candidate next-positions at each control step; publishes /cmd_vel.

The node maintains an internal lightweight representation of the bloom (via
the received grid) so that the kernelcal MaxCal functional can evaluate
bloom concentration and gradient at any candidate position through bilinear
interpolation.

Topics subscribed
-----------------
/rover/pose           geometry_msgs/PoseStamped
    Current rover pose (x, y, yaw).

/bloom/field          std_msgs/Float32MultiArray
    Bloom concentration grid (ny × nx, row-major).

/bloom/params         std_msgs/Float32MultiArray
    [Lx, Ly, nx, ny, t] — domain metadata.

Topics published
----------------
/cmd_vel              geometry_msgs/Twist
    Velocity command (linear.x = v, angular.z = ω).

/maxcal/waypoint      geometry_msgs/PointStamped
    Currently selected MaxCal waypoint.

/maxcal/distribution  std_msgs/Float32MultiArray
    MaxCal probability distribution over candidate positions.
    data = [p_0, x_0, y_0, p_1, x_1, y_1, …]

/maxcal/diagnostics   std_msgs/Float32MultiArray
    [entropy, hs_distance, lambda_bloom, lambda_grad, lambda_dist,
     bloom_obs, gradient_mag, landauer_nJ]

Parameters
----------
control_rate        (float, 2.0)   — controller Hz
n_candidates        (int,   32)    — candidate positions per step
lookahead_min       (float, 3.0)   — inner lookahead ring (m)
lookahead_max       (float, 10.0)  — outer lookahead ring (m)
sigma_q             (float, 6.0)   — reference prior width (m)
bloom_target_q      (float, 0.65)  — bloom concentration target quantile
gradient_target_q   (float, 0.55)  — gradient target quantile
distance_target_frac (float, 0.50) — distance target fraction of lookahead_max
kernel_length_scale (float, 5.0)   — RBF length scale for HS tracking
v_max               (float, 1.2)   — max velocity (m/s)
k_omega             (float, 1.8)   — heading proportional gain
arrival_radius      (float, 2.5)   — waypoint arrival distance (m)
domain_lx           (float, 100.0)
domain_ly           (float, 100.0)
seed                (int,   7)
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import rclpy
from rclpy.node import Node
import rclpy.qos as rqos
from geometry_msgs.msg import PoseStamped, PointStamped, Twist
from std_msgs.msg import Float32MultiArray
import numpy as np

_PKG_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')
)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# Try to add the kernelcal manuscript root too
_MANUSCRIPT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', '..')
)
if _MANUSCRIPT_ROOT not in sys.path:
    sys.path.insert(0, _MANUSCRIPT_ROOT)

from bloom_maxcal_sim.maxcal_bloom_follower import MaxCalBloomFollower, MaxCalConfig


# ---------------------------------------------------------------------------
# Lightweight grid-based bloom proxy
# ---------------------------------------------------------------------------

class GridBloomProxy:
    """Wraps a cached concentration grid so MaxCal can query it point-wise."""

    def __init__(self) -> None:
        self._grid: Optional[np.ndarray] = None   # (ny, nx) float32
        self._Lx: float = 100.0
        self._Ly: float = 100.0
        self._nx: int = 1
        self._ny: int = 1

    def update(self, grid: np.ndarray, Lx: float, Ly: float) -> None:
        self._grid = grid.astype(np.float64)
        self._ny, self._nx = grid.shape
        self._Lx, self._Ly = Lx, Ly

    def ready(self) -> bool:
        return self._grid is not None

    def _interp(self, x: float, y: float) -> float:
        g = self._grid
        ny, nx = self._ny, self._nx
        xi = float(np.clip((x / self._Lx) * (nx - 1), 0, nx - 2))
        yi = float(np.clip((y / self._Ly) * (ny - 1), 0, ny - 2))
        ix, iy = int(xi), int(yi)
        fx, fy = xi - ix, yi - iy
        return float(
            g[iy, ix]       * (1-fx) * (1-fy)
            + g[iy, ix+1]   * fx     * (1-fy)
            + g[iy+1, ix]   * (1-fx) * fy
            + g[iy+1, ix+1] * fx     * fy
        )

    def concentration_at(self, x: float, y: float) -> float:
        if not self.ready():
            return 0.0
        return max(0.0, self._interp(x, y))

    def gradient_at(self, x: float, y: float) -> tuple:
        if not self.ready():
            return 0.0, 0.0
        g = self._grid
        ny, nx = self._ny, self._nx
        Lx, Ly = self._Lx, self._Ly
        xi = float(np.clip((x / Lx) * (nx - 1), 1, nx - 2))
        yi = float(np.clip((y / Ly) * (ny - 1), 1, ny - 2))
        ix, iy = int(xi), int(yi)
        dx_phys = Lx / (nx - 1)
        dy_phys = Ly / (ny - 1)
        gx = (float(g[iy, ix+1]) - float(g[iy, ix-1])) / (2.0 * dx_phys)
        gy = (float(g[iy+1, ix]) - float(g[iy-1, ix])) / (2.0 * dy_phys)
        return gx, gy

    def gradient_magnitude_at(self, x: float, y: float) -> float:
        gx, gy = self.gradient_at(x, y)
        return float(math.hypot(gx, gy))


# ---------------------------------------------------------------------------
# Fake rover wrapper for MaxCalBloomFollower
# ---------------------------------------------------------------------------

class _FakeRover:
    """Minimal rover-like object for MaxCalBloomFollower.update() interface."""

    def __init__(self, x: float, y: float, theta: float, seed: int = 7) -> None:
        self.x = x
        self.y = y
        self.theta = theta
        self._rng = np.random.default_rng(seed)

    def pose(self):
        return self.x, self.y, self.theta

    def observe_bloom(self, bloom):
        return bloom.concentration_at(self.x, self.y)

    def observe_gradient_magnitude(self, bloom):
        return bloom.gradient_magnitude_at(self.x, self.y)

    class _Cfg:
        sigma_obs = 0.02

    cfg = _Cfg()

    @property
    def rng(self):
        return self._rng


# ---------------------------------------------------------------------------
# Controller node
# ---------------------------------------------------------------------------

class MaxCalControllerNode(Node):
    """ROS2 node: MaxCal bloom-following velocity controller."""

    def __init__(self) -> None:
        super().__init__('maxcal_controller_node')

        # Parameters
        self.declare_parameter('control_rate', 2.0)
        self.declare_parameter('n_candidates', 32)
        self.declare_parameter('lookahead_min', 3.0)
        self.declare_parameter('lookahead_max', 10.0)
        self.declare_parameter('sigma_q', 6.0)
        self.declare_parameter('bloom_target_q', 0.65)
        self.declare_parameter('gradient_target_q', 0.55)
        self.declare_parameter('distance_target_frac', 0.50)
        self.declare_parameter('kernel_length_scale', 5.0)
        self.declare_parameter('v_max', 1.2)
        self.declare_parameter('k_omega', 1.8)
        self.declare_parameter('arrival_radius', 2.5)
        self.declare_parameter('domain_lx', 100.0)
        self.declare_parameter('domain_ly', 100.0)
        self.declare_parameter('seed', 7)

        Lx = float(self.get_parameter('domain_lx').value)
        Ly = float(self.get_parameter('domain_ly').value)

        mc_cfg = MaxCalConfig(
            n_candidates=int(self.get_parameter('n_candidates').value),
            lookahead_min=float(self.get_parameter('lookahead_min').value),
            lookahead_max=float(self.get_parameter('lookahead_max').value),
            sigma_q=float(self.get_parameter('sigma_q').value),
            bloom_target_quantile=float(self.get_parameter('bloom_target_q').value),
            gradient_target_quantile=float(self.get_parameter('gradient_target_q').value),
            distance_target_fraction=float(self.get_parameter('distance_target_frac').value),
            kernel_length_scale=float(self.get_parameter('kernel_length_scale').value),
            v_max=float(self.get_parameter('v_max').value),
            k_omega=float(self.get_parameter('k_omega').value),
            arrival_radius=float(self.get_parameter('arrival_radius').value),
            domain_x=(0.0, Lx),
            domain_y=(0.0, Ly),
        )
        self._follower = MaxCalBloomFollower(config=mc_cfg)
        self._bloom_proxy = GridBloomProxy()
        self._rng = np.random.default_rng(int(self.get_parameter('seed').value))

        # State
        self._rover_x: float = Lx / 2.0
        self._rover_y: float = Ly / 2.0
        self._rover_theta: float = 0.0
        self._pose_received: bool = False

        # QoS profiles
        best_effort = rqos.QoSProfile(
            reliability=rqos.ReliabilityPolicy.BEST_EFFORT,
            history=rqos.HistoryPolicy.KEEP_LAST, depth=1,
        )
        reliable = rqos.QoSProfile(
            reliability=rqos.ReliabilityPolicy.RELIABLE,
            history=rqos.HistoryPolicy.KEEP_LAST, depth=5,
        )

        # Subscribers
        self.create_subscription(PoseStamped, '/rover/pose', self._pose_cb, best_effort)
        self.create_subscription(Float32MultiArray, '/bloom/field', self._bloom_field_cb, best_effort)
        self.create_subscription(Float32MultiArray, '/bloom/params', self._bloom_params_cb, reliable)

        # Publishers
        self._pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', best_effort)
        self._pub_waypoint = self.create_publisher(PointStamped, '/maxcal/waypoint', reliable)
        self._pub_dist = self.create_publisher(Float32MultiArray, '/maxcal/distribution', reliable)
        self._pub_diag = self.create_publisher(Float32MultiArray, '/maxcal/diagnostics', reliable)

        # Control timer
        rate = float(self.get_parameter('control_rate').value)
        self._control_dt = 1.0 / rate
        self._timer = self.create_timer(self._control_dt, self._control_cb)

        self.get_logger().info(
            f"MaxCalControllerNode started: rate={rate} Hz, "
            f"n_candidates={mc_cfg.n_candidates}, "
            f"lookahead=[{mc_cfg.lookahead_min},{mc_cfg.lookahead_max}] m"
        )

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._rover_x = msg.pose.position.x
        self._rover_y = msg.pose.position.y
        q = msg.pose.orientation
        # yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._rover_theta = math.atan2(siny_cosp, cosy_cosp)
        self._pose_received = True

    def _bloom_field_cb(self, msg: Float32MultiArray) -> None:
        dims = msg.layout.dim
        if len(dims) >= 2:
            ny = dims[0].size
            nx = dims[1].size
            grid = np.array(msg.data, dtype=np.float32).reshape(ny, nx)
            self._bloom_proxy.update(grid, self._bloom_proxy._Lx, self._bloom_proxy._Ly)

    def _bloom_params_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 2:
            self._bloom_proxy._Lx = float(msg.data[0])
            self._bloom_proxy._Ly = float(msg.data[1])
            if len(msg.data) >= 4:
                self._bloom_proxy._nx = int(msg.data[2])
                self._bloom_proxy._ny = int(msg.data[3])

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_cb(self) -> None:
        if not self._pose_received or not self._bloom_proxy.ready():
            # Publish zero velocity while waiting
            self._pub_cmd_vel.publish(Twist())
            return

        # Fake rover object for the follower interface
        fake_rover = _FakeRover(
            self._rover_x, self._rover_y, self._rover_theta
        )
        fake_rover._rng = self._rng

        # Run MaxCal step
        v_cmd, omega_cmd = self._follower.update(
            fake_rover, self._bloom_proxy, self._control_dt, rng=self._rng
        )

        # Publish cmd_vel
        twist = Twist()
        twist.linear.x = float(v_cmd)
        twist.angular.z = float(omega_cmd)
        self._pub_cmd_vel.publish(twist)

        now = self.get_clock().now().to_msg()

        # Publish waypoint
        wp = self._follower.current_waypoint()
        if wp is not None:
            msg_wp = PointStamped()
            msg_wp.header.stamp = now
            msg_wp.header.frame_id = 'bloom_field'
            msg_wp.point.x = float(wp[0])
            msg_wp.point.y = float(wp[1])
            self._pub_waypoint.publish(msg_wp)

        # Publish diagnostics
        rec = self._follower.last_record()
        if rec is not None:
            lambdas = rec.lagrange_multipliers
            diag = Float32MultiArray()
            diag.data = [
                float(rec.entropy_nats),
                float(rec.hs_distance),
                float(lambdas[0]) if len(lambdas) > 0 else 0.0,
                float(lambdas[1]) if len(lambdas) > 1 else 0.0,
                float(lambdas[2]) if len(lambdas) > 2 else 0.0,
                float(rec.bloom_obs),
                float(rec.gradient_mag),
                float(rec.landauer_bound_nJ),
            ]
            self._pub_diag.publish(diag)

        # Log every 20 steps
        if self._follower._step % 20 == 0:
            stats = self._follower.statistics()
            self.get_logger().info(
                f"[step={stats.get('steps',0)}] "
                f"bloom={stats.get('mean_bloom_obs',0):.3f}, "
                f"HS_dist={stats.get('mean_hs_distance',0):.4f}, "
                f"H={stats.get('mean_entropy_nats',0):.3f} nats, "
                f"Landauer={stats.get('total_landauer_nJ',0):.2f} nJ"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = MaxCalControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

"""
ROS2 node: 2D Differential-Drive Rover Simulator.

Integrates the unicycle rover model and publishes odometry, pose, and bloom
observations.  Subscribes to /cmd_vel for velocity commands.

Topics subscribed
-----------------
/cmd_vel              geometry_msgs/Twist
    Linear velocity in x (v, m/s) and angular velocity in z (ω, rad/s).

/bloom/field          std_msgs/Float32MultiArray
    Bloom concentration grid (see bloom_field_node).  Used for observation
    simulation without needing direct access to the bloom object.

/bloom/params         std_msgs/Float32MultiArray
    [Lx, Ly, nx, ny, t] for grid interpolation.

Topics published
----------------
/rover/odom           nav_msgs/Odometry
    Full odometry: pose (x, y, quaternion from yaw) and velocity (vx, ωz).

/rover/pose           geometry_msgs/PoseStamped
    Current 2D pose.

/rover/bloom_obs      std_msgs/Float32
    Noisy bloom concentration observation at current rover position.

/rover/gradient_dir   std_msgs/Float32
    Noisy bloom gradient direction (rad, CCW from +x).

/rover/path           nav_msgs/Path
    Accumulated rover trajectory for RViz display.

Parameters
----------
sim_dt         (float, 0.05) — rover integration step (s)
publish_rate   (float, 20.0) — publish frequency (Hz)
x0             (float, 50.0) — initial x (m)
y0             (float, 50.0) — initial y (m)
theta0         (float, 0.0)  — initial heading (rad)
v_max          (float, 1.5)  — maximum linear speed (m/s)
omega_max      (float, 1.2)  — maximum angular speed (rad/s)
sigma_obs      (float, 0.02) — concentration observation noise
sigma_dir      (float, 0.15) — gradient direction noise (rad)
seed           (int,   123)
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import rclpy
from rclpy.node import Node
import rclpy.qos as rqos
from geometry_msgs.msg import (
    PoseStamped, Quaternion, Twist, TransformStamped
)
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32, Float32MultiArray
from tf2_ros import TransformBroadcaster

import numpy as np

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from bloom_maxcal_sim.rover_model import DifferentialDriveRover, RoverConfig


class RoverSimNode(Node):
    """ROS2 node that simulates the 2D rover and publishes its state."""

    def __init__(self) -> None:
        super().__init__('rover_sim_node')

        # Parameters
        self.declare_parameter('sim_dt', 0.05)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('x0', 50.0)
        self.declare_parameter('y0', 50.0)
        self.declare_parameter('theta0', 0.0)
        self.declare_parameter('v_max', 1.5)
        self.declare_parameter('omega_max', 1.2)
        self.declare_parameter('sigma_obs', 0.02)
        self.declare_parameter('sigma_dir', 0.15)
        self.declare_parameter('seed', 123)

        cfg = RoverConfig(
            x0=float(self.get_parameter('x0').value),
            y0=float(self.get_parameter('y0').value),
            theta0=float(self.get_parameter('theta0').value),
            v_max=float(self.get_parameter('v_max').value),
            omega_max=float(self.get_parameter('omega_max').value),
            sigma_obs=float(self.get_parameter('sigma_obs').value),
            sigma_dir=float(self.get_parameter('sigma_dir').value),
        )
        seed = int(self.get_parameter('seed').value)
        self._rover = DifferentialDriveRover(config=cfg, seed=seed)
        self._sim_dt = float(self.get_parameter('sim_dt').value)

        # Commanded velocities (filled by /cmd_vel subscriber)
        self._v_cmd: float = 0.0
        self._omega_cmd: float = 0.0

        # Bloom grid cache (for observation simulation)
        self._bloom_grid: Optional[np.ndarray] = None   # (ny, nx)
        self._bloom_Lx: float = 100.0
        self._bloom_Ly: float = 100.0
        self._bloom_nx: int = 1
        self._bloom_ny: int = 1

        # QoS
        sensor_qos = rqos.QoSProfile(
            reliability=rqos.ReliabilityPolicy.BEST_EFFORT,
            history=rqos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = rqos.QoSProfile(
            reliability=rqos.ReliabilityPolicy.RELIABLE,
            history=rqos.HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Subscribers
        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, sensor_qos)
        self.create_subscription(Float32MultiArray, '/bloom/field', self._bloom_field_cb, sensor_qos)
        self.create_subscription(Float32MultiArray, '/bloom/params', self._bloom_params_cb, reliable_qos)

        # Publishers
        self._pub_odom = self.create_publisher(Odometry, '/rover/odom', reliable_qos)
        self._pub_pose = self.create_publisher(PoseStamped, '/rover/pose', reliable_qos)
        self._pub_bloom_obs = self.create_publisher(Float32, '/rover/bloom_obs', sensor_qos)
        self._pub_grad_dir = self.create_publisher(Float32, '/rover/gradient_dir', sensor_qos)
        self._pub_path = self.create_publisher(Path, '/rover/path', reliable_qos)

        # TF broadcaster for RViz
        self._tf_broadcaster = TransformBroadcaster(self)

        # Path message (accumulated)
        self._path_msg = Path()
        self._path_msg.header.frame_id = 'bloom_field'

        # Simulation timer
        rate = float(self.get_parameter('publish_rate').value)
        self._steps_per_tick = max(1, round(1.0 / (rate * self._sim_dt)))
        self._timer = self.create_timer(1.0 / rate, self._timer_cb)

        self.get_logger().info(
            f"RoverSimNode started: x0={cfg.x0}, y0={cfg.y0}, "
            f"v_max={cfg.v_max} m/s, dt={self._sim_dt}s, rate={rate} Hz"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self._v_cmd = float(msg.linear.x)
        self._omega_cmd = float(msg.angular.z)

    def _bloom_field_cb(self, msg: Float32MultiArray) -> None:
        dims = msg.layout.dim
        if len(dims) >= 2:
            ny = dims[0].size
            nx = dims[1].size
            self._bloom_grid = np.array(msg.data, dtype=np.float32).reshape(ny, nx)
            self._bloom_ny = ny
            self._bloom_nx = nx

    def _bloom_params_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 4:
            self._bloom_Lx = float(msg.data[0])
            self._bloom_Ly = float(msg.data[1])

    # ------------------------------------------------------------------
    # Simulation tick
    # ------------------------------------------------------------------

    def _timer_cb(self) -> None:
        # Integrate rover kinematics
        for _ in range(self._steps_per_tick):
            self._rover.step(self._v_cmd, self._omega_cmd, self._sim_dt)

        now = self.get_clock().now().to_msg()
        x, y, theta = self._rover.pose()

        # Odometry
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'bloom_field'
        odom.child_frame_id = 'rover'
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = _yaw_to_quaternion(theta)
        odom.twist.twist.linear.x = self._rover.v
        odom.twist.twist.angular.z = self._rover.omega
        self._pub_odom.publish(odom)

        # PoseStamped
        ps = PoseStamped()
        ps.header = odom.header
        ps.pose = odom.pose.pose
        self._pub_pose.publish(ps)

        # TF transform: bloom_field → rover
        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = 'bloom_field'
        tf.child_frame_id = 'rover'
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = 0.0
        tf.transform.rotation = odom.pose.pose.orientation
        self._tf_broadcaster.sendTransform(tf)

        # Accumulated path
        self._path_msg.header.stamp = now
        self._path_msg.poses.append(ps)
        if len(self._path_msg.poses) > 2000:   # keep last 2000 poses
            self._path_msg.poses.pop(0)
        self._pub_path.publish(self._path_msg)

        # Bloom observation via grid interpolation
        bloom_obs, grad_dir = self._observe_from_grid(x, y)
        msg_obs = Float32()
        msg_obs.data = float(bloom_obs)
        self._pub_bloom_obs.publish(msg_obs)

        msg_gd = Float32()
        msg_gd.data = float(grad_dir)
        self._pub_grad_dir.publish(msg_gd)

    # ------------------------------------------------------------------
    # Grid-based observation
    # ------------------------------------------------------------------

    def _observe_from_grid(self, x: float, y: float) -> tuple:
        """Bilinearly interpolate bloom concentration from cached grid."""
        if self._bloom_grid is None:
            return 0.0, 0.0

        g = self._bloom_grid
        ny, nx = g.shape
        Lx, Ly = self._bloom_Lx, self._bloom_Ly

        # Normalised coordinates
        xi = (x / Lx) * (nx - 1)
        yi = (y / Ly) * (ny - 1)
        xi = float(np.clip(xi, 0, nx - 2))
        yi = float(np.clip(yi, 0, ny - 2))

        ix = int(xi)
        iy = int(yi)
        fx = xi - ix
        fy = yi - iy

        # Bilinear interpolation
        c = (
            g[iy, ix]     * (1 - fx) * (1 - fy)
            + g[iy, ix+1]   * fx       * (1 - fy)
            + g[iy+1, ix]   * (1 - fx) * fy
            + g[iy+1, ix+1] * fx       * fy
        )

        # Gradient direction via central differences
        dx = (g[iy, min(ix+1, nx-1)] - g[iy, max(ix-1, 0)]) / 2.0
        dy = (g[min(iy+1, ny-1), ix] - g[max(iy-1, 0), ix]) / 2.0
        grad_dir = math.atan2(float(dy), float(dx))

        # Add observation noise
        rng = self._rover.rng
        c_obs = float(max(0.0, c + rng.normal(0.0, self._rover.cfg.sigma_obs)))
        gd_obs = float((grad_dir + rng.normal(0.0, self._rover.cfg.sigma_dir) + math.pi)
                       % (2 * math.pi) - math.pi)
        return c_obs, gd_obs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = float(math.cos(yaw / 2.0))
    q.x = 0.0
    q.y = 0.0
    q.z = float(math.sin(yaw / 2.0))
    return q


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoverSimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

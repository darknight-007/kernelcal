"""
ROS2 node: Algae Bloom Field Simulator.

Publishes the spatiotemporally evolving double-gyre advecting Gaussian bloom
field at a configurable rate and exposes the current state on standard topics.

Topics published
----------------
/bloom/field          std_msgs/Float32MultiArray
    Flattened (row-major) concentration grid of shape (ny, nx).
    Layout metadata: [data_offset, dim[0].label="y", dim[0].size=ny,
                      dim[0].stride=ny*nx, dim[1].label="x",
                      dim[1].size=nx, dim[1].stride=nx]

/bloom/center         geometry_msgs/PointStamped
    Concentration-weighted centre of mass of the bloom field.

/bloom/peak           geometry_msgs/PointStamped
    Grid point with maximum concentration.

/bloom/params         std_msgs/Float32MultiArray
    Domain metadata [Lx, Ly, nx, ny, t] for downstream consumers.

/bloom/velocity_field  std_msgs/Float32MultiArray
    Flattened (U, V) double-gyre velocity field, shape (2, ny, nx).
    Allows the controller to access the advection velocity if desired.

Parameters
----------
sim_dt        (float, default 0.5) — bloom simulation step (s)
publish_rate  (float, default 2.0) — publish frequency (Hz)
domain_lx     (float, default 100.0) — domain width (m)
domain_ly     (float, default 100.0) — domain height (m)
grid_nx       (int,   default 120)   — grid columns
grid_ny       (int,   default 120)   — grid rows
gyre_A        (float, default 0.10)  — gyre amplitude (m²/s)
gyre_eps      (float, default 0.25)  — gyre oscillation ε
gyre_omega    (float, default 0.021) — gyre oscillation ω (rad/s)
sigma_wind    (float, default 0.008) — turbulent noise amplitude
seed          (int,   default 42)    — RNG seed
"""

from __future__ import annotations

import sys
import os

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from geometry_msgs.msg import PointStamped

import numpy as np

# Resolve kernelcal from the manuscript repo
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from bloom_maxcal_sim.bloom_field import (
    AlgaeBloomField,
    BloomFieldConfig,
    DoubleGyreParams,
)


class BloomFieldNode(Node):
    """ROS2 node that simulates and publishes the algae bloom field."""

    def __init__(self) -> None:
        super().__init__('bloom_field_node')

        # Declare parameters
        self.declare_parameter('sim_dt', 0.5)
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('domain_lx', 100.0)
        self.declare_parameter('domain_ly', 100.0)
        self.declare_parameter('grid_nx', 120)
        self.declare_parameter('grid_ny', 120)
        self.declare_parameter('gyre_A', 0.10)
        self.declare_parameter('gyre_eps', 0.25)
        self.declare_parameter('gyre_omega', 2.0 * np.pi / 300.0)
        self.declare_parameter('sigma_wind', 0.008)
        self.declare_parameter('seed', 42)

        # Build bloom model
        self._bloom = self._build_bloom()

        # Publishers
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub_field = self.create_publisher(Float32MultiArray, '/bloom/field', qos)
        self._pub_center = self.create_publisher(PointStamped, '/bloom/center', qos)
        self._pub_peak = self.create_publisher(PointStamped, '/bloom/peak', qos)
        self._pub_params = self.create_publisher(Float32MultiArray, '/bloom/params', qos)
        self._pub_vel = self.create_publisher(Float32MultiArray, '/bloom/velocity_field', qos)

        # Timer
        dt = self.get_parameter('sim_dt').value
        rate = self.get_parameter('publish_rate').value
        self._sim_dt = float(dt)
        self._steps_per_publish = max(1, int(1.0 / (rate * dt)))
        self._sim_step = 0

        period = 1.0 / float(rate)
        self._timer = self.create_timer(period, self._timer_cb)
        self.get_logger().info(
            f"BloomFieldNode started: domain=[0,{self._bloom.cfg.gyre.Lx}]×"
            f"[0,{self._bloom.cfg.gyre.Ly}] m, "
            f"dt={dt}s, rate={rate} Hz"
        )

    def _build_bloom(self) -> AlgaeBloomField:
        Lx = float(self.get_parameter('domain_lx').value)
        Ly = float(self.get_parameter('domain_ly').value)
        nx = int(self.get_parameter('grid_nx').value)
        ny = int(self.get_parameter('grid_ny').value)
        A = float(self.get_parameter('gyre_A').value)
        eps = float(self.get_parameter('gyre_eps').value)
        omega = float(self.get_parameter('gyre_omega').value)
        sw = float(self.get_parameter('sigma_wind').value)
        seed = int(self.get_parameter('seed').value)
        cfg = BloomFieldConfig(
            gyre=DoubleGyreParams(A=A, eps=eps, omega=omega, Lx=Lx, Ly=Ly),
            sigma_wind=sw,
            nx=nx,
            ny=ny,
        )
        return AlgaeBloomField(config=cfg, seed=seed)

    def _timer_cb(self) -> None:
        # Advance bloom simulation
        for _ in range(self._steps_per_publish):
            self._bloom.step(self._sim_dt)
        self._sim_step += self._steps_per_publish

        now = self.get_clock().now().to_msg()

        # Publish concentration field
        grid = self._bloom.field_grid().astype(np.float32)
        ny, nx = grid.shape
        msg_field = Float32MultiArray()
        msg_field.layout = _make_layout(['y', 'x'], [ny, nx])
        msg_field.data = grid.flatten().tolist()
        self._pub_field.publish(msg_field)

        # Publish velocity field (U, V stacked)
        U, V = self._bloom.advection_field_grid()
        uv = np.stack([U, V], axis=0).astype(np.float32)   # (2, ny, nx)
        msg_vel = Float32MultiArray()
        msg_vel.layout = _make_layout(['component', 'y', 'x'], [2, ny, nx])
        msg_vel.data = uv.flatten().tolist()
        self._pub_vel.publish(msg_vel)

        # Publish bloom centre
        cx, cy = self._bloom.bloom_center_of_mass()
        msg_ctr = PointStamped()
        msg_ctr.header.stamp = now
        msg_ctr.header.frame_id = 'bloom_field'
        msg_ctr.point.x = cx
        msg_ctr.point.y = cy
        msg_ctr.point.z = 0.0
        self._pub_center.publish(msg_ctr)

        # Publish peak location
        px, py = self._bloom.peak_location()
        msg_pk = PointStamped()
        msg_pk.header.stamp = now
        msg_pk.header.frame_id = 'bloom_field'
        msg_pk.point.x = px
        msg_pk.point.y = py
        msg_pk.point.z = 0.0
        self._pub_peak.publish(msg_pk)

        # Publish domain params [Lx, Ly, nx, ny, t]
        p = self._bloom.cfg.gyre
        msg_params = Float32MultiArray()
        msg_params.data = [float(p.Lx), float(p.Ly), float(nx), float(ny), float(self._bloom.t)]
        self._pub_params.publish(msg_params)


# ---------------------------------------------------------------------------
# Layout helper
# ---------------------------------------------------------------------------

def _make_layout(labels: list, sizes: list) -> MultiArrayLayout:
    dims = []
    stride = 1
    for lbl, sz in zip(reversed(labels), reversed(sizes)):
        stride *= sz
        d = MultiArrayDimension()
        d.label = lbl
        d.size = sz
        d.stride = stride
        dims.append(d)
    dims.reverse()
    layout = MultiArrayLayout()
    layout.dim = dims
    layout.data_offset = 0
    return layout


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = BloomFieldNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

"""
ROS2 bridge for kernelcal navigation components.

Wraps SemanticSLAMKernelTracker, InformativePathPlanner, and thermodynamic
monitoring in ROS2 nodes that integrate with the Earth Rover's existing
`deepgis_vehicles` stack.

Requires: rclpy, sensor_msgs, geometry_msgs, nav_msgs, mavros_msgs.
ROS2 is an optional dependency — this module gracefully fails to import
when rclpy is unavailable.

Integration points in earth-rover:
  deepgis_telemetry_publisher.py  ← subscribe to /kernelcal/metrics
  vehicle_interface_node.cpp      ← publishes /mavros/battery
  launch/earth_rover.launch.py    ← add KernelcalNavigationNode
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    )
    from sensor_msgs.msg import BatteryState, NavSatFix
    from geometry_msgs.msg import PoseStamped, Point
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float32, String, Float32MultiArray
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False


# ---------------------------------------------------------------------------
# ROSPowerMonitor — wraps /mavros/battery instead of nvidia-smi
# ---------------------------------------------------------------------------

class ROSPowerMonitor:
    """Subscribes to /mavros/battery and tracks energy consumption.

    Drop-in analog of kernelcal.thermodynamics.PowerMonitor for the rover,
    using the Pixhawk/MAVROS battery telemetry instead of GPU power polling.

    Parameters
    ----------
    node : rclpy.Node — the parent ROS2 node.
    battery_topic : str — default '/mavros/battery'.
    nominal_voltage : float — nominal pack voltage (V) for energy estimation.
    """

    def __init__(
        self,
        node,
        battery_topic: str = "/mavros/battery",
        nominal_voltage: float = 57.6,
    ):
        if not _ROS_AVAILABLE:
            raise ImportError("rclpy is required for ROSPowerMonitor.")

        self._node = node
        self._nominal_voltage = nominal_voltage
        self._latest_battery: Optional[object] = None
        self._start_charge_ah: Optional[float] = None
        self._start_time: Optional[float] = None
        self._samples = []

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._sub = node.create_subscription(
            BatteryState, battery_topic, self._battery_callback, qos
        )

    def _battery_callback(self, msg) -> None:
        self._latest_battery = msg
        t = time.time()
        # BatteryState.current is in Amps (positive = discharging on Pixhawk)
        power_w = abs(msg.voltage * msg.current) if msg.current != 0 else 0.0
        self._samples.append((t, power_w))

        if self._start_charge_ah is None and msg.charge > 0:
            self._start_charge_ah = msg.charge
            self._start_time = t

    def start(self) -> None:
        self._samples.clear()
        self._start_charge_ah = None
        self._start_time = None

    def total_energy_joules(self) -> float:
        """Trapezoidal integration of power samples."""
        if len(self._samples) < 2:
            return 0.0
        times = np.array([s[0] for s in self._samples])
        powers = np.array([s[1] for s in self._samples])
        return float(np.trapz(powers, times))

    def energy_from_charge_delta(self) -> float:
        """Energy from Ah consumed × nominal voltage (coarser but robust)."""
        if self._latest_battery is None or self._start_charge_ah is None:
            return 0.0
        delta_ah = self._start_charge_ah - self._latest_battery.charge
        return max(delta_ah, 0.0) * self._nominal_voltage * 3600.0  # Ah → J

    def battery_fraction(self) -> float:
        if self._latest_battery is None:
            return 1.0
        return float(self._latest_battery.percentage)

    def remaining_energy_joules(self) -> float:
        """Estimate remaining energy from battery percentage × capacity."""
        if self._latest_battery is None:
            return 0.0
        # 25 Ah traction battery at 57.6V nominal
        capacity_j = 25.0 * self._nominal_voltage * 3600.0
        return capacity_j * self.battery_fraction()

    def summary(self) -> dict:
        return {
            "n_samples": len(self._samples),
            "total_energy_J": self.total_energy_joules(),
            "battery_fraction": self.battery_fraction(),
            "remaining_J": self.remaining_energy_joules(),
        }


# ---------------------------------------------------------------------------
# KernelcalNavigationNode — main ROS2 node
# ---------------------------------------------------------------------------

class KernelcalNavigationNode:
    """ROS2 node integrating kernelcal navigation into the earth-rover stack.

    Subscribes to:
      /mavros/battery          → ROSPowerMonitor
      /mavros/global_position/global → current GPS position
      /kernelcal/slam_novelty  → Float32 novelty score from SLAM tracker

    Publishes:
      /kernelcal/next_waypoint → NavSatFix  (next informative waypoint)
      /kernelcal/metrics       → String     (JSON metrics for DeepGIS telemetry)
      /kernelcal/patrol_status → String     ('transient'|'stable_fp'|...)

    Parameters
    ----------
    candidate_waypoints : (N, 2) array of (lon, lat) survey grid.
    energy_budget_joules : float — initial battery budget.
    update_rate_hz : float — how often to re-solve the MaxCal distribution.
    """

    def __init__(
        self,
        candidate_waypoints: np.ndarray,
        energy_budget_joules: float = 100_000.0,
        update_rate_hz: float = 0.1,
    ):
        if not _ROS_AVAILABLE:
            raise ImportError("rclpy is required for KernelcalNavigationNode.")

        from ..navigation.planner import InformativePathPlanner

        rclpy.init()
        self._node = Node("kernelcal_navigation")

        self._planner = InformativePathPlanner(
            candidate_waypoints=candidate_waypoints,
            energy_budget_joules=energy_budget_joules,
        )
        self._power_monitor = ROSPowerMonitor(self._node)
        self._current_position: Optional[np.ndarray] = None
        self._latest_novelty: float = 0.0
        self._novelty_map = np.zeros(len(candidate_waypoints))

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscribers
        self._gps_sub = self._node.create_subscription(
            NavSatFix,
            "/mavros/global_position/global",
            self._gps_callback,
            qos,
        )
        self._novelty_sub = self._node.create_subscription(
            Float32,
            "/kernelcal/slam_novelty",
            self._novelty_callback,
            qos,
        )
        self._novelty_map_sub = self._node.create_subscription(
            Float32MultiArray,
            "/kernelcal/novelty_map",
            self._novelty_map_callback,
            qos,
        )

        # Publishers
        self._wp_pub = self._node.create_publisher(
            NavSatFix, "/kernelcal/next_waypoint", 10
        )
        self._metrics_pub = self._node.create_publisher(
            String, "/kernelcal/metrics", 10
        )
        self._status_pub = self._node.create_publisher(
            String, "/kernelcal/patrol_status", 10
        )

        # Update timer
        self._timer = self._node.create_timer(
            1.0 / update_rate_hz, self._update_callback
        )

        self._node.get_logger().info("KernelcalNavigationNode initialised.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _gps_callback(self, msg) -> None:
        self._current_position = np.array([msg.longitude, msg.latitude])

    def _novelty_callback(self, msg) -> None:
        self._latest_novelty = float(msg.data)

    def _novelty_map_callback(self, msg) -> None:
        data = np.array(msg.data)
        if len(data) == len(self._novelty_map):
            self._novelty_map = data

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------

    def _update_callback(self) -> None:
        import json
        from std_msgs.msg import String as StringMsg

        self._planner.update(
            current_position=self._current_position,
            battery_joules_remaining=self._power_monitor.remaining_energy_joules(),
            semantic_scores=self._novelty_map,
        )

        # Publish next waypoint
        wp = self._planner.next_waypoint()
        wp_msg = NavSatFix()
        wp_msg.header.stamp = self._node.get_clock().now().to_msg()
        wp_msg.longitude = float(wp[0])
        wp_msg.latitude = float(wp[1])
        self._wp_pub.publish(wp_msg)

        # Publish metrics (for DeepGIS telemetry extension)
        stats = self._planner.statistics()
        stats["battery_remaining_J"] = self._power_monitor.remaining_energy_joules()
        stats["battery_fraction"] = self._power_monitor.battery_fraction()
        stats["slam_novelty"] = self._latest_novelty
        metrics_msg = StringMsg()
        metrics_msg.data = json.dumps(stats, default=float)
        self._metrics_pub.publish(metrics_msg)

        # Publish patrol status
        status_msg = StringMsg()
        status_msg.data = self._planner.classify()
        self._status_pub.publish(status_msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def spin(self) -> None:
        rclpy.spin(self._node)

    def destroy(self) -> None:
        self._node.destroy_node()
        rclpy.shutdown()


# ---------------------------------------------------------------------------
# KernelVelocityNode — terrain-aware velocity control
# ---------------------------------------------------------------------------

class KernelVelocityNode:
    """ROS2 node: subscribes to ORB-SLAM3 outputs, publishes velocity commands.

    Reads from ros2_ws/src packages:
      /orb_slam3/tracking_state   (std_msgs/Int32)
      /orb_slam3/map_points       (sensor_msgs/PointCloud2)
      /orb_slam3/camera_pose      (geometry_msgs/PoseStamped)
      /kernelcal/slam_novelty     (std_msgs/Float32, from SemanticSLAMKernelTracker)
      /kernelcal/patrol_status    (std_msgs/String)

    Publishes to:
      /kernelcal/velocity_cmd         (std_msgs/Float32)  — scalar forward speed m/s
      /fmu/in/trajectory_setpoint     (px4_msgs/TrajectorySetpoint) — PX4 offboard
      /kernelcal/velocity_factors     (std_msgs/Float32MultiArray) — debug factors
      /kernelcal/velocity_metrics     (std_msgs/String)   — JSON summary

    Parameters
    ----------
    v_max : float — maximum forward speed in m/s (default 3.0 for Earth Rover trike)
    v_crawl : float — speed when tracking is degraded
    use_px4_setpoint : bool — also publish to /fmu/in/trajectory_setpoint
    """

    def __init__(
        self,
        v_max: float = 3.0,
        v_crawl: float = 0.3,
        v_min: float = 0.0,
        use_px4_setpoint: bool = False,
        update_rate_hz: float = 10.0,
    ):
        if not _ROS_AVAILABLE:
            raise ImportError("rclpy is required for KernelVelocityNode.")

        from .velocity import (
            TerrainKernelVelocityController, VelocityBand,
            map_points_to_kernel, TRACKING_OK, TRACKING_LOST,
        )

        rclpy.init()
        self._node = Node("kernelcal_velocity")

        self._ctrl = TerrainKernelVelocityController(
            band=VelocityBand(v_min=v_min, v_max=v_max, v_crawl=v_crawl),
            use_look_ahead=True,
        )
        self._map_points_to_kernel = map_points_to_kernel
        self._TRACKING_OK   = TRACKING_OK
        self._TRACKING_LOST = TRACKING_LOST

        # State
        self._tracking_state: int = TRACKING_OK
        self._novelty: float = 0.0
        self._stability: float = 0.0
        self._latest_kernel = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Subscribers ───────────────────────────────────────────────
        self._node.create_subscription(
            _ros_import("std_msgs.msg", "Int32"),
            "/orb_slam3/tracking_state",
            self._tracking_state_cb, qos,
        )
        self._node.create_subscription(
            _ros_import("sensor_msgs.msg", "PointCloud2"),
            "/orb_slam3/map_points",
            self._map_points_cb, qos,
        )
        self._node.create_subscription(
            _ros_import("std_msgs.msg", "Float32"),
            "/kernelcal/slam_novelty",
            self._novelty_cb, qos,
        )
        self._node.create_subscription(
            _ros_import("std_msgs.msg", "Float32"),
            "/kernelcal/map_stability",
            self._stability_cb, qos,
        )

        # ── Publishers ────────────────────────────────────────────────
        self._v_pub = self._node.create_publisher(
            _ros_import("std_msgs.msg", "Float32"), "/kernelcal/velocity_cmd", 10
        )
        self._factors_pub = self._node.create_publisher(
            _ros_import("std_msgs.msg", "Float32MultiArray"),
            "/kernelcal/velocity_factors", 10,
        )
        self._metrics_pub = self._node.create_publisher(
            _ros_import("std_msgs.msg", "String"), "/kernelcal/velocity_metrics", 10
        )
        self._use_px4 = use_px4_setpoint
        if use_px4_setpoint:
            try:
                TrajectorySetpoint = _ros_import("px4_msgs.msg", "TrajectorySetpoint")
                self._px4_pub = self._node.create_publisher(
                    TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10
                )
            except Exception as e:
                self._node.get_logger().warn(
                    f"px4_msgs not available, disabling PX4 setpoint: {e}"
                )
                self._use_px4 = False

        self._timer = self._node.create_timer(
            1.0 / update_rate_hz, self._update_cb
        )
        self._node.get_logger().info("KernelVelocityNode initialised.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _tracking_state_cb(self, msg) -> None:
        self._tracking_state = int(msg.data)

    def _novelty_cb(self, msg) -> None:
        self._novelty = float(msg.data)

    def _stability_cb(self, msg) -> None:
        self._stability = float(msg.data)

    def _map_points_cb(self, msg) -> None:
        """Convert PointCloud2 to numpy xyz and build local kernel."""
        try:
            import struct
            pts = []
            point_step = msg.point_step
            for i in range(msg.width):
                offset = i * point_step
                x, y, z = struct.unpack_from("fff", msg.data, offset)
                pts.append([x, y, z])
            if pts:
                self._latest_kernel = self._map_points_to_kernel(
                    np.array(pts), fov_radius=5.0, n_sample=50
                )
        except Exception as e:
            self._node.get_logger().warn(f"map_points decode error: {e}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _update_cb(self) -> None:
        import json
        Float32     = _ros_import("std_msgs.msg", "Float32")
        Float32MA   = _ros_import("std_msgs.msg", "Float32MultiArray")
        StringMsg   = _ros_import("std_msgs.msg", "String")

        v_cmd = self._ctrl.update(
            novelty=self._novelty,
            stability=self._stability,
            current_kernel=self._latest_kernel,
            tracking_state=self._tracking_state,
        )

        # Publish scalar velocity
        v_msg = Float32()
        v_msg.data = v_cmd
        self._v_pub.publish(v_msg)

        # Publish PX4 setpoint if enabled
        if self._use_px4:
            sp = _ros_import("px4_msgs.msg", "TrajectorySetpoint")()
            sp.timestamp = int(self._node.get_clock().now().nanoseconds / 1000)
            # Forward velocity only — heading maintained by existing yaw controller
            sp.velocity[0] = v_cmd   # NED x = forward
            sp.velocity[1] = 0.0
            sp.velocity[2] = 0.0
            sp.yawspeed = float("nan")
            self._px4_pub.publish(sp)

        # Publish factor decomposition for debugging
        factors = self._ctrl.factor_histories()
        fma_msg = Float32MA()
        if factors["novelty_factor"].size > 0:
            fma_msg.data = [
                float(factors["novelty_factor"][-1]),
                float(factors["stability_factor"][-1]),
                float(factors["complexity_factor"][-1]),
                float(factors["tracking_factor"][-1]),
            ]
        self._factors_pub.publish(fma_msg)

        # JSON metrics
        metrics = self._ctrl.summary()
        metrics["v_cmd"] = v_cmd
        metrics["tracking_state"] = self._tracking_state
        m_msg = StringMsg()
        m_msg.data = json.dumps(metrics, default=float)
        self._metrics_pub.publish(m_msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def spin(self) -> None:
        rclpy.spin(self._node)

    def destroy(self) -> None:
        self._node.destroy_node()
        rclpy.shutdown()


def _ros_import(module: str, cls: str):
    """Lazy import helper for optional ROS message types."""
    import importlib
    return getattr(importlib.import_module(module), cls)

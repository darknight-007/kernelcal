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

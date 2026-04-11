"""ROS2 node: DepthStreamCodecNode.

Subscribes to a PointCloud2 or depth image topic, compresses each frame
via ``DepthStreamCodec``, and publishes:

  /kernelcal/depth/compressed      std_msgs/UInt8MultiArray — NPZ payload
  /kernelcal/depth/novelty         std_msgs/Float32         — HS novelty score
  /kernelcal/depth/is_keyframe     std_msgs/Bool            — keyframe flag
  /kernelcal/depth/metrics         std_msgs/String (JSON)   — rolling summary

Topic naming mirrors the pattern in ``kernelcal.navigation.ros_bridge``.

Graceful no-ROS fallback: importing this module without rclpy only raises
ImportError when the node is instantiated, not at import time.

Launch fragment (add to your earth_rover.launch.py or similar):
::
    from launch_ros.actions import Node

    depth_codec = Node(
        package="kernelcal_ros",
        executable="depth_codec_node",
        name="kernelcal_depth_codec",
        parameters=[{
            "input_topic":          "/camera/depth/points",
            "max_points":           256,
            "n_modes_keyframe":     32,
            "n_modes_delta":        16,
            "novelty_keyframe":     0.15,
            "force_keyframe_every": 30,
            "publish_rate_hz":      10.0,
        }],
    )

Standalone test (no real camera needed):
::
    from kernelcal.video.ros_bridge import run_demo
    run_demo(n_frames=50, fps=10)
"""

from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np

from .depth_stream import (
    DepthStreamCodec,
    DepthStreamConfig,
    CompressedDepthFrame,
    depth_image_to_xyz,
    pointcloud2_to_xyz,
)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    )
    from sensor_msgs.msg import PointCloud2, Image
    from std_msgs.msg import Float32, Bool, String, UInt8MultiArray
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False


class DepthStreamCodecNode:
    """ROS2 node wrapping DepthStreamCodec.

    Parameters
    ----------
    input_topic : str
        PointCloud2 or depth Image topic to subscribe to.
    input_type : 'pointcloud2' | 'depth_image'
        Message type on ``input_topic``.
    camera_intrinsics : dict | None
        Required when ``input_type='depth_image'``:
        {'fx': ..., 'fy': ..., 'cx': ..., 'cy': ...,
         'depth_scale': 0.001, 'max_depth': 10.0}
    config : DepthStreamConfig | None
        Codec configuration.  Defaults to DepthStreamConfig().
    """

    def __init__(
        self,
        input_topic: str = "/camera/depth/points",
        input_type: str = "pointcloud2",
        camera_intrinsics: dict | None = None,
        config: DepthStreamConfig | None = None,
        node_name: str = "kernelcal_depth_codec",
    ):
        if not _ROS_AVAILABLE:
            raise ImportError(
                "rclpy is required for DepthStreamCodecNode. "
                "Install ROS2 or run the standalone demo: "
                "from kernelcal.video.ros_bridge import run_demo; run_demo()"
            )
        rclpy.init()
        self._node = rclpy.create_node(node_name)
        self._codec = DepthStreamCodec(config)
        self._intrinsics = camera_intrinsics or {}
        self._input_type = input_type

        # QoS: best-effort, sensor data
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Subscriber
        if input_type == "pointcloud2":
            self._sub = self._node.create_subscription(
                PointCloud2, input_topic, self._pc2_callback, qos_sensor
            )
        else:
            self._sub = self._node.create_subscription(
                Image, input_topic, self._depth_image_callback, qos_sensor
            )

        # Publishers
        self._pub_compressed = self._node.create_publisher(
            UInt8MultiArray, "/kernelcal/depth/compressed", qos_latched
        )
        self._pub_novelty = self._node.create_publisher(
            Float32, "/kernelcal/depth/novelty", qos_sensor
        )
        self._pub_keyframe = self._node.create_publisher(
            Bool, "/kernelcal/depth/is_keyframe", qos_sensor
        )
        self._pub_metrics = self._node.create_publisher(
            String, "/kernelcal/depth/metrics", qos_latched
        )

        self._node.get_logger().info(
            f"DepthStreamCodecNode ready  topic={input_topic}  type={input_type}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _pc2_callback(self, msg) -> None:
        try:
            pts = pointcloud2_to_xyz(msg)
        except Exception as e:
            self._node.get_logger().warn(f"PointCloud2 parse error: {e}")
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._process(pts, t)

    def _depth_image_callback(self, msg) -> None:
        try:
            depth = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(
                msg.height, msg.width
            )
            pts = depth_image_to_xyz(
                depth,
                fx=self._intrinsics.get("fx", 600.0),
                fy=self._intrinsics.get("fy", 600.0),
                cx=self._intrinsics.get("cx", msg.width / 2),
                cy=self._intrinsics.get("cy", msg.height / 2),
                depth_scale=self._intrinsics.get("depth_scale", 0.001),
                max_depth=self._intrinsics.get("max_depth", 10.0),
            )
        except Exception as e:
            self._node.get_logger().warn(f"Depth image parse error: {e}")
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._process(pts, t)

    def _process(self, pts: np.ndarray, timestamp: float) -> None:
        if len(pts) < 4:
            return
        frame = self._codec.push(pts, timestamp=timestamp)
        self._publish(frame)

    def _publish(self, frame: CompressedDepthFrame) -> None:
        # Compressed payload
        payload = list(frame.to_bytes())
        msg_comp = UInt8MultiArray()
        msg_comp.data = payload
        self._pub_compressed.publish(msg_comp)

        # Novelty score
        msg_nov = Float32()
        msg_nov.data = float(frame.hs_novelty)
        self._pub_novelty.publish(msg_nov)

        # Keyframe flag
        msg_kf = Bool()
        msg_kf.data = frame.is_keyframe
        self._pub_keyframe.publish(msg_kf)

        # Rolling summary (every 10 frames)
        if frame.frame_idx % 10 == 0:
            summary = self._codec.compression_summary()
            msg_m = String()
            msg_m.data = json.dumps(summary)
            self._pub_metrics.publish(msg_m)

    def spin(self) -> None:
        rclpy.spin(self._node)

    def destroy(self) -> None:
        self._node.destroy_node()
        rclpy.shutdown()


# ---------------------------------------------------------------------------
# Standalone demo — no ROS2 required
# ---------------------------------------------------------------------------

def run_demo(
    n_frames: int = 60,
    fps: float = 10.0,
    n_points: int = 500,
    config: DepthStreamConfig | None = None,
    verbose: bool = True,
) -> DepthStreamCodec:
    """Simulate a depth stream with synthetic point clouds and print metrics.

    Parameters
    ----------
    n_frames  : number of frames to simulate
    fps       : simulated frame rate
    n_points  : points per frame (before codec subsampling)
    config    : codec configuration; defaults to DepthStreamConfig()
    verbose   : print per-frame log

    Returns
    -------
    The DepthStreamCodec after processing all frames.
    """
    cfg   = config or DepthStreamConfig()
    codec = DepthStreamCodec(cfg)
    rng   = np.random.default_rng(0)

    # Simulate a slowly drifting terrain + occasional rapid change
    base = rng.standard_normal((n_points, 3)) * 0.5
    dt   = 1.0 / fps

    print(f"\nDepth stream codec demo  ({n_frames} frames @ {fps:.0f} fps)")
    print(f"Config: max_points={cfg.max_points}  "
          f"modes kf/delta={cfg.n_modes_keyframe}/{cfg.n_modes_delta}  "
          f"novelty_kf={cfg.novelty_keyframe}")
    print(f"{'Frame':>5}  {'t':>6}  {'pts':>5}  {'modes':>5}  "
          f"{'novelty':>8}  {'kf':>3}  {'bytes':>8}")
    print("-" * 55)

    for i in range(n_frames):
        t = i * dt
        # Slow drift every frame
        drift = rng.standard_normal((n_points, 3)) * 0.02
        # Sudden jump at frame 20 (simulates fast camera motion)
        if i == 20:
            base = rng.standard_normal((n_points, 3)) * 0.5
        base += drift
        pts = base + rng.standard_normal((n_points, 3)) * 0.005

        frame = codec.push(pts, timestamp=t, seed=i)
        if verbose:
            print(
                f"{frame.frame_idx:>5}  {t:>6.2f}  "
                f"{frame.n_points_raw:>5}  {frame.n_modes:>5}  "
                f"{frame.hs_novelty:>8.4f}  "
                f"{'KF' if frame.is_keyframe else '  ':>3}  "
                f"{len(frame.to_bytes()):>8,}"
            )

    print("\nRolling summary:")
    for k, v in codec.compression_summary().items():
        print(f"  {k:<35s} {v}")
    print(f"\nTrajectory path length: {float(codec.trajectory.cumulative_length()[-1]):.4f}")
    return codec

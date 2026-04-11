"""kernelcal.video — spectral compression for sensor streams.

Depth / LiDAR streams
---------------------
``DepthStreamCodec`` compresses each PointCloud2 or depth image frame via
``compress_point_cloud`` and maintains a Hilbert-Schmidt novelty trajectory.
``DepthStreamCodecNode`` wraps it in a ROS2 node (requires rclpy).

Quick-start (no ROS2):
::
    from kernelcal.video import DepthStreamCodec, DepthStreamConfig, run_demo

    run_demo(n_frames=60, fps=10)

    # Or drive manually:
    codec = DepthStreamCodec(DepthStreamConfig(max_points=256, n_modes_keyframe=32))
    for t, pts in your_stream:
        frame = codec.push(pts, timestamp=t)
        print(frame.hs_novelty, frame.is_keyframe)

ROS2 node:
::
    from kernelcal.video.ros_bridge import DepthStreamCodecNode
    node = DepthStreamCodecNode(input_topic="/camera/depth/points")
    node.spin()
"""

from .depth_stream import (
    CompressedDepthFrame,
    DepthStreamCodec,
    DepthStreamConfig,
    depth_image_to_xyz,
    pointcloud2_to_xyz,
)
from .ros_bridge import run_demo

__all__ = [
    "CompressedDepthFrame",
    "DepthStreamCodec",
    "DepthStreamConfig",
    "depth_image_to_xyz",
    "pointcloud2_to_xyz",
    "run_demo",
]

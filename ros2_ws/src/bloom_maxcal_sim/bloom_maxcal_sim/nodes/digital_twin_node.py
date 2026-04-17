"""
ROS2 node: Spectral Digital Twin — Subscriber / Decoder / Publisher.

Bridges the kernelcal spectral encoder (rover-side) and the Blender/MeshLab
visualization pipeline (ground-station-side).

Topics subscribed
-----------------
/twin/spectral_update   std_msgs/UInt8MultiArray
    Serialised LargeMeshCompressed NPZ payload (output of encoder).

/twin/telemetry         std_msgs/Float32MultiArray
    Per-frame diagnostics from the rover:
    [D_t, spectral_entropy, delta_prime, curl_energy, latent_code,
     beta_0, beta_1, beta_2, timestamp]

/twin/landmarks         sensor_msgs/PointCloud2
    Sparse high-confidence LiDAR landmarks for Poisson pinning.

Topics published
----------------
/twin/decoded_mesh      visualization_msgs/Marker
    Base skeleton mesh (LIST_TRIANGLES) for RViz display.

/twin/detail_mesh       visualization_msgs/Marker
    Full synthesized mesh after detail_synthesis (if bandwidth allows).

/twin/diagnostics       std_msgs/String
    JSON diagnostics string (D_t, detail_level, patch_request, ...).

/twin/patch_request     std_msgs/Bool
    True when D_t exceeds patch threshold — triggers re-transmission request.

/twin/curl_heatmap      sensor_msgs/PointCloud2
    Per-vertex curl weight map for RViz colouring (x,y,z,intensity).

Parameters
----------
sigma2        (float, default 1.0)   — Gaussian MI source sigma^2
mu2           (float, default 2.0)   — MaxCal source weight mu_2
publish_rate  (float, default 1.0)   — decode loop rate (Hz)
export_ply    (bool,  default False)  — write PLY per frame (for MeshLab)
export_dir    (str,   default /tmp)  — output directory for PLY/OBJ files
seed          (int,   default 0)     — RNG seed for procedural synthesis
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Float32MultiArray, String, UInt8MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration

try:
    from sensor_msgs.msg import PointCloud2, PointField
    import sensor_msgs_py.point_cloud2 as pc2
    _HAS_PC2 = True
except ImportError:
    _HAS_PC2 = False

# Resolve kernelcal from manuscript repo
_PKG_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), *[".."] * 6)
)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from kernelcal.geo3d.large_mesh import LargeMeshCompressed
from kernelcal.geo3d.decoder import (
    SpectralTelemetry,
    decode,
    DetailLevel,
)
from kernelcal.geo3d.detail_synthesis import synthesize


class DigitalTwinNode(Node):
    """Ground-station digital twin decoder and visualization publisher."""

    def __init__(self) -> None:
        super().__init__("digital_twin_node")

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter("sigma2",       1.0)
        self.declare_parameter("mu2",          2.0)
        self.declare_parameter("publish_rate", 1.0)
        self.declare_parameter("export_ply",   False)
        self.declare_parameter("export_dir",   "/tmp/twin_frames")
        self.declare_parameter("seed",         0)

        self._sigma2  = self.get_parameter("sigma2").value
        self._mu2     = self.get_parameter("mu2").value
        self._rate    = self.get_parameter("publish_rate").value
        self._export  = self.get_parameter("export_ply").value
        self._out_dir = Path(self.get_parameter("export_dir").value)
        self._seed    = self.get_parameter("seed").value
        self._out_dir.mkdir(parents=True, exist_ok=True)

        # ── State ────────────────────────────────────────────────────────
        self._pending_payload: bytes | None = None
        self._pending_diag: list[float] = []
        self._pending_landmarks: np.ndarray | None = None
        self._frame_count = 0

        # ── Subscribers ──────────────────────────────────────────────────
        self.create_subscription(
            UInt8MultiArray, "/twin/spectral_update",
            self._cb_spectral, 10,
        )
        self.create_subscription(
            Float32MultiArray, "/twin/telemetry",
            self._cb_telemetry, 10,
        )
        if _HAS_PC2:
            self.create_subscription(
                PointCloud2, "/twin/landmarks",
                self._cb_landmarks, 10,
            )

        # ── Publishers ───────────────────────────────────────────────────
        self._pub_decoded   = self.create_publisher(Marker,              "/twin/decoded_mesh",  10)
        self._pub_detail    = self.create_publisher(Marker,              "/twin/detail_mesh",   10)
        self._pub_diag      = self.create_publisher(String,              "/twin/diagnostics",   10)
        self._pub_patch     = self.create_publisher(Bool,                "/twin/patch_request", 10)
        if _HAS_PC2:
            self._pub_heatmap = self.create_publisher(PointCloud2,       "/twin/curl_heatmap",  10)

        # ── Decode timer ─────────────────────────────────────────────────
        period = 1.0 / max(self._rate, 0.1)
        self.create_timer(period, self._decode_and_publish)

        self.get_logger().info("DigitalTwinNode ready  rate=%.1f Hz", self._rate)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _cb_spectral(self, msg: UInt8MultiArray) -> None:
        self._pending_payload = bytes(msg.data)
        self.get_logger().debug("Received spectral payload %d bytes", len(msg.data))

    def _cb_telemetry(self, msg: Float32MultiArray) -> None:
        self._pending_diag = list(msg.data)
        self.get_logger().debug("Received telemetry: %s", self._pending_diag)

    def _cb_landmarks(self, msg: "PointCloud2") -> None:
        if not _HAS_PC2:
            return
        pts = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        self._pending_landmarks = np.array(pts, dtype=float) if pts else None

    # ── Decode + publish cycle ────────────────────────────────────────────────

    def _decode_and_publish(self) -> None:
        if self._pending_payload is None:
            return

        try:
            compressed = LargeMeshCompressed.from_bytes(self._pending_payload)
        except Exception as e:
            self.get_logger().warning("Failed to deserialise payload: %s", e)
            return

        # Parse telemetry: [D_t, H, Delta', E_curl, lc, b0, b1, b2, ts]
        d = self._pending_diag
        D_t   = float(d[0]) if len(d) > 0 else None
        H     = float(d[1]) if len(d) > 1 else None
        dp    = float(d[2]) if len(d) > 2 else None
        E_c   = float(d[3]) if len(d) > 3 else None
        lc    = int(d[4])   if len(d) > 4 else None
        b0    = int(d[5])   if len(d) > 5 else 1
        b1    = int(d[6])   if len(d) > 6 else 0
        b2    = int(d[7])   if len(d) > 7 else 0
        ts    = float(d[8]) if len(d) > 8 else time.time()

        # Reconstruct per-mode D_m from D_t (uniform approximation)
        k = compressed.meta.get("n_modes", 1)
        D_m_residuals = np.full(k, abs(D_t) / k) if D_t is not None else None

        tel = SpectralTelemetry(
            compressed=compressed,
            betti=(b0, b1, b2),
            D_m_residuals=D_m_residuals,
            spectral_entropy=H,
            delta_prime=dp,
            curl_energy=E_c,
            latent_code=lc,
            landmark_xyz=self._pending_landmarks,
            timestamp=ts,
            frame_id="map",
        )

        # Stage 1+2: decode skeleton + triage
        twin = decode(tel, compute_curl=(E_c is None))

        # Stage 3: detail synthesis
        ply_path = None
        obj_path = None
        if self._export:
            base = self._out_dir / f"frame_{self._frame_count:05d}"
            ply_path = str(base.with_suffix(".ply"))
            obj_path = str(base.with_suffix(".obj"))

        synth = synthesize(
            twin,
            seed=self._seed + self._frame_count,
            export_ply_path=ply_path,
            export_obj_path=obj_path,
        )

        # Publish
        stamp = self.get_clock().now().to_msg()
        self._publish_mesh_marker(
            self._pub_decoded, twin.vertices_base,  twin.faces, stamp,
            ns="skeleton", color=(0.6, 0.6, 0.6, 0.5),
        )
        self._publish_mesh_marker(
            self._pub_detail, synth.vertices_detailed, synth.faces, stamp,
            ns="detail",   color=(0.8, 0.7, 0.5, 1.0),
        )
        self._pub_diag.publish(String(data=json.dumps(synth.diagnostics)))
        self._pub_patch.publish(Bool(data=twin.request_patch))

        if _HAS_PC2 and hasattr(self, "_pub_heatmap"):
            self._publish_curl_heatmap(
                synth.vertices_detailed, synth.texture_weights, stamp
            )

        self.get_logger().info(
            "Frame %d: level=%s D_t=%.2f rms_disp=%.4f patch=%s",
            self._frame_count,
            synth.detail_level.value,
            synth.diagnostics.get("D_t", 0),
            synth.diagnostics.get("disp_rms", 0),
            twin.request_patch,
        )
        self._frame_count += 1

    # ── RViz marker helpers ───────────────────────────────────────────────────

    def _publish_mesh_marker(
        self,
        pub,
        vertices: np.ndarray,
        faces: np.ndarray,
        stamp,
        *,
        ns: str = "twin",
        color: tuple[float, float, float, float] = (0.7, 0.7, 0.5, 1.0),
    ) -> None:
        """Publish a TRIANGLE_LIST Marker from vertex/face arrays."""
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp    = stamp
        m.ns              = ns
        m.id              = 0
        m.type            = Marker.TRIANGLE_LIST
        m.action          = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.lifetime        = Duration(sec=5, nanosec=0)

        # Subsample to ≤ 10 000 faces to stay within RViz limits
        f = faces
        if len(f) > 10_000:
            idx = np.random.default_rng(0).choice(len(f), 10_000, replace=False)
            f   = f[idx]

        for tri in f:
            for vi in tri:
                p = Point()
                p.x, p.y, p.z = float(vertices[vi, 0]), float(vertices[vi, 1]), float(vertices[vi, 2])
                m.points.append(p)
        pub.publish(m)

    def _publish_curl_heatmap(
        self,
        vertices: np.ndarray,
        weights: np.ndarray,
        stamp,
    ) -> None:
        """Publish per-vertex curl intensity as a PointCloud2."""
        if not _HAS_PC2:
            return
        fields = [
            PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pts = np.column_stack([vertices[:, :3].astype(np.float32),
                               weights.astype(np.float32)])
        msg = pc2.create_cloud(
            header=self._make_header(stamp, "map"),
            fields=fields,
            points=pts.tolist(),
        )
        self._pub_heatmap.publish(msg)

    @staticmethod
    def _make_header(stamp, frame_id: str):
        from std_msgs.msg import Header
        h = Header()
        h.stamp    = stamp
        h.frame_id = frame_id
        return h


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = DigitalTwinNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

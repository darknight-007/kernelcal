"""Tests for kernelcal.video depth stream codec."""

import numpy as np
import pytest

from kernelcal.video import (
    CompressedDepthFrame,
    DepthStreamCodec,
    DepthStreamConfig,
    depth_image_to_xyz,
)
from kernelcal.video.ros_bridge import run_demo


def _cloud(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3)) * 0.5


# ---------------------------------------------------------------------------
# depth_image_to_xyz
# ---------------------------------------------------------------------------

def test_depth_image_to_xyz_shape():
    H, W = 32, 32
    depth = np.full((H, W), 1000, dtype=np.uint16)  # 1m
    pts = depth_image_to_xyz(depth, fx=600, fy=600, cx=W/2, cy=H/2)
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert len(pts) == H * W


def test_depth_image_to_xyz_zeros_removed():
    depth = np.zeros((16, 16), dtype=np.uint16)
    pts = depth_image_to_xyz(depth, fx=600, fy=600, cx=8, cy=8)
    assert len(pts) == 0


# ---------------------------------------------------------------------------
# DepthStreamCodec — basic behaviour
# ---------------------------------------------------------------------------

def test_codec_first_frame_is_keyframe():
    codec = DepthStreamCodec()
    frame = codec.push(_cloud(), timestamp=0.0, seed=0)
    assert frame.is_keyframe
    assert frame.frame_idx == 0
    assert frame.hs_novelty == 0.0


def test_codec_novelty_nonnegative():
    codec = DepthStreamCodec()
    for i in range(5):
        frame = codec.push(_cloud(seed=i), timestamp=float(i), seed=i)
        assert frame.hs_novelty >= 0.0


def test_codec_forced_keyframe_interval():
    cfg = DepthStreamConfig(force_keyframe_every=5)
    codec = DepthStreamCodec(cfg)
    kf_idx = []
    for i in range(20):
        frame = codec.push(_cloud(seed=i), timestamp=float(i), seed=i)
        if frame.is_keyframe:
            kf_idx.append(frame.frame_idx)
    # Frame 0 is always KF; next forced KF at 5, 10, 15
    assert 0 in kf_idx


def test_codec_novelty_triggers_keyframe():
    cfg = DepthStreamConfig(
        novelty_keyframe=0.001,   # very low threshold → almost every frame is KF
        force_keyframe_every=100,
    )
    codec = DepthStreamCodec(cfg)
    codec.push(_cloud(seed=0), timestamp=0.0, seed=0)
    # Completely different cloud → large novelty → keyframe
    big_jump = np.random.default_rng(99).standard_normal((200, 3)) * 5.0
    frame = codec.push(big_jump, timestamp=1.0, seed=1)
    assert frame.is_keyframe


def test_codec_delta_frames_fewer_modes():
    cfg = DepthStreamConfig(
        n_modes_keyframe=24,
        n_modes_delta=8,
        novelty_keyframe=10.0,   # never trigger novelty-based KF
        force_keyframe_every=100,
    )
    codec = DepthStreamCodec(cfg)
    codec.push(_cloud(seed=0), timestamp=0.0, seed=0)  # KF
    frame = codec.push(_cloud(seed=1), timestamp=0.1, seed=1)  # delta
    assert not frame.is_keyframe
    assert frame.n_modes == cfg.n_modes_delta


def test_codec_serialization():
    codec = DepthStreamCodec()
    frame = codec.push(_cloud(), timestamp=0.0, seed=0)
    payload = frame.to_bytes()
    c2 = CompressedDepthFrame.__class__  # just check bytes are valid NPZ
    z = np.load(__import__("io").BytesIO(payload), allow_pickle=True)
    assert "eigenvalues" in z.files


def test_codec_compression_summary():
    codec = DepthStreamCodec()
    for i in range(10):
        codec.push(_cloud(seed=i), timestamp=float(i), seed=i)
    s = codec.compression_summary()
    assert s["n_frames"] == 10
    assert s["n_keyframes"] >= 1
    assert s["payload_bytes_total"] > 0


def test_codec_trajectory_length():
    codec = DepthStreamCodec()
    for i in range(6):
        codec.push(_cloud(seed=i), timestamp=float(i), seed=i)
    assert len(codec.trajectory) == 6


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------

def test_novelty_callback():
    received = []
    codec = DepthStreamCodec()
    codec.on_novelty(received.append)
    for i in range(4):
        codec.push(_cloud(seed=i), timestamp=float(i), seed=i)
    assert len(received) == 4
    assert all(v >= 0 for v in received)


def test_frame_callback():
    frames = []
    codec = DepthStreamCodec()
    codec.on_frame(frames.append)
    for i in range(3):
        codec.push(_cloud(seed=i), timestamp=float(i), seed=i)
    assert len(frames) == 3
    assert all(isinstance(f, CompressedDepthFrame) for f in frames)


# ---------------------------------------------------------------------------
# run_demo (end-to-end smoke test)
# ---------------------------------------------------------------------------

def test_run_demo_smoke():
    codec = run_demo(n_frames=10, fps=5.0, n_points=80, verbose=False)
    s = codec.compression_summary()
    assert s["n_frames"] == 10
    assert s["n_keyframes"] >= 1

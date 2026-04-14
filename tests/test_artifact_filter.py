"""
tests/test_artifact_filter.py
==============================
Test suite for DEM swath-artifact detection in the Jezero rook-adjacency pipeline.

Tests verify that:
  1. The Hough line detector finds near-vertical artifact lines in a synthetic mask
  2. Detected artifact pixels are removed after masking
  3. Curved/diagonal real channels are preserved (no over-filtering)
  4. A clean mask (no straight lines) triggers no false positives
  5. 45-degree diagonal channels are not mistaken for N-S seam artifacts
  6. After the full three-layer + Hough pipeline, no long straight line survives

Run with:
    cd software-kernelcal-deepgis-integration
    python3 -m pytest tests/test_artifact_filter.py -v
"""

from __future__ import annotations
import sys
import math
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2', reason='opencv-python-headless required')

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifact_filter import detect_and_mask_hough_lines


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic mask factories
# ══════════════════════════════════════════════════════════════════════════════

def _vertical_artifact(rows=200, cols=100, col=40, length_frac=0.80) -> np.ndarray:
    """Single near-vertical artifact line at `col`."""
    mask = np.zeros((rows, cols), dtype=bool)
    start = int(rows * (1 - length_frac) / 2)
    mask[start: rows - start, col] = True
    return mask


def _diagonal_channel(rows=150, cols=150, angle_deg=45.0) -> np.ndarray:
    """A channel at `angle_deg` from horizontal, spanning the full image height.

    dc/dr = cot(angle_deg) = 1/tan(angle_deg):
      89° from horizontal → almost vertical   (dc/dr ≈ 0.017)
      45° from horizontal → true diagonal     (dc/dr = 1.0)
      30° from horizontal → gentle slope      (dc/dr ≈ 1.73)
    """
    mask = np.zeros((rows, cols), dtype=bool)
    rad = math.radians(angle_deg)
    # avoid division by zero for exactly horizontal line
    col_per_row = 0.0 if abs(math.sin(rad)) < 1e-9 else math.cos(rad) / math.sin(rad)
    c_start = cols // 2
    for r in range(rows):
        c = int(round(c_start + r * col_per_row))
        if 0 <= c < cols:
            mask[r, c] = True
    return mask


def _curved_channel(rows=150, cols=100) -> np.ndarray:
    """A sinusoidal channel — real-looking, never triggers Hough."""
    mask = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        c = int(cols // 2 + 15 * math.sin(r * 0.12))
        if 0 <= c < cols:
            mask[r, c] = True
    return mask


def _mixed_mask(rows=300, cols=150) -> tuple[np.ndarray, list[int]]:
    """Realistic mixed mask: 2 vertical artifact lines + 3 real channels."""
    mask = np.zeros((rows, cols), dtype=bool)
    artifact_cols = [30, 110]
    # Artifacts: near-vertical, long
    for c in artifact_cols:
        mask[20: rows - 20, c] = True
    # Real 1: diagonal upper-left
    for i in range(80):
        r, c = 10 + i, 60 + i // 3
        if c < cols:
            mask[r, c] = True
    # Real 2: sinusoidal
    for r in range(rows // 2, rows - 10):
        c = int(75 + 12 * math.sin((r - rows // 2) * 0.15))
        if 0 <= c < cols:
            mask[r, c] = True
    # Real 3: short branching segment
    for i in range(30):
        r, c = 50 + i, 90 + i // 2
        if c < cols:
            mask[r, c] = True
    return mask, artifact_cols


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHoughDetectsArtifacts:

    def test_detects_single_vertical_line(self):
        """A single tall near-vertical line is detected."""
        mask = _vertical_artifact(rows=200, cols=100, col=50)
        _, n = detect_and_mask_hough_lines(mask, min_line_length=25)
        assert n >= 1, f'Expected ≥1 line, got {n}'

    def test_detects_two_artifact_lines(self):
        """Both artifact columns in the mixed mask are detected."""
        mask, _ = _mixed_mask()
        _, n = detect_and_mask_hough_lines(mask, min_line_length=25)
        assert n >= 2, f'Expected ≥2 lines, got {n}'

    def test_diagonal_45_not_flagged(self):
        """A 45-degree diagonal is NOT a near-vertical seam artifact."""
        mask = _diagonal_channel(angle_deg=45.0)
        _, n = detect_and_mask_hough_lines(
            mask, min_line_length=40, angle_tolerance_deg=8.0)
        assert n == 0, f'45° channel falsely flagged ({n} lines)'

    def test_diagonal_30_not_flagged(self):
        """A 30-degree channel is well outside the 8-degree vertical tolerance."""
        mask = _diagonal_channel(rows=200, cols=200, angle_deg=30.0)
        _, n = detect_and_mask_hough_lines(
            mask, min_line_length=40, angle_tolerance_deg=8.0)
        assert n == 0, f'30° channel falsely flagged ({n} lines)'

    def test_curved_channel_not_flagged(self):
        """A sinusoidal channel never forms a long straight segment."""
        mask = _curved_channel()
        _, n = detect_and_mask_hough_lines(mask, min_line_length=30)
        assert n == 0, f'Curved channel falsely flagged ({n} lines)'

    def test_empty_mask_no_crash(self):
        """Empty channel mask should not raise and should detect 0 lines."""
        mask = np.zeros((100, 80), dtype=bool)
        keep, n = detect_and_mask_hough_lines(mask, min_line_length=20)
        assert n == 0
        assert keep.shape == mask.shape


class TestHoughRemovesArtifactPixels:

    def test_artifact_column_emptied(self):
        """After masking, the artifact column retains fewer than 5 pixels."""
        artifact_col = 40
        mask = _vertical_artifact(rows=200, cols=100, col=artifact_col)
        keep, _ = detect_and_mask_hough_lines(mask, min_line_length=25)
        cleaned = mask & keep
        remaining = int(cleaned[:, artifact_col].sum())
        assert remaining < 5, (
            f'Artifact column {artifact_col} still has {remaining} pixels')

    def test_both_artifact_columns_emptied(self):
        """Both artifact columns in the mixed mask are cleaned."""
        mask, artifact_cols = _mixed_mask()
        keep, _ = detect_and_mask_hough_lines(mask, min_line_length=25)
        cleaned = mask & keep
        for c in artifact_cols:
            remaining = int(cleaned[:, c].sum())
            assert remaining < 8, (
                f'Artifact col {c}: {remaining} pixels remain after filter')

    def test_returns_correct_shape(self):
        """keep_mask must have the same shape as the input."""
        mask = _vertical_artifact()
        keep, _ = detect_and_mask_hough_lines(mask, min_line_length=25)
        assert keep.shape == mask.shape, (
            f'Shape mismatch: {keep.shape} vs {mask.shape}')

    def test_keep_is_boolean(self):
        mask = _vertical_artifact()
        keep, _ = detect_and_mask_hough_lines(mask, min_line_length=25)
        assert keep.dtype == bool


class TestHoughPreservesRealChannels:

    def test_real_channels_survive(self):
        """At least 40% of non-artifact pixels survive after filtering."""
        mask, artifact_cols = _mixed_mask(rows=300, cols=150)
        # Count pixels NOT on artifact columns
        non_art = mask.copy()
        for c in artifact_cols:
            non_art[:, c] = False
        n_real_before = int(non_art.sum())

        keep, _ = detect_and_mask_hough_lines(mask, min_line_length=25)
        n_real_after = int((non_art & keep).sum())
        frac = n_real_after / max(n_real_before, 1)
        assert frac >= 0.40, (
            f'Only {frac:.1%} of real-channel pixels survived filtering')

    def test_curved_channel_fully_preserved(self):
        """A purely curved channel mask survives intact."""
        mask = _curved_channel()
        n_before = int(mask.sum())
        keep, _ = detect_and_mask_hough_lines(mask, min_line_length=30)
        n_after = int((mask & keep).sum())
        assert n_after >= n_before, (
            f'Curved channel pixels reduced: {n_before} → {n_after}')


class TestHoughAngleTolerance:

    @pytest.mark.parametrize('angle_deg,should_flag', [
        (89.0, True),   # nearly vertical — artifact
        (85.0, True),   # within 8° tolerance — artifact
        (80.0, False),  # 10° from vertical — real channel
        (45.0, False),  # diagonal — real channel
    ])
    def test_angle_boundary(self, angle_deg: float, should_flag: bool):
        """Channels at various angles should be correctly classified."""
        mask = _diagonal_channel(rows=200, cols=200, angle_deg=angle_deg)
        _, n = detect_and_mask_hough_lines(
            mask, min_line_length=40, angle_tolerance_deg=8.0)
        if should_flag:
            assert n >= 1, (
                f'{angle_deg}° channel should be flagged but n={n}')
        else:
            assert n == 0, (
                f'{angle_deg}° channel should NOT be flagged but n={n}')


class TestHoughMinLineLength:

    def test_short_line_not_detected(self):
        """A vertical segment shorter than min_line_length is not an artifact."""
        mask = np.zeros((100, 60), dtype=bool)
        mask[40:55, 30] = True   # 15-pixel segment, well below threshold
        _, n = detect_and_mask_hough_lines(mask, min_line_length=25)
        assert n == 0, f'Short 15-px segment falsely flagged ({n} lines)'

    def test_long_line_is_detected(self):
        """A vertical segment longer than min_line_length is detected."""
        mask = np.zeros((120, 60), dtype=bool)
        mask[10:100, 30] = True  # 90-pixel segment
        _, n = detect_and_mask_hough_lines(mask, min_line_length=25)
        assert n >= 1, f'90-px vertical segment not detected (n={n})'


class TestEndToEndPipelineClean:
    """Integration test: run the full Hough filter on a realistic artifact mask
    and verify that no Hough-detectable straight lines remain in the output."""

    def test_no_long_lines_after_filtering(self):
        """After detect_and_mask_hough_lines, re-applying Hough finds 0 lines."""
        mask, _ = _mixed_mask(rows=300, cols=150)
        keep, n_before = detect_and_mask_hough_lines(mask, min_line_length=25)
        assert n_before >= 1, 'Precondition: artifacts must be present before filter'

        cleaned = mask & keep
        _, n_after = detect_and_mask_hough_lines(cleaned, min_line_length=25)
        assert n_after == 0, (
            f'After filtering, re-applying Hough still finds {n_after} line(s)')

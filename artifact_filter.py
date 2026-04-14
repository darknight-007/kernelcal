"""
artifact_filter.py
==================
General-purpose DEM and channel-mask artifact detection utilities.

Currently provides:
    detect_and_mask_hough_lines — OpenCV probabilistic Hough transform that
        finds near-vertical (N-S swath-seam) artifact lines in a binary
        channel mask and returns a keep-mask with those pixels erased.

These utilities are independent of any specific study site.  The companion
test suite lives in tests/test_artifact_filter.py.

Dependencies
------------
    numpy
    opencv-python-headless  (pip install opencv-python-headless)
"""

from __future__ import annotations
import math
import numpy as np


def detect_and_mask_hough_lines(
        mask: np.ndarray,
        min_line_length: int = 25,
        max_line_gap: int = 8,
        angle_tolerance_deg: float = 8.0,
        line_buffer_px: int = 2,
) -> tuple[np.ndarray, int]:
    """OpenCV probabilistic Hough transform to detect and remove N-S artifact lines.

    Swath-seam artifacts in orbital DEMs appear as long straight near-vertical
    lines in the D8 channel mask — structurally impossible for real dendritic
    drainage networks.

    Parameters
    ----------
    mask              : bool array (rows × cols), True = channel pixel
    min_line_length   : minimum segment length (pixels) to be classified as
                        an artifact line
    max_line_gap      : gap allowed within one line segment (pixels)
    angle_tolerance_deg : maximum angular deviation from vertical (90°) for a
                        segment to be treated as an artifact; 0° = exact N-S
    line_buffer_px    : half-width of the erasing brush drawn around each
                        detected artifact segment

    Returns
    -------
    keep_mask : bool array (same shape as mask), True = pixel retained
    n_lines   : number of near-vertical artifact segments detected
    """
    import cv2  # lazy import — optional dependency

    img = (mask.astype(np.uint8) * 255)
    if img.sum() == 0:
        return np.ones(mask.shape, dtype=bool), 0

    lines = cv2.HoughLinesP(
        img,
        rho=1,
        theta=np.pi / 180,
        threshold=12,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        print('  Hough filter: no lines detected')
        return np.ones(mask.shape, dtype=bool), 0

    # Retain only near-vertical segments (angle from vertical ≤ tolerance)
    artifact_segs = []
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dy < 1:
            continue  # horizontal segment — not a seam artifact
        angle_from_vertical = math.degrees(math.atan2(dx, dy))
        if angle_from_vertical <= angle_tolerance_deg:
            artifact_segs.append((x1, y1, x2, y2))

    if not artifact_segs:
        print('  Hough filter: no near-vertical artifact segments')
        return np.ones(mask.shape, dtype=bool), 0

    # Rasterise a buffer around each detected segment
    rows, cols = mask.shape
    art_img = np.zeros((rows, cols), dtype=np.uint8)
    for x1, y1, x2, y2 in artifact_segs:
        cv2.line(art_img, (x1, y1), (x2, y2), 1,
                 thickness=line_buffer_px * 2 + 1)

    keep = (art_img == 0)
    n_masked = int((~keep & mask).sum())
    print(f'  Hough filter: {len(artifact_segs)} near-vertical segment(s) → '
          f'{n_masked} channel pixels removed  (buffer={line_buffer_px} px)')
    return keep, len(artifact_segs)

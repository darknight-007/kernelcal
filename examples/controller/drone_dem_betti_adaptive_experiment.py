#!/usr/bin/env python3
"""Drone DEM camera simulation with Betti-adaptive mapping.

Dependencies: numpy, scipy, matplotlib

What this script does
---------------------
1) Emulates a drone with a nadir DEM camera.
2) Uses a square camera footprint derived from altitude + FOV angle.
3) Captures DEM patches along a flight path.
4) Computes stream-like topology per patch: Betti numbers β₀ (connected
   components) and β₁ (loops) of the stream-pixel graph.
5) Chooses the next waypoint by splitting the current FoV into four diagonal
   quadrants (NW/NE/SW/SE), scoring each by its local Betti topology::

       score = w_beta1 * (β₁/n)   # topological complexity — P4 Δβ₁ signal
             - w_beta0 * (β₀/n)   # fragmentation penalty (high β₀ = noisy mask)
             + w_unseen * unseen   # exploration incentive

   β₁/n measures braided-loop density in the channel graph — the per-node
   Δβ₁ signal from P4 ("Spectral Kernel Dynamics as a Biosignature Framework").
   β₀/n is negated: a single connected network (β₀=1) is preferred over many
   disconnected fragments (β₀>>1).  n = graph node count normalises both to
   [0, 1].  The drone moves to the outer corner of the winning quadrant
   (step = side_px // 2 in each diagonal direction).

kernelcal integration
---------------------
- ``d8_flow_direction`` / ``flow_accumulation`` → kernelcal.terrain.dem
- Betti numbers (graph topology, consistent across extractors)
  → kernelcal.terrain.graph_codec.combinatorial_betti
- Abiotic null model for Δβ₁
  → kernelcal.terrain.channels.abiotic_beta1_channels
- Biosignature computation → kernelcal.terrain.biosig.topological_biosignature

6) Writes summary plots of coverage, path, and topology history.

Example (Tonto Basin HydroShed)
--------------------------------
::

  python3 examples/controller/drone_dem_betti_adaptive_experiment.py \\
    --dem-tiff datasets/hydroshed-dem/na_con_3s/na_con_3s.tif \\
    --bbox-lonlat=-112.6,33.2,-110.6,34.3 \\
    --nodata-value 32767 --dem-resolution-m 90 \\
    --altitude-m 2000 --fov-deg 100 --steps 200 \\
    --channel-extractor rivgraph --rivgraph-prune-dangling \\
    --rivgraph-repo /home/jdas/Documents/manuscripts/rivgraph \\
    --w-beta1 2.5 --w-beta0 0.5 --w-unseen 5.0 \\
    --realtime --realtime-block \\
    --output-dir datasets/hydroshed-dem/drone_betti_run
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import ndimage
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from matplotlib import animation as mpl_animation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import colormaps as mpl_colormaps

# ---------------------------------------------------------------------------
# kernelcal.terrain integration
# The script lives inside the kernelcal repo so these are always available.
# They replace local duplicate implementations below.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from kernelcal.terrain.dem import (
        d8_flow_direction as _kt_d8_flow_direction,
        flow_accumulation as _kt_flow_accumulation,
    )
    from kernelcal.terrain.graph_codec import combinatorial_betti as _kt_combinatorial_betti
    from kernelcal.terrain.channels import abiotic_beta1_channels as _kt_abiotic_beta1
    _KT_AVAILABLE = True
except ImportError:  # graceful degradation when running outside repo
    _KT_AVAILABLE = False

# Shared quadrant-Betti scoring + tie-break + revisit penalty lives in
# ``kernelcal.graph_explorer`` so the bishop-rocks k-NN explorer uses the
# exact same policy as this one without duplicating the formula.
from kernelcal.graph_explorer import (
    BettiWeights,
    CameraModel,
    Candidate,
    QUADRANT_OFFSETS_IMAGE,
    choose_best_candidate,
)


_CC_CMAP = mpl_colormaps.get_cmap("tab20")


def _cc_colors_for_labels(labels: np.ndarray) -> np.ndarray:
    """Map integer CC labels to cycling RGBA colors (unknown -> gray)."""
    if labels.size == 0:
        return np.empty((0, 4), dtype=float)
    lab = np.asarray(labels, dtype=int)
    n = int(_CC_CMAP.N)
    idx = np.mod(np.maximum(lab, 0), n) / float(max(1, n - 1))
    rgba = _CC_CMAP(idx)
    unknown = lab < 0
    if np.any(unknown):
        rgba[unknown] = (0.6, 0.6, 0.6, 1.0)
    return rgba


_D8_OFFSETS = np.array(
    [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ],
    dtype=int,
)
_D8_DIST = np.array([np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0, 1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0)])


@dataclass
class CaptureRecord:
    step: int
    row: int
    col: int
    beta0: int
    beta1: int
    fiedler: float
    stream_fraction: float
    unseen_fraction: float
    score: float


def build_exploration_graph_edges(
    records: list[CaptureRecord],
    radius_px: float,
    k_nearest: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Build an exploration graph from capture nodes.

    Temporal edges connect consecutive captures.
    Proximity edges connect each node to up to k nearest prior nodes
    within radius_px.
    """
    n = len(records)
    if n <= 1:
        return [], []

    pts = np.array([[r.row, r.col] for r in records], dtype=float)
    temporal = [(i - 1, i) for i in range(1, n)]
    prox_set: set[tuple[int, int]] = set()
    k_nearest = max(0, int(k_nearest))
    radius2 = float(radius_px) ** 2

    for i in range(1, n):
        prev = pts[:i]
        d2 = np.sum((prev - pts[i]) ** 2, axis=1)
        ids = np.where(d2 <= radius2)[0]
        if ids.size == 0:
            continue
        order = ids[np.argsort(d2[ids])]
        for j in order[:k_nearest]:
            a, b = (int(j), i) if int(j) < i else (i, int(j))
            # Keep temporal chain and proximity graph visually distinct.
            if b - a > 1:
                prox_set.add((a, b))
    return temporal, sorted(prox_set)


def edges_to_segments(
    records: list[CaptureRecord],
    edges: list[tuple[int, int]],
) -> list[np.ndarray]:
    if len(edges) == 0:
        return []
    pts = np.array([[r.col, r.row] for r in records], dtype=float)
    return [np.array([pts[i], pts[j]], dtype=float) for i, j in edges]


def parse_bbox_lonlat(spec: str) -> tuple[float, float, float, float]:
    """Parse bbox string: 'lon_min,lat_min,lon_max,lat_max'."""
    try:
        parts = [float(x.strip()) for x in spec.split(",")]
    except Exception as e:
        raise ValueError(
            "Invalid --bbox-lonlat. Expected 'lon_min,lat_min,lon_max,lat_max'."
        ) from e
    if len(parts) != 4:
        raise ValueError("Invalid --bbox-lonlat. Expected exactly 4 comma-separated values.")
    lon_min, lat_min, lon_max, lat_max = parts
    if lon_max <= lon_min or lat_max <= lat_min:
        raise ValueError("Invalid bbox ordering. Require lon_max>lon_min and lat_max>lat_min.")
    return lon_min, lat_min, lon_max, lat_max


def crop_geotiff_by_bbox(
    src_tiff: Path,
    bbox_lonlat: tuple[float, float, float, float],
    dst_tiff: Path,
) -> Path:
    """Crop GeoTIFF using GDAL with geographic bbox (EPSG:4326-style lon/lat)."""
    lon_min, lat_min, lon_max, lat_max = bbox_lonlat
    dst_tiff.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gdal_translate",
        "-projwin",
        str(lon_min),   # upper-left x (lon_min)
        str(lat_max),   # upper-left y (lat_max)
        str(lon_max),   # lower-right x (lon_max)
        str(lat_min),   # lower-right y (lat_min)
        str(src_tiff),
        str(dst_tiff),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "gdal_translate not found. Install GDAL or pre-crop the DEM externally."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"gdal_translate failed for bbox {bbox_lonlat}.\nSTDERR:\n{e.stderr}"
        ) from e
    return dst_tiff


def parse_window_spec(spec: str, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Parse a window string like 'r0:r1,c0:c1' into clipped bounds."""
    try:
        row_part, col_part = spec.split(",")
        r0_s, r1_s = row_part.split(":")
        c0_s, c1_s = col_part.split(":")
        r0, r1 = int(r0_s), int(r1_s)
        c0, c1 = int(c0_s), int(c1_s)
    except Exception as e:
        raise ValueError("Invalid --tiff-window format. Expected 'r0:r1,c0:c1'.") from e

    nrows, ncols = shape
    r0 = int(np.clip(r0, 0, nrows))
    r1 = int(np.clip(r1, 0, nrows))
    c0 = int(np.clip(c0, 0, ncols))
    c1 = int(np.clip(c1, 0, ncols))
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"Invalid clipped window bounds: {(r0, r1, c0, c1)}")
    return r0, r1, c0, c1


def centered_window_for_max_cells(shape: tuple[int, int], max_cells: int) -> tuple[int, int, int, int]:
    """Centered crop window with at most max_cells pixels."""
    nrows, ncols = shape
    if nrows * ncols <= max_cells:
        return 0, nrows, 0, ncols

    side = int(np.sqrt(max_cells))
    side = max(32, min(side, nrows, ncols))
    cr = nrows // 2
    cc = ncols // 2
    r0 = max(0, cr - side // 2)
    c0 = max(0, cc - side // 2)
    r1 = min(nrows, r0 + side)
    c1 = min(ncols, c0 + side)
    return r0, r1, c0, c1


def load_tiff_dem(
    path: Path,
    nodata_value: float | None,
    window_spec: str | None,
    max_cells: int,
) -> np.ndarray:
    """Load a TIFF/GeoTIFF DEM as a 2D float array.

    Tries `tifffile` first (best support for large/scientific TIFF),
    then falls back to matplotlib image loading.
    """
    p = path.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"TIFF file not found: {p}")

    arr = None
    last_err: Exception | None = None
    try:
        import tifffile  # type: ignore

        with tifffile.TiffFile(str(p)) as tf:
            shape = tuple(int(x) for x in tf.pages[0].shape[:2])
        if window_spec:
            r0, r1, c0, c1 = parse_window_spec(window_spec, shape)
        else:
            r0, r1, c0, c1 = centered_window_for_max_cells(shape, max_cells=max_cells)
            if (r0, r1, c0, c1) != (0, shape[0], 0, shape[1]):
                print(
                    "[info] Large TIFF detected; using centered crop "
                    f"rows[{r0}:{r1}] cols[{c0}:{c1}] "
                    f"({(r1-r0)*(c1-c0)} cells). "
                    "Override with --tiff-window."
                )
        mm = tifffile.memmap(str(p))
        arr = mm[r0:r1, c0:c1]
    except Exception as e:
        last_err = e

    if arr is None:
        try:
            from matplotlib import image as mpimg

            arr = mpimg.imread(str(p))
            if window_spec:
                r0, r1, c0, c1 = parse_window_spec(window_spec, arr.shape[:2])
                arr = arr[r0:r1, c0:c1]
        except Exception as e:
            msg = (
                f"Could not read TIFF: {p}\n"
                "Install tifffile (`pip install tifffile`) for large GeoTIFF support.\n"
                f"Primary loader error: {last_err}\nFallback error: {e}"
            )
            raise RuntimeError(msg) from e

    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        # Keep first band if TIFF contains multiple channels/bands.
        # Supports both (rows, cols, bands) and (bands, rows, cols).
        if arr.shape[-1] <= 8 and arr.shape[0] > 8 and arr.shape[1] > 8:
            arr = arr[..., 0]
        else:
            arr = arr[0, ...]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D DEM after TIFF load, got shape {arr.shape}.")

    dem = arr.astype(float)
    if nodata_value is not None:
        dem[np.isclose(dem, float(nodata_value))] = np.nan
    return dem


def synthetic_dem(nrows: int, ncols: int, seed: int) -> np.ndarray:
    """Generate a smooth DEM with channel-like valleys."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, ncols)
    y = np.linspace(0.0, 1.0, nrows)
    xx, yy = np.meshgrid(x, y)

    base = 900.0 - 350.0 * yy
    terrain = (
        70.0 * np.sin(2.5 * np.pi * xx) * np.cos(1.5 * np.pi * yy)
        + 35.0 * np.sin(5.2 * np.pi * xx + 0.7)
        + 22.0 * np.cos(3.1 * np.pi * yy)
    )

    valley = np.zeros_like(base)
    for k in range(4):
        phase = 0.3 + 0.9 * k
        center = 0.18 + 0.2 * k + 0.07 * np.sin(6.0 * yy + phase)
        valley -= 60.0 * np.exp(-((xx - center) ** 2) / (2.0 * (0.018 + 0.005 * k) ** 2))

    noise = ndimage.gaussian_filter(rng.normal(0.0, 1.0, size=(nrows, ncols)), sigma=2.5)
    noise = 9.0 * noise / (np.std(noise) + 1e-12)
    dem = base + terrain + valley + noise
    return dem.astype(float)


def d8_flow_direction(dem: np.ndarray, resolution_m: float) -> np.ndarray:
    """D8 steepest-descent routing (-1 marks sink).

    Delegates to kernelcal.terrain.dem when available (avoids duplication).
    """
    if _KT_AVAILABLE:
        return _kt_d8_flow_direction(dem, dx=resolution_m, dy=resolution_m)
    # Local fallback (identical logic, kept for standalone use outside repo).
    z = np.asarray(dem, dtype=float)
    nrows, ncols = z.shape
    out = np.full((nrows, ncols), -1, dtype=np.int8)
    scales = resolution_m * _D8_DIST
    for r in range(nrows):
        for c in range(ncols):
            if not np.isfinite(z[r, c]):
                continue
            best = -1
            best_slope = 0.0
            z0 = z[r, c]
            for d, (dr, dc) in enumerate(_D8_OFFSETS):
                rr, cc = r + dr, c + dc
                if 0 <= rr < nrows and 0 <= cc < ncols and np.isfinite(z[rr, cc]):
                    drop = z0 - z[rr, cc]
                    slope = drop / scales[d]
                    if slope > best_slope:
                        best_slope = slope
                        best = d
            out[r, c] = best
    return out


def flow_accumulation(fdir: np.ndarray) -> np.ndarray:
    """Accumulate upstream contributing area (in cell counts).

    Delegates to kernelcal.terrain.dem when available (avoids duplication).
    """
    if _KT_AVAILABLE:
        return _kt_flow_accumulation(fdir)
    # Local fallback.
    nrows, ncols = fdir.shape
    acc = np.ones((nrows, ncols), dtype=np.int32)
    indeg = np.zeros((nrows, ncols), dtype=np.int32)
    for r in range(nrows):
        for c in range(ncols):
            d = int(fdir[r, c])
            if d >= 0:
                dr, dc = _D8_OFFSETS[d]
                rr, cc = r + dr, c + dc
                if 0 <= rr < nrows and 0 <= cc < ncols:
                    indeg[rr, cc] += 1
    q = [(r, c) for r in range(nrows) for c in range(ncols) if indeg[r, c] == 0]
    head = 0
    while head < len(q):
        r, c = q[head]
        head += 1
        d = int(fdir[r, c])
        if d >= 0:
            dr, dc = _D8_OFFSETS[d]
            rr, cc = r + dr, c + dc
            if 0 <= rr < nrows and 0 <= cc < ncols:
                acc[rr, cc] += acc[r, c]
                indeg[rr, cc] -= 1
                if indeg[rr, cc] == 0:
                    q.append((rr, cc))
    return acc


def stream_mask_from_patch(
    patch_dem: np.ndarray,
    resolution_m: float,
    percentile: float,
    *,
    close_iters: int = 0,
    dilate_iters: int = 0,
) -> np.ndarray:
    """Infer stream-like binary mask from local flow accumulation.

    close_iters: morphological closing iterations (bridges small gaps).
    dilate_iters: morphological dilation iterations applied after closing
        (thickens channels so skeletonization fuses nearby arms).
    Both use a 3x3 structuring element (8-connectivity).
    """
    if patch_dem.size == 0:
        return np.zeros_like(patch_dem, dtype=bool)
    valid = np.isfinite(patch_dem)
    if float(np.mean(valid)) < 0.25:
        return np.zeros_like(patch_dem, dtype=bool)
    work = np.asarray(patch_dem, dtype=float).copy()
    if not np.all(valid):
        fill = float(np.nanmean(work)) if np.isfinite(np.nanmean(work)) else 0.0
        work[~valid] = fill
    fdir = d8_flow_direction(work, resolution_m=resolution_m)
    acc = flow_accumulation(fdir).astype(float)
    thr = np.percentile(acc, percentile)
    mask = (acc >= max(2.0, thr)) & valid
    if int(close_iters) > 0:
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=int(close_iters))
    if int(dilate_iters) > 0:
        mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=int(dilate_iters))
    return mask & valid


def _count_components_graph(n_nodes: int, edges: list[tuple[int, int]]) -> int:
    if n_nodes <= 0:
        return 0
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    return len({find(i) for i in range(n_nodes)})


def _betti_from_rivgraph_links_nodes(links: dict, nodes: dict) -> tuple[int, int]:
    n = len(nodes.get("idx", []))
    if n == 0:
        return 0, 0
    seen: set[tuple[int, int]] = set()
    for conn in links.get("conn", []):
        if conn is None or len(conn) != 2:
            continue
        a, b = int(conn[0]), int(conn[1])
        if a == b:
            continue
        i, j = (a, b) if a < b else (b, a)
        seen.add((i, j))
    edge_list = sorted(seen)
    e = len(edge_list)
    beta0 = _count_components_graph(n, edge_list)
    beta1 = max(0, e - n + beta0)
    return int(beta0), int(beta1)




def _fiedler_from_graph(
    n_nodes: int,
    edges: list[tuple[int, int]],
    *,
    normalized: bool = True,
    mode: str = "largest_cc",
) -> float:
    """Algebraic connectivity (2nd-smallest Laplacian eigenvalue).

    Uses the normalized Laplacian by default so values live in [0, 2]
    independent of graph size, which makes cross-patch comparison meaningful.

    mode:
      - "strict"       : returns 0.0 if the full graph is disconnected
      - "largest_cc"   : returns lambda_2 of the largest connected component
                         (the standard convention for fragmented channel graphs)
    """
    m = int(n_nodes)
    if m <= 1:
        return 0.0
    if len(edges) == 0:
        return 0.0

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for a, b in edges:
        ai, bi = int(a), int(b)
        if ai == bi:
            continue
        if not (0 <= ai < m and 0 <= bi < m):
            continue
        rows.extend([ai, bi])
        cols.extend([bi, ai])
        data.extend([1.0, 1.0])
    if len(data) == 0:
        return 0.0

    adj = sparse.coo_matrix((data, (rows, cols)), shape=(m, m)).tocsr()
    n_comp, labels = csgraph.connected_components(adj, directed=False)

    if n_comp > 1:
        if mode != "largest_cc":
            return 0.0
        counts = np.bincount(labels, minlength=n_comp)
        biggest = int(np.argmax(counts))
        keep = np.where(labels == biggest)[0]
        if keep.size <= 1:
            return 0.0
        remap = -np.ones(m, dtype=np.int64)
        remap[keep] = np.arange(keep.size, dtype=np.int64)
        sub_rows = remap[np.asarray(rows, dtype=np.int64)]
        sub_cols = remap[np.asarray(cols, dtype=np.int64)]
        mask = (sub_rows >= 0) & (sub_cols >= 0)
        sub_rows = sub_rows[mask]
        sub_cols = sub_cols[mask]
        sub_data = np.asarray(data, dtype=float)[mask]
        adj = sparse.coo_matrix(
            (sub_data, (sub_rows, sub_cols)),
            shape=(int(keep.size), int(keep.size)),
        ).tocsr()
        m = int(keep.size)

    lap = csgraph.laplacian(adj, normed=bool(normalized))

    def _dense_lambda2() -> float:
        vals = np.linalg.eigvalsh(np.asarray(lap.toarray(), dtype=float))
        vals = np.sort(np.real(vals))
        return float(max(0.0, vals[1])) if vals.size >= 2 else 0.0

    if m <= 4:
        return _dense_lambda2()

    try:
        vals = eigsh(lap.astype(float), k=2, which="SA", return_eigenvectors=False)
        vals = np.sort(np.real(vals))
        lam2 = float(max(0.0, vals[1])) if vals.size >= 2 else 0.0
        if lam2 > 0.0:
            return lam2
    except Exception:
        pass
    return _dense_lambda2()


def _fiedler_ipr_from_graph(
    n_nodes: int,
    edges: list[tuple[int, int]],
    *,
    normalized: bool = True,
    mode: str = "largest_cc",
) -> tuple[float, float]:
    """Return (lambda_2, IPR(v2)) for a graph Laplacian.

    IPR is the inverse participation ratio of the Fiedler eigenvector:
        IPR(v) = sum(v^4) / (sum(v^2)^2)
    Larger IPR indicates stronger localization.
    """
    m = int(n_nodes)
    if m <= 1 or len(edges) == 0:
        return 0.0, 0.0

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for a, b in edges:
        ai, bi = int(a), int(b)
        if ai == bi:
            continue
        if not (0 <= ai < m and 0 <= bi < m):
            continue
        rows.extend([ai, bi])
        cols.extend([bi, ai])
        data.extend([1.0, 1.0])
    if len(data) == 0:
        return 0.0, 0.0

    adj = sparse.coo_matrix((data, (rows, cols)), shape=(m, m)).tocsr()
    n_comp, labels = csgraph.connected_components(adj, directed=False)

    if n_comp > 1:
        if mode != "largest_cc":
            return 0.0, 0.0
        counts = np.bincount(labels, minlength=n_comp)
        biggest = int(np.argmax(counts))
        keep = np.where(labels == biggest)[0]
        if keep.size <= 1:
            return 0.0, 0.0
        remap = -np.ones(m, dtype=np.int64)
        remap[keep] = np.arange(keep.size, dtype=np.int64)
        sub_rows = remap[np.asarray(rows, dtype=np.int64)]
        sub_cols = remap[np.asarray(cols, dtype=np.int64)]
        mask = (sub_rows >= 0) & (sub_cols >= 0)
        sub_rows = sub_rows[mask]
        sub_cols = sub_cols[mask]
        sub_data = np.asarray(data, dtype=float)[mask]
        adj = sparse.coo_matrix(
            (sub_data, (sub_rows, sub_cols)),
            shape=(int(keep.size), int(keep.size)),
        ).tocsr()
        m = int(keep.size)

    if m <= 1:
        return 0.0, 0.0
    lap = csgraph.laplacian(adj, normed=bool(normalized))

    try:
        k = 2 if m <= 32 else 3
        vals_raw, vecs = eigsh(lap.astype(float), k=k, which="SA", return_eigenvectors=True)
        order = np.argsort(np.real(vals_raw))
        vals = np.real(vals_raw)[order]
        if vals.size < 2:
            return 0.0, 0.0
        lam2 = float(max(0.0, vals[1]))
        vecs = np.real(vecs[:, order])
        v2 = vecs[:, 1] if vecs.shape[1] > 1 else None
        if v2 is None:
            return lam2, 0.0
        denom = float(np.sum(v2 * v2))
        if denom <= 1e-12:
            return lam2, 0.0
        ipr = float(np.sum(v2**4) / (denom * denom))
        return lam2, ipr
    except Exception:
        return _fiedler_from_graph(n_nodes, edges, normalized=normalized, mode=mode), 0.0


def _fiedler_from_rivgraph(links: dict, nodes: dict) -> float:
    """Fiedler value from a RivGraph links/nodes graph, with safe ID remapping.

    RivGraph node IDs are not guaranteed to be 0..n-1. We remap them here
    so the Fiedler routine sees a clean contiguous graph.
    """
    node_ids = list(nodes.get("id", []))
    if len(node_ids) == 0:
        node_ids = list(range(len(nodes.get("idx", []))))
    id_map = {int(nid): k for k, nid in enumerate(node_ids)}
    m = len(id_map)
    if m <= 1:
        return 0.0
    edges: set[tuple[int, int]] = set()
    for conn in links.get("conn", []):
        if conn is None or len(conn) != 2:
            continue
        a, b = int(conn[0]), int(conn[1])
        if a == b or a not in id_map or b not in id_map:
            continue
        i, j = id_map[a], id_map[b]
        if i == j:
            continue
        edges.add((i, j) if i < j else (j, i))
    return _fiedler_from_graph(m, sorted(edges), normalized=True)


def _fiedler_ipr_from_rivgraph(links: dict, nodes: dict) -> tuple[float, float]:
    node_ids = list(nodes.get("id", []))
    if len(node_ids) == 0:
        node_ids = list(range(len(nodes.get("idx", []))))
    id_map = {int(nid): k for k, nid in enumerate(node_ids)}
    m = len(id_map)
    if m <= 1:
        return 0.0, 0.0
    edges: set[tuple[int, int]] = set()
    for conn in links.get("conn", []):
        if conn is None or len(conn) != 2:
            continue
        a, b = int(conn[0]), int(conn[1])
        if a == b or a not in id_map or b not in id_map:
            continue
        i, j = id_map[a], id_map[b]
        if i == j:
            continue
        edges.add((i, j) if i < j else (j, i))
    return _fiedler_ipr_from_graph(m, sorted(edges), normalized=True)


def _mask_graph_edge_list(mask: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    """Convert foreground mask to an undirected 8-neighbor graph."""
    m = np.asarray(mask, dtype=bool)
    rows, cols = np.where(m)
    n = int(rows.size)
    if n == 0:
        return 0, []
    id_map = -np.ones(m.shape, dtype=np.int64)
    id_map[rows, cols] = np.arange(n, dtype=np.int64)

    neigh = [(0, 1), (1, 0), (1, 1), (1, -1)]
    nrows, ncols = m.shape
    edges: list[tuple[int, int]] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        i = int(id_map[r, c])
        for dr, dc in neigh:
            rr, cc = r + dr, c + dc
            if 0 <= rr < nrows and 0 <= cc < ncols and m[rr, cc]:
                j = int(id_map[rr, cc])
                if i != j:
                    a, b = (i, j) if i < j else (j, i)
                    edges.append((a, b))
    if len(edges) == 0:
        return n, []
    return n, sorted(set(edges))


_RIVGRAPH_CACHE: dict[str, object] = {}


def configure_rivgraph_import(rivgraph_repo: Path | None) -> None:
    """Optionally add RivGraph repo and _deps to sys.path."""
    if rivgraph_repo is None:
        return
    repo = Path(rivgraph_repo).resolve()
    for p in (repo, repo / "_deps"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _load_rivgraph_modules() -> tuple[object, object]:
    """Lazy-load RivGraph only when requested."""
    if "m2g" in _RIVGRAPH_CACHE and "lnu" in _RIVGRAPH_CACHE:
        return _RIVGRAPH_CACHE["m2g"], _RIVGRAPH_CACHE["lnu"]
    try:
        from rivgraph import mask_to_graph as m2g  # type: ignore
        from rivgraph import ln_utils as lnu  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "RivGraph extractor selected, but rivgraph import failed. "
            "Install/configure RivGraph or use --channel-extractor simple."
        ) from e
    _RIVGRAPH_CACHE["m2g"] = m2g
    _RIVGRAPH_CACHE["lnu"] = lnu
    return m2g, lnu


def _betti_from_edges(n_nodes: int, edge_list: list[tuple[int, int]]) -> tuple[int, int]:
    """(β₀, β₁) from an explicit (n_nodes, edges) graph description.

    Delegates to kernelcal.terrain.graph_codec.combinatorial_betti when
    available; otherwise uses the local union-find fallback.
    """
    if n_nodes <= 0:
        return 0, 0
    if _KT_AVAILABLE:
        edges_arr = (np.array(edge_list, dtype=np.int64).reshape(-1, 2)
                     if edge_list else np.empty((0, 2), dtype=np.int64))
        b0, b1 = _kt_combinatorial_betti(n_nodes, edges_arr)
        return int(b0), int(b1)
    b0 = _count_components_graph(n_nodes, edge_list)
    b1 = max(0, len(edge_list) - n_nodes + b0)
    return int(b0), int(b1)


def patch_betti_n(
    patch_dem: np.ndarray,
    resolution_m: float,
    percentile: float,
    extractor: str,
    rivgraph_prune_dangling: bool,
    *,
    mask_close_px: int = 0,
    mask_dilate_px: int = 0,
) -> tuple[int, int, int]:
    """Fast variant for scoring: returns (beta0, beta1, n_graph_nodes) only.

    Skips Fiedler / IPR eigenvector computation — those are expensive and
    are not used in the quadrant scoring formula.
    """
    smask = stream_mask_from_patch(
        patch_dem,
        resolution_m=resolution_m,
        percentile=percentile,
        close_iters=int(mask_close_px),
        dilate_iters=int(mask_dilate_px),
    )
    if extractor == "simple":
        n_nodes, edge_list = _mask_graph_edge_list(smask)
        b0, b1 = _betti_from_edges(n_nodes, edge_list)
        return b0, b1, int(n_nodes)

    # RivGraph path: mask → skeleton → links/nodes → graph Betti (no spectral).
    try:
        m2g, lnu = _load_rivgraph_modules()
        skel = m2g.skeletonize_mask(np.asarray(smask, dtype=bool))
        links, nodes = m2g.skel_to_graph(skel)
        if rivgraph_prune_dangling:
            dangling = [conn[0] for conn in nodes.get("conn", []) if len(conn) == 1]
            for lid in dangling:
                links, nodes = lnu.delete_link(links, nodes, lid)
        b0, b1 = _betti_from_rivgraph_links_nodes(links, nodes)
        n_nodes = int(len(nodes.get("idx", [])))
    except Exception:
        b0, b1, n_nodes = 0, 0, 0
    return int(b0), int(b1), n_nodes


def patch_channel_spectral_metrics(
    patch_dem: np.ndarray,
    resolution_m: float,
    percentile: float,
    extractor: str,
    rivgraph_prune_dangling: bool,
    *,
    mask_close_px: int = 0,
    mask_dilate_px: int = 0,
) -> tuple[int, int, float, float, float, int]:
    """Compute (beta0, beta1, stream_fraction, fiedler, ipr_v2, n_graph_nodes)."""
    smask = stream_mask_from_patch(
        patch_dem,
        resolution_m=resolution_m,
        percentile=percentile,
        close_iters=int(mask_close_px),
        dilate_iters=int(mask_dilate_px),
    )
    stream_frac = float(np.mean(smask))
    if extractor == "simple":
        # Graph-topology Betti (cycle rank, consistent with rivgraph extractor).
        n_nodes, edge_list = _mask_graph_edge_list(smask)
        beta0, beta1 = _betti_from_edges(n_nodes, edge_list)
        fiedler, ipr = _fiedler_ipr_from_graph(n_nodes, edge_list)
        return beta0, beta1, stream_frac, fiedler, ipr, int(n_nodes)

    # RivGraph path: mask -> skeleton -> links/nodes -> graph Betti.
    m2g, lnu = _load_rivgraph_modules()
    try:
        skel = m2g.skeletonize_mask(np.asarray(smask, dtype=bool))
        links, nodes = m2g.skel_to_graph(skel)
        if rivgraph_prune_dangling:
            dangling = []
            for conn in nodes.get("conn", []):
                if len(conn) == 1:
                    dangling.append(conn[0])
            for lid in dangling:
                links, nodes = lnu.delete_link(links, nodes, lid)
        beta0, beta1 = _betti_from_rivgraph_links_nodes(links, nodes)
        fiedler, ipr = _fiedler_ipr_from_rivgraph(links, nodes)
        n_nodes = int(len(nodes.get("idx", [])))
    except Exception:
        beta0, beta1, fiedler, ipr, n_nodes = 0, 0, 0.0, 0.0, 0
    return beta0, beta1, stream_frac, fiedler, ipr, n_nodes


def patch_channel_metrics(
    patch_dem: np.ndarray,
    resolution_m: float,
    percentile: float,
    extractor: str,
    rivgraph_prune_dangling: bool,
    *,
    mask_close_px: int = 0,
    mask_dilate_px: int = 0,
) -> tuple[int, int, float, float]:
    """Compute (beta0, beta1, stream_fraction, fiedler) from a DEM patch."""
    beta0, beta1, stream_frac, fiedler, _ipr, _n_nodes = patch_channel_spectral_metrics(
        patch_dem,
        resolution_m=resolution_m,
        percentile=percentile,
        extractor=extractor,
        rivgraph_prune_dangling=rivgraph_prune_dangling,
        mask_close_px=mask_close_px,
        mask_dilate_px=mask_dilate_px,
    )
    return beta0, beta1, stream_frac, fiedler


class _UnionFind:
    """Minimal union-find over integer labels."""

    def __init__(self, labels: np.ndarray) -> None:
        self.parent: dict[int, int] = {int(x): int(x) for x in labels.tolist()}

    def add(self, x: int) -> None:
        x = int(x)
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent.get(x, x) != x:
            p = self.parent[x]
            self.parent[x] = self.parent.get(p, p)
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _bridge_components(
    segments: list[np.ndarray],
    segment_cc: np.ndarray,
    node_cols: np.ndarray,
    node_rows: np.ndarray,
    node_cc: np.ndarray,
    bridge_px: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Merge distinct CCs whose nodes come within `bridge_px` pixels.

    Adds a straight-line bridging segment for each merged pair and rewrites
    segment_cc / node_cc to use canonical (union-find root) labels.
    """
    if bridge_px <= 0.0 or node_cc.size == 0:
        return segments, segment_cc.copy(), node_cc.copy()

    try:
        from scipy.spatial import cKDTree  # local import to keep top imports tight
    except Exception:
        return segments, segment_cc.copy(), node_cc.copy()

    pts = np.column_stack((np.asarray(node_cols, dtype=float), np.asarray(node_rows, dtype=float)))
    if pts.shape[0] < 2:
        return segments, segment_cc.copy(), node_cc.copy()

    uf = _UnionFind(np.unique(np.concatenate([node_cc, segment_cc])).astype(int))
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=float(bridge_px))

    bridging_segments: list[np.ndarray] = []
    bridging_labels: list[int] = []
    merged: set[tuple[int, int]] = set()
    for i, j in pairs:
        ci, cj = int(node_cc[i]), int(node_cc[j])
        if ci < 0 or cj < 0 or ci == cj:
            continue
        ri, rj = uf.find(ci), uf.find(cj)
        if ri == rj:
            continue
        key = (ri, rj) if ri < rj else (rj, ri)
        if key in merged:
            continue
        merged.add(key)
        uf.union(ci, cj)
        seg = np.array([[pts[i, 0], pts[i, 1]], [pts[j, 0], pts[j, 1]]], dtype=float)
        bridging_segments.append(seg)
        bridging_labels.append(uf.find(ci))

    new_node_cc = np.array([uf.find(int(x)) for x in node_cc.tolist()], dtype=int)
    new_seg_cc = np.array([uf.find(int(x)) for x in segment_cc.tolist()], dtype=int)
    if bridging_segments:
        segments = list(segments) + bridging_segments
        new_seg_cc = np.concatenate([new_seg_cc, np.asarray(bridging_labels, dtype=int)])
    return segments, new_seg_cc, new_node_cc


def _mask_graph_segments(
    mask: np.ndarray,
    *,
    max_edges: int = 4000,
    max_nodes: int = 2500,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a lightweight pixel-graph visualization from a binary mask.

    Returns:
      - segments (flat list of XY line segments),
      - segment_cc (connected-component label per segment),
      - node_rows,
      - node_cols,
      - node_cc (connected-component label per node).
    """
    m = np.asarray(mask, dtype=bool)
    rows, cols = np.where(m)
    if rows.size == 0:
        return [], np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=int)

    # 8-connected CC labels of the foreground mask.
    cc_struct = np.ones((3, 3), dtype=np.int8)
    cc_labels, _ = ndimage.label(m, structure=cc_struct)

    if rows.size > max_nodes:
        sel = np.linspace(0, rows.size - 1, max_nodes, dtype=int)
        rows_show = rows[sel].astype(float)
        cols_show = cols[sel].astype(float)
        node_cc = cc_labels[rows[sel], cols[sel]].astype(int)
    else:
        rows_show = rows.astype(float)
        cols_show = cols.astype(float)
        node_cc = cc_labels[rows, cols].astype(int)

    segments: list[np.ndarray] = []
    segment_cc: list[int] = []
    neigh = [(0, 1), (1, 0), (1, 1), (1, -1)]
    nrows, ncols = m.shape
    for r, c in zip(rows.tolist(), cols.tolist()):
        lab = int(cc_labels[r, c])
        for dr, dc in neigh:
            rr, cc = r + dr, c + dc
            if 0 <= rr < nrows and 0 <= cc < ncols and m[rr, cc]:
                seg = np.array([[float(c), float(r)], [float(cc), float(rr)]], dtype=float)
                segments.append(seg)
                segment_cc.append(lab)
                if len(segments) >= max_edges:
                    return segments, np.asarray(segment_cc, dtype=int), rows_show, cols_show, node_cc
    return segments, np.asarray(segment_cc, dtype=int), rows_show, cols_show, node_cc


def patch_channel_observation(
    patch_dem: np.ndarray,
    resolution_m: float,
    percentile: float,
    extractor: str,
    rivgraph_prune_dangling: bool,
    *,
    mask_close_px: int = 0,
    mask_dilate_px: int = 0,
    bridge_endpoints_px: float = 0.0,
) -> dict[str, object]:
    """Per-patch channel extraction outputs for visualization and scoring."""
    smask = stream_mask_from_patch(
        patch_dem,
        resolution_m=resolution_m,
        percentile=percentile,
        close_iters=int(mask_close_px),
        dilate_iters=int(mask_dilate_px),
    )
    stream_frac = float(np.mean(smask))
    if extractor == "rivgraph":
        try:
            m2g, lnu = _load_rivgraph_modules()
            skel = m2g.skeletonize_mask(np.asarray(smask, dtype=bool))
            links, nodes = m2g.skel_to_graph(skel)
            if rivgraph_prune_dangling:
                dangling = []
                for conn in nodes.get("conn", []):
                    if len(conn) == 1:
                        dangling.append(conn[0])
                for lid in dangling:
                    links, nodes = lnu.delete_link(links, nodes, lid)
            beta0, beta1 = _betti_from_rivgraph_links_nodes(links, nodes)
            fiedler = _fiedler_from_rivgraph(links, nodes)

            node_ids = list(nodes.get("id", []))
            if len(node_ids) == 0:
                node_ids = list(range(len(nodes.get("idx", []))))
            id_map = {int(nid): k for k, nid in enumerate(node_ids)}
            n_nodes_g = len(id_map)

            # Adjacency for CC labeling.
            rr_adj: list[int] = []
            cc_adj: list[int] = []
            for conn in links.get("conn", []):
                if conn is None or len(conn) != 2:
                    continue
                a, b = int(conn[0]), int(conn[1])
                if a == b or a not in id_map or b not in id_map:
                    continue
                i, j = id_map[a], id_map[b]
                rr_adj.extend([i, j])
                cc_adj.extend([j, i])
            if n_nodes_g > 0:
                if rr_adj:
                    adj = sparse.coo_matrix(
                        (np.ones(len(rr_adj), dtype=float), (rr_adj, cc_adj)),
                        shape=(n_nodes_g, n_nodes_g),
                    ).tocsr()
                    _, cc_labels_nodes = csgraph.connected_components(adj, directed=False)
                else:
                    cc_labels_nodes = np.arange(n_nodes_g, dtype=int)
            else:
                cc_labels_nodes = np.array([], dtype=int)

            segments: list[np.ndarray] = []
            segment_cc: list[int] = []
            for conn, lidcs in zip(links.get("conn", []), links.get("idx", [])):
                pix = np.asarray(list(lidcs), dtype=np.int64)
                if pix.size < 2:
                    continue
                rc = np.unravel_index(pix, smask.shape)
                seg = np.column_stack((rc[1].astype(float), rc[0].astype(float)))
                lab = -1
                if conn is not None and len(conn) >= 1:
                    a = int(conn[0])
                    if a in id_map:
                        lab = int(cc_labels_nodes[id_map[a]])
                segments.append(seg)
                segment_cc.append(lab)

            nrows, ncols = smask.shape
            node_r = np.zeros(len(nodes.get("idx", [])), dtype=float)
            node_c = np.zeros(len(nodes.get("idx", [])), dtype=float)
            node_cc = np.full(len(nodes.get("idx", [])), -1, dtype=int)
            for i, pix in enumerate(nodes.get("idx", [])):
                r, c = np.unravel_index(int(pix), (nrows, ncols))
                node_r[i] = float(r)
                node_c[i] = float(c)
                nid = int(node_ids[i]) if i < len(node_ids) else i
                if nid in id_map:
                    node_cc[i] = int(cc_labels_nodes[id_map[nid]])

            seg_cc_arr = np.asarray(segment_cc, dtype=int)
            if float(bridge_endpoints_px) > 0.0:
                segments, seg_cc_arr, node_cc = _bridge_components(
                    segments, seg_cc_arr, node_c, node_r, node_cc, float(bridge_endpoints_px)
                )
            return {
                "mask": smask,
                "beta0": int(beta0),
                "beta1": int(beta1),
                "fiedler": float(fiedler),
                "stream_fraction": stream_frac,
                "segments": segments,
                "segment_cc": seg_cc_arr,
                "node_rows": node_r,
                "node_cols": node_c,
                "node_cc": node_cc,
            }
        except Exception:
            # Robust fallback if RivGraph fails on a given small patch.
            pass

    # Graph-topology Betti (consistent with rivgraph branch above and scoring).
    n_nodes, edge_list = _mask_graph_edge_list(smask)
    beta0, beta1 = _betti_from_edges(n_nodes, edge_list)
    fiedler = _fiedler_from_graph(n_nodes, edge_list)
    segments, segment_cc, node_r, node_c, node_cc = _mask_graph_segments(smask)
    if float(bridge_endpoints_px) > 0.0:
        segments, segment_cc, node_cc = _bridge_components(
            segments, segment_cc, node_c, node_r, node_cc, float(bridge_endpoints_px)
        )
    return {
        "mask": smask,
        "beta0": int(beta0),
        "beta1": int(beta1),
        "fiedler": float(fiedler),
        "stream_fraction": stream_frac,
        "segments": segments,
        "segment_cc": segment_cc,
        "node_rows": node_r,
        "node_cols": node_c,
        "node_cc": node_cc,
    }


def clip_center(row: int, col: int, half_side: int, shape: tuple[int, int]) -> tuple[int, int]:
    """Keep center inside valid square-capture bounds."""
    nrows, ncols = shape
    r = int(np.clip(row, half_side, nrows - half_side - 1))
    c = int(np.clip(col, half_side, ncols - half_side - 1))
    return r, c


def capture_square(dem: np.ndarray, row: int, col: int, side_px: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return square patch and bounds (r0, r1, c0, c1)."""
    half = side_px // 2
    row, col = clip_center(row, col, half, dem.shape)
    r0 = row - half
    r1 = r0 + side_px
    c0 = col - half
    c1 = c0 + side_px
    return dem[r0:r1, c0:c1], (r0, r1, c0, c1)


# Diagonal quadrant definitions: (name, row_sign, col_sign)
# Row grows south (+), col grows east (+).  Re-exported from the shared
# ``kernelcal.graph_explorer`` subpackage so the bishop-rocks explorer can
# use the same quadrant convention.
_QUAD_DIRS: tuple[tuple[str, int, int], ...] = tuple(
    (name, dr, dc) for name, (dr, dc) in QUADRANT_OFFSETS_IMAGE.items()
)



def choose_next_location(
    dem: np.ndarray,
    visited_mask: np.ndarray,
    current: tuple[int, int],
    records: list[CaptureRecord],
    side_px: int,
    resolution_m: float,
    stream_percentile: float,
    channel_extractor: str,
    rivgraph_prune_dangling: bool,
    w_beta1: float,
    w_beta0: float,
    w_unseen: float,
    revisit_penalty: float,
    min_valid_fraction: float,
    mask_close_px: int = 0,
    mask_dilate_px: int = 0,
) -> tuple[tuple[int, int], float, float, int, int]:
    """Score four diagonal quadrants and return the best next waypoint.

    The current FoV patch is divided into four diagonal quadrants (NW/NE/SW/SE).
    Each quadrant's Betti topology is scored, and the drone targets the outer
    corner of the winning quadrant (= outer edge of that quadrant relative to
    the current footprint).

    Scoring, revisit penalty, and cyclic tie-break are delegated to
    :func:`kernelcal.graph_explorer.choose_best_candidate` so the bishop-rocks
    explorer can use the exact same policy.  This function only owns the
    DEM-specific bit: extracting each quadrant's sub-graph to compute
    ``(β₀, β₁, n_nodes, unseen_frac)``.

    Score = w_beta1*clip(β₁/n) − w_beta0*clip(β₀/n) + w_unseen*unseen

    Returns (center, best_score, unseen_frac, beta0, beta1).
    """
    # Outer-edge step: half the footprint in each diagonal direction.
    half = max(2, side_px // 2)

    # Capture current patch and split into 4 diagonal sub-patches.
    cur_patch, _ = capture_square(dem, current[0], current[1], side_px=side_px)
    h, w = cur_patch.shape
    hr, wc = h // 2, w // 2
    quad_slices = {
        "NW": (slice(0, hr),  slice(0, wc)),
        "NE": (slice(0, hr),  slice(wc, w)),
        "SW": (slice(hr, h),  slice(0, wc)),
        "SE": (slice(hr, h),  slice(wc, w)),
    }

    candidates: list[Candidate] = []
    for name, dr, dc in _QUAD_DIRS:
        sub = cur_patch[quad_slices[name]]

        # Skip quadrants with too much missing data.
        if float(np.mean(np.isfinite(sub))) < min_valid_fraction:
            continue

        # Betti numbers on the sub-patch (fast path: no Fiedler/IPR).
        b0, b1, n_nodes = patch_betti_n(
            sub, resolution_m=resolution_m, percentile=stream_percentile,
            extractor=channel_extractor, rivgraph_prune_dangling=rivgraph_prune_dangling,
            mask_close_px=int(mask_close_px), mask_dilate_px=int(mask_dilate_px),
        )

        # Target = outer corner of this quadrant.
        target_r = int(np.clip(current[0] + dr * half, half, dem.shape[0] - half - 1))
        target_c = int(np.clip(current[1] + dc * half, half, dem.shape[1] - half - 1))

        # Unseen fraction at the target position.
        _, (tr0, tr1, tc0, tc1) = capture_square(dem, target_r, target_c, side_px=side_px)
        unseen = 1.0 - float(np.mean(visited_mask[tr0:tr1, tc0:tc1]))

        candidates.append(Candidate(
            name=name,
            position=(target_r, target_c),
            beta0=int(b0),
            beta1=int(b1),
            n_nodes=int(n_nodes),
            unseen_frac=float(unseen),
        ))

    if not candidates:
        # cur_patch already captured at function entry; reuse via patch_channel_metrics.
        b0, b1, _, _ = patch_channel_metrics(
            cur_patch, resolution_m=resolution_m, percentile=stream_percentile,
            extractor=channel_extractor, rivgraph_prune_dangling=rivgraph_prune_dangling,
        )
        return current, 0.0, 0.0, b0, b1

    weights = BettiWeights(
        w_beta1=float(w_beta1),
        w_beta0=float(w_beta0),
        w_unseen=float(w_unseen),
        revisit_penalty=float(revisit_penalty),
    )
    recent_positions = [(r.row, r.col) for r in records[-20:]] if records else []
    best, best_score, _scored = choose_best_candidate(
        candidates,
        weights,
        recent_positions=recent_positions,
        tie_break_index=len(records),
    )
    assert best is not None  # candidates is non-empty
    return (
        (int(best.position[0]), int(best.position[1])),
        float(best_score),
        float(best.unseen_frac),
        int(best.beta0),
        int(best.beta1),
    )



def nearest_finite_cell(dem: np.ndarray) -> tuple[int, int]:
    """Find a start location near center that has finite elevation."""
    nrows, ncols = dem.shape
    cr, cc = nrows // 2, ncols // 2
    finite = np.isfinite(dem)
    if finite[cr, cc]:
        return cr, cc
    ys, xs = np.where(finite)
    if len(ys) == 0:
        return cr, cc
    d2 = (ys - cr) ** 2 + (xs - cc) ** 2
    i = int(np.argmin(d2))
    return int(ys[i]), int(xs[i])


def run_experiment(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[CaptureRecord], int]:
    if args.dem_npy:
        dem = np.load(args.dem_npy)
        if dem.ndim != 2:
            raise ValueError("--dem-npy must contain a 2D array.")
    elif args.dem_tiff:
        tiff_path = args.dem_tiff
        if args.bbox_lonlat is not None:
            bbox = parse_bbox_lonlat(args.bbox_lonlat)
            crop_name = args.bbox_crop_name or "bbox_crop.tif"
            crop_path = args.output_dir.resolve() / crop_name
            tiff_path = crop_geotiff_by_bbox(args.dem_tiff, bbox, crop_path)
            print(
                "[info] Cropped DEM from bbox "
                f"(lon_min={bbox[0]}, lat_min={bbox[1]}, lon_max={bbox[2]}, lat_max={bbox[3]}) "
                f"-> {tiff_path}"
            )
        dem = load_tiff_dem(
            tiff_path,
            nodata_value=args.nodata_value,
            window_spec=args.tiff_window,
            max_cells=args.max_dem_cells,
        )
    else:
        dem = synthetic_dem(args.dem_rows, args.dem_cols, seed=args.seed)

    return run_experiment_on_dem(dem, args)


def run_experiment_on_dem(
    dem: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[CaptureRecord], int]:
    cam = CameraModel(
        altitude_m=args.altitude_m,
        fov_deg=args.fov_deg,
        resolution_m=args.dem_resolution_m,
    )
    side_px = cam.footprint_side_px
    mission_params_line = (
        f"mission: alt={args.altitude_m:g}m | fov={args.fov_deg:g}deg | "
        f"wβ1={args.w_beta1:g} | wβ0={args.w_beta0:g}"
    )
    visited = np.zeros_like(dem, dtype=bool)
    finite_frac = float(np.mean(np.isfinite(dem)))
    if args.channel_extractor == "rivgraph":
        configure_rivgraph_import(args.rivgraph_repo)
    print(
        f"[info] DEM shape={dem.shape}, finite_fraction={finite_frac:.3f}, "
        f"footprint_px={side_px}, extractor={args.channel_extractor}, "
        f"planner=quadrant-betti"
    )
    if side_px < 24:
        print(
            "[warn] Small footprint (side < 24 px). For regional DEMs this often yields sparse topology. "
            "Increase altitude or FOV for richer Betti signals."
        )

    center = nearest_finite_cell(dem)
    records: list[CaptureRecord] = []
    for step in range(args.steps):
        patch, (r0, r1, c0, c1) = capture_square(dem, center[0], center[1], side_px=side_px)
        unseen_frac = 1.0 - float(np.mean(visited[r0:r1, c0:c1]))
        visited[r0:r1, c0:c1] = True

        obs = patch_channel_observation(
            patch,
            resolution_m=args.dem_resolution_m,
            percentile=args.stream_percentile,
            extractor=args.channel_extractor,
            rivgraph_prune_dangling=args.rivgraph_prune_dangling,
            mask_close_px=int(getattr(args, "mask_close_px", 0)),
            mask_dilate_px=int(getattr(args, "mask_dilate_px", 0)),
            bridge_endpoints_px=float(getattr(args, "bridge_endpoints_px", 0.0)),
        )
        beta0 = int(obs["beta0"])
        beta1 = int(obs["beta1"])
        fiedler = float(obs.get("fiedler", 0.0))
        stream_frac = float(obs["stream_fraction"])

        rec = CaptureRecord(
            step=step,
            row=int((r0 + r1) // 2),
            col=int((c0 + c1) // 2),
            beta0=beta0,
            beta1=beta1,
            fiedler=fiedler,
            stream_fraction=stream_frac,
            unseen_fraction=unseen_frac,
            score=0.0,
        )
        records.append(rec)

        if step < args.steps - 1:
            center, score, _, _, _ = choose_next_location(
                dem=dem,
                visited_mask=visited,
                current=center,
                records=records,
                side_px=side_px,
                resolution_m=args.dem_resolution_m,
                stream_percentile=args.stream_percentile,
                channel_extractor=args.channel_extractor,
                rivgraph_prune_dangling=args.rivgraph_prune_dangling,
                w_beta1=args.w_beta1,
                w_beta0=args.w_beta0,
                w_unseen=args.w_unseen,
                revisit_penalty=args.revisit_penalty,
                min_valid_fraction=args.min_valid_fraction,
                mask_close_px=int(getattr(args, "mask_close_px", 0)),
                mask_dilate_px=int(getattr(args, "mask_dilate_px", 0)),
            )
            rec.score = float(score)
        else:
            # Last step has no next-move score.
            rec.score = 0.0
    return dem, visited, records, side_px


def run_experiment_realtime(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[CaptureRecord], int]:
    """Run simulation with live matplotlib updates each step."""
    # Reuse the same DEM-loading path as batch mode.
    if args.dem_npy:
        dem = np.load(args.dem_npy)
        if dem.ndim != 2:
            raise ValueError("--dem-npy must contain a 2D array.")
    elif args.dem_tiff:
        tiff_path = args.dem_tiff
        if args.bbox_lonlat is not None:
            bbox = parse_bbox_lonlat(args.bbox_lonlat)
            crop_name = args.bbox_crop_name or "bbox_crop.tif"
            crop_path = args.output_dir.resolve() / crop_name
            tiff_path = crop_geotiff_by_bbox(args.dem_tiff, bbox, crop_path)
            print(
                "[info] Cropped DEM from bbox "
                f"(lon_min={bbox[0]}, lat_min={bbox[1]}, lon_max={bbox[2]}, lat_max={bbox[3]}) "
                f"-> {tiff_path}"
            )
        dem = load_tiff_dem(
            tiff_path,
            nodata_value=args.nodata_value,
            window_spec=args.tiff_window,
            max_cells=args.max_dem_cells,
        )
    else:
        dem = synthetic_dem(args.dem_rows, args.dem_cols, seed=args.seed)

    cam = CameraModel(
        altitude_m=args.altitude_m,
        fov_deg=args.fov_deg,
        resolution_m=args.dem_resolution_m,
    )
    side_px = cam.footprint_side_px
    mission_params_line = (
        f"mission: alt={args.altitude_m:g}m | fov={args.fov_deg:g}deg | "
        f"wβ1={args.w_beta1:g} | wβ0={args.w_beta0:g}"
    )
    visited = np.zeros_like(dem, dtype=bool)
    finite_frac = float(np.mean(np.isfinite(dem)))
    if args.channel_extractor == "rivgraph":
        configure_rivgraph_import(args.rivgraph_repo)
    print(
        f"[info] DEM shape={dem.shape}, finite_fraction={finite_frac:.3f}, "
        f"footprint_px={side_px}, extractor={args.channel_extractor}, "
        f"planner=quadrant-betti"
    )
    if side_px < 24:
        print(
            "[warn] Small footprint (side < 24 px). For regional DEMs this often yields sparse topology. "
            "Increase altitude or FOV for richer Betti signals."
        )

    # Live figure state
    plt.ion()
    fig, ax = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    ax00, ax01, ax02 = ax[0, 0], ax[0, 1], ax[0, 2]
    ax10, ax11, ax12 = ax[1, 0], ax[1, 1], ax[1, 2]
    mdem = np.ma.masked_invalid(dem)
    # Lock the elevation colormap limits to the full DEM range so the FoV
    # panel (ax02) uses the *same* colors as the global DEM panel (ax00).
    finite_dem = dem[np.isfinite(dem)]
    if finite_dem.size:
        dem_vmin = float(np.nanpercentile(finite_dem, 1.0))
        dem_vmax = float(np.nanpercentile(finite_dem, 99.0))
        if dem_vmax <= dem_vmin:
            dem_vmin, dem_vmax = float(finite_dem.min()), float(finite_dem.max()) + 1e-6
    else:
        dem_vmin, dem_vmax = 0.0, 1.0
    im0 = ax00.imshow(mdem, cmap="terrain", origin="upper",
                       vmin=dem_vmin, vmax=dem_vmax)
    fig.colorbar(im0, ax=ax00, shrink=0.75, label="Elevation")
    ax00.set_title("DEM with live drone path")
    ax00.set_axis_off()
    path_line, = ax00.plot([], [], "-o", color="black", ms=3, lw=1, alpha=0.9)
    fov_rect = Rectangle((0, 0), side_px, side_px, linewidth=1.2, edgecolor="white", facecolor="none")
    ax00.add_patch(fov_rect)

    cov_img = ax01.imshow(visited.astype(float), cmap="viridis", origin="upper", vmin=0.0, vmax=1.0)
    ax01.set_title("Coverage + exploration graph")
    ax01.set_axis_off()
    lc_temporal = LineCollection([], colors="white", linewidths=1.1, alpha=0.9)
    lc_prox = LineCollection([], colors="cyan", linewidths=0.8, alpha=0.55)
    ax01.add_collection(lc_prox)
    ax01.add_collection(lc_temporal)
    node_scatter = ax01.scatter([], [], c=[], cmap="plasma", s=12, edgecolors="none")
    fig.colorbar(node_scatter, ax=ax01, shrink=0.75, label="Node beta1")
    ax01.legend(
        handles=[
            Line2D([0], [0], color="white", lw=1.2, label="temporal edge"),
            Line2D([0], [0], color="cyan", lw=1.0, label="proximity edge"),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.7,
    )

    # Local FOV DEM panel — share elevation colormap limits with ax00 so the
    # per-patch colors match the global DEM rather than auto-scaling each frame.
    local_dem_placeholder = np.ma.masked_all((side_px, side_px), dtype=float)
    local_dem_img = ax02.imshow(local_dem_placeholder, cmap="terrain",
                                 origin="upper", vmin=dem_vmin, vmax=dem_vmax)
    fig.colorbar(local_dem_img, ax=ax02, shrink=0.75, label="Elevation")
    lc_dem_graph = LineCollection([], linewidths=1.0, alpha=0.9)
    ax02.add_collection(lc_dem_graph)
    dem_nodes = ax02.scatter([], [], s=9, edgecolors="none")
    ax02.set_title("DEM inside current FOV")
    ax02.set_axis_off()
    ax02.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", lw=1.0, label="edge (colored by component)"),
            Line2D([0], [0], marker="o", color="tab:blue", lw=0, markersize=5, label="node (colored by component)"),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.7,
    )
    # Local extracted mask + local graph panel
    local_mask_img = ax12.imshow(np.zeros((side_px, side_px), dtype=float), cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
    lc_local_graph = LineCollection([], linewidths=1.0, alpha=0.9)
    ax12.add_collection(lc_local_graph)
    local_nodes = ax12.scatter([], [], s=8, edgecolors="none")
    ax12.set_title("Extracted stream mask + local graph")
    ax12.set_axis_off()
    ax12.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", lw=1.0, label="edge (colored by component)"),
            Line2D([0], [0], marker="o", color="tab:blue", lw=0, markersize=5, label="node (colored by component)"),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.7,
    )

    ax10.set_title("Topology history per capture")
    ax10.set_xlabel("Step")
    ax10.set_ylabel("Topology metric")
    ax10.grid(alpha=0.25)
    beta0_line, = ax10.plot([], [], label="beta0", color="tab:blue")
    beta1_line, = ax10.plot([], [], label="beta1", color="tab:red")
    fiedler_line, = ax10.plot([], [], label="fiedler", color="tab:brown")
    ax10.legend()

    ax11.set_title("Adaptive mapping diagnostics")
    ax11.set_xlabel("Step")
    ax11.grid(alpha=0.25)
    score_line, = ax11.plot([], [], label="adaptive score", color="tab:purple")
    unseen_line, = ax11.plot([], [], label="unseen fraction", color="tab:green")
    stream_line, = ax11.plot([], [], label="stream fraction", color="tab:orange")
    ax11.legend()

    center = nearest_finite_cell(dem)
    records: list[CaptureRecord] = []
    xs: list[float] = []
    ys: list[float] = []
    beta0_hist: list[float] = []
    beta1_hist: list[float] = []
    fiedler_hist: list[float] = []
    score_hist: list[float] = []
    unseen_hist: list[float] = []
    stream_hist: list[float] = []

    for step in range(args.steps):
        patch, (r0, r1, c0, c1) = capture_square(dem, center[0], center[1], side_px=side_px)
        unseen_frac = 1.0 - float(np.mean(visited[r0:r1, c0:c1]))
        visited[r0:r1, c0:c1] = True

        obs = patch_channel_observation(
            patch,
            resolution_m=args.dem_resolution_m,
            percentile=args.stream_percentile,
            extractor=args.channel_extractor,
            rivgraph_prune_dangling=args.rivgraph_prune_dangling,
            mask_close_px=int(getattr(args, "mask_close_px", 0)),
            mask_dilate_px=int(getattr(args, "mask_dilate_px", 0)),
            bridge_endpoints_px=float(getattr(args, "bridge_endpoints_px", 0.0)),
        )
        beta0 = int(obs["beta0"])
        beta1 = int(obs["beta1"])
        fiedler = float(obs.get("fiedler", 0.0))
        stream_frac = float(obs["stream_fraction"])

        rec = CaptureRecord(
            step=step,
            row=int((r0 + r1) // 2),
            col=int((c0 + c1) // 2),
            beta0=beta0,
            beta1=beta1,
            fiedler=fiedler,
            stream_fraction=stream_frac,
            unseen_fraction=unseen_frac,
            score=0.0,
        )
        records.append(rec)

        if step < args.steps - 1:
            center, score, _, _, _ = choose_next_location(
                dem=dem,
                visited_mask=visited,
                current=center,
                records=records,
                side_px=side_px,
                resolution_m=args.dem_resolution_m,
                stream_percentile=args.stream_percentile,
                channel_extractor=args.channel_extractor,
                rivgraph_prune_dangling=args.rivgraph_prune_dangling,
                w_beta1=args.w_beta1,
                w_beta0=args.w_beta0,
                w_unseen=args.w_unseen,
                revisit_penalty=args.revisit_penalty,
                min_valid_fraction=args.min_valid_fraction,
                mask_close_px=int(getattr(args, "mask_close_px", 0)),
                mask_dilate_px=int(getattr(args, "mask_dilate_px", 0)),
            )
            rec.score = float(score)
        else:
            # Last step has no next-move score.
            rec.score = 0.0

        # Live update
        xs.append(rec.col)
        ys.append(rec.row)
        beta0_hist.append(rec.beta0)
        beta1_hist.append(rec.beta1)
        fiedler_hist.append(rec.fiedler)
        score_hist.append(rec.score)
        unseen_hist.append(rec.unseen_fraction)
        stream_hist.append(rec.stream_fraction)

        path_line.set_data(xs, ys)
        fov_rect.set_xy((c0, r0))
        fov_rect.set_width(c1 - c0)
        fov_rect.set_height(r1 - r0)
        cov_img.set_data(visited.astype(float))
        temporal, proximity = build_exploration_graph_edges(
            records, radius_px=float(args.graph_radius_px), k_nearest=int(args.graph_k_nearest)
        )
        lc_temporal.set_segments(edges_to_segments(records, temporal))
        lc_prox.set_segments(edges_to_segments(records, proximity))
        nxs = np.array([r.col for r in records], dtype=float)
        nys = np.array([r.row for r in records], dtype=float)
        nb1 = np.array([r.beta1 for r in records], dtype=float)
        node_scatter.set_offsets(np.column_stack((nxs, nys)))
        node_scatter.set_array(nb1)
        ax01.set_title(
            f"Coverage + graph (visited={np.mean(visited):.5f}, "
            f"nodes={len(records)}, edges={len(temporal)+len(proximity)})"
        )

        # Update local FOV DEM and local extracted channel graph.
        local_dem = np.ma.masked_invalid(patch)
        local_dem_img.set_data(local_dem)
        mask = np.asarray(obs["mask"], dtype=bool)
        local_mask_img.set_data(mask.astype(float))
        segs_all = obs.get("segments", []) or []
        seg_cc = np.asarray(obs.get("segment_cc", np.array([], dtype=int)), dtype=int)
        edge_rgba = _cc_colors_for_labels(seg_cc) if len(segs_all) > 0 else np.empty((0, 4))
        lc_dem_graph.set_segments(segs_all)
        lc_local_graph.set_segments(segs_all)
        if edge_rgba.size > 0:
            lc_dem_graph.set_colors(edge_rgba)
            lc_local_graph.set_colors(edge_rgba)

        nr = np.asarray(obs["node_rows"], dtype=float)
        nc = np.asarray(obs["node_cols"], dtype=float)
        ncc = np.asarray(obs.get("node_cc", np.array([], dtype=int)), dtype=int)
        if nr.size > 0:
            node_rgba = _cc_colors_for_labels(ncc) if ncc.size == nr.size else None
            dem_nodes.set_offsets(np.column_stack((nc, nr)))
            local_nodes.set_offsets(np.column_stack((nc, nr)))
            if node_rgba is not None:
                dem_nodes.set_facecolors(node_rgba)
                local_nodes.set_facecolors(node_rgba)
        else:
            dem_nodes.set_offsets(np.empty((0, 2)))
            local_nodes.set_offsets(np.empty((0, 2)))
        ax02.set_title(
            f"DEM in FOV ({patch.shape[0]}x{patch.shape[1]})"
        )
        n_components_local = int(len(set(int(x) for x in ncc.tolist()))) if ncc.size > 0 else 0
        ax12.set_title(
            f"Mask+graph in FOV (beta0={beta0}, beta1={beta1}, fiedler={fiedler:.3f}, "
            f"components={n_components_local})"
        )

        x_axis = np.arange(len(records))
        beta0_line.set_data(x_axis, beta0_hist)
        beta1_line.set_data(x_axis, beta1_hist)
        fiedler_line.set_data(x_axis, fiedler_hist)
        score_line.set_data(x_axis, score_hist)
        unseen_line.set_data(x_axis, unseen_hist)
        stream_line.set_data(x_axis, stream_hist)

        ax10.set_xlim(0, max(1, args.steps - 1))
        ax10.set_ylim(-0.1, max(1.0, max(beta0_hist + beta1_hist + fiedler_hist) + 0.5))
        ax11.set_xlim(0, max(1, args.steps - 1))
        ax11.set_ylim(-0.05, max(1.0, max(score_hist + unseen_hist + stream_hist) + 0.1))

        fig.suptitle(
            f"Realtime adaptive flight step {step + 1}/{args.steps}\n{mission_params_line}",
            fontsize=11,
        )
        fig.canvas.draw_idle()
        plt.pause(max(0.001, float(args.realtime_pause_s)))

    if args.realtime_block:
        print("[info] Realtime window complete. Close plot window to continue.")
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)
    return dem, visited, records, side_px


def save_plot(dem: np.ndarray, visited: np.ndarray, records: list[CaptureRecord], side_px: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    ax00 = ax[0, 0]
    mdem = np.ma.masked_invalid(dem)
    im0 = ax00.imshow(mdem, cmap="terrain", origin="upper")
    rr = np.array([r.row for r in records], dtype=float)
    cc = np.array([r.col for r in records], dtype=float)
    ax00.plot(cc, rr, "-o", color="black", ms=3, lw=1, alpha=0.9)
    draw_n = min(8, len(records))
    for rec in records[-draw_n:]:
        rect = Rectangle(
            (rec.col - side_px // 2, rec.row - side_px // 2),
            side_px,
            side_px,
            linewidth=1.0,
            edgecolor="white",
            facecolor="none",
            alpha=0.7,
        )
        ax00.add_patch(rect)
    ax00.set_title("DEM with drone path and recent FOV squares")
    ax00.set_axis_off()
    fig.colorbar(im0, ax=ax00, shrink=0.75, label="Elevation")

    ax01 = ax[0, 1]
    ax01.imshow(visited.astype(float), cmap="viridis", origin="upper", vmin=0.0, vmax=1.0)
    temporal, proximity = build_exploration_graph_edges(
        records, radius_px=160.0, k_nearest=2
    )
    seg_t = edges_to_segments(records, temporal)
    seg_p = edges_to_segments(records, proximity)
    if seg_p:
        ax01.add_collection(LineCollection(seg_p, colors="cyan", linewidths=0.8, alpha=0.55))
    if seg_t:
        ax01.add_collection(LineCollection(seg_t, colors="white", linewidths=1.2, alpha=0.9))
    node_x = np.array([r.col for r in records], dtype=float)
    node_y = np.array([r.row for r in records], dtype=float)
    node_beta1 = np.array([r.beta1 for r in records], dtype=float)
    sc_graph = ax01.scatter(node_x, node_y, c=node_beta1, cmap="plasma", s=12, edgecolors="none")
    fig.colorbar(sc_graph, ax=ax01, shrink=0.75, label="Node beta1")
    ax01.set_title(
        f"Coverage + exploration graph (visited={np.mean(visited):.5f}, "
        f"nodes={len(records)}, edges={len(temporal)+len(proximity)})"
    )
    ax01.set_axis_off()

    ax10 = ax[1, 0]
    beta0 = np.array([r.beta0 for r in records], dtype=float)
    beta1 = np.array([r.beta1 for r in records], dtype=float)
    fiedler = np.array([r.fiedler for r in records], dtype=float)
    ax10.plot(beta0, label="beta0", color="tab:blue")
    ax10.plot(beta1, label="beta1", color="tab:red")
    ax10.plot(fiedler, label="fiedler", color="tab:brown")
    ax10.set_title("Topology history per capture")
    ax10.set_xlabel("Step")
    ax10.set_ylabel("Topology metric")
    ax10.grid(alpha=0.25)
    ax10.legend()

    ax11 = ax[1, 1]
    score = np.array([r.score for r in records], dtype=float)
    unseen = np.array([r.unseen_fraction for r in records], dtype=float)
    stream_frac = np.array([r.stream_fraction for r in records], dtype=float)
    ax11.plot(score, label="adaptive score", color="tab:purple")
    ax11.plot(unseen, label="unseen fraction", color="tab:green")
    ax11.plot(stream_frac, label="stream fraction", color="tab:orange")
    ax11.set_title("Adaptive mapping diagnostics")
    ax11.set_xlabel("Step")
    ax11.grid(alpha=0.25)
    ax11.legend()

    out_path = out_dir / "drone_betti_adaptive_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_animation(
    dem: np.ndarray,
    records: list[CaptureRecord],
    side_px: int,
    out_dir: Path,
    fps: int,
    resolution_m: float,
    stream_percentile: float,
    channel_extractor: str,
    rivgraph_prune_dangling: bool,
    altitude_m: float,
    fov_deg: float,
    w_beta1: float,
    w_beta0: float,
    mask_close_px: int = 0,
    mask_dilate_px: int = 0,
    bridge_endpoints_px: float = 0.0,
) -> Path:
    """Save adaptive exploration as a matplotlib animation (GIF preferred)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(records) == 0:
        raise ValueError("No records to animate.")

    mdem = np.ma.masked_invalid(dem)
    nrows, ncols = dem.shape
    visited_prog = np.zeros((nrows, ncols), dtype=bool)
    half = side_px // 2
    mission_params_line = (
        f"mission: alt={altitude_m:g}m | fov={fov_deg:g}deg | "
        f"wβ1={w_beta1:g} | wβ0={w_beta0:g}"
    )

    fig, ax = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    ax00, ax01, ax02 = ax[0, 0], ax[0, 1], ax[0, 2]
    ax10, ax11, ax12 = ax[1, 0], ax[1, 1], ax[1, 2]

    # Lock the elevation colormap limits to the full DEM range so the FoV
    # panel (ax02) uses the *same* colors as the global DEM panel (ax00).
    finite_dem = dem[np.isfinite(dem)]
    if finite_dem.size:
        dem_vmin = float(np.nanpercentile(finite_dem, 1.0))
        dem_vmax = float(np.nanpercentile(finite_dem, 99.0))
        if dem_vmax <= dem_vmin:
            dem_vmin, dem_vmax = float(finite_dem.min()), float(finite_dem.max()) + 1e-6
    else:
        dem_vmin, dem_vmax = 0.0, 1.0
    im0 = ax00.imshow(mdem, cmap="terrain", origin="upper",
                       vmin=dem_vmin, vmax=dem_vmax)
    path_line, = ax00.plot([], [], "-o", color="black", ms=3, lw=1, alpha=0.9)
    fov_rect = Rectangle((0, 0), side_px, side_px, linewidth=1.2, edgecolor="white", facecolor="none")
    ax00.add_patch(fov_rect)
    ax00.set_title("DEM with drone path and current FOV")
    ax00.set_axis_off()
    fig.colorbar(im0, ax=ax00, shrink=0.75, label="Elevation")

    cov_img = ax01.imshow(visited_prog.astype(float), cmap="viridis", origin="upper", vmin=0.0, vmax=1.0)
    ax01.set_title("Coverage + exploration graph")
    ax01.set_axis_off()
    lc_temporal = LineCollection([], colors="white", linewidths=1.1, alpha=0.9)
    lc_prox = LineCollection([], colors="cyan", linewidths=0.8, alpha=0.55)
    ax01.add_collection(lc_prox)
    ax01.add_collection(lc_temporal)
    node_scatter = ax01.scatter([], [], c=[], cmap="plasma", s=12, edgecolors="none")
    fig.colorbar(node_scatter, ax=ax01, shrink=0.75, label="Node beta1")
    ax01.legend(
        handles=[
            Line2D([0], [0], color="white", lw=1.2, label="temporal edge"),
            Line2D([0], [0], color="cyan", lw=1.0, label="proximity edge"),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.7,
    )

    # Local FOV DEM panel — share elevation colormap limits with ax00.
    local_dem_placeholder = np.ma.masked_all((side_px, side_px), dtype=float)
    local_dem_img = ax02.imshow(local_dem_placeholder, cmap="terrain",
                                 origin="upper", vmin=dem_vmin, vmax=dem_vmax)
    fig.colorbar(local_dem_img, ax=ax02, shrink=0.75, label="Elevation")
    lc_dem_graph = LineCollection([], linewidths=1.0, alpha=0.9)
    ax02.add_collection(lc_dem_graph)
    dem_nodes = ax02.scatter([], [], s=9, edgecolors="none")
    ax02.set_title("DEM inside current FOV")
    ax02.set_axis_off()
    ax02.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", lw=1.0, label="edge (colored by component)"),
            Line2D([0], [0], marker="o", color="tab:blue", lw=0, markersize=5, label="node (colored by component)"),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.7,
    )

    local_mask_img = ax12.imshow(np.zeros((side_px, side_px), dtype=float), cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
    lc_local_graph = LineCollection([], linewidths=1.0, alpha=0.9)
    ax12.add_collection(lc_local_graph)
    local_nodes = ax12.scatter([], [], s=8, edgecolors="none")
    ax12.set_title("Extracted stream mask + local graph")
    ax12.set_axis_off()
    ax12.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", lw=1.0, label="edge (colored by component)"),
            Line2D([0], [0], marker="o", color="tab:blue", lw=0, markersize=5, label="node (colored by component)"),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.7,
    )

    ax10.set_title("Topology history per capture")
    ax10.set_xlabel("Step")
    ax10.set_ylabel("Topology metric")
    ax10.grid(alpha=0.25)
    beta0_line, = ax10.plot([], [], label="beta0", color="tab:blue")
    beta1_line, = ax10.plot([], [], label="beta1", color="tab:red")
    fiedler_line, = ax10.plot([], [], label="fiedler", color="tab:brown")
    ax10.legend()

    ax11.set_title("Adaptive mapping diagnostics")
    ax11.set_xlabel("Step")
    ax11.grid(alpha=0.25)
    score_line, = ax11.plot([], [], label="adaptive score", color="tab:purple")
    unseen_line, = ax11.plot([], [], label="unseen fraction", color="tab:green")
    stream_line, = ax11.plot([], [], label="stream fraction", color="tab:orange")
    ax11.legend()

    xs = np.array([r.col for r in records], dtype=float)
    ys = np.array([r.row for r in records], dtype=float)
    beta0 = np.array([r.beta0 for r in records], dtype=float)
    beta1 = np.array([r.beta1 for r in records], dtype=float)
    fiedler = np.array([r.fiedler for r in records], dtype=float)
    score = np.array([r.score for r in records], dtype=float)
    unseen = np.array([r.unseen_fraction for r in records], dtype=float)
    stream = np.array([r.stream_fraction for r in records], dtype=float)
    t = np.arange(len(records))

    ax10.set_xlim(0, max(1, len(records) - 1))
    yb_max = max(1.0, float(max(np.max(beta0), np.max(beta1), np.max(fiedler)) + 0.5))
    ax10.set_ylim(-0.1, yb_max)
    ax11.set_xlim(0, max(1, len(records) - 1))
    yd_max = max(1.0, float(max(np.max(score), np.max(unseen), np.max(stream)) + 0.1))
    ax11.set_ylim(-0.05, yd_max)

    def update(frame_idx: int):
        rec = records[frame_idx]
        r0 = max(0, rec.row - half)
        r1 = min(nrows, r0 + side_px)
        c0 = max(0, rec.col - half)
        c1 = min(ncols, c0 + side_px)
        visited_prog[r0:r1, c0:c1] = True

        path_line.set_data(xs[: frame_idx + 1], ys[: frame_idx + 1])
        fov_rect.set_xy((c0, r0))
        fov_rect.set_width(c1 - c0)
        fov_rect.set_height(r1 - r0)

        cov_img.set_data(visited_prog.astype(float))
        sub_records = records[: frame_idx + 1]
        temporal, proximity = build_exploration_graph_edges(
            sub_records, radius_px=160.0, k_nearest=2
        )
        lc_temporal.set_segments(edges_to_segments(sub_records, temporal))
        lc_prox.set_segments(edges_to_segments(sub_records, proximity))
        nxs = np.array([r.col for r in sub_records], dtype=float)
        nys = np.array([r.row for r in sub_records], dtype=float)
        nb1 = np.array([r.beta1 for r in sub_records], dtype=float)
        node_scatter.set_offsets(np.column_stack((nxs, nys)))
        node_scatter.set_array(nb1)
        ax01.set_title(
            f"Coverage + graph (visited={np.mean(visited_prog):.5f}, "
            f"nodes={len(sub_records)}, edges={len(temporal)+len(proximity)})"
        )

        patch, _ = capture_square(dem, rec.row, rec.col, side_px=side_px)
        obs = patch_channel_observation(
            patch,
            resolution_m=resolution_m,
            percentile=stream_percentile,
            extractor=channel_extractor,
            rivgraph_prune_dangling=rivgraph_prune_dangling,
            mask_close_px=int(mask_close_px),
            mask_dilate_px=int(mask_dilate_px),
            bridge_endpoints_px=float(bridge_endpoints_px),
        )
        local_dem_img.set_data(np.ma.masked_invalid(patch))
        mask = np.asarray(obs["mask"], dtype=bool)
        local_mask_img.set_data(mask.astype(float))
        segs_all = obs.get("segments", []) or []
        seg_cc = np.asarray(obs.get("segment_cc", np.array([], dtype=int)), dtype=int)
        edge_rgba = _cc_colors_for_labels(seg_cc) if len(segs_all) > 0 else np.empty((0, 4))
        lc_dem_graph.set_segments(segs_all)
        lc_local_graph.set_segments(segs_all)
        if edge_rgba.size > 0:
            lc_dem_graph.set_colors(edge_rgba)
            lc_local_graph.set_colors(edge_rgba)

        nr = np.asarray(obs["node_rows"], dtype=float)
        nc = np.asarray(obs["node_cols"], dtype=float)
        ncc = np.asarray(obs.get("node_cc", np.array([], dtype=int)), dtype=int)
        if nr.size > 0:
            dem_nodes.set_offsets(np.column_stack((nc, nr)))
            local_nodes.set_offsets(np.column_stack((nc, nr)))
            if ncc.size == nr.size:
                node_rgba = _cc_colors_for_labels(ncc)
                dem_nodes.set_facecolors(node_rgba)
                local_nodes.set_facecolors(node_rgba)
        else:
            dem_nodes.set_offsets(np.empty((0, 2)))
            local_nodes.set_offsets(np.empty((0, 2)))
        ax02.set_title(f"DEM in FOV ({patch.shape[0]}x{patch.shape[1]})")
        n_components_local = int(len(set(int(x) for x in ncc.tolist()))) if ncc.size > 0 else 0
        ax12.set_title(
            f"Mask+graph in FOV (beta0={int(obs['beta0'])}, beta1={int(obs['beta1'])}, "
            f"fiedler={float(obs.get('fiedler', 0.0)):.3f}, components={n_components_local})"
        )

        beta0_line.set_data(t[: frame_idx + 1], beta0[: frame_idx + 1])
        beta1_line.set_data(t[: frame_idx + 1], beta1[: frame_idx + 1])
        fiedler_line.set_data(t[: frame_idx + 1], fiedler[: frame_idx + 1])
        score_line.set_data(t[: frame_idx + 1], score[: frame_idx + 1])
        unseen_line.set_data(t[: frame_idx + 1], unseen[: frame_idx + 1])
        stream_line.set_data(t[: frame_idx + 1], stream[: frame_idx + 1])

        fig.suptitle(
            f"Adaptive flight step {frame_idx + 1}/{len(records)}\n{mission_params_line}",
            fontsize=11,
        )
        return (
            path_line,
            fov_rect,
            cov_img,
            lc_temporal,
            lc_prox,
            node_scatter,
            local_dem_img,
            lc_dem_graph,
            dem_nodes,
            local_mask_img,
            lc_local_graph,
            local_nodes,
            beta0_line,
            beta1_line,
            fiedler_line,
            score_line,
            unseen_line,
            stream_line,
        )

    ani = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=len(records),
        interval=max(20, int(1000 / max(1, fps))),
        blit=False,
        repeat=False,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = (
        f"drone_betti_steps{len(records)}"
        f"_alt{altitude_m:g}m_fov{fov_deg:g}deg"
        f"_wb1_{w_beta1:g}_wb0_{w_beta0:g}"
        f"_{channel_extractor}_fps{max(1, fps)}_{ts}"
    )
    mp4_path = out_dir / f"{base_name}.mp4"
    gif_path = out_dir / f"{base_name}.gif"
    try:
        # libx264 + yuv420p chroma subsampling requires both pixel
        # dimensions to be even.  matplotlib's constrained-layout can
        # render an odd width/height (e.g. 1920x1165 on some displays),
        # which makes ffmpeg abort with "height not divisible by 2".
        # The ``pad=ceil(iw/2)*2:...`` video filter appends at most one
        # black pixel on the right / bottom edge so the encoder always
        # sees an even canvas.
        mp4_writer = mpl_animation.FFMpegWriter(
            fps=max(1, fps),
            extra_args=[
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black",
                "-pix_fmt",
                "yuv420p",
            ],
        )
        ani.save(mp4_path, writer=mp4_writer, dpi=120)
        if not (mp4_path.exists() and mp4_path.stat().st_size > 0):
            raise RuntimeError(f"MP4 writer completed but file is empty: {mp4_path}")

        gif_writer = mpl_animation.PillowWriter(fps=max(1, fps))
        ani.save(gif_path, writer=gif_writer, dpi=120)
        if not (gif_path.exists() and gif_path.stat().st_size > 0):
            raise RuntimeError(f"GIF writer completed but file is empty: {gif_path}")

        plt.close(fig)
        return mp4_path
    except Exception as export_err:
        plt.close(fig)
        raise RuntimeError(
            "Failed to create required animation outputs (MP4 + GIF). "
            f"Expected files: {mp4_path} and {gif_path}"
        ) from export_err


def write_csv(records: list[CaptureRecord], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "capture_metrics.csv"
    header = "step,row,col,beta0,beta1,fiedler,stream_fraction,unseen_fraction,score\n"
    lines = [header]
    for r in records:
        lines.append(
            f"{r.step},{r.row},{r.col},{r.beta0},{r.beta1},{r.fiedler:.6f},"
            f"{r.stream_fraction:.6f},{r.unseen_fraction:.6f},{r.score:.6f}\n"
        )
    out.write_text("".join(lines), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dem-npy", type=Path, default=None, help="Optional path to a 2D DEM .npy array.")
    p.add_argument("--dem-tiff", type=Path, default=None, help="Optional path to a DEM TIFF/GeoTIFF file.")
    p.add_argument(
        "--rivgraph-repo",
        type=Path,
        default=None,
        help="Optional path to a RivGraph clone root; adds repo and _deps to sys.path.",
    )
    p.add_argument(
        "--bbox-lonlat",
        type=str,
        default=None,
        help=(
            "Optional geographic crop bbox as 'lon_min,lat_min,lon_max,lat_max'. "
            "If set with --dem-tiff, script crops with gdal_translate before simulation."
        ),
    )
    p.add_argument(
        "--bbox-crop-name",
        type=str,
        default="bbox_crop.tif",
        help="Filename for bbox-cropped TIFF written under --output-dir.",
    )
    p.add_argument(
        "--nodata-value",
        type=float,
        default=None,
        help="Optional DEM NoData sentinel value to map to NaN (e.g., -9999).",
    )
    p.add_argument(
        "--tiff-window",
        type=str,
        default=None,
        help="Optional TIFF crop window as 'r0:r1,c0:c1'.",
    )
    p.add_argument(
        "--max-dem-cells",
        type=int,
        default=4_000_000,
        help="When TIFF is huge and no window is given, auto-center-crop to this many cells.",
    )
    p.add_argument("--dem-rows", type=int, default=320)
    p.add_argument("--dem-cols", type=int, default=320)
    p.add_argument("--dem-resolution-m", type=float, default=30.0)

    p.add_argument("--altitude-m", type=float, default=120.0)
    p.add_argument("--fov-deg", type=float, default=55.0, help="Full camera FOV angle in degrees.")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=11)

    p.add_argument("--stream-percentile", type=float, default=85.0)
    p.add_argument(
        "--channel-extractor",
        choices=("simple", "rivgraph"),
        default="simple",
        help="Channel-to-topology extraction backend for patch Betti metrics.",
    )
    p.add_argument(
        "--rivgraph-prune-dangling",
        action="store_true",
        help="When using rivgraph extractor, prune dangling one-link branches before Betti.",
    )
    p.add_argument("--w-beta1", type=float, default=2.5,
                   help="Weight on normalised β₁ (loop density) per candidate patch.")
    p.add_argument("--w-beta0", type=float, default=0.5,
                   help="Penalty weight on normalised β₀ (fragmentation). "
                        "Higher values steer away from patches with many disconnected "
                        "components (typical of noisy masks).")
    p.add_argument("--w-unseen", type=float, default=5.0,
                   help="Weight on unseen (unexplored) fraction of candidate patch.")
    p.add_argument("--revisit-penalty", type=float, default=0.5,
                   help="Score penalty applied when a candidate was recently visited.")
    p.add_argument(
        "--target-overlap",
        type=float,
        default=0.3,
        help="(Retained for backward compat; step is fixed at side_px // 2 in quadrant mode.)",
    )
    p.add_argument("--min-valid-fraction", type=float, default=0.6)
    p.add_argument(
        "--mask-close-px",
        type=int,
        default=0,
        help=(
            "Morphological closing iterations on the stream mask (3x3 struct). "
            "Bridges small gaps and produces fewer, larger connected channels. "
            "Affects beta0/beta1/fiedler and the skeleton fed to rivgraph."
        ),
    )
    p.add_argument(
        "--mask-dilate-px",
        type=int,
        default=0,
        help=(
            "Morphological dilation iterations applied AFTER closing (3x3 struct). "
            "Thickens channels so skeletonization fuses diagonal/near-parallel arms."
        ),
    )
    p.add_argument(
        "--bridge-endpoints-px",
        type=float,
        default=0.0,
        help=(
            "Post-graph stitching: merge connected components whose nodes fall within "
            "this pixel radius, adding straight-line bridging edges. Affects the "
            "rendered component count and coloring (does not re-run Betti)."
        ),
    )
    p.add_argument("--graph-radius-px", type=float, default=160.0)
    p.add_argument("--graph-k-nearest", type=int, default=2)
    p.add_argument("--animation-fps", type=int, default=6)
    p.add_argument(
        "--no-animation",
        action="store_true",
        help="Disable matplotlib animation export.",
    )
    p.add_argument(
        "--realtime",
        action="store_true",
        help="Run with live matplotlib updates during simulation.",
    )
    p.add_argument(
        "--realtime-pause-s",
        type=float,
        default=0.08,
        help="Pause duration between live frames (seconds).",
    )
    p.add_argument(
        "--realtime-block",
        action="store_true",
        help="Keep realtime plot open at the end until manually closed.",
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jdas/Documents/kernelcal/video-demos/hydroshed"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.realtime:
        dem, visited, records, side_px = run_experiment_realtime(args)
    else:
        dem, visited, records, side_px = run_experiment(args)
    out_dir = args.output_dir.resolve()
    png = save_plot(dem, visited, records, side_px=side_px, out_dir=out_dir)
    anim_path = None
    if not args.no_animation:
        anim_path = save_animation(
            dem,
            records,
            side_px=side_px,
            out_dir=out_dir,
            fps=max(1, int(args.animation_fps)),
            resolution_m=args.dem_resolution_m,
            stream_percentile=args.stream_percentile,
            channel_extractor=args.channel_extractor,
            rivgraph_prune_dangling=args.rivgraph_prune_dangling,
            altitude_m=args.altitude_m,
            fov_deg=args.fov_deg,
            w_beta1=args.w_beta1,
            w_beta0=args.w_beta0,
            mask_close_px=int(getattr(args, "mask_close_px", 0)),
            mask_dilate_px=int(getattr(args, "mask_dilate_px", 0)),
            bridge_endpoints_px=float(getattr(args, "bridge_endpoints_px", 0.0)),
        )
    csv_path = write_csv(records, out_dir=out_dir)

    print("Drone DEM Betti-adaptive experiment complete.")
    print(f"Camera footprint side: {side_px} px")
    print(f"Coverage fraction    : {np.mean(visited):.4f}")
    print(f"Summary plot         : {png}")
    if anim_path is not None:
        print(f"Animation (MP4)      : {anim_path}")
        gif_sibling = anim_path.with_suffix(".gif")
        if gif_sibling.exists():
            print(f"Animation (GIF)      : {gif_sibling}")
    print(f"Capture CSV          : {csv_path}")


if __name__ == "__main__":
    main()

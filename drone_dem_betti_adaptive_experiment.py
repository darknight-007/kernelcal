#!/usr/bin/env python3
"""Drone DEM camera simulation with Betti-adaptive mapping.

Dependencies: numpy, scipy, matplotlib

What this script does
---------------------
1) Emulates a drone with a nadir DEM camera.
2) Uses a square camera footprint derived from altitude + FOV angle.
3) Captures DEM patches along a flight path.
4) Computes stream-like topology per patch and Betti numbers (beta0, beta1).
5) Chooses next waypoint with a Betti-adaptive mapping policy.
6) Writes summary plots of coverage, path, and topology history.

Example
-------
python3 drone_dem_betti_adaptive_experiment.py \
  --steps 35 --altitude-m 120 --fov-deg 55 --dem-resolution-m 30 \
  --output-dir datasets/hydroshed-dem/drone_betti_experiment
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import animation as mpl_animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


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


@dataclass(frozen=True)
class CameraModel:
    altitude_m: float
    fov_deg: float
    dem_resolution_m: float

    @property
    def footprint_side_m(self) -> float:
        half_angle = np.deg2rad(self.fov_deg * 0.5)
        return 2.0 * self.altitude_m * np.tan(half_angle)

    @property
    def footprint_side_px(self) -> int:
        px = int(round(self.footprint_side_m / self.dem_resolution_m))
        return max(5, px)


@dataclass
class CaptureRecord:
    step: int
    row: int
    col: int
    beta0: int
    beta1: int
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
    """D8 steepest-descent routing (-1 marks sink)."""
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
    """Accumulate upstream contributing area (in cell counts)."""
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


def stream_mask_from_patch(patch_dem: np.ndarray, resolution_m: float, percentile: float) -> np.ndarray:
    """Infer stream-like binary mask from local flow accumulation."""
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
    return (acc >= max(2.0, thr)) & valid


def betti_numbers_binary(mask: np.ndarray) -> tuple[int, int]:
    """Compute beta0 and beta1 for a 2D binary mask.

    beta0: # connected components in foreground.
    beta1: # holes in foreground (background components not touching border).
    """
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return 0, 0

    structure = np.ones((3, 3), dtype=np.int8)
    _, n_fg = ndimage.label(m, structure=structure)

    bg = ~m
    labels_bg, n_bg = ndimage.label(bg, structure=structure)
    border_ids = set()
    border_ids.update(np.unique(labels_bg[0, :]).tolist())
    border_ids.update(np.unique(labels_bg[-1, :]).tolist())
    border_ids.update(np.unique(labels_bg[:, 0]).tolist())
    border_ids.update(np.unique(labels_bg[:, -1]).tolist())
    holes = 0
    for lab in range(1, n_bg + 1):
        if lab not in border_ids:
            holes += 1
    return int(n_fg), int(holes)


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


def patch_channel_metrics(
    patch_dem: np.ndarray,
    resolution_m: float,
    percentile: float,
    extractor: str,
    rivgraph_prune_dangling: bool,
) -> tuple[int, int, float]:
    """Compute (beta0, beta1, stream_fraction) from a DEM patch."""
    smask = stream_mask_from_patch(
        patch_dem,
        resolution_m=resolution_m,
        percentile=percentile,
    )
    stream_frac = float(np.mean(smask))
    if extractor == "simple":
        beta0, beta1 = betti_numbers_binary(smask)
        return beta0, beta1, stream_frac

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
    except Exception:
        beta0, beta1 = 0, 0
    return beta0, beta1, stream_frac


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


def candidate_moves(center: tuple[int, int], step_px: int) -> list[tuple[int, int]]:
    r, c = center
    candidates = [(r, c)]
    for dr in (-step_px, 0, step_px):
        for dc in (-step_px, 0, step_px):
            if dr == 0 and dc == 0:
                continue
            candidates.append((r + dr, c + dc))
    return candidates


def choose_next_location(
    dem: np.ndarray,
    visited_mask: np.ndarray,
    current: tuple[int, int],
    side_px: int,
    resolution_m: float,
    stream_percentile: float,
    channel_extractor: str,
    rivgraph_prune_dangling: bool,
    w_beta1: float,
    w_unseen: float,
    w_relief: float,
    min_valid_fraction: float,
    stay_penalty: float,
) -> tuple[tuple[int, int], float, float, int, int]:
    """Choose next waypoint by maximizing topology-aware utility."""
    step_px = max(3, int(round(0.7 * side_px)))
    best = None
    best_score = -np.inf
    for cand in candidate_moves(current, step_px=step_px):
        patch, (r0, r1, c0, c1) = capture_square(dem, cand[0], cand[1], side_px=side_px)
        valid_frac = float(np.mean(np.isfinite(patch)))
        if valid_frac < min_valid_fraction:
            continue
        unseen = 1.0 - float(np.mean(visited_mask[r0:r1, c0:c1]))
        beta0, beta1, _ = patch_channel_metrics(
            patch,
            resolution_m=resolution_m,
            percentile=stream_percentile,
            extractor=channel_extractor,
            rivgraph_prune_dangling=rivgraph_prune_dangling,
        )
        relief = float(np.nanstd(patch))
        score = w_beta1 * beta1 + w_unseen * unseen + w_relief * (relief / 50.0)
        if cand == current:
            score -= stay_penalty
        if score > best_score:
            center = ((r0 + r1) // 2, (c0 + c1) // 2)
            best_score = score
            best = (center, unseen, beta0, beta1)
    if best is None:
        patch, (r0, r1, c0, c1) = capture_square(dem, current[0], current[1], side_px=side_px)
        beta0, beta1, _ = patch_channel_metrics(
            patch,
            resolution_m=resolution_m,
            percentile=stream_percentile,
            extractor=channel_extractor,
            rivgraph_prune_dangling=rivgraph_prune_dangling,
        )
        return current, 0.0, 0.0, beta0, beta1
    center, unseen, beta0, beta1 = best
    return center, float(best_score), unseen, beta0, beta1


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
    if args.dem_npy:
        pass

    cam = CameraModel(
        altitude_m=args.altitude_m,
        fov_deg=args.fov_deg,
        dem_resolution_m=args.dem_resolution_m,
    )
    side_px = cam.footprint_side_px
    visited = np.zeros_like(dem, dtype=bool)
    finite_frac = float(np.mean(np.isfinite(dem)))
    if args.channel_extractor == "rivgraph":
        configure_rivgraph_import(args.rivgraph_repo)
    print(
        f"[info] DEM shape={dem.shape}, finite_fraction={finite_frac:.3f}, "
        f"footprint_px={side_px}, extractor={args.channel_extractor}"
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

        beta0, beta1, stream_frac = patch_channel_metrics(
            patch,
            resolution_m=args.dem_resolution_m,
            percentile=args.stream_percentile,
            extractor=args.channel_extractor,
            rivgraph_prune_dangling=args.rivgraph_prune_dangling,
        )

        if step < args.steps - 1:
            center, score, _, _, _ = choose_next_location(
                dem=dem,
                visited_mask=visited,
                current=center,
                side_px=side_px,
                resolution_m=args.dem_resolution_m,
                stream_percentile=args.stream_percentile,
                channel_extractor=args.channel_extractor,
                rivgraph_prune_dangling=args.rivgraph_prune_dangling,
                w_beta1=args.w_beta1,
                w_unseen=args.w_unseen,
                w_relief=args.w_relief,
                min_valid_fraction=args.min_valid_fraction,
                stay_penalty=args.stay_penalty,
            )
        else:
            score = float(beta1)

        records.append(
            CaptureRecord(
                step=step,
                row=int((r0 + r1) // 2),
                col=int((c0 + c1) // 2),
                beta0=beta0,
                beta1=beta1,
                stream_fraction=stream_frac,
                unseen_fraction=unseen_frac,
                score=float(score),
            )
        )
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
        dem_resolution_m=args.dem_resolution_m,
    )
    side_px = cam.footprint_side_px
    visited = np.zeros_like(dem, dtype=bool)
    finite_frac = float(np.mean(np.isfinite(dem)))
    if args.channel_extractor == "rivgraph":
        configure_rivgraph_import(args.rivgraph_repo)
    print(
        f"[info] DEM shape={dem.shape}, finite_fraction={finite_frac:.3f}, "
        f"footprint_px={side_px}, extractor={args.channel_extractor}"
    )
    if side_px < 24:
        print(
            "[warn] Small footprint (side < 24 px). For regional DEMs this often yields sparse topology. "
            "Increase altitude or FOV for richer Betti signals."
        )

    # Live figure state
    plt.ion()
    fig, ax = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax00, ax01 = ax[0, 0], ax[0, 1]
    ax10, ax11 = ax[1, 0], ax[1, 1]
    mdem = np.ma.masked_invalid(dem)
    im0 = ax00.imshow(mdem, cmap="terrain", origin="upper")
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

    ax10.set_title("Betti history per capture")
    ax10.set_xlabel("Step")
    ax10.set_ylabel("Count")
    ax10.grid(alpha=0.25)
    beta0_line, = ax10.plot([], [], label="beta0", color="tab:blue")
    beta1_line, = ax10.plot([], [], label="beta1", color="tab:red")
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
    score_hist: list[float] = []
    unseen_hist: list[float] = []
    stream_hist: list[float] = []

    for step in range(args.steps):
        patch, (r0, r1, c0, c1) = capture_square(dem, center[0], center[1], side_px=side_px)
        unseen_frac = 1.0 - float(np.mean(visited[r0:r1, c0:c1]))
        visited[r0:r1, c0:c1] = True

        beta0, beta1, stream_frac = patch_channel_metrics(
            patch,
            resolution_m=args.dem_resolution_m,
            percentile=args.stream_percentile,
            extractor=args.channel_extractor,
            rivgraph_prune_dangling=args.rivgraph_prune_dangling,
        )

        if step < args.steps - 1:
            center, score, _, _, _ = choose_next_location(
                dem=dem,
                visited_mask=visited,
                current=center,
                side_px=side_px,
                resolution_m=args.dem_resolution_m,
                stream_percentile=args.stream_percentile,
                channel_extractor=args.channel_extractor,
                rivgraph_prune_dangling=args.rivgraph_prune_dangling,
                w_beta1=args.w_beta1,
                w_unseen=args.w_unseen,
                w_relief=args.w_relief,
                min_valid_fraction=args.min_valid_fraction,
                stay_penalty=args.stay_penalty,
            )
        else:
            score = float(beta1)

        rec = CaptureRecord(
            step=step,
            row=int((r0 + r1) // 2),
            col=int((c0 + c1) // 2),
            beta0=beta0,
            beta1=beta1,
            stream_fraction=stream_frac,
            unseen_fraction=unseen_frac,
            score=float(score),
        )
        records.append(rec)

        # Live update
        xs.append(rec.col)
        ys.append(rec.row)
        beta0_hist.append(rec.beta0)
        beta1_hist.append(rec.beta1)
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

        x_axis = np.arange(len(records))
        beta0_line.set_data(x_axis, beta0_hist)
        beta1_line.set_data(x_axis, beta1_hist)
        score_line.set_data(x_axis, score_hist)
        unseen_line.set_data(x_axis, unseen_hist)
        stream_line.set_data(x_axis, stream_hist)

        ax10.set_xlim(0, max(1, args.steps - 1))
        ax10.set_ylim(-0.1, max(1.0, max(beta0_hist + beta1_hist) + 0.5))
        ax11.set_xlim(0, max(1, args.steps - 1))
        ax11.set_ylim(-0.05, max(1.0, max(score_hist + unseen_hist + stream_hist) + 0.1))

        fig.suptitle(f"Realtime adaptive flight step {step + 1}/{args.steps}", fontsize=12)
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
    ax10.plot(beta0, label="beta0", color="tab:blue")
    ax10.plot(beta1, label="beta1", color="tab:red")
    ax10.set_title("Betti history per capture")
    ax10.set_xlabel("Step")
    ax10.set_ylabel("Count")
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
) -> Path:
    """Save adaptive exploration as a matplotlib animation (GIF preferred)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(records) == 0:
        raise ValueError("No records to animate.")

    mdem = np.ma.masked_invalid(dem)
    nrows, ncols = dem.shape
    visited_prog = np.zeros((nrows, ncols), dtype=bool)
    half = side_px // 2

    fig, ax = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax00, ax01 = ax[0, 0], ax[0, 1]
    ax10, ax11 = ax[1, 0], ax[1, 1]

    im0 = ax00.imshow(mdem, cmap="terrain", origin="upper")
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

    ax10.set_title("Betti history per capture")
    ax10.set_xlabel("Step")
    ax10.set_ylabel("Count")
    ax10.grid(alpha=0.25)
    beta0_line, = ax10.plot([], [], label="beta0", color="tab:blue")
    beta1_line, = ax10.plot([], [], label="beta1", color="tab:red")
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
    score = np.array([r.score for r in records], dtype=float)
    unseen = np.array([r.unseen_fraction for r in records], dtype=float)
    stream = np.array([r.stream_fraction for r in records], dtype=float)
    t = np.arange(len(records))

    ax10.set_xlim(0, max(1, len(records) - 1))
    yb_max = max(1.0, float(max(np.max(beta0), np.max(beta1)) + 0.5))
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

        beta0_line.set_data(t[: frame_idx + 1], beta0[: frame_idx + 1])
        beta1_line.set_data(t[: frame_idx + 1], beta1[: frame_idx + 1])
        score_line.set_data(t[: frame_idx + 1], score[: frame_idx + 1])
        unseen_line.set_data(t[: frame_idx + 1], unseen[: frame_idx + 1])
        stream_line.set_data(t[: frame_idx + 1], stream[: frame_idx + 1])

        fig.suptitle(f"Adaptive flight step {frame_idx + 1}/{len(records)}", fontsize=12)
        return (
            path_line,
            fov_rect,
            cov_img,
            lc_temporal,
            lc_prox,
            node_scatter,
            beta0_line,
            beta1_line,
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

    gif_path = out_dir / "drone_betti_adaptive_animation.gif"
    try:
        writer = mpl_animation.PillowWriter(fps=max(1, fps))
        ani.save(gif_path, writer=writer, dpi=120)
        plt.close(fig)
        return gif_path
    except Exception:
        mp4_path = out_dir / "drone_betti_adaptive_animation.mp4"
        writer = mpl_animation.FFMpegWriter(fps=max(1, fps))
        ani.save(mp4_path, writer=writer, dpi=120)
        plt.close(fig)
        return mp4_path


def write_csv(records: list[CaptureRecord], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "capture_metrics.csv"
    header = "step,row,col,beta0,beta1,stream_fraction,unseen_fraction,score\n"
    lines = [header]
    for r in records:
        lines.append(
            f"{r.step},{r.row},{r.col},{r.beta0},{r.beta1},"
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
    p.add_argument("--w-beta1", type=float, default=2.5)
    p.add_argument("--w-unseen", type=float, default=1.5)
    p.add_argument("--w-relief", type=float, default=0.7)
    p.add_argument("--min-valid-fraction", type=float, default=0.6)
    p.add_argument("--stay-penalty", type=float, default=0.25)
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
        default=Path("datasets/hydroshed-dem/drone_betti_experiment"),
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
        )
    csv_path = write_csv(records, out_dir=out_dir)

    print("Drone DEM Betti-adaptive experiment complete.")
    print(f"Camera footprint side: {side_px} px")
    print(f"Coverage fraction    : {np.mean(visited):.4f}")
    print(f"Summary plot         : {png}")
    if anim_path is not None:
        print(f"Animation            : {anim_path}")
    print(f"Capture CSV          : {csv_path}")


if __name__ == "__main__":
    main()

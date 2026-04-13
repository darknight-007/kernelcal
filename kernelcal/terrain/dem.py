"""Digital Elevation Model → graph pipeline.

Converts a 2-D elevation array (DEM) into graph structures suitable for
spectral kernel dynamics and topological biosignature analysis.

Key outputs
-----------
* Grid graph  : 4- or 8-connected graph over DEM pixels, edge weights from
                elevation differences or slopes.
* Flow graph  : directed D8 drainage network (each cell flows to the steepest
                downhill neighbour).
* Point cloud : (row, col, elev) triples for compatibility with geo3d modules.

Physical conventions
--------------------
* DEM values are in metres of elevation.
* dx, dy are horizontal cell spacings in metres (default 1 m; set to 30 m for
  SRTM, 0.25 m for UAV LiDAR, etc.).
* NoData cells should be marked with np.nan before calling these functions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Slope and curvature (cell-level)
# ---------------------------------------------------------------------------

def slope(dem: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Gradient magnitude (rise / run) at each interior cell.

    Uses central differences; boundary cells are filled with np.nan.

    Parameters
    ----------
    dem : (nrows, ncols) float array — elevation in metres
    dx  : column spacing in metres
    dy  : row spacing in metres

    Returns
    -------
    (nrows, ncols) float array — slope in m/m
    """
    z = np.asarray(dem, dtype=float)
    dz_dy, dz_dx = np.gradient(z, dy, dx)
    return np.hypot(dz_dx, dz_dy)


def curvature_planform(dem: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Planform (contour) curvature — positive = concave (channel), negative = convex (ridge).

    Computed from second-order finite differences of the DEM.
    """
    z = np.asarray(dem, dtype=float)
    # Second derivatives via central differences
    d2z_dx2 = np.gradient(np.gradient(z, dx, axis=1), dx, axis=1)
    d2z_dy2 = np.gradient(np.gradient(z, dy, axis=0), dy, axis=0)
    return d2z_dx2 + d2z_dy2


def curvature_profile(dem: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Profile curvature — curvature in the direction of steepest descent.

    Positive = slope increasing downhill (convergent), negative = decreasing.
    """
    z = np.asarray(dem, dtype=float)
    dz_dy, dz_dx = np.gradient(z, dy, dx)
    d2z_dx2 = np.gradient(np.gradient(z, dx, axis=1), dx, axis=1)
    d2z_dy2 = np.gradient(np.gradient(z, dy, axis=0), dy, axis=0)
    d2z_dxdy = np.gradient(np.gradient(z, dx, axis=1), dy, axis=0)
    denom = dz_dx**2 + dz_dy**2 + 1e-12
    return (dz_dx**2 * d2z_dx2 + 2 * dz_dx * dz_dy * d2z_dxdy + dz_dy**2 * d2z_dy2) / denom


# ---------------------------------------------------------------------------
# D8 flow routing
# ---------------------------------------------------------------------------

# 8-neighbourhood offsets (row, col) and labels 0..7
_D8_OFFSETS = np.array([
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),           ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
], dtype=int)
# Diagonal distance multiplier (1 for cardinal, √2 for diagonal)
_D8_DIST = np.array([np.sqrt(2), 1, np.sqrt(2),
                     1,              1,
                     np.sqrt(2), 1, np.sqrt(2)])


def d8_flow_direction(dem: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """D8 single-flow-direction routing.

    Each cell is assigned an integer 0–7 indicating which of its 8 neighbours
    it drains to (index into _D8_OFFSETS), or -1 if it is a local minimum / pit.

    Parameters
    ----------
    dem : (nrows, ncols) elevation array
    dx, dy : horizontal cell spacings

    Returns
    -------
    (nrows, ncols) int8 array — flow direction (-1 = sink / flat)
    """
    z = np.asarray(dem, dtype=float)
    nrows, ncols = z.shape
    fdir = np.full((nrows, ncols), -1, dtype=np.int8)
    scales = np.array([dx * _D8_DIST[i] if (_D8_OFFSETS[i, 1] != 0) else dy
                       for i in range(8)])

    for r in range(nrows):
        for c in range(ncols):
            if np.isnan(z[r, c]):
                continue
            best_slope = 0.0
            best_dir = -1
            for d, (dr, dc) in enumerate(_D8_OFFSETS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols and not np.isnan(z[nr, nc]):
                    drop = z[r, c] - z[nr, nc]
                    s = drop / scales[d]
                    if s > best_slope:
                        best_slope = s
                        best_dir = d
            fdir[r, c] = best_dir
    return fdir


def flow_accumulation(fdir: np.ndarray) -> np.ndarray:
    """Upstream area (cell count) via recursive D8 accumulation.

    Parameters
    ----------
    fdir : (nrows, ncols) int8 flow-direction array from d8_flow_direction

    Returns
    -------
    (nrows, ncols) int32 array — number of upstream cells including self
    """
    nrows, ncols = fdir.shape
    acc = np.ones((nrows, ncols), dtype=np.int32)
    # Build in-degree count
    indegree = np.zeros((nrows, ncols), dtype=np.int32)
    for r in range(nrows):
        for c in range(ncols):
            d = int(fdir[r, c])
            if d >= 0:
                dr, dc = _D8_OFFSETS[d]
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    indegree[nr, nc] += 1

    # Topological sort (Kahn's algorithm)
    from collections import deque
    queue = deque()
    for r in range(nrows):
        for c in range(ncols):
            if indegree[r, c] == 0:
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        d = int(fdir[r, c])
        if d >= 0:
            dr, dc = _D8_OFFSETS[d]
            nr, nc = r + dr, c + dc
            if 0 <= nr < nrows and 0 <= nc < ncols:
                acc[nr, nc] += acc[r, c]
                indegree[nr, nc] -= 1
                if indegree[nr, nc] == 0:
                    queue.append((nr, nc))
    return acc


def channel_mask(acc: np.ndarray, threshold: int) -> np.ndarray:
    """Boolean mask of cells with upstream area ≥ threshold (channel cells)."""
    return acc >= threshold


# ---------------------------------------------------------------------------
# Graph construction from DEM
# ---------------------------------------------------------------------------

@dataclass
class TerrainGraph:
    """Undirected weighted graph built from a DEM.

    Attributes
    ----------
    positions   : (N, 2) float — (row, col) pixel coordinates of nodes
    elevations  : (N,)   float — elevation of each node
    edges       : (E, 2) int  — node index pairs
    weights     : (E,)   float — edge weights (elevation difference or 1.0)
    shape       : (nrows, ncols) — DEM shape
    cell_index  : (nrows, ncols) int — node index at each pixel (-1 = excluded)
    """
    positions:  np.ndarray
    elevations: np.ndarray
    edges:      np.ndarray
    weights:    np.ndarray
    shape:      tuple[int, int]
    cell_index: np.ndarray


def dem_to_graph(
    dem: np.ndarray,
    connectivity: Literal[4, 8] = 8,
    weight: Literal["elev_diff", "slope", "uniform"] = "elev_diff",
    dx: float = 1.0,
    dy: float = 1.0,
    mask: np.ndarray | None = None,
) -> TerrainGraph:
    """Build an undirected grid graph from a DEM.

    Parameters
    ----------
    dem          : (nrows, ncols) elevation array
    connectivity : 4 (cardinal) or 8 (cardinal + diagonal) neighbours
    weight       : edge weight type —
                   'elev_diff'  absolute elevation difference |z_i - z_j|
                   'slope'      slope magnitude (elev_diff / distance)
                   'uniform'    all weights = 1.0
    dx, dy       : horizontal cell spacings (used when weight='slope')
    mask         : (nrows, ncols) bool — include only True cells (default all)

    Returns
    -------
    TerrainGraph
    """
    z = np.asarray(dem, dtype=float)
    nrows, ncols = z.shape

    if mask is None:
        mask = ~np.isnan(z)
    else:
        mask = np.asarray(mask, dtype=bool) & ~np.isnan(z)

    # Map (r, c) → node index
    cell_idx = np.full((nrows, ncols), -1, dtype=np.int32)
    node_rc = np.argwhere(mask)                   # (N, 2)
    cell_idx[mask] = np.arange(len(node_rc))

    # Offsets for 4- or 8-connectivity
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diag_dist = [dy, dy, dx, dx]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        diag_dist += [np.hypot(dx, dy)] * 4

    edge_list: list[tuple[int, int]] = []
    weight_list: list[float] = []
    seen: set[tuple[int, int]] = set()

    for idx, (r, c) in enumerate(node_rc):
        for k, (dr, dc) in enumerate(offsets):
            nr, nc = r + dr, c + dc
            if 0 <= nr < nrows and 0 <= nc < ncols and cell_idx[nr, nc] >= 0:
                j = int(cell_idx[nr, nc])
                pair = (min(idx, j), max(idx, j))
                if pair not in seen:
                    seen.add(pair)
                    edge_list.append(pair)
                    dz = abs(float(z[r, c]) - float(z[nr, nc]))
                    if weight == "uniform":
                        w = 1.0
                    elif weight == "slope":
                        w = dz / diag_dist[k]
                    else:           # elev_diff
                        w = dz
                    weight_list.append(max(w, 1e-8))

    edges   = np.array(edge_list,  dtype=np.int32)
    weights = np.array(weight_list, dtype=float)

    return TerrainGraph(
        positions=node_rc.astype(float),
        elevations=z[mask],
        edges=edges,
        weights=weights,
        shape=(nrows, ncols),
        cell_index=cell_idx,
    )


def terrain_graph_laplacian(tg: TerrainGraph) -> np.ndarray:
    """Dense combinatorial Laplacian L = D - W for a TerrainGraph."""
    n = len(tg.elevations)
    W = np.zeros((n, n), dtype=float)
    for (i, j), w in zip(tg.edges, tg.weights):
        W[i, j] = w
        W[j, i] = w
    D = np.diag(W.sum(axis=1))
    return D - W


def dem_to_point_cloud(dem: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Convert DEM to (N, 3) XYZ point cloud (x=col*dx, y=row*dy, z=elevation).

    NaN cells are excluded.
    """
    z = np.asarray(dem, dtype=float)
    nrows, ncols = z.shape
    valid = ~np.isnan(z)
    rows, cols = np.where(valid)
    xs = cols * dx
    ys = rows * dy
    zs = z[valid]
    return np.stack([xs, ys, zs], axis=1)


# ---------------------------------------------------------------------------
# Synthetic DEMs for testing
# ---------------------------------------------------------------------------

def synthetic_crater_dem(
    nrows: int = 64,
    ncols: int = 64,
    center: tuple[int, int] | None = None,
    radius: float = 12.0,
    depth: float = 5.0,
    rim_height: float = 2.0,
    noise_std: float = 0.1,
) -> np.ndarray:
    """Synthetic DEM with a single circular impact crater.

    The crater has a flat floor, a raised rim ring, and a flat surrounding
    plain.  Used for testing crater Betti-number detection.

    Parameters
    ----------
    nrows, ncols  : DEM dimensions in pixels
    center        : (row, col) of crater centre (default: grid centre)
    radius        : crater rim radius in pixels
    depth         : depth of crater floor below plain level
    rim_height    : height of rim above plain level
    noise_std     : Gaussian noise standard deviation

    Returns
    -------
    (nrows, ncols) float DEM in metres
    """
    if center is None:
        center = (nrows // 2, ncols // 2)
    cr, cc = center
    rows, cols = np.mgrid[0:nrows, 0:ncols]
    r_dist = np.hypot(rows - cr, cols - cc)

    dem = np.zeros((nrows, ncols), dtype=float)
    # Crater interior: depressed
    dem[r_dist < radius * 0.8] = -depth
    # Rim ring
    rim_mask = (r_dist >= radius * 0.8) & (r_dist <= radius * 1.2)
    dem[rim_mask] = rim_height * np.exp(-((r_dist[rim_mask] - radius) ** 2) / (0.1 * radius) ** 2)

    rng = np.random.default_rng(42)
    dem += rng.standard_normal((nrows, ncols)) * noise_std
    return dem


def synthetic_channel_dem(
    nrows: int = 64,
    ncols: int = 64,
    n_tributaries: int = 3,
    slope_angle: float = 0.05,
    valley_depth: float = 3.0,
    noise_std: float = 0.2,
) -> np.ndarray:
    """Synthetic DEM with a main channel and branching tributaries.

    Creates a sloped planar surface with incised V-shaped channels —
    a simple abiotic drainage network with known β₁ = n_tributaries - 1.

    Parameters
    ----------
    nrows, ncols   : DEM dimensions
    n_tributaries  : number of tributary channels (main channel counts as one)
    slope_angle    : overall slope (rise/run) in the row direction
    valley_depth   : incision depth of channels below surrounding terrain
    noise_std      : Gaussian noise

    Returns
    -------
    (nrows, ncols) float DEM
    """
    rows, cols = np.mgrid[0:nrows, 0:ncols]
    # Base sloped plane
    dem = (nrows - 1 - rows) * slope_angle

    # Main stem: centre column, full length
    main_col = ncols // 2
    dist_main = np.abs(cols - main_col).astype(float)
    dem -= valley_depth * np.exp(-dist_main**2 / (2.0 * 2.0**2))

    # Tributaries branch from main channel at regular intervals
    branch_rows = np.linspace(nrows // 4, 3 * nrows // 4, n_tributaries - 1).astype(int)
    for br in branch_rows:
        # Diagonal tributary going upper-left
        for offset in range(min(nrows, ncols) // 3):
            tr, tc = br - offset, main_col - offset
            if 0 <= tr < nrows and 0 <= tc < ncols:
                dist_trib = np.sqrt((rows - tr)**2 + (cols - tc)**2)
                dem -= valley_depth * 0.5 * np.exp(-dist_trib**2 / (2.0 * 1.5**2))

    rng = np.random.default_rng(7)
    dem += rng.standard_normal((nrows, ncols)) * noise_std
    return dem

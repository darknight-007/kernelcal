#!/usr/bin/env python3
"""Graph-based bishop-rocks explorer with live visualization.

Analogue of ``drone_dem_betti_adaptive_experiment.py`` in the
``software-kernelcal-deepgis-integration`` package, but the underlying graph is
built over **rock centroids** (point data) instead of DEM pixels.

Inputs (expected under ``unprocessed/bishop-root/`` next to this script):
  - ``rocks-coord-list.csv`` — no header, (lon, lat)      ~82k centroids
  - ``rock_traits_full.csv`` — lon, lat, area_m2, major_axis_m,
    minor_axis_m, eccentricity, orientation_deg, elevation_rel   ~14k rocks

Coord rocks without a matching row in ``rock_traits_full.csv`` are imputed as
round 2 cm pebbles (``--fallback-diameter-m``; disable with
``--no-impute-missing-traits``): ``diameter = 2 cm``, ``eccentricity = 0``,
``area = π · 0.01² ≈ 3.14 × 10⁻⁴ m²``. After imputation every rock is
edge-eligible.

The explorer sweeps a **square** scan footprint over the traits bounding box,
with side ``scan_side_m = 2 · altitude · tan(fov/2)`` derived from the shared
:class:`kernelcal.graph_explorer.CameraModel` (same knobs as the drone-DEM
explorer: ``--altitude-m``, ``--fov-deg``).  At every step it:

1. Projects all lon/lat to a local equirectangular frame in **metres**.
2. Paints the current footprint on a shared
   :class:`kernelcal.graph_explorer.CoverageRaster` (bishop analog of the
   DEM explorer's ``visited`` numpy array) at ``--scarp-resolution-m`` pixel
   size, so ``unseen_frac`` at every future candidate uses the exact same
   ``1 − mean(visited[target_patch])`` rule as the DEM explorer.
3. Collects the trait rocks inside the square window (imputed pebbles
   included).
4. Builds a local **k-NN graph** where edges are added between rocks that
   have trait rows in ``rock_traits_full.csv`` (with
   ``--no-impute-missing-traits`` only the measured rocks are edge-eligible;
   imputed 2 cm pebbles are treated the same as measured rocks and contribute
   edges by default).  **This step is the only algorithmic difference from
   the drone-DEM explorer** (which extracts a channel-network graph from a
   DEM patch).
5. Runs scipy's connected-components and the smallest Laplacian eigenvalue
   (Fiedler) on the k-NN graph.
6. Picks the next waypoint with the **quadrant-Betti** planner — identical
   to ``choose_next_location`` in the drone-DEM explorer.  The current
   window is split into four diagonal rock quadrants (NW/NE/SW/SE); each
   quadrant's β₀/β₁ on its sub-k-NN graph are combined with an area-based
   unseen fraction via the shared ``kernelcal.graph_explorer`` planner
   (``w_beta1·clip(β₁/n) − w_beta0·clip(β₀/n) + w_unseen·unseen``, minus a
   revisit penalty for recently-visited targets, with a deterministic
   cyclic tie-break).  The explorer teleports to the outer corner of the
   winner (same motion model as DEM's ``center = (target_r, target_c)``).
7. Updates a live 2x3 matplotlib figure:
     (0,0) full map, rocks colored by area + window + path
     (0,1) local graph — nodes colored by component, cyan = k-NN edges
     (0,2) area histogram over rocks inside the window
     (1,0) rolling topology history (n_nodes, n_components, Fiedler)
     (1,1) rolling trait stats (median area, median eccentricity)
     (1,2) cumulative explored: eccentricity vs area scatter, colored by step

A summary PNG is always written. By default the explorer also writes **both**
MP4 and GIF animations into ``--out`` with auto-generated, timestamped
filenames encoding the mission parameters (parity with
``drone_dem_betti_adaptive_experiment.py``); pass ``--no-animation`` to skip
that, or ``--save-mp4 PATH`` / ``--save-gif PATH`` to override the
destination for that format.

Usage
-----
    python3 bishop_rocks_graph_explorer.py                            # default scan + auto MP4/GIF
    python3 bishop_rocks_graph_explorer.py --altitude-m 50 --fov-deg 90  # custom camera
    python3 bishop_rocks_graph_explorer.py --show                     # interactive window
    python3 bishop_rocks_graph_explorer.py --steps 120 --knn 8
    python3 bishop_rocks_graph_explorer.py --no-animation             # PNG+CSV only
    python3 bishop_rocks_graph_explorer.py --save-mp4 run.mp4         # custom MP4 path
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation as mpl_animation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

# Ensure direct script execution can import the local ``kernelcal`` package.
# This mirrors sibling example scripts moved under ``examples/<category>/``.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Shared lon/lat -> local metres projection lives in the kernelcal package
# so this script, the drone-DEM explorer, and any future examples share the
# same implementation instead of redefining it. Imported names are kept at
# module scope for back-compat with callers such as ``tests/test_bishop_rocks_explorer.py``
# that reach for ``bishop_rocks_graph_explorer.LocalFrame``.
from kernelcal.geo3d import METERS_PER_DEG_LAT, LocalFrame  # noqa: F401

# The exploration policy (scoring, revisit penalty, cyclic tie-break) is
# identical to the one in ``examples/controller/drone_dem_betti_adaptive_experiment.py``
# and now lives in ``kernelcal.graph_explorer`` so both scripts call the same
# function.  The domain-specific work — splitting the scan window into four
# diagonal rock quadrants — stays in this module; scoring / ranking does not.
from kernelcal.graph_explorer import (
    BettiWeights,
    CameraModel,
    Candidate,
    CoverageRaster,
    QUADRANT_NAMES,
    QUADRANT_OFFSETS_METRIC,
    choose_best_candidate,
)

HERE = Path(__file__).resolve().parent
# Matches sibling scripts (``bishop_kernelcal.py``, ``bishop_trait_analysis.py``):
# rock centroids + per-rock traits CSVs are expected under
# ``<repo>/datasets/bishop_scarp/`` by default.  Env vars allow local override
# without editing the script.
DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "KERNELCAL_BISHOP_DATA_DIR",
        str(REPO_ROOT / "datasets" / "bishop_scarp"),
    )
)
DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "KERNELCAL_BISHOP_FIG_DIR",
        "/home/jdas/Documents/kernelcal/video-demos/bishop",
    )
)


# ---------------------------------------------------------------------------
# Fallback traits for rocks without measurements
# ---------------------------------------------------------------------------
# Coord rocks that have no matching row in ``rock_traits_full.csv`` are
# imputed as perfectly round 2 cm pebbles:
#   diameter      = 2 cm  = 0.02 m
#   radius        = 0.01 m
#   area_m2       = π · 0.01² ≈ 3.14 × 10⁻⁴ m²
#   eccentricity  = 0  (circular — major_axis == minor_axis == 0.02 m)
#   orientation   = 0°
#   elevation_rel = NaN (unknown; not inferred from diameter)

_FALLBACK_DIAMETER_M: float = 0.02
_FALLBACK_ECCENTRICITY: float = 0.0


def _circular_area_m2(diameter_m: float) -> float:
    """Area of a circle with the given diameter in metres."""
    r = 0.5 * float(diameter_m)
    return float(np.pi * r * r)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, LocalFrame]:
    coords = pd.read_csv(data_dir / "rocks-coord-list.csv", header=None,
                         names=["lon", "lat"])
    traits = pd.read_csv(data_dir / "rock_traits_full.csv")

    lon0 = float(traits["lon"].mean())
    lat0 = float(traits["lat"].mean())
    frame = LocalFrame(lon0=lon0, lat0=lat0)

    traits["x_m"], traits["y_m"] = frame.to_xy(traits["lon"].to_numpy(),
                                               traits["lat"].to_numpy())
    coords["x_m"], coords["y_m"] = frame.to_xy(coords["lon"].to_numpy(),
                                               coords["lat"].to_numpy())
    return coords, traits, frame


# ---------------------------------------------------------------------------
# Graph construction over a point set
# ---------------------------------------------------------------------------


def _edge_set_from_neighbors(neighbors: list[np.ndarray], *,
                             skip_self: bool = True) -> set[tuple[int, int]]:
    es: set[tuple[int, int]] = set()
    for i, idxs in enumerate(neighbors):
        for j in idxs:
            if skip_self and int(j) == i:
                continue
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            es.add((a, b))
    return es


def knn_edges(
    xy: np.ndarray,
    k: int,
    *,
    max_edge_m: float | None = None,
) -> set[tuple[int, int]]:
    """k-nearest-neighbour edge set over 2-D points.

    When ``max_edge_m`` is a positive finite number, edges whose Euclidean
    length exceeds ``max_edge_m`` are dropped, so a point may end up with
    fewer than ``k`` neighbours (possibly zero).  ``max_edge_m`` of 0,
    ``None``, ``inf`` or ``nan`` disables the cap and recovers the
    uncapped k-NN behaviour used by earlier Bishop runs.
    """
    if len(xy) < 2:
        return set()
    tree = cKDTree(xy)
    kq = min(k + 1, len(xy))   # +1 because query returns self
    dist, idx = tree.query(xy, k=kq)
    if idx.ndim == 1:
        idx = idx[:, None]
        dist = dist[:, None]

    has_cap = (
        max_edge_m is not None
        and np.isfinite(max_edge_m)
        and float(max_edge_m) > 0.0
    )
    if not has_cap:
        return _edge_set_from_neighbors([row for row in idx])

    cap = float(max_edge_m)
    n_pts = len(xy)
    es: set[tuple[int, int]] = set()
    for i in range(n_pts):
        for d_ij, j in zip(dist[i], idx[i]):
            j_int = int(j)
            if j_int == i:
                continue
            if not np.isfinite(d_ij) or float(d_ij) > cap:
                continue
            a, b = (i, j_int) if i < j_int else (j_int, i)
            es.add((a, b))
    return es


def trait_mask_for_coords(
    coords: pd.DataFrame,
    traits: pd.DataFrame,
    *,
    decimals: int = 8,
    tol_m: float = 1.0,
) -> np.ndarray:
    """Boolean mask over `coords`: True when that rock has trait data.

    Prefers nearest-neighbour matching in projected metres (`x_m`,`y_m`) with
    tolerance `tol_m`, which is robust to CSV precision drift between the two
    source files. Falls back to rounded lon/lat key matching when projected
    columns are unavailable.
    """
    if len(coords) == 0:
        return np.empty((0,), dtype=bool)
    if len(traits) == 0:
        return np.zeros((len(coords),), dtype=bool)

    if {"x_m", "y_m"}.issubset(coords.columns) and {"x_m", "y_m"}.issubset(traits.columns):
        cxy = np.c_[coords["x_m"].to_numpy(dtype=float), coords["y_m"].to_numpy(dtype=float)]
        txy = np.c_[traits["x_m"].to_numpy(dtype=float), traits["y_m"].to_numpy(dtype=float)]
        tree = cKDTree(cxy)
        d, idx = tree.query(txy, k=1)
        mask = np.zeros((len(coords),), dtype=bool)
        good = np.isfinite(d) & (d <= float(tol_m))
        if np.any(good):
            mask[np.asarray(idx[good], dtype=int)] = True
        return mask

    tr_lon = np.round(traits["lon"].to_numpy(dtype=float), decimals=decimals)
    tr_lat = np.round(traits["lat"].to_numpy(dtype=float), decimals=decimals)
    trait_keys = {(float(lo), float(la)) for lo, la in zip(tr_lon.tolist(), tr_lat.tolist())}

    co_lon = np.round(coords["lon"].to_numpy(dtype=float), decimals=decimals)
    co_lat = np.round(coords["lat"].to_numpy(dtype=float), decimals=decimals)
    mask = np.array(
        [(float(lo), float(la)) in trait_keys for lo, la in zip(co_lon.tolist(), co_lat.tolist())],
        dtype=bool,
    )
    return mask


def impute_missing_traits(
    coords: pd.DataFrame,
    traits: pd.DataFrame,
    *,
    tol_m: float = 1.0,
    diameter_m: float = _FALLBACK_DIAMETER_M,
    eccentricity: float = _FALLBACK_ECCENTRICITY,
) -> pd.DataFrame:
    """Append synthetic trait rows for coord rocks that have no measured traits.

    Each missing-trait rock is treated as a circular pebble with diameter
    ``diameter_m`` (default 2 cm) and ``eccentricity`` (default 0), which
    implies ``area_m2 = π · (diameter_m / 2)²``. The imputed rows inherit the
    coord rock's ``lon``, ``lat``, and projected ``x_m`` / ``y_m`` so nearest-
    neighbour matching marks every coord rock as trait-linked afterwards.

    ``elevation_rel`` is set to NaN for imputed rocks because diameter carries
    no information about elevation.

    Returns a new DataFrame (``traits`` is not modified in place). If every
    coord rock already has a matching trait within ``tol_m``, ``traits`` is
    returned unchanged.
    """
    mask = trait_mask_for_coords(coords, traits, tol_m=float(tol_m))
    missing = coords.loc[~mask]
    if missing.empty:
        return traits

    n = int(len(missing))
    area = _circular_area_m2(diameter_m)
    imputed_cols: dict[str, np.ndarray] = {
        "lon": missing["lon"].to_numpy(dtype=float),
        "lat": missing["lat"].to_numpy(dtype=float),
        "area_m2": np.full(n, area, dtype=float),
        "major_axis_m": np.full(n, float(diameter_m), dtype=float),
        "minor_axis_m": np.full(n, float(diameter_m), dtype=float),
        "eccentricity": np.full(n, float(eccentricity), dtype=float),
        "orientation_deg": np.zeros(n, dtype=float),
        "elevation_rel": np.full(n, np.nan, dtype=float),
    }
    if "x_m" in missing.columns and "y_m" in missing.columns:
        imputed_cols["x_m"] = missing["x_m"].to_numpy(dtype=float)
        imputed_cols["y_m"] = missing["y_m"].to_numpy(dtype=float)
    imputed = pd.DataFrame(imputed_cols)
    # ``sort=False`` keeps the existing column order; missing columns in the
    # original ``traits`` frame are added at the end and filled with NaN for
    # pre-existing rows, which matches pandas' default concat behaviour.
    return pd.concat([traits, imputed], ignore_index=True, sort=False)


def coord_diameter_m_for_coords(
    coords: pd.DataFrame,
    traits: pd.DataFrame,
    *,
    tol_m: float = 1.0,
) -> np.ndarray:
    """Per-coord circular-equivalent diameter in metres.

    For each coord rock, finds the nearest trait row within ``tol_m`` (in
    projected metric ``x_m``/``y_m``) and returns its equivalent-circle
    diameter ``d = 2·sqrt(area_m2 / π)``.  Returns ``NaN`` for coords with
    no trait match inside the tolerance.

    Used by the ``--min-edge-diameter-m`` CLI filter to drop small pebbles
    from the k-NN edge set without removing them as nodes: the scan window
    still counts them, they just don't wire into the graph topology.
    """
    n = len(coords)
    if n == 0:
        return np.empty((0,), dtype=float)
    out = np.full((n,), np.nan, dtype=float)
    if (
        len(traits) == 0
        or "area_m2" not in traits.columns
        or not {"x_m", "y_m"}.issubset(coords.columns)
        or not {"x_m", "y_m"}.issubset(traits.columns)
    ):
        return out
    cxy = np.c_[coords["x_m"].to_numpy(dtype=float),
                coords["y_m"].to_numpy(dtype=float)]
    txy = np.c_[traits["x_m"].to_numpy(dtype=float),
                traits["y_m"].to_numpy(dtype=float)]
    tree = cKDTree(txy)
    d, idx = tree.query(cxy, k=1)
    areas = traits["area_m2"].to_numpy(dtype=float)
    good = (
        np.isfinite(d)
        & (d <= float(tol_m))
        & np.isfinite(areas[idx])
        & (areas[idx] > 0.0)
    )
    if np.any(good):
        out[good] = 2.0 * np.sqrt(areas[idx[good]] / np.pi)
    return out


def knn_edges_trait_only(
    xy: np.ndarray,
    k: int,
    has_trait: np.ndarray,
    *,
    max_edge_m: float | None = None,
) -> set[tuple[int, int]]:
    """k-NN edges between edge-eligible nodes only; others stay isolated.

    ``max_edge_m`` forwards to :func:`knn_edges` and caps the maximum
    Euclidean edge length (see that function's docstring).
    """
    n = int(len(xy))
    if n < 2:
        return set()
    m = np.asarray(has_trait, dtype=bool)
    if m.size != n:
        raise ValueError("has_trait mask length must match xy length")
    active = np.where(m)[0]
    if active.size < 2:
        return set()

    sub_edges = knn_edges(xy[active], k, max_edge_m=max_edge_m)
    out: set[tuple[int, int]] = set()
    for i_sub, j_sub in sub_edges:
        i = int(active[int(i_sub)])
        j = int(active[int(j_sub)])
        a, b = (i, j) if i < j else (j, i)
        out.add((a, b))
    return out


def adjacency_from_edges(n: int, edges: Iterable[tuple[int, int]]) -> csr_matrix:
    e = list(edges)
    if not e:
        return csr_matrix((n, n))
    arr = np.asarray(e, dtype=int)
    rows = np.concatenate([arr[:, 0], arr[:, 1]])
    cols = np.concatenate([arr[:, 1], arr[:, 0]])
    data = np.ones(len(rows), dtype=float)
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def fiedler_value(A: csr_matrix) -> float:
    n = A.shape[0]
    if n < 3 or A.nnz == 0:
        return 0.0
    L = laplacian(A, normed=False).astype(float)
    k = min(2, n - 1)
    try:
        vals = eigsh(L, k=k, sigma=0.0, which="LM", return_eigenvectors=False)
    except Exception:
        try:
            vals = eigsh(L, k=k, which="SM", return_eigenvectors=False)
        except Exception:
            return float("nan")
    vals = np.sort(np.real(vals))
    # Fiedler = 2nd smallest eigenvalue of L
    return float(vals[1]) if len(vals) > 1 else float(vals[0])


# ---------------------------------------------------------------------------
# Quadrant-Betti candidate builder
# ---------------------------------------------------------------------------


def _square_window_mask(
    coord_xy: np.ndarray, cx: float, cy: float, side_m: float
) -> np.ndarray:
    """Bool mask of rocks inside the ``side_m × side_m`` square at ``(cx, cy)``.

    Square footprint semantics mirror the drone-DEM explorer's
    ``capture_square`` which uses ``[r0:r1, c0:c1]`` slicing; a rock is
    inside iff ``|x - cx| <= side/2`` AND ``|y - cy| <= side/2``.  The
    half-open upper edge (``<``) is not important here because float
    coordinates almost never land exactly on the boundary.
    """
    half = 0.5 * float(side_m)
    dx = coord_xy[:, 0] - float(cx)
    dy = coord_xy[:, 1] - float(cy)
    return (np.abs(dx) <= half) & (np.abs(dy) <= half)


def build_betti_quadrant_candidates(
    cx: float,
    cy: float,
    scan_side_m: float,
    step_m: float,
    *,
    coord_xy: np.ndarray,
    coord_has_trait: np.ndarray,
    knn: int,
    visited_raster: CoverageRaster | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    knn_max_edge_m: float | None = None,
) -> list[Candidate]:
    """Build 4 quadrant candidates for the shared Betti scorer.

    Mirrors the quadrant split in ``choose_next_location`` of the drone-DEM
    explorer (``examples/controller/drone_dem_betti_adaptive_experiment.py``):
    split the current square scan footprint of side ``scan_side_m`` into
    four diagonal sub-quadrants (NW/NE/SW/SE) around ``(cx, cy)``, compute
    each quadrant's k-NN graph Betti numbers ``(β₀, β₁, n_nodes)``, and
    target the outer corner of that quadrant in metric space (``step_m``
    in each diagonal axis).

    ``coord_has_trait`` is the edge-eligibility mask (see
    :func:`knn_edges_trait_only`).  ``visited_raster`` is the shared
    coverage mask; if provided, ``unseen_frac`` is computed at each
    candidate target with the exact same
    ``1 - mean(visited[target_patch])`` rule as the drone-DEM explorer's
    ``visited[tr0:tr1, tc0:tc1]`` (see
    :meth:`~kernelcal.graph_explorer.CoverageRaster.unseen_fraction_at`).
    When not provided the unseen fraction is set to ``1.0`` (fully
    unexplored).

    ``bbox`` = ``(xmin, xmax, ymin, ymax)`` clamps the candidate target
    positions; defaults to no clamping.
    """
    # Rocks inside the current square window — shared across all 4 quadrants.
    in_sq = _square_window_mask(coord_xy, cx, cy, scan_side_m)
    if not np.any(in_sq):
        return []
    win_xy = coord_xy[in_sq]
    win_has_trait = coord_has_trait[in_sq]
    # Quadrant membership in the current window: sign of relative position.
    rel = win_xy - np.asarray([[cx, cy]], dtype=float)
    # Rocks exactly on an axis (rel == 0) default to the positive half so
    # every rock lands in exactly one quadrant.  Matches the DEM slicing
    # convention where the lower-right corner of NW goes to NE / SE via
    # ``slice(hr, h)`` / ``slice(wc, w)``.
    east = rel[:, 0] >= 0.0
    north = rel[:, 1] >= 0.0

    quad_masks: dict[str, np.ndarray] = {
        "NW": (~east) & north,
        "NE": east & north,
        "SW": (~east) & (~north),
        "SE": east & (~north),
    }

    if bbox is not None:
        xmin, xmax, ymin, ymax = (float(b) for b in bbox)
    else:
        xmin = xmax = ymin = ymax = float("nan")

    candidates: list[Candidate] = []
    for name in QUADRANT_NAMES:
        q_mask = quad_masks[name]
        n_q = int(np.count_nonzero(q_mask))
        if n_q == 0:
            continue
        q_xy = win_xy[q_mask]
        q_has_trait = win_has_trait[q_mask]

        # k-NN graph on the quadrant's rocks; edges only between trait-having
        # rocks (same rule as the main window graph).  ``knn_max_edge_m``
        # propagates the max-edge-length cap from the CLI so per-quadrant
        # Betti numbers match the local graph that ends up rendered.
        q_edges = (
            knn_edges_trait_only(
                q_xy, int(knn), q_has_trait, max_edge_m=knn_max_edge_m
            )
            if n_q >= 2 else set()
        )
        q_A = adjacency_from_edges(max(n_q, 1), q_edges)
        if q_A.nnz > 0:
            n_comp, _ = connected_components(q_A, directed=False)
        else:
            n_comp = n_q  # all-isolated
        beta1 = max(0, int(q_A.nnz // 2) - n_q + int(n_comp))

        # Target = outer corner of this quadrant in metric space.
        dx, dy = QUADRANT_OFFSETS_METRIC[name]
        tx = cx + float(step_m) * float(dx)
        ty = cy + float(step_m) * float(dy)
        if bbox is not None:
            tx = float(np.clip(tx, xmin, xmax))
            ty = float(np.clip(ty, ymin, ymax))

        # Unseen fraction = fraction of the target footprint's ground area
        # not yet covered by any past scan.  Semantically identical to the
        # drone-DEM explorer's
        # ``1 - mean(visited_mask[tr0:tr1, tc0:tc1])`` — the
        # :class:`CoverageRaster` is the bishop analog of DEM's numpy
        # ``visited`` array.  Callers that pass ``visited_raster=None``
        # get ``unseen = 1.0`` (treat everything as unexplored).
        if visited_raster is not None:
            unseen = float(
                visited_raster.unseen_fraction_at(tx, ty, float(scan_side_m))
            )
        else:
            unseen = 1.0

        # Position key for the revisit-penalty equality check.  Rounded to
        # 1 cm so tiny float drift between consecutive visits doesn't hide
        # a true revisit.  Must match the grid used by the caller when it
        # records past centers against ``recent_positions`` — see the
        # betti-quadrant branch in ``run()``.
        pos_key = (round(tx, 2), round(ty, 2))
        candidates.append(Candidate(
            name=name,
            position=pos_key,
            beta0=int(n_comp),
            beta1=int(beta1),
            n_nodes=int(n_q),
            unseen_frac=float(unseen),
            extra={"target_xy": (float(tx), float(ty))},
        ))
    return candidates


# ---------------------------------------------------------------------------
# Explorer
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    cx: float
    cy: float
    n_nodes: int
    n_edges_knn: int
    n_components: int     # β₀
    beta1: int            # cycles in local k-NN graph
    fiedler: float        # λ₂
    median_area: float
    median_ecc: float
    chosen_direction: str  # winning quadrant name: NW / NE / SW / SE
    chosen_score: float


# ---------------------------------------------------------------------------
# Animation helpers (ported from examples/controller/drone_dem_betti_adaptive_experiment.py)
# ---------------------------------------------------------------------------


def _build_animation_base_name(
    args: argparse.Namespace, step_m: float, scan_side_m: float
) -> str:
    """Auto-generated animation stem mirroring the DEM explorer's convention.

    Pattern:
    ``bishop_rocks_steps{N}_alt{A}m_fov{F}deg_side{W}m_step{S}m_knn{K}_fps{F}_{YYYYMMDD_HHMMSS}``.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"bishop_rocks_steps{int(args.steps)}"
        f"_alt{args.altitude_m:g}m_fov{args.fov_deg:g}deg"
        f"_side{scan_side_m:g}m_step{step_m:g}m"
        f"_knn{int(args.knn)}"
        f"_fps{max(1, int(args.fps))}_{ts}"
    )


def _save_animation_mp4_and_gif(
    ani: mpl_animation.FuncAnimation,
    out_dir: Path,
    base_name: str,
    fps: int,
    *,
    mp4_override: Path | None = None,
    gif_override: Path | None = None,
    dpi: int = 120,
) -> tuple[Path, Path]:
    """Write both MP4 and GIF from one animation, with post-write validation.

    Ported from ``save_animation`` in
    ``examples/controller/drone_dem_betti_adaptive_experiment.py``: both
    formats are always produced from the same ``FuncAnimation`` object, each
    file is checked for existence and non-zero size, and any failure is
    re-raised as ``RuntimeError`` with both expected paths in the message so
    callers can surface a clear diagnostic.

    ``mp4_override`` / ``gif_override`` (when set) replace the auto-generated
    path for that format; they still produce *both* outputs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = (
        Path(mp4_override).resolve() if mp4_override else (out_dir / f"{base_name}.mp4").resolve()
    )
    gif_path = (
        Path(gif_override).resolve() if gif_override else (out_dir / f"{base_name}.gif").resolve()
    )
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # libx264 + yuv420p chroma subsampling requires both pixel
        # dimensions to be even.  matplotlib's constrained-layout can
        # produce an odd width/height at save time (e.g. 1920x1165 on
        # some displays), which makes ffmpeg abort with
        # "height not divisible by 2".  The ``pad=ceil(iw/2)*2:...``
        # video filter appends at most one black pixel on the right /
        # bottom edge so the encoder always sees an even canvas.
        mp4_writer = mpl_animation.FFMpegWriter(
            fps=max(1, int(fps)),
            extra_args=[
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black",
                "-pix_fmt",
                "yuv420p",
            ],
        )
        ani.save(str(mp4_path), writer=mp4_writer, dpi=dpi)
        if not (mp4_path.exists() and mp4_path.stat().st_size > 0):
            raise RuntimeError(f"MP4 writer completed but file is empty: {mp4_path}")

        gif_writer = mpl_animation.PillowWriter(fps=max(1, int(fps)))
        ani.save(str(gif_path), writer=gif_writer, dpi=dpi)
        if not (gif_path.exists() and gif_path.stat().st_size > 0):
            raise RuntimeError(f"GIF writer completed but file is empty: {gif_path}")
        return mp4_path, gif_path
    except Exception as export_err:
        raise RuntimeError(
            "Failed to create required animation outputs (MP4 + GIF). "
            f"Expected files: {mp4_path} and {gif_path}"
        ) from export_err


def run(args: argparse.Namespace) -> None:
    coords, traits, _frame = load(args.data_dir)
    print(f"coords CSV : {len(coords):>7,} rocks")
    print(f"traits CSV : {len(traits):>7,} rocks, columns={list(traits.columns)}")

    if not args.no_impute_missing_traits:
        n_before = len(traits)
        traits = impute_missing_traits(
            coords, traits,
            tol_m=float(args.trait_match_tol_m),
            diameter_m=float(args.fallback_diameter_m),
            eccentricity=_FALLBACK_ECCENTRICITY,
        )
        n_imputed = len(traits) - n_before
        if n_imputed > 0:
            imp_area = _circular_area_m2(args.fallback_diameter_m)
            print(
                f"imputed    : {n_imputed:>7,} rocks as circular "
                f"{args.fallback_diameter_m * 100.0:.1f} cm pebbles "
                f"(area={imp_area:.3e} m², ecc={_FALLBACK_ECCENTRICITY:g})"
            )
        else:
            print("imputed    :       0 rocks (every coord rock already has a trait row)")

    tx = traits["x_m"].to_numpy()
    ty = traits["y_m"].to_numpy()
    area = traits["area_m2"].to_numpy()
    ecc = traits["eccentricity"].to_numpy()
    elev = traits["elevation_rel"].to_numpy() if "elevation_rel" in traits.columns else None
    cx_bg = coords["x_m"].to_numpy()
    cy_bg = coords["y_m"].to_numpy()

    xmin, xmax = float(tx.min()), float(tx.max())
    ymin, ymax = float(ty.min()), float(ty.max())
    print(f"traits bbox : x {xmin:.1f} .. {xmax:.1f} m   "
          f"y {ymin:.1f} .. {ymax:.1f} m   "
          f"(Δx={xmax-xmin:.0f}m, Δy={ymax-ymin:.0f}m)")

    # Camera model (shared with drone-DEM explorer).  ``scan_side_m`` is
    # the edge of the square nadir footprint: rocks are considered "in
    # the scan window" iff both |x - cx| and |y - cy| are within
    # ``scan_side_m / 2``, matching the DEM's ``capture_square`` slicing.
    cam = CameraModel(
        altitude_m=float(args.altitude_m),
        fov_deg=float(args.fov_deg),
        resolution_m=float(args.scarp_resolution_m),
    )
    scan_side_m = float(cam.footprint_side_m)
    scan_half_m = 0.5 * scan_side_m

    # Bishop analog of the drone-DEM ``visited = np.zeros_like(dem, bool)``.
    # A single coverage raster over the traits bbox (plus a one-pixel pad
    # on each side) lets us compute ``unseen = 1 - mean(visited[target])``
    # at any future candidate position via the shared ``CoverageRaster``.
    raster_pad = float(args.scarp_resolution_m)
    visited_raster = CoverageRaster(
        bbox=(xmin - raster_pad, xmax + raster_pad,
              ymin - raster_pad, ymax + raster_pad),
        resolution_m=float(args.scarp_resolution_m),
    )

    # Step = scan_side_m / 2 per axis (matches DEM's ``half = side_px // 2``
    # diagonal step exactly).
    step_m = args.step_m if args.step_m > 0 else 0.5 * scan_side_m
    mission_params_line = (
        f"mission: steps={args.steps} | "
        f"alt={args.altitude_m:g}m | fov={args.fov_deg:g}deg | "
        f"scan_side={scan_side_m:g}m | step_m={step_m:g} | "
        f"raster={args.scarp_resolution_m:g}m/px "
        f"({visited_raster.shape[0]}x{visited_raster.shape[1]}) | "
        f"knn={args.knn} | "
        f"knn_max_edge={args.knn_max_edge_m:g}m | "
        f"fps={args.fps}"
    )

    # All coord rocks are candidate nodes; only nodes with trait rows
    # (and ≥ ``--min-edge-diameter-m``) can receive k-NN edges.  Scan
    # window membership uses a square bbox test (see ``_square_window_mask``).
    coord_xy_stack = np.c_[cx_bg, cy_bg]
    trait_xy_stack = np.c_[tx, ty]
    coord_has_trait = trait_mask_for_coords(coords, traits)
    n_linked = int(np.count_nonzero(coord_has_trait))
    edge_note = (
        "(all edge-eligible)"
        if n_linked == len(coords)
        else "(only these are edge-eligible)"
    )
    print(
        f"trait-linked coord rocks: {n_linked:,} / {len(coords):,} {edge_note}"
    )

    # Optional size filter: rocks whose circular-equivalent diameter is
    # smaller than ``--min-edge-diameter-m`` are kept as nodes (they still
    # count inside the scan window and against β₀/n) but excluded from the
    # k-NN edge set.  Matches the common field-geology question "how does
    # the graph change if we ignore sub-cm pebbles?" without having to
    # re-run imputation with a different fallback diameter.
    min_edge_d = float(args.min_edge_diameter_m)
    if min_edge_d > 0.0:
        coord_diameter_m = coord_diameter_m_for_coords(
            coords, traits, tol_m=float(args.trait_match_tol_m)
        )
        size_eligible = np.where(
            np.isfinite(coord_diameter_m),
            coord_diameter_m >= min_edge_d,
            False,
        )
        coord_edge_eligible = coord_has_trait & size_eligible
        n_dropped = int(np.count_nonzero(coord_has_trait & ~size_eligible))
        n_eligible = int(np.count_nonzero(coord_edge_eligible))
        print(
            f"min-edge-diameter : {min_edge_d * 100.0:.2f} cm -> "
            f"{n_dropped:,} rocks excluded from edges "
            f"(edge-eligible: {n_eligible:,} / {len(coords):,})"
        )
    else:
        coord_edge_eligible = coord_has_trait

    # Optional per-edge length cap: k-NN edges longer than
    # ``--knn-max-edge-m`` are dropped before any topology measurement.
    # Threshold must be positive and finite; 0/negative/NaN/None disables.
    # A rock may end up with fewer than ``--knn`` neighbours once the cap
    # is in effect, and can even become isolated (β₀ goes up, β₁ stays
    # pinned to 0 for that node).  This is useful when you want k-NN's
    # adaptive density behaviour but also a hard "don't link rocks farther
    # apart than D metres" rule — for example to avoid long cross-scarp
    # edges that are geometrically spurious.
    knn_max_edge_raw = float(args.knn_max_edge_m)
    knn_max_edge_m: float | None = (
        knn_max_edge_raw
        if np.isfinite(knn_max_edge_raw) and knn_max_edge_raw > 0.0
        else None
    )
    if knn_max_edge_m is not None:
        print(f"knn-max-edge     : {knn_max_edge_m:g} m "
              f"(edges longer than this are dropped from the k-NN graph)")

    # Path is built online (the quadrant-Betti planner chooses the next
    # centre from the 4 diagonal rock quadrants).
    path_xy: list[tuple[float, float]] = []
    seed_xy = (
        float(args.seed_x) if args.seed_x is not None else 0.5 * (xmin + xmax),
        float(args.seed_y) if args.seed_y is not None else 0.5 * (ymin + ymax),
    )

    # ------------------------------------------------------------------
    # Figure layout — mirrors the 2x3 live panel from the DEM explorer.
    # ------------------------------------------------------------------
    if args.show:
        plt.ion()
    fig, ax = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    ax00, ax01, ax02 = ax[0, 0], ax[0, 1], ax[0, 2]
    ax10, ax11, ax12 = ax[1, 0], ax[1, 1], ax[1, 2]

    # (0,0) Full map
    ax00.scatter(cx_bg, cy_bg, s=0.8, color="lightgray", alpha=0.35, linewidths=0)
    if elev is not None:
        elev_arr = np.asarray(elev, dtype=float)
        finite_elev = elev_arr[np.isfinite(elev_arr)]
        if finite_elev.size:
            vmin = float(np.nanpercentile(finite_elev, 2.0))
            vmax = float(np.nanpercentile(finite_elev, 98.0))
            if vmax <= vmin:
                vmin = float(np.nanmin(finite_elev))
                vmax = float(np.nanmax(finite_elev) + 1e-6)
        else:
            vmin, vmax = 0.0, 1.0
        color_values = elev_arr
        color_norm = Normalize(vmin=vmin, vmax=vmax)
        color_label = "relative elevation"
        title_suffix = "elevation"
    else:
        # Backward-compatible fallback for older trait exports.
        area_clip = np.clip(area, max(area.min(), 0.05), area.max())
        color_values = area_clip
        color_norm = Normalize(vmin=float(np.nanmin(area_clip)),
                               vmax=float(np.nanmax(area_clip)))
        color_label = "area (m²)"
        title_suffix = "area"
    sc_all = ax00.scatter(
        tx, ty,
        c=color_values, cmap="viridis",
        norm=color_norm,
        s=np.clip(area * 1.2, 1.5, 40),
        linewidths=0, alpha=0.85,
    )
    fig.colorbar(sc_all, ax=ax00, shrink=0.8, pad=0.02, label=color_label)
    # Scan footprint artist: matches the DEM explorer's square ``fov_rect``.
    # ``xy`` is the lower-left corner in data coords.
    window = Rectangle(
        xy=(seed_xy[0] - scan_half_m, seed_xy[1] - scan_half_m),
        width=scan_side_m, height=scan_side_m,
        ec="crimson", fc="none", lw=1.6, alpha=0.9,
    )
    ax00.add_patch(window)
    path_line, = ax00.plot([], [], "-", color="crimson", lw=0.8, alpha=0.8)
    # Yellow arrow showing chosen next direction.
    dir_arrow, = ax00.plot([], [], "-", color="gold", lw=2.0, alpha=0.9)
    dir_head = ax00.scatter([], [], marker=(3, 0, 0), s=60, color="gold")
    ax00.set_aspect("equal")
    ax00.set_xlabel("x (m, local)")
    ax00.set_ylabel("y (m, local)")
    ax00.set_title(f"Bishop scarp — rocks colored by {title_suffix}\n"
                   "(red: scan window, gold: next move)")
    ax00.grid(alpha=0.3)

    # (0,1) Local graph
    ax01.set_aspect("equal")
    ax01.set_xlabel("x (m, local)")
    ax01.set_ylabel("y (m, local)")
    ax01.set_title(
        f"Local k-NN graph — all rocks "
        f"(square {scan_side_m:g} × {scan_side_m:g} m)"
    )
    ax01.grid(alpha=0.3)
    lc_knn = LineCollection([], colors="cyan", linewidths=1.0, alpha=0.9)
    ax01.add_collection(lc_knn)
    local_nodes = ax01.scatter([], [], s=[], c=[], cmap="tab20", linewidths=0)
    ax01.legend(
        handles=[
            Line2D([0], [0], color="cyan", lw=1.2,
                   label=f"k-NN edge  (k={args.knn})"),
            Line2D([0], [0], marker="o", color="k", lw=0, markersize=6,
                   label="node = any rock (color = component)"),
        ],
        loc="lower left", fontsize=8, framealpha=0.85,
    )
    # Panel background so white edges are visible.
    ax01.set_facecolor("#1f1f1f")

    # (0,2) local area histogram
    ax02.set_title("Area histogram (rocks inside window)")
    ax02.set_xlabel("area (m²)")
    ax02.set_ylabel("count")
    ax02.set_yscale("log")
    ax02.grid(alpha=0.3)
    # Floor chosen so imputed 2 cm pebbles (~3.14×10⁻⁴ m²) fall inside the
    # geomspace range rather than being silently clipped out of the histogram.
    hist_bins = np.geomspace(max(float(np.nanmin(area)), 1e-6), float(np.nanmax(area)), 40)
    (hist_bar,) = ax02.plot([], [], drawstyle="steps-mid", color="steelblue", lw=1.2)

    # (1,0) topology history
    ax10.set_title("Topology per step (k-NN graph)")
    ax10.set_xlabel("step")
    ax10.grid(alpha=0.3)
    nodes_line, = ax10.plot([], [], label="n_nodes", color="tab:blue")
    comp_line, = ax10.plot([], [], label="n_components", color="tab:red")
    fied_line, = ax10.plot([], [], label="Fiedler × 10", color="tab:brown")
    ax10.legend(loc="upper left", fontsize=8)

    # (1,1) trait stats
    ax11.set_title("Trait medians per step (inside window)")
    ax11.set_xlabel("step")
    ax11.grid(alpha=0.3)
    med_area_line, = ax11.plot([], [], label="median area_m²", color="tab:green")
    med_ecc_line, = ax11.plot([], [], label="median eccentricity × 50",
                              color="tab:purple")
    ax11.legend(loc="upper left", fontsize=8)

    # (1,2) cumulative explored: ecc vs area colored by step
    ax12.set_title("Cumulative explored rocks (ecc vs area)")
    ax12.set_xlabel("area (m²)")
    ax12.set_ylabel("eccentricity")
    ax12.set_xscale("log")
    ax12.grid(alpha=0.3)
    cum_scat = ax12.scatter([], [], c=[], s=8, cmap="plasma", linewidths=0)
    cbar_cum = fig.colorbar(cum_scat, ax=ax12, shrink=0.8, pad=0.02, label="step")

    # ------------------------------------------------------------------
    # Step loop  (online: plan next center from directional candidates)
    # ------------------------------------------------------------------
    records: list[StepRecord] = []
    seen_nodes: set[int] = set()         # coord indices — all rocks seen so far
    seen_trait_nodes: set[int] = set()   # trait indices — for cumulative scatter
    next_center: list[tuple[float, float]] = [seed_xy]  # mutable across frames

    # Update ax10 (topology history) to include β₁.
    beta1_line, = ax10.plot([], [], label="β₁ (cycles)", color="tab:orange")
    ax10.legend(loc="upper left", fontsize=8)

    def update_step(step: int) -> Iterable:
        # Frame-0 reset makes update_step idempotent across repeated
        # FuncAnimation passes (needed to save MP4 and GIF from the same
        # animation object without corrupting cumulative sim state).
        if int(step) == 0:
            records.clear()
            path_xy.clear()
            seen_nodes.clear()
            seen_trait_nodes.clear()
            next_center[0] = seed_xy

        cx, cy = next_center[0]
        path_xy_snapshot = path_xy + [(cx, cy)]

        # Square scan footprint (matches DEM's ``capture_square``).  A rock
        # is inside iff both |x - cx| and |y - cy| are within scan_half_m.
        coord_in_sq = _square_window_mask(coord_xy_stack, cx, cy, scan_side_m)
        coord_idx = np.nonzero(coord_in_sq)[0]
        n_local = int(coord_idx.size)
        local_xy = np.c_[cx_bg[coord_idx], cy_bg[coord_idx]] if n_local else np.empty((0, 2))

        # Trait rocks inside the same square window — for stats / histogram.
        trait_in_sq = _square_window_mask(trait_xy_stack, cx, cy, scan_side_m)
        trait_idx = np.nonzero(trait_in_sq)[0]
        n_traits_local = int(trait_idx.size)

        # Paint the visited raster *now*, before scoring the candidates, so
        # the current footprint counts toward the unseen fraction at every
        # future target.  Matches the DEM explorer which writes
        # ``visited[r0:r1, c0:c1] = True`` after each capture.
        visited_raster.mark_square(cx, cy, scan_side_m)

        local_has_trait = coord_edge_eligible[coord_idx] if n_local else np.empty((0,), dtype=bool)
        if n_local >= 2:
            e_knn = knn_edges_trait_only(
                local_xy,
                args.knn,
                local_has_trait,
                max_edge_m=knn_max_edge_m,
            )
        else:
            e_knn = set()

        A_knn = adjacency_from_edges(max(n_local, 1), e_knn)
        if n_local > 0:
            # Always run connected-components, even for empty-edge graphs.
            # (If there are N isolated nodes, beta0 must be N, not 1.)
            n_comp, labels = connected_components(A_knn, directed=False)
        else:
            n_comp, labels = (0, np.empty((0,), dtype=int))
        fied_full = fiedler_value(A_knn) if n_local >= 3 else 0.0
        n_edges_full = int(A_knn.nnz // 2)
        beta1_full = max(0, n_edges_full - n_local + int(n_comp))

        # ---- move selection (quadrant-Betti, shared with DEM) -------------
        # Split the current window into 4 diagonal rock quadrants, score
        # each by Betti topology + area-based unseen fraction, and move to
        # the outer corner of the winner.  Scoring, revisit penalty, and
        # cyclic tie-break all live in ``kernelcal.graph_explorer`` so this
        # script and the DEM explorer stay in lock-step.
        q_cands = build_betti_quadrant_candidates(
            cx, cy,
            scan_side_m=scan_side_m,
            step_m=float(step_m),
            coord_xy=coord_xy_stack,
            coord_has_trait=coord_edge_eligible,
            knn=int(args.knn),
            visited_raster=visited_raster,
            bbox=(xmin, xmax, ymin, ymax),
            knn_max_edge_m=knn_max_edge_m,
        )
        weights = BettiWeights(
            w_beta1=float(args.w_beta1),
            w_beta0=float(args.w_beta0),
            w_unseen=float(args.w_unseen),
            revisit_penalty=float(args.revisit_penalty),
        )
        # Past visit centres keyed the same way as each candidate's
        # ``position`` (rounded to 1 cm) so the revisit penalty fires on a
        # genuine revisit of a prior target.
        recent_pos = (
            [(round(r.cx, 2), round(r.cy, 2)) for r in records[-20:]]
            if records else []
        )
        best_cand, best_score_q, _scored_q = choose_best_candidate(
            q_cands, weights,
            recent_positions=recent_pos,
            tie_break_index=len(records),
        )

        if best_cand is not None:
            best_name = best_cand.name
            best_score = float(best_score_q)
            target_xy = (
                float(best_cand.extra["target_xy"][0]),
                float(best_cand.extra["target_xy"][1]),
            )
            new_cx = float(np.clip(target_xy[0], xmin, xmax))
            new_cy = float(np.clip(target_xy[1], ymin, ymax))
        else:
            # Empty window: drift one step toward the bbox centre so we
            # escape voids instead of stalling.  Matches DEM's bbox-clamp
            # fallback when the candidate list is empty.
            best_name = "STAY"
            best_score = 0.0
            tgt = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)])
            vec = tgt - np.array([cx, cy])
            nrm = float(np.linalg.norm(vec))
            dirv = (vec / nrm) if nrm > 1e-6 else np.array([1.0, 0.0])
            new_cx = float(np.clip(cx + step_m * dirv[0], xmin, xmax))
            new_cy = float(np.clip(cy + step_m * dirv[1], ymin, ymax))
        next_center[0] = (new_cx, new_cy)

        med_area = float(np.median(area[trait_idx])) if n_traits_local else float("nan")
        med_ecc = float(np.median(ecc[trait_idx])) if n_traits_local else float("nan")

        rec = StepRecord(
            step=step, cx=cx, cy=cy,
            n_nodes=n_local,
            n_edges_knn=len(e_knn),
            n_components=int(n_comp),
            beta1=int(beta1_full),
            fiedler=float(fied_full),
            median_area=med_area,
            median_ecc=med_ecc,
            chosen_direction=best_name,
            chosen_score=float(best_score if np.isfinite(best_score) else 0.0),
        )
        if records and records[-1].step == step:
            records[-1] = rec          # FuncAnimation can call frame 0 twice
        else:
            records.append(rec)
            if len(path_xy) < step + 1:
                path_xy.append((cx, cy))
        seen_nodes.update(int(i) for i in coord_idx)
        seen_trait_nodes.update(int(i) for i in trait_idx)

        # ------------------------- update artists --------------------------
        # Rectangle uses lower-left-corner anchor, not centre.
        window.set_xy((cx - scan_half_m, cy - scan_half_m))
        # Path history trail.
        pxy = np.asarray(path_xy_snapshot, dtype=float)
        path_line.set_data(pxy[:, 0], pxy[:, 1])

        # Chosen direction arrow — point to the actual next center so the
        # arrow matches the motion (and, for betti-quadrant, the exact
        # target used for scoring).
        arrow_tip = (new_cx, new_cy)
        dir_arrow.set_data([cx, arrow_tip[0]], [cy, arrow_tip[1]])
        dir_head.set_offsets(np.array([[arrow_tip[0], arrow_tip[1]]]))

        # Local graph panel.
        knn_segs = [np.array([local_xy[a], local_xy[b]]) for a, b in e_knn]
        lc_knn.set_segments(knn_segs)

        node_sizes = np.full(n_local, 8.0) if n_local else np.array([])
        local_nodes.set_offsets(local_xy if n_local else np.empty((0, 2)))
        local_nodes.set_sizes(node_sizes)
        local_nodes.set_array(labels.astype(float) if n_local else np.array([]))

        # Local-graph panel axis limits: show the full square footprint
        # plus a 15 % margin so edges near the perimeter aren't clipped.
        pad = 0.15 * scan_half_m
        lim = scan_half_m + pad
        ax01.set_xlim(cx - lim, cx + lim)
        ax01.set_ylim(cy - lim, cy + lim)

        # Area histogram panel — trait rocks in FoV only.
        if n_traits_local:
            hist, edges = np.histogram(area[trait_idx], bins=hist_bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            hist_bar.set_data(centers, np.maximum(hist, 1e-1))
            ax02.set_xlim(hist_bins[0], hist_bins[-1])
            ax02.set_xscale("log")
            ax02.set_ylim(0.8, max(10, hist.max() * 1.5))

        # Topology history.
        xs = [r.step for r in records]
        nodes_line.set_data(xs, [r.n_nodes for r in records])
        comp_line.set_data(xs, [r.n_components for r in records])
        fied_line.set_data(xs, [10.0 * r.fiedler for r in records])
        beta1_line.set_data(xs, [r.beta1 for r in records])
        ax10.relim(); ax10.autoscale_view()

        # Trait stats.
        med_area_line.set_data(xs, [r.median_area for r in records])
        med_ecc_line.set_data(xs, [50.0 * r.median_ecc for r in records])
        ax11.relim(); ax11.autoscale_view()

        # Cumulative explored scatter — trait rocks only (have area + ecc).
        if seen_trait_nodes:
            sn = np.fromiter(seen_trait_nodes, dtype=int)
            a_seen = area[sn]
            e_seen = ecc[sn]
            cum_scat.set_offsets(np.c_[a_seen, e_seen])
            cum_scat.set_array(np.arange(len(sn)))
            cbar_cum.mappable.set_clim(0, max(1, len(sn)))
            ax12.set_xlim(max(float(np.nanmin(a_seen)), 1e-6) * 0.8, float(np.nanmax(a_seen)) * 1.2)
            ax12.set_ylim(-0.02, max(1.02, float(np.nanmax(e_seen)) + 0.02))

        fig.suptitle(
            f"Bishop rocks explorer — step {step+1}/{args.steps} | "
            f"window=({cx:.1f}, {cy:.1f}) m | graph_nodes={n_local} | traits={n_traits_local} | "
            f"β₀={n_comp} | β₁={beta1_full} | λ₂={fied_full:.3g} | →{best_name} (score={best_score:.2f})\n"
            f"{mission_params_line}",
            fontsize=11,
        )

        return (
            path_line, window, dir_arrow, dir_head,
            lc_knn, local_nodes,
            hist_bar,
            nodes_line, comp_line, fied_line, beta1_line,
            med_area_line, med_ecc_line, cum_scat,
        )

    if args.show:
        # Live interactive mode runs update_step once per step with plt.pause;
        # state is cleared on frame 0 inside update_step so a subsequent
        # animation save starts from a clean slate.
        for s in range(args.steps):
            update_step(s)
            fig.canvas.draw_idle()
            plt.pause(args.pause_s)
        plt.ioff()

    anim_mp4: Path | None = None
    anim_gif: Path | None = None
    if not args.no_animation:
        ani = mpl_animation.FuncAnimation(
            fig, update_step, frames=args.steps,
            interval=max(20, int(args.pause_s * 1000)),
            blit=False, repeat=False,
        )
        base_name = _build_animation_base_name(args, step_m=step_m, scan_side_m=scan_side_m)
        mp4_override = Path(args.save_mp4) if args.save_mp4 else None
        gif_override = Path(args.save_gif) if args.save_gif else None
        anim_mp4, anim_gif = _save_animation_mp4_and_gif(
            ani,
            out_dir=args.out,
            base_name=base_name,
            fps=max(1, int(args.fps)),
            mp4_override=mp4_override,
            gif_override=gif_override,
            dpi=max(50, int(args.animation_dpi)),
        )
        print(f"wrote MP4  : {anim_mp4}")
        print(f"wrote GIF  : {anim_gif}")
    elif not args.show:
        # No live display and no animation: still need to advance the sim so
        # the final figure (saved below) reflects the full trajectory.
        for s in range(args.steps):
            update_step(s)

    args.out.mkdir(parents=True, exist_ok=True)
    final_png = args.out / "bishop_rocks_explorer_final.png"
    fig.savefig(final_png, dpi=180)
    print(f"wrote final: {final_png}")

    # Summary CSV of per-step records
    summary = pd.DataFrame(
        dict(
            step=[r.step for r in records],
            cx_m=[r.cx for r in records],
            cy_m=[r.cy for r in records],
            n_nodes=[r.n_nodes for r in records],
            n_edges_knn=[r.n_edges_knn for r in records],
            beta0_components=[r.n_components for r in records],
            beta1_cycles=[r.beta1 for r in records],
            fiedler=[r.fiedler for r in records],
            median_area_m2=[r.median_area for r in records],
            median_eccentricity=[r.median_ecc for r in records],
            chosen_direction=[r.chosen_direction for r in records],
            chosen_score=[r.chosen_score for r in records],
        )
    )
    summary_csv = args.out / "bishop_rocks_explorer_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"wrote summary: {summary_csv}  ({len(summary)} steps, "
          f"{len(seen_nodes):,} rocks explored)")

    if args.show:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help=f"Folder with the two CSVs (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Output folder (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--steps", type=int, default=80,
                   help="Number of explorer steps (default: 80)")
    # Camera model (shared with the drone-DEM explorer; see
    # ``kernelcal.graph_explorer.CameraModel``).  Square nadir footprint
    # side = 2·altitude·tan(fov/2).  Defaults give ~71.5 m side (≈ prior
    # 40 m disk radius) so existing runs keep the same ground coverage.
    p.add_argument("--altitude-m", type=float, default=30.0,
                   help="Drone altitude above the scarp in metres "
                        "(default: 30). Feeds the shared CameraModel: "
                        "scan footprint side = 2·altitude·tan(fov/2).")
    p.add_argument("--fov-deg", type=float, default=100.0,
                   help="Camera field-of-view angle in degrees (default: 100). "
                        "Same knob as the drone-DEM explorer.")
    p.add_argument("--scarp-resolution-m", type=float, default=0.5,
                   help="Pixel size (metres) of the coverage raster used to "
                        "compute unseen_frac with the same "
                        "`1 - mean(visited[target_patch])` rule as the "
                        "drone-DEM explorer (default: 0.5 m).")
    p.add_argument("--step-m", type=float, default=0.0,
                   help="Move step size in metres (default: scan_side_m / 2, "
                        "matching the drone-DEM explorer's "
                        "`half = side_px // 2` diagonal step).")
    p.add_argument(
        "--knn-max-edge-m",
        type=float,
        default=0.0,
        help=(
            "Maximum Euclidean edge length (metres) in the k-NN graph. "
            "Edges longer than this threshold are dropped, so a rock may "
            "end up with fewer than --knn neighbours (or none). Default 0 "
            "= no cap (pure k-NN). Example: --knn-max-edge-m 5.0 keeps "
            "k-NN adaptivity but forbids cross-scarp links longer than "
            "5 m."
        ),
    )
    p.add_argument("--knn", type=int, default=6,
                   help="k for k-NN graph (default: 6)")
    p.add_argument(
        "--fallback-diameter-m",
        type=float,
        default=_FALLBACK_DIAMETER_M,
        help=(
            "Diameter (metres) assigned to coord rocks that have no matching "
            "row in rock_traits_full.csv. Such rocks are treated as circular "
            "pebbles (eccentricity=0, area=π·(d/2)²). Default: 0.02 m = 2 cm."
        ),
    )
    p.add_argument(
        "--trait-match-tol-m",
        type=float,
        default=1.0,
        help=(
            "Nearest-neighbour tolerance (metres) for matching a coord rock "
            "to a row in rock_traits_full.csv. Coord rocks farther than this "
            "from any trait row are treated as missing-trait."
        ),
    )
    p.add_argument(
        "--no-impute-missing-traits",
        action="store_true",
        help=(
            "Disable imputation of missing-trait rocks as 2 cm round pebbles. "
            "Reverts to the pre-imputation behaviour where only measured "
            "rocks contribute edges to the k-NN graph."
        ),
    )
    p.add_argument(
        "--min-edge-diameter-m",
        type=float,
        default=0.0,
        help=(
            "Minimum circular-equivalent diameter (metres) for a rock to be "
            "eligible for k-NN edges.  Rocks below this threshold stay as "
            "nodes (they still count in the scan window and in β₀/n) but "
            "are excluded from the edge set, so they cannot close cycles. "
            "Default 0 = no filter.  Example: --min-edge-diameter-m 0.05 "
            "links only rocks >= 5 cm, dropping imputed 2 cm pebbles from "
            "the graph topology without re-running imputation."
        ),
    )
    p.add_argument("--seed-x", type=float, default=None,
                   help="Starting x in metres (default: bbox centre)")
    p.add_argument("--seed-y", type=float, default=None,
                   help="Starting y in metres (default: bbox centre)")
    # Defaults match the shared kernelcal.graph_explorer.BettiWeights — same
    # weights as the drone-DEM explorer's --w-beta1 / --w-beta0 / --w-unseen.
    p.add_argument("--w-beta1", type=float, default=2.5,
                   help="Weight on clip(β₁/n, 0, 1) (cycle density) in the "
                        "quadrant-Betti score.")
    p.add_argument("--w-beta0", type=float, default=0.5,
                   help="Penalty weight on clip(β₀/n, 0, 1) (fragmentation).")
    p.add_argument("--w-unseen", type=float, default=5.0,
                   help="Weight on unseen-area fraction at the candidate target.")
    p.add_argument("--revisit-penalty", type=float, default=0.5,
                   help="Score penalty applied to candidates whose target "
                        "position matches one of the last 20 visited centres.")
    p.add_argument("--pause-s", type=float, default=0.08,
                   help="Seconds to pause between frames when --show")
    p.add_argument("--show", action="store_true",
                   help="Open a live interactive window (uses plt.ion)")
    p.add_argument(
        "--save-mp4",
        type=str,
        default="",
        help=(
            "Explicit MP4 output path (overrides the auto-generated filename "
            "for the MP4 output only). Both MP4 and GIF are still written."
        ),
    )
    p.add_argument(
        "--save-gif",
        type=str,
        default="",
        help=(
            "Explicit GIF output path (overrides the auto-generated filename "
            "for the GIF output only). Both MP4 and GIF are still written."
        ),
    )
    p.add_argument("--fps", type=int, default=12, help="FPS for saved animation")
    p.add_argument(
        "--animation-dpi",
        type=int,
        default=120,
        help="Render DPI for MP4/GIF frames (parity with DEM explorer).",
    )
    p.add_argument(
        "--no-animation",
        action="store_true",
        help=(
            "Skip MP4/GIF export. By default the explorer writes both formats "
            "to --out with an auto-generated timestamped filename encoding "
            "steps/scan_side/step/knn/fps."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()

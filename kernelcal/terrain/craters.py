"""Crater graph analysis for planetary terrain biosignature detection.

Craters are the dominant topological features on the Moon and Mars.
They produce well-defined β₁ (rim loops) and β₂ (enclosed voids) that
constitute the abiotic topological null model against which any biosignature
Δβ₁ must be measured.

This module:
  1. Detects crater rims from DEM curvature / ring-fit heuristics.
  2. Builds a crater-rim graph (rim pixels as nodes, adjacency as edges).
  3. Computes Betti numbers for crater topology.
  4. Provides the abiotic null model β₁^abio and β₂^abio for a given
     crater population (set of circular rim features).
  5. Computes the spectral entropy and Fiedler value diagnostics on
     the crater graph for phase-transition early-warning.

Physical context (P2, §6.1):
  - β₂ = number of enclosed crater voids (one per crater interior)
  - β₁ = number of independent rim loops (one per closed rim)
  - kmin = β₀ + β₁ (topologically obligate mode count, Theorem 1 of P2)

Abiotic null model:
  - An impact crater with a complete rim gives β₁ = 1 (the rim loop)
    and β₂ = 1 (the enclosed floor void) per crater.
  - A degraded crater (incomplete rim) gives β₁ = 0, β₂ = 0.
  - Therefore β₁^abio = n_intact_craters for a field of n craters.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

from .dem import TerrainGraph, dem_to_graph, terrain_graph_laplacian, curvature_planform


# ---------------------------------------------------------------------------
# Crater detection — ring Hough transform on curvature image
# ---------------------------------------------------------------------------

@dataclass
class CraterCandidate:
    """A detected crater candidate."""
    row: int
    col: int
    radius: float          # in pixels
    rim_completeness: float  # 0–1 (1 = fully closed rim)
    curvature_contrast: float  # rim curvature vs surroundings


def detect_craters(
    dem: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    min_radius: float = 3.0,
    max_radius: float = 20.0,
    n_radii: int = 10,
    curvature_threshold: float = 0.1,
    completeness_threshold: float = 0.6,
) -> list[CraterCandidate]:
    """Detect crater rims from planform curvature using circular Hough transform.

    Crater rims appear as rings of high negative-then-positive curvature
    (concave interior → convex rim crest → flat exterior).  We accumulate
    votes in a Hough space (row, col, radius) and return candidates above
    threshold.

    Parameters
    ----------
    dem                    : (nrows, ncols) elevation array
    dx, dy                 : cell spacings
    min_radius, max_radius : search range in pixels
    n_radii                : number of radius steps to test
    curvature_threshold    : minimum |curvature| to vote
    completeness_threshold : minimum fraction of rim arc that must be present

    Returns
    -------
    List of CraterCandidate objects sorted by radius descending.
    """
    z   = np.asarray(dem, dtype=float)
    nrows, ncols = z.shape
    curv = curvature_planform(z, dx=dx, dy=dy)

    # Rim pixels: high absolute curvature
    rim_mask = np.abs(curv) > curvature_threshold

    radii = np.linspace(min_radius, max_radius, n_radii)
    # Hough accumulator
    accumulator = np.zeros((nrows, ncols, n_radii), dtype=np.int32)

    rim_rows, rim_cols = np.where(rim_mask)
    for pr, pc in zip(rim_rows, rim_cols):
        for ri, rad in enumerate(radii):
            # Vote for all circle centres that this rim pixel could belong to
            for theta in np.linspace(0, 2 * np.pi, 36, endpoint=False):
                cr = int(round(pr - rad * np.sin(theta)))
                cc = int(round(pc - rad * np.cos(theta)))
                if 0 <= cr < nrows and 0 <= cc < ncols:
                    accumulator[cr, cc, ri] += 1

    # Peak detection: simple local maxima above threshold
    max_votes = int(2 * np.pi * np.max(radii) * 0.5)  # half-circumference as baseline
    candidates: list[CraterCandidate] = []
    used = np.zeros((nrows, ncols), dtype=bool)

    for ri, rad in enumerate(radii):
        acc_slice = accumulator[:, :, ri]
        # Non-maximum suppression at scale rad
        for cr in range(nrows):
            for cc in range(ncols):
                votes = acc_slice[cr, cc]
                min_votes = int(completeness_threshold * 2 * np.pi * rad)
                if votes >= max(min_votes, 4):
                    # Check not already claimed by a larger crater
                    r0, r1 = max(0, cr - int(rad)), min(nrows, cr + int(rad) + 1)
                    c0, c1 = max(0, cc - int(rad)), min(ncols, cc + int(rad) + 1)
                    if not used[r0:r1, c0:c1].any():
                        completeness = votes / (2 * np.pi * rad)
                        # Curvature contrast at rim vs background
                        rows_rim = np.array([cr + int(rad * np.sin(t))
                                             for t in np.linspace(0, 2*np.pi, 36)
                                             if 0 <= cr + int(rad * np.sin(t)) < nrows])
                        cols_rim = np.array([cc + int(rad * np.cos(t))
                                             for t in np.linspace(0, 2*np.pi, 36)
                                             if 0 <= cc + int(rad * np.cos(t)) < ncols])
                        if len(rows_rim) > 0:
                            rim_curv = float(np.mean(np.abs(curv[rows_rim[:len(cols_rim)],
                                                                  cols_rim[:len(rows_rim)]])))
                        else:
                            rim_curv = 0.0

                        candidates.append(CraterCandidate(
                            row=cr, col=cc, radius=float(rad),
                            rim_completeness=min(completeness, 1.0),
                            curvature_contrast=rim_curv,
                        ))
                        used[r0:r1, c0:c1] = True

    return sorted(candidates, key=lambda c: c.radius, reverse=True)


# ---------------------------------------------------------------------------
# Crater-rim mask and graph
# ---------------------------------------------------------------------------

def crater_rim_mask(
    shape: tuple[int, int],
    craters: list[CraterCandidate],
    rim_width: int = 2,
) -> np.ndarray:
    """Boolean mask marking rim pixels for a list of craters.

    Parameters
    ----------
    shape     : (nrows, ncols) DEM shape
    craters   : detected crater candidates
    rim_width : half-width of rim band in pixels

    Returns
    -------
    (nrows, ncols) bool array — True at rim pixels
    """
    nrows, ncols = shape
    mask = np.zeros((nrows, ncols), dtype=bool)
    rows, cols = np.mgrid[0:nrows, 0:ncols]
    for c in craters:
        d = np.hypot(rows - c.row, cols - c.col)
        mask |= (np.abs(d - c.radius) <= rim_width)
    return mask


def crater_rim_graph(
    dem: np.ndarray,
    craters: list[CraterCandidate],
    rim_width: int = 2,
    connectivity: int = 8,
) -> TerrainGraph:
    """Build a terrain graph restricted to crater rim pixels.

    Each crater rim forms a (approximately) closed loop — a topological
    1-cycle with β₁ contribution = 1 per complete rim.

    Parameters
    ----------
    dem         : (nrows, ncols) elevation array
    craters     : detected or manually specified craters
    rim_width   : width of rim band to include
    connectivity: 4 or 8 pixel connectivity

    Returns
    -------
    TerrainGraph restricted to rim pixels
    """
    mask = crater_rim_mask(dem.shape, craters, rim_width)
    return dem_to_graph(dem, connectivity=connectivity, weight="elev_diff", mask=mask)


# ---------------------------------------------------------------------------
# Betti numbers for crater graphs
# ---------------------------------------------------------------------------

def crater_betti_numbers(tg: TerrainGraph) -> dict[str, int]:
    """Compute Betti numbers (β₀, β₁) for a crater-rim graph.

    Uses the Euler-characteristic relation on the graph:
        β₀ = #components
        β₁ = #edges - #nodes + β₀   (for a graph without 2-simplices)

    These are the GRAPH Betti numbers, not the full Hodge complex.
    For closed rim loops: β₁ = 1 per complete loop.

    Returns
    -------
    dict with keys 'beta0', 'beta1', 'euler_characteristic', 'n_nodes', 'n_edges'
    """
    n_nodes = len(tg.elevations)
    n_edges = len(tg.edges)

    if n_nodes == 0:
        return {"beta0": 0, "beta1": 0, "euler_characteristic": 0,
                "n_nodes": 0, "n_edges": 0}

    # β₀ via union-find
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[ry] = rx
        return True

    for i, j in tg.edges:
        union(int(i), int(j))

    beta0 = len(set(find(i) for i in range(n_nodes)))
    beta1 = n_edges - n_nodes + beta0
    chi   = n_nodes - n_edges  # = β₀ - β₁ for a graph

    return {
        "beta0": beta0,
        "beta1": max(0, beta1),
        "euler_characteristic": chi,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
    }


# ---------------------------------------------------------------------------
# Abiotic null model for impact craters
# ---------------------------------------------------------------------------

def abiotic_beta1_craters(
    n_intact: int,
    n_degraded: int = 0,
) -> dict[str, int]:
    """Abiotic β₁ prediction for a crater field.

    Physical model:
      - Each intact crater (complete rim) contributes β₁ = 1 (the rim loop)
        and β₂ = 1 (the enclosed void — handled separately in Hodge complex).
      - Each degraded crater (incomplete rim, <50% arc) contributes β₁ = 0.
      - Multiple overlapping craters can create additional complex loops, but
        the simple lower bound is n_intact.

    Parameters
    ----------
    n_intact   : number of craters with complete or near-complete rims
    n_degraded : number of degraded craters with incomplete rims

    Returns
    -------
    dict with 'beta1_abio', 'beta2_abio', 'kmin_abio'
    """
    beta1_abio = n_intact   # one loop per complete rim
    beta2_abio = n_intact   # one enclosed void per crater (needs Hodge L2)
    beta0_abio = 1          # terrain is connected
    kmin_abio  = beta0_abio + beta1_abio

    return {
        "beta1_abio": beta1_abio,
        "beta2_abio": beta2_abio,
        "beta0_abio": beta0_abio,
        "kmin_abio":  kmin_abio,
        "n_intact":   n_intact,
        "n_degraded": n_degraded,
    }


# ---------------------------------------------------------------------------
# Spectral signature of crater graph
# ---------------------------------------------------------------------------

def crater_spectral_signature(
    tg: TerrainGraph,
    n_modes: int = 20,
) -> dict[str, float | np.ndarray]:
    """Compute spectral kernel diagnostics on a crater-rim graph.

    Returns the Fiedler value, spectral entropy of the uniform kernel,
    and the first n_modes eigenvalues.

    Parameters
    ----------
    tg      : TerrainGraph (typically crater-rim restricted)
    n_modes : number of eigenvalues to compute

    Returns
    -------
    dict with 'fiedler', 'spectral_entropy', 'eigenvalues', 'n_nodes',
              'beta0', 'beta1'
    """
    n = len(tg.elevations)
    if n < 3:
        return {"fiedler": 0.0, "spectral_entropy": 0.0,
                "eigenvalues": np.array([]), "n_nodes": n,
                "beta0": 0, "beta1": 0}

    L = terrain_graph_laplacian(tg)
    k = min(n_modes, n)
    eigvals = np.linalg.eigvalsh(L)[:k]
    eigvals = np.maximum(eigvals, 0.0)

    fiedler = float(eigvals[1]) if len(eigvals) > 1 else 0.0

    # Spectral entropy of uniform h = 1/N on eigenvalues
    pos = eigvals[eigvals > 1e-10]
    h_bar = pos / pos.sum() if pos.sum() > 0 else np.ones(len(pos)) / len(pos)
    h_entropy = float(-np.sum(h_bar * np.log(h_bar + 1e-12)))

    betti = crater_betti_numbers(tg)

    return {
        "fiedler":        fiedler,
        "spectral_entropy": h_entropy,
        "eigenvalues":    eigvals,
        "n_nodes":        n,
        "beta0":          betti["beta0"],
        "beta1":          betti["beta1"],
    }

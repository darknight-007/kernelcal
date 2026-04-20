"""
kernelcal.integrations.rivgraph
===============================
Strategy A: georeferenced channel mask -> RivGraph skeleton and network ->
kernelcal spectral diagnostics on a graph Laplacian.

This module is the canonical home for the RivGraph bridge. The legacy
``rivgraph_kernelcal_bridge`` module at the repo root is now a thin
back-compat shim that forwards every attribute lookup here.

**Laplacian modes** (``--laplacian-mode``):

- ``skeleton`` — **Least structural assumption:** nodes are **skeleton pixels**
  (``Iskel`` from the mask), edges are **uniform** 4- or 8-neighbour links
  between adjacent skeleton cells only. No RivGraph link/node or junction
  semantics. Implemented with ``kernelcal.terrain.dem.dem_to_graph`` on a
  zero DEM with ``mask=Iskel``. Dense eigen-solve ⇒ keep
  ``#skeleton_pixels <= --laplacian-max-nodes`` (crop or coarsen the mask if
  needed; full braided tiles are often too large).

- ``rivgraph`` — **RivGraph-native graph:** ``delta_metrics.graphiphy`` →
  ``networkx.DiGraph``, symmetrised to undirected, combinatorial Laplacian on
  ``sorted(G.nodes())``. Matches RivGraph’s own adjacency construction (see
  ``rivnetwork.adjacency_matrix``). Optional edge weights via
  ``--rivgraph-edge-weight``.

- ``junction`` — Legacy hand-built graph from ``links['conn']`` with deduped
  undirected edges (can differ from ``graphiphy`` when parallel links exist).

- ``auto`` (default) — ``skeleton`` if small enough, else ``rivgraph`` (not
  ``junction``).

**Plotting:** Figures ``02_*`` and ``04_*`` draw RivGraph **centerline polylines**
(``links['idx']``) for geometry; Fiedler colouring is on whichever graph ``tg``
uses (skeleton pixels or junction nodes). With ``--plot-eigenmodes K``,
``05_eigenvector_maps.png`` adds spatial maps of ``u_1..u_K`` (``u_1`` = Fiedler)
and ``06_eigenvector_heatmap.png`` shows the matrix of eigenvector components.

Default **river** dataset: Brahmaputra mask. With ``--delta``, defaults switch to
the **Colville** delta (mask + shoreline + inlet shapefiles from RivGraph
``examples/data/Colville_Delta/``), matching ``examples/delta_example.py``.

Requirements
------------
- RivGraph importable (``pip install -e /path/to/RivGraph`` or add repo root and
  ``_deps`` to PYTHONPATH as in RivGraph README experiments).

Usage
-----
  # Preferred: console-script entry point (installed by ``pip install -e .``):
  kernelcal-rivgraph-bridge --rivgraph-repo /path/to/rivgraph --plot

  # Equivalent invocations:
  python -m kernelcal.integrations.rivgraph --rivgraph-repo /path/to/rivgraph --plot
  python rivgraph_kernelcal_bridge.py --rivgraph-repo /path/to/rivgraph --plot   # back-compat shim

  # Explicit mask + exit sides (river mode):
  kernelcal-rivgraph-bridge --mask /path/to/mask.tif --exit-sides NS

  # Laplacian eigenvector maps + heatmap (modes 1..K; 1 = Fiedler):
  kernelcal-rivgraph-bridge --plot --plot-eigenmodes 6 --rivgraph-repo /path/to/rivgraph

  # Colville delta (pruned network) -> kernelcal:
  kernelcal-rivgraph-bridge --delta --plot --rivgraph-repo /path/to/rivgraph
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, cast

import numpy as np

from kernelcal.terrain.dem import TerrainGraph, dem_to_graph, terrain_graph_laplacian
from kernelcal.terrain.diagnostics import (
    spectral_entropy_from_laplacian,
    fixed_point_kernel,
    fiedler_mode_gap,
    stability_conservation_tradeoff,
)

MU2 = 2.0
SIGMA2 = 1.0


def _default_rivgraph_repo() -> Path:
    return (Path(__file__).resolve().parent.parent / "rivgraph").resolve()


def _default_paths_river(repo: Path) -> tuple[Path, Path]:
    mask = repo / "examples" / "data" / "Brahmaputra_Braided_River" / "Brahmaputra_mask.tif"
    results = repo / "examples" / "data" / "Brahmaputra_Braided_River" / "Results_kernelcal_bridge"
    return mask, results


def _default_paths_delta(repo: Path) -> tuple[Path, Path, Path, Path]:
    """mask, results_folder, shoreline.shp, inlet_nodes.shp"""
    base = repo / "examples" / "data" / "Colville_Delta"
    mask = base / "Colville_mask.tif"
    results = base / "Results_kernelcal_bridge"
    shore = base / "Colville_shoreline.shp"
    inlets = base / "Colville_inlet_nodes.shp"
    return mask, results, shore, inlets


def _rivgraph_link_polyline_xy(
    links: dict,
    imshape: tuple[int, int],
) -> list[np.ndarray]:
    """Centerline polylines in matplotlib (x=col, y=row) coords, RivGraph-style.

    Matches ``ln_utils.plot_network`` geometry: each ``links['idx']`` chain is
    unraveled to pixel (row,col) then plotted as x=col, y=row with ``origin='upper'``.
    """
    segs: list[np.ndarray] = []
    for lidcs in links["idx"]:
        pix = np.asarray(list(lidcs), dtype=np.int64)
        if pix.size < 2:
            continue
        rc = np.unravel_index(pix, imshape)
        segs.append(np.column_stack((rc[1].astype(float), rc[0].astype(float))))
    return segs


def terrain_graph_from_graphiphy(
    links: dict,
    nodes: dict,
    imshape: tuple[int, int],
    *,
    weight_key: str | None,
) -> TerrainGraph:
    """Build ``TerrainGraph`` / Laplacian topology from RivGraph ``graphiphy``."""
    from rivgraph.deltas import delta_metrics as dm

    G = dm.graphiphy(links, nodes, weight=weight_key)
    nodelist = sorted(G.nodes())
    id_to_i = {int(nid): i for i, nid in enumerate(nodelist)}
    n = len(nodelist)

    id_to_rc: dict[int, tuple[float, float]] = {}
    for nid, pix in zip(nodes["id"], nodes["idx"]):
        r, c = np.unravel_index(int(pix), imshape)
        id_to_rc[int(nid)] = (float(r), float(c))

    cr, cc = 0.5 * (imshape[0] - 1), 0.5 * (imshape[1] - 1)
    positions = np.zeros((n, 2), dtype=float)
    for i, nid in enumerate(nodelist):
        nid_i = int(nid)
        positions[i] = id_to_rc.get(nid_i, (cr, cc))

    cell_index = np.full(imshape, -1, dtype=np.int32)
    for i, nid in enumerate(nodelist):
        nid_i = int(nid)
        if nid_i in id_to_rc:
            r, c = int(id_to_rc[nid_i][0]), int(id_to_rc[nid_i][1])
            cell_index[r, c] = int(i)

    U = G.to_undirected()
    edge_list: list[tuple[int, int]] = []
    weight_list: list[float] = []
    seen: set[tuple[int, int]] = set()
    for u, v, d in U.edges(data=True):
        i, j = id_to_i[int(u)], id_to_i[int(v)]
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        w = float(d.get("weight", 1.0))
        w = max(w, 1e-12)
        edge_list.append((a, b))
        weight_list.append(w)

    edges = np.array(edge_list, dtype=np.int32) if edge_list else np.empty((0, 2), dtype=np.int32)
    weights = np.array(weight_list, dtype=float) if weight_list else np.empty((0,), dtype=float)

    return TerrainGraph(
        positions=positions,
        elevations=np.zeros(n, dtype=float),
        edges=edges,
        weights=weights,
        shape=tuple(int(x) for x in imshape),
        cell_index=cell_index,
    )


def skeleton_pixel_terrain_graph(
    iskel: np.ndarray,
    *,
    connectivity: Literal[4, 8] = 8,
) -> TerrainGraph:
    """8- or 4-connected combinatorial graph on skeleton pixels (mask-only topology).

    Uses ``dem_to_graph`` with a flat DEM and ``mask=Iskel``: the only
    assumptions are (i) RivGraph's skeletonisation of the channel mask and
    (ii) chosen grid adjacency.
    """
    sk = np.asarray(iskel, dtype=bool)
    z = np.zeros(sk.shape, dtype=float)
    return dem_to_graph(
        z,
        connectivity=cast(Literal[4, 8], int(connectivity)),
        weight="uniform",
        mask=sk,
    )


def resolve_laplacian_terrain_graph(
    *,
    mode: str,
    iskel: np.ndarray,
    links: dict,
    nodes: dict,
    imshape: tuple[int, int],
    weight_by_length: bool,
    max_skel_nodes: int,
    skeleton_connectivity: Literal[4, 8],
    rivgraph_weight: str | None,
) -> tuple[TerrainGraph, str]:
    """Build ``TerrainGraph`` and return ``(tg, resolved_label)`` for Laplacian."""
    sk = np.asarray(iskel, dtype=bool)
    n_skel = int(sk.sum())

    if mode == "junction":
        tg = rivgraph_to_terrain_graph(
            links, nodes, imshape, weight_by_length=weight_by_length
        )
        return tg, f"junction (RivGraph conn dedup, N={len(nodes['idx'])})"

    if mode == "rivgraph":
        tg = terrain_graph_from_graphiphy(
            links, nodes, imshape, weight_key=rivgraph_weight
        )
        wlab = rivgraph_weight or "unweighted"
        return tg, f"rivgraph graphiphy ({wlab}, N={tg.elevations.shape[0]})"

    want_skel = mode in ("auto", "skeleton")
    if want_skel and n_skel <= max_skel_nodes:
        tg = skeleton_pixel_terrain_graph(sk, connectivity=skeleton_connectivity)
        ctag = "4" if skeleton_connectivity == 4 else "8"
        return tg, f"skeleton ({ctag}-neigh, N={n_skel})"

    if mode == "skeleton":
        raise SystemExit(
            f"Laplacian mode 'skeleton' requires at most {max_skel_nodes} skeleton "
            f"pixels (got {n_skel}). Crop or coarsen the mask, raise "
            "`--laplacian-max-nodes` if you accept dense eigen cost, or use "
            "`--laplacian-mode rivgraph` / `junction` / `auto`."
        )

    tg = terrain_graph_from_graphiphy(
        links, nodes, imshape, weight_key=rivgraph_weight
    )
    wlab = rivgraph_weight or "unweighted"
    print(
        f"  [laplacian] auto: skeleton has {n_skel} pixels > max "
        f"{max_skel_nodes} → rivgraph graphiphy ({wlab})."
    )
    return tg, f"rivgraph graphiphy ({wlab}, N={tg.elevations.shape[0]})"


def rivgraph_to_terrain_graph(
    links: dict,
    nodes: dict,
    imshape: tuple[int, int],
    *,
    weight_by_length: bool = False,
) -> TerrainGraph:
    """Build a kernelcal TerrainGraph from RivGraph links/nodes dicts.

    RivGraph node ``id`` values are contiguous integers 0 .. N-1 matching the
    order of ``nodes['idx']`` after extraction. Edges are undirected unique
    pairs from ``links['conn']`` — the **junction graph** (not the polyline
    pixel chain graph). For RivGraph-native topology export see
    ``rivgraph.deltas.delta_metrics.graphiphy(links, nodes)``.

    Parameters
    ----------
    links, nodes
        RivGraph structures after ``compute_network()`` (``river`` or ``delta``).
    imshape
        ``(nrows, ncols)`` of the mask / skeleton used for RivGraph.
    weight_by_length
        If True and ``links`` contains ``len``, use link length as edge weight;
        otherwise unit weight.
    """
    n = len(nodes["idx"])
    if len(nodes["id"]) != n:
        raise ValueError("RivGraph nodes['id'] and nodes['idx'] length mismatch.")

    positions = np.zeros((n, 2), dtype=float)
    cell_index = np.full(imshape, -1, dtype=np.int32)
    for i, pix in enumerate(nodes["idx"]):
        pix_i = int(pix)
        r, c = np.unravel_index(pix_i, imshape)
        positions[i, 0] = float(r)
        positions[i, 1] = float(c)
        cell_index[r, c] = int(i)

    has_len = weight_by_length and "len" in links
    seen: set[tuple[int, int]] = set()
    edge_list: list[tuple[int, int]] = []
    weight_list: list[float] = []

    link_ids = list(links["id"])
    for li, lid in enumerate(link_ids):
        conn = links["conn"][li]
        if conn is None or len(conn) != 2:
            continue
        a, b = int(conn[0]), int(conn[1])
        if a == b:
            continue
        t, u = (a, b) if a < b else (b, a)
        if (t, u) in seen:
            continue
        seen.add((t, u))
        edge_list.append((t, u))
        if has_len:
            w = float(links["len"][li])
            w = max(w, 1e-8)
        else:
            w = 1.0
        weight_list.append(w)

    edges = np.array(edge_list, dtype=np.int32) if edge_list else np.empty((0, 2), dtype=np.int32)
    weights = np.array(weight_list, dtype=float) if weight_list else np.empty((0,), dtype=float)

    return TerrainGraph(
        positions=positions,
        elevations=np.zeros(n, dtype=float),
        edges=edges,
        weights=weights,
        shape=tuple(int(x) for x in imshape),
        cell_index=cell_index,
    )


def betti_from_laplacian(L: np.ndarray) -> tuple[int, int]:
    """beta0 from small eigenvalues; beta1 from Euler characteristic (graph)."""
    eigvals = np.linalg.eigvalsh(L)
    beta0 = int(np.sum(np.abs(eigvals) < 1e-6))
    v = L.shape[0]
    a = np.diag(np.diag(L)) - L
    e = int(np.sum(a > 1e-10)) // 2
    beta1 = max(0, e - v + beta0)
    return beta0, beta1


def ensure_rivgraph_import(rivgraph_repo: Path | None) -> Path:
    """Prep sys.path and return resolved RivGraph repo root."""
    repo = (rivgraph_repo or _default_rivgraph_repo()).resolve()
    for p in (repo, repo / "_deps"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        import rivgraph  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Could not import rivgraph. Install RivGraph (e.g. "
            "`pip install -e /path/to/RivGraph`) or pass --rivgraph-repo pointing "
            "to the cloned repo (optionally with a `_deps` tree from "
            "`pip install -t _deps .`).\n"
            f"Original error: {e}"
        ) from e
    return repo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RivGraph mask -> kernelcal bridge demo.")
    p.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Channel mask GeoTIFF. Default: Brahmaputra (river) or Colville (with --delta).",
    )
    p.add_argument(
        "--results",
        type=Path,
        default=None,
        help="RivGraph outputs directory. Default follows --mask example location.",
    )
    p.add_argument(
        "--name",
        default=None,
        help="RivGraph run name prefix. Default: Brahma_kcal or Colville_kcal when --delta.",
    )
    p.add_argument(
        "--delta",
        action="store_true",
        help="Use rivgraph.classes.delta + prune_network (Colville defaults for paths).",
    )
    p.add_argument(
        "--shoreline",
        type=Path,
        default=None,
        help="Delta shoreline vector (default: Colville_shoreline.shp next to mask).",
    )
    p.add_argument(
        "--inlet-nodes",
        type=Path,
        default=None,
        help="Delta inlet points vector (default: Colville_inlet_nodes.shp).",
    )
    p.add_argument(
        "--prune-less",
        action="store_true",
        help="Forward prune_less=True to delta.prune_network (see RivGraph docs).",
    )
    p.add_argument(
        "--exit-sides",
        default="NS",
        help="River exit sides for rivgraph.classes.river (ignored with --delta).",
    )
    p.add_argument(
        "--rivgraph-repo",
        type=Path,
        default=None,
        help="RivGraph clone root; adds repo and _deps to sys.path if present.",
    )
    p.add_argument(
        "--weight-by-length",
        action="store_true",
        help="Use link geometric length as edge weight when available.",
    )
    p.add_argument("--verbose", action="store_true", help="RivGraph verbose logging.")
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save matplotlib figures (mask/skel, network, spectrum, Fiedler map).",
    )
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Figure output directory (default: <results>/plots).",
    )
    p.add_argument(
        "--spectrum-modes",
        type=int,
        default=48,
        help="Number of smallest Laplacian eigenvalues to show in spectrum plot.",
    )
    p.add_argument(
        "--plot-eigenmodes",
        type=int,
        default=0,
        metavar="K",
        help="If >0 with --plot, save spatial maps for Laplacian eigenvectors u_1..u_K (u_1=Fiedler) and a heatmap.",
    )
    p.add_argument(
        "--laplacian-mode",
        choices=("auto", "skeleton", "junction", "rivgraph"),
        default="auto",
        help="Graph for L: skeleton, RivGraph graphiphy, legacy junction conn, or auto.",
    )
    p.add_argument(
        "--rivgraph-edge-weight",
        choices=("none", "len_adj", "wid_adj"),
        default="none",
        help="graphiphy / rivgraph mode edge attribute (needs compute_link_width_and_length).",
    )
    p.add_argument(
        "--laplacian-max-nodes",
        type=int,
        default=12000,
        help="Max skeleton pixels for dense 'skeleton' Laplacian; auto falls back beyond this.",
    )
    p.add_argument(
        "--skeleton-connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Neighbourhood on Iskel for skeleton Laplacian (4 or 8).",
    )
    return p.parse_args()


def _junction_pixel_coords(nodes: dict, imshape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    n = len(nodes["idx"])
    rows = np.zeros(n, dtype=float)
    cols = np.zeros(n, dtype=float)
    for i, pix in enumerate(nodes["idx"]):
        r, c = np.unravel_index(int(pix), imshape)
        rows[i] = float(r)
        cols[i] = float(c)
    return rows, cols


def write_visualizations(
    *,
    plot_dir: Path,
    rnet,
    tg: TerrainGraph,
    L: np.ndarray,
    h_star: np.ndarray,
    spectrum_modes: int,
    laplacian_label: str = "",
    plot_eigenmodes_k: int = 0,
) -> None:
    """Save PNG summaries (requires matplotlib)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    plot_dir.mkdir(parents=True, exist_ok=True)

    rows, cols = tg.positions[:, 0], tg.positions[:, 1]
    polylines = _rivgraph_link_polyline_xy(rnet.links, rnet.imshape)
    skel_laplacian = laplacian_label.startswith("skeleton")

    # --- 1) mask and skeleton ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].imshow(rnet.Imask, cmap="bone", origin="upper")
    axes[0].set_title("Channel mask")
    axes[0].set_axis_off()
    axes[1].imshow(rnet.Iskel, cmap="inferno", origin="upper")
    axes[1].set_title("Skeleton (RivGraph)")
    axes[1].set_axis_off()
    fig.savefig(plot_dir / "01_mask_skeleton.png", dpi=160)
    plt.close(fig)

    # --- 2) centerline geometry + junction nodes (RivGraph-style drawing) -----
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.imshow(rnet.Imask, cmap="Greys", alpha=0.45, origin="upper")
    if polylines:
        ax.add_collection(
            LineCollection(
                polylines,
                colors="dodgerblue",
                linewidths=0.45,
                alpha=0.75,
            )
        )
    if skel_laplacian:
        ax.scatter(cols, rows, s=0.35, c="yellow", alpha=0.12, linewidths=0, zorder=3)
        jrows, jcols = _junction_pixel_coords(rnet.nodes, rnet.imshape)
        ax.scatter(jcols, jrows, s=5.0, c="crimson", alpha=0.55, linewidths=0, zorder=5)
    else:
        ax.scatter(cols, rows, s=4.0, c="crimson", alpha=0.65, linewidths=0, zorder=5)
    ax.set_title(
        "RivGraph centerlines + nodes | Laplacian: "
        + (laplacian_label or "see console")
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.savefig(plot_dir / "02_network_on_mask.png", dpi=160)
    plt.close(fig)

    # --- shared spectrum decomposition (single eigh) ---------------------------
    n_lap = int(L.shape[0])
    if n_lap > 0:
        ev_full, U_full = np.linalg.eigh(L)
        ordv = np.argsort(ev_full)
        ev_full = np.maximum(ev_full[ordv], 0.0)
        U_full = U_full[:, ordv]
    else:
        ev_full = np.zeros(0)
        U_full = np.zeros((0, 0))

    # --- 3) Laplacian spectrum + fixed-point h* -------------------------------
    w = ev_full
    k = min(spectrum_modes, len(w))
    idx = np.arange(k)
    fig, ax1 = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax1.bar(idx, w[:k], color="steelblue", alpha=0.85, label=r"$\lambda_\ell$")
    ax1.set_xlabel("Mode index (smallest eigenvalues)")
    ax1.set_ylabel(r"Laplacian eigenvalue $\lambda$")
    ax1.set_title("Smallest Laplacian eigenvalues | " + (laplacian_label or "?"))
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    hs = np.asarray(h_star, dtype=float)[:k]
    ax2.plot(idx, hs, color="darkorange", linewidth=2.0, marker=".", markersize=3, label=r"$h^*$ (fixed point)")
    ax2.set_ylabel(r"Fixed-point kernel $h^*_\ell$")
    fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), bbox_transform=ax1.transAxes)
    fig.savefig(plot_dir / "03_spectrum_and_hstar.png", dpi=160)
    plt.close(fig)

    # --- 4) Fiedler vector (2nd eigenvector) on network ----------------------
    if n_lap > 2:
        fiedler = U_full[:, 1]
    else:
        fiedler = np.zeros(max(n_lap, 0))

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.imshow(rnet.Imask, cmap="Greys", alpha=0.35, origin="upper")
    if polylines:
        ax.add_collection(
            LineCollection(
                polylines,
                colors="0.45",
                linewidths=0.35,
                alpha=0.55,
            )
        )
    pt_size = 0.6 if skel_laplacian else 6.0
    pt_alpha = 0.35 if skel_laplacian else 0.9
    sc = ax.scatter(
        cols,
        rows,
        c=fiedler,
        cmap="coolwarm",
        s=pt_size,
        linewidths=0,
        alpha=pt_alpha,
        rasterized=skel_laplacian,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, label="Fiedler component")
    cb.ax.tick_params(labelsize=8)
    ax.set_title(
        "Fiedler (2nd eigenvector) | "
        + (laplacian_label or "?")
        + " — centerlines = RivGraph geometry"
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.savefig(plot_dir / "04_fiedler_on_nodes.png", dpi=160)
    plt.close(fig)

    # --- 5) eigenvector spatial maps u_1 .. u_K (u_1 = Fiedler) --------------
    if plot_eigenmodes_k > 0 and n_lap > 2:
        k_sp = min(plot_eigenmodes_k, n_lap - 1, 9)
        ncols = int(np.ceil(np.sqrt(k_sp)))
        nrows = int(np.ceil(k_sp / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.4 * ncols, 3.2 * nrows),
            constrained_layout=True,
            squeeze=False,
        )
        axes_flat = np.atleast_1d(axes).ravel()
        for j in range(k_sp):
            axm = axes_flat[j]
            mode_idx = j + 1
            axm.imshow(rnet.Imask, cmap="Greys", alpha=0.32, origin="upper")
            if polylines:
                axm.add_collection(
                    LineCollection(
                        polylines,
                        colors="0.5",
                        linewidths=0.2,
                        alpha=0.45,
                    )
                )
            uj = U_full[:, mode_idx]
            scj = axm.scatter(
                cols,
                rows,
                c=uj,
                cmap="coolwarm",
                s=0.55 if skel_laplacian else 5.0,
                linewidths=0,
                alpha=0.85 if not skel_laplacian else 0.4,
                rasterized=skel_laplacian,
            )
            fig.colorbar(scj, ax=axm, shrink=0.72, fraction=0.046)
            axm.set_title(rf"$u_{{{mode_idx}}}$  $\lambda={ev_full[mode_idx]:.3g}$")
            axm.set_aspect("equal")
            axm.set_axis_off()
        for j in range(k_sp, len(axes_flat)):
            axes_flat[j].set_axis_off()
        fig.suptitle(
            "Laplacian eigenvectors on graph nodes | " + (laplacian_label or "?"),
            fontsize=11,
        )
        fig.savefig(plot_dir / "05_eigenvector_maps.png", dpi=160)
        plt.close(fig)

        # --- 6) eigenvector matrix (nodes × mode) --------------------------------
        k_mat = min(plot_eigenmodes_k, n_lap - 1, 48)
        block = U_full[:, 1 : k_mat + 1]
        max_rows = 2400
        if block.shape[0] > max_rows:
            sel = np.linspace(0, block.shape[0] - 1, max_rows, dtype=int)
            block = block[sel, :]
            row_note = f" (subsampled {max_rows}/{n_lap} nodes)"
        else:
            row_note = ""
        fig, axh = plt.subplots(figsize=(min(0.35 * k_mat + 2, 14), 7.2), constrained_layout=True)
        im = axh.imshow(block, aspect="auto", cmap="coolwarm", interpolation="nearest")
        axh.set_xlabel("Laplacian mode index (column j = $u_j$)")
        axh.set_ylabel("Graph node index (sorted by eigh order)" + row_note)
        axh.set_title("Eigenvector matrix $U_{i,j}$ (columns = $u_j$)" + row_note)
        fig.colorbar(im, ax=axh, shrink=0.65, label="component")
        fig.savefig(plot_dir / "06_eigenvector_heatmap.png", dpi=160)
        plt.close(fig)

    print(f"  Figures written  : {plot_dir.resolve()}")


def main() -> None:
    args = parse_args()
    repo = ensure_rivgraph_import(args.rivgraph_repo)

    from rivgraph.classes import delta as delta_cls
    from rivgraph.classes import river

    if args.delta:
        dm, dr, dsh, din = _default_paths_delta(repo)
        mask_path = (args.mask or dm).resolve()
        results_folder = (args.results or dr).resolve()
        run_name = args.name or "Colville_kcal"
        shoreline = (args.shoreline or dsh).resolve()
        inlet_nodes = (args.inlet_nodes or din).resolve()
    else:
        bm, br = _default_paths_river(repo)
        mask_path = (args.mask or bm).resolve()
        results_folder = (args.results or br).resolve()
        run_name = args.name or "Brahma_kcal"
        shoreline = None
        inlet_nodes = None

    if not mask_path.is_file():
        raise SystemExit(f"Mask not found: {mask_path}")

    results_folder.mkdir(parents=True, exist_ok=True)

    for stale in (f"{run_name}_fixlinks.csv", "fixlinks.csv"):
        sp = results_folder / stale
        if sp.exists():
            sp.unlink()

    print()
    print("RIVGRAPH -> KERNELCAL bridge")
    print(f"  mode           : {'delta' if args.delta else 'river'}")
    print(f"  mask           : {mask_path}")
    print(f"  results_folder : {results_folder}")
    if args.delta:
        print(f"  shoreline      : {shoreline}")
        print(f"  inlet_nodes    : {inlet_nodes}")
        if not shoreline.is_file():
            raise SystemExit(f"Shoreline vector not found: {shoreline}")
        if not inlet_nodes.is_file():
            raise SystemExit(f"Inlet nodes vector not found: {inlet_nodes}")
    else:
        print(f"  exit_sides     : {args.exit_sides}")

    if args.delta:
        rnet = delta_cls(
            run_name,
            str(mask_path),
            str(results_folder),
            verbose=args.verbose,
        )
        rnet.skeletonize()
        rnet.compute_network()
        rnet.prune_network(
            path_shoreline=str(shoreline),
            path_inletnodes=str(inlet_nodes),
            prune_less=args.prune_less,
        )
    else:
        rnet = river(
            run_name,
            str(mask_path),
            str(results_folder),
            exit_sides=args.exit_sides,
            verbose=args.verbose,
        )
        rnet.skeletonize()
        rnet.compute_network()

    riv_w = None if args.rivgraph_edge_weight == "none" else args.rivgraph_edge_weight
    if args.weight_by_length or riv_w in ("len_adj", "wid_adj"):
        rnet.compute_link_width_and_length()

    n_nodes = len(rnet.nodes["idx"])
    n_links = len(rnet.links["id"])
    print(f"  RivGraph nodes : {n_nodes}")
    print(f"  RivGraph links : {n_links}")

    sk_conn = cast(Literal[4, 8], int(args.skeleton_connectivity))
    tg, laplacian_label = resolve_laplacian_terrain_graph(
        mode=args.laplacian_mode,
        iskel=rnet.Iskel,
        links=rnet.links,
        nodes=rnet.nodes,
        imshape=rnet.imshape,
        weight_by_length=args.weight_by_length,
        max_skel_nodes=args.laplacian_max_nodes,
        skeleton_connectivity=sk_conn,
        rivgraph_weight=riv_w,
    )
    L = terrain_graph_laplacian(tg)
    print(f"  Laplacian      : {laplacian_label}")
    print(f"  TerrainGraph   : N={tg.elevations.shape[0]}  E={len(tg.edges)}")

    beta0, beta1 = betti_from_laplacian(L)
    print(f"  beta0 (est.)   : {beta0}")
    print(f"  beta1 (est.)   : {beta1}")

    H = spectral_entropy_from_laplacian(L, tau=1.0)
    print(f"  H[h] (heat)    : {H:.5f} nats")

    h_star, info = fixed_point_kernel(L, mu2=MU2, sigma2=SIGMA2)
    conv = "yes" if info["converged"] else f"NO (res={info['residual']:.2e})"
    print(f"  fixed_point h* : converged={conv}  iters={info['n_iter']}")

    delta_prime = fiedler_mode_gap(h_star, L, mu2=MU2, sigma2=SIGMA2)
    print(f"  Delta'         : {delta_prime:.6f}")

    sct = stability_conservation_tradeoff(h_star, L, mu2=MU2, sigma2=SIGMA2)
    print(f"  D_m deficit    : {sct['conservation_deficit']:.6f}")

    if args.plot:
        pdir = args.plot_dir if args.plot_dir is not None else (results_folder / "plots")
        write_visualizations(
            plot_dir=pdir,
            rnet=rnet,
            tg=tg,
            L=L,
            h_star=h_star,
            spectrum_modes=args.spectrum_modes,
            laplacian_label=laplacian_label,
            plot_eigenmodes_k=args.plot_eigenmodes,
        )

    print()
    print("Done. kernelcal diagnostics on dense L; see --laplacian-mode for graph definition.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
rivgraph_graph_analysis.py
============================
Use RivGraph's native graph objects (``graphiphy`` → ``networkx.DiGraph``,
``adjacency_matrix``, delta ``compute_topologic_metrics``) and optionally
compare to kernelcal spectral scalars on the **symmetrized** undirected Laplacian.

Examples
--------
  cd manuscripts/software-kernelcal-deepgis-integration
  PYTHONPATH=/path/to/RivGraph/_deps MPLBACKEND=Agg \\
    python3 rivgraph_graph_analysis.py --example meander

  PYTHONPATH=... python3 rivgraph_graph_analysis.py --example colville

  # Full braided river directionality (slow): brahmaputra
  PYTHONPATH=... python3 rivgraph_graph_analysis.py --example brahmaputra

  # Meander: skip prune to keep the full extracted tree for graph stats
  PYTHONPATH=... python3 rivgraph_graph_analysis.py --example meander --no-prune
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

KCAL_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(KCAL_ROOT))

from kernelcal.terrain.dem import terrain_graph_laplacian
from kernelcal.terrain.diagnostics import spectral_entropy_from_laplacian


def _default_repo() -> Path:
    return (KCAL_ROOT.parent / "rivgraph").resolve()


def ensure_rivgraph(repo: Path | None) -> Path:
    r = (repo or _default_repo()).resolve()
    for p in (r, r / "_deps"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import rivgraph  # noqa: F401
    return r


def _nx_report(G: Any) -> dict[str, Any]:
    import networkx as nx

    Gu = G.to_undirected()
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "is_directed": G.is_directed(),
        "is_dag": nx.is_directed_acyclic_graph(G) if G.is_directed() else None,
        "n_weakly_connected": nx.number_weakly_connected_components(G)
        if G.is_directed()
        else nx.number_connected_components(Gu),
        "n_strongly_connected": nx.number_strongly_connected_components(G)
        if G.is_directed()
        else None,
        "avg_out_degree": sum(d for _, d in G.out_degree()) / max(G.number_of_nodes(), 1)
        if G.is_directed()
        else None,
    }


def _kernelcal_on_symmetric_adj(A: np.ndarray) -> dict[str, float]:
    """Symmetric combinatorial Laplacian from weighted undirected adjacency."""
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    n = L.shape[0]
    if n == 0:
        return {"H_heat": float("nan")}
    H = spectral_entropy_from_laplacian(L, tau=1.0)
    ev = np.linalg.eigvalsh(L)
    beta0 = int(np.sum(np.abs(ev) < 1e-6))
    e = int(np.sum(A > 1e-10)) // 2
    beta1 = max(0, e - n + beta0)
    return {"H_heat": float(H), "beta0_est": float(beta0), "beta1_est": float(beta1)}


def analyze_graphiphy(links, nodes, *, weight_key: str | None) -> None:
    import networkx as nx
    from rivgraph.deltas import delta_metrics as dm

    G = dm.graphiphy(links, nodes, weight=weight_key)
    rep = _nx_report(G)
    print("\n--- graphiphy (RivGraph → networkx) ---")
    print(f"  weight         : {weight_key!r}")
    for k, v in rep.items():
        print(f"  {k:24s}: {v}")

    nodelist = sorted(G.nodes())
    A = nx.to_numpy_array(G.to_undirected(), nodelist=nodelist, weight="weight")
    kc = _kernelcal_on_symmetric_adj(A)
    print("  kernelcal (undirected symmetrise, heat entropy): ", kc)


def analyze_meander(repo: Path, results: Path, *, prune: bool) -> None:
    from rivgraph.classes import river

    mask = repo / "examples/data/Meandering_River/meander_mask.tif"
    r = river("rg_an", str(mask), str(results), exit_sides="NS", verbose=False)
    r.skeletonize()
    r.compute_network()
    if prune:
        r.prune_network()
    r.compute_link_width_and_length()

    print("\n=== Meandering river (RivGraph network) ===")
    print(f"  nodes: {len(r.nodes['idx'])}  links: {len(r.links['id'])}")

    analyze_graphiphy(r.links, r.nodes, weight_key=None)
    if "len_adj" in r.links:
        analyze_graphiphy(r.links, r.nodes, weight_key="len_adj")

    A0 = r.adjacency_matrix(weight=None, normalized=False)
    print("\n--- rivnetwork.adjacency_matrix (unweighted) ---")
    print(f"  shape {A0.shape}  nnz {(A0 > 0).sum()}")

    kc = _kernelcal_on_symmetric_adj(A0)
    print("  kernelcal on adjacency symmetrise:", kc)

    # Cross-check: Laplacian from TerrainGraph built like bridge junction graph
    from rivgraph_kernelcal_bridge import rivgraph_to_terrain_graph

    tg = rivgraph_to_terrain_graph(r.links, r.nodes, r.imshape, weight_by_length=False)
    L = terrain_graph_laplacian(tg)
    H2 = spectral_entropy_from_laplacian(L, tau=1.0)
    print(f"\n  bridge-style junction Laplacian H(heat) = {H2:.5f} nats")


def analyze_brahmaputra(repo: Path, results: Path, *, prune: bool) -> None:
    from rivgraph.classes import river

    mask = repo / "examples/data/Brahmaputra_Braided_River/Brahmaputra_mask.tif"
    r = river("rg_an", str(mask), str(results), exit_sides="NS", verbose=False)
    r.skeletonize()
    r.compute_network()
    if prune:
        r.prune_network()
    r.compute_link_width_and_length()
    r.compute_centerline()
    r.compute_mesh()
    r.compute_distance_transform()
    r.assign_flow_directions()

    print("\n=== Brahmaputra (directed DAG after RivGraph directionality) ===")
    print(f"  nodes: {len(r.nodes['idx'])}  links: {len(r.links['id'])}")

    analyze_graphiphy(r.links, r.nodes, weight_key=None)
    analyze_graphiphy(r.links, r.nodes, weight_key="len_adj")

    print("\n--- adjacency_matrix wid_adj (first 5×5 block) ---")
    Aw = r.adjacency_matrix(weight="wid_adj", normalized=False)
    print(Aw[:5, :5])


def analyze_colville(repo: Path, results: Path) -> None:
    from rivgraph.classes import delta as delta_cls
    from rivgraph.deltas.delta_metrics import compute_steady_state_link_fluxes

    base = repo / "examples/data/Colville_Delta"
    mask = base / "Colville_mask.tif"
    shore = base / "Colville_shoreline.shp"
    inlets = base / "Colville_inlet_nodes.shp"

    d = delta_cls("rg_an", str(mask), str(results), verbose=False)
    d.skeletonize()
    d.compute_network()
    d.prune_network(path_shoreline=str(shore), path_inletnodes=str(inlets))
    d.compute_link_width_and_length()
    d.assign_flow_directions()
    d.compute_junction_angles(weight=None)

    d.links = compute_steady_state_link_fluxes(
        None,
        d.links,
        d.nodes,
        weight_name="flux_ss",
        routing="width",
        inlet="equal",
    )

    print("\n=== Colville delta (steady flux + topology metrics) ===")
    print(f"  nodes: {len(d.nodes['idx'])}  links: {len(d.links['id'])}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        d.compute_topologic_metrics(inlet="equal")

    print("\n--- topo_metrics (keys) ---")
    for k in sorted(d.topo_metrics.keys()):
        v = d.topo_metrics[k]
        if isinstance(v, np.ndarray):
            print(f"  {k}: array shape {v.shape}")
        else:
            s = str(v)
            print(f"  {k}: {s[:120]}{'...' if len(s) > 120 else ''}")

    analyze_graphiphy(d.links, d.nodes, weight_key="wid_adj")

    A = d.adjacency_matrix(weight="wid_adj", normalized=False)
    print(f"\n--- adjacency wid_adj nnz={(A > 0).sum()} ---")
    print("  kernelcal:", _kernelcal_on_symmetric_adj(A))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RivGraph native graph analysis + kernelcal hook.")
    p.add_argument(
        "--example",
        choices=("meander", "brahmaputra", "colville"),
        default="meander",
        help="Which packaged RivGraph example to run.",
    )
    p.add_argument("--rivgraph-repo", type=Path, default=None)
    p.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Scratch results folder for RivGraph logs.",
    )
    p.add_argument(
        "--no-prune",
        action="store_true",
        help="Skip prune_network on river examples (richer graph for small masks).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = ensure_rivgraph(args.rivgraph_repo)
    results = args.results or (KCAL_ROOT / "rivgraph_analysis_results")
    results.mkdir(parents=True, exist_ok=True)

    prune = not args.no_prune
    if args.example == "meander":
        analyze_meander(repo, results / "meander", prune=prune)
    elif args.example == "brahmaputra":
        if args.no_prune:
            print(
                "[note] brahmaputra analysis requires prune + directionality; "
                "ignoring --no-prune.",
            )
        analyze_brahmaputra(repo, results / "brahmaputra", prune=True)
    else:
        analyze_colville(repo, results / "colville")

    print("\nDone.")


if __name__ == "__main__":
    main()

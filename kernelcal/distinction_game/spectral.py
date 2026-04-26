"""
Spectral diagnostics on the per-tile region graph (PR-3, §5).

Once a :class:`SceneGraph` carries a CityGraph backbone (PR-2 Option A
``use_city_graph_regions=True``), we have an explicit graph Laplacian
``L`` over the region-anchor nodes (buildings) plus the road-aware
edges. This module provides two utilities that consume that structure:

1. :func:`graph_smooth_posteriors` — Tikhonov smoothing of per-node
   category posteriors against the graph Laplacian. Solves
   ``(I + tau * L) P_smooth = P`` column-by-column (one column per
   category). Can also use the heat-kernel form ``exp(-tau L) P``
   when ``kernel="heat"`` (computed in eigenbasis if eigvecs/eigvals
   are supplied, else densely via :func:`scipy.linalg.expm`).

2. :func:`spectral_consistency_score` — the Dirichlet quadratic form
   ``tr(P^T L P) / (n_nodes * n_categories)``. Small values mean the
   posteriors are smooth across the graph (i.e. spatially adjacent
   regions agree about category), which is the right inductive bias
   for urban scenes with lots of building-rows.

Both functions are happy with either a sparse or dense Laplacian. They
fall back gracefully if scipy is missing (the rest of kernelcal does
not require scipy, so this module imports it lazily).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

LaplacianLike = Union[np.ndarray, "scipy.sparse.spmatrix"]  # noqa: F821


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_import_scipy() -> Tuple[Optional[object], Optional[object]]:
    """Lazy scipy import. Returns (sparse, linalg) or (None, None)."""
    try:
        from scipy import sparse as sp_sparse           # noqa
        from scipy.sparse import linalg as sp_splinalg  # noqa
        return sp_sparse, sp_splinalg
    except Exception:
        return None, None


def _is_sparse(L: LaplacianLike) -> bool:
    sp_sparse, _ = _try_import_scipy()
    if sp_sparse is None:
        return False
    return sp_sparse.issparse(L)


def _to_dense(L: LaplacianLike) -> np.ndarray:
    if _is_sparse(L):
        return np.asarray(L.toarray())
    return np.asarray(L)


# ---------------------------------------------------------------------------
# Tikhonov smoother  (I + tau L) P_smooth = P
# ---------------------------------------------------------------------------

def _tikhonov_smooth(
    P: np.ndarray,
    L: LaplacianLike,
    tau: float,
) -> np.ndarray:
    """Column-wise Tikhonov smoothing.

    Closed form on dense L for small graphs, sparse CG for big ones.
    Each column of ``P`` is renormalised to a probability vector
    afterward (the smoother does not preserve column-stochasticity
    on its own).
    """
    n = P.shape[0]
    sp_sparse, sp_splinalg = _try_import_scipy()
    if _is_sparse(L) and sp_sparse is not None and sp_splinalg is not None:
        I = sp_sparse.eye(n, format="csr")
        A = (I + tau * L).tocsr()
        P_smooth = np.zeros_like(P)
        for k in range(P.shape[1]):
            x, info = sp_splinalg.cg(A, P[:, k], rtol=1e-7, atol=1e-9, maxiter=2 * n + 50)
            if info != 0:
                # CG failed to converge; fall back to a dense solve.
                P_smooth[:, k] = np.linalg.solve(_to_dense(A), P[:, k])
            else:
                P_smooth[:, k] = x
    else:
        L_d = _to_dense(L)
        A = np.eye(n) + tau * L_d
        P_smooth = np.linalg.solve(A, P)

    P_smooth = np.maximum(P_smooth, 0.0)
    row_sums = P_smooth.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return P_smooth / row_sums


# ---------------------------------------------------------------------------
# Heat-kernel smoother  exp(-tau L) P  (eigenbasis preferred)
# ---------------------------------------------------------------------------

def _heat_smooth(
    P: np.ndarray,
    L: Optional[LaplacianLike] = None,
    *,
    tau: float = 1.0,
    eigvals: Optional[np.ndarray] = None,
    eigvecs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Heat-kernel smoothing.

    If ``eigvals`` and ``eigvecs`` (full or truncated) are provided,
    compute ``U diag(exp(-tau lambda)) U^T P`` directly — robust on
    big graphs and respects truncation.

    Otherwise fall back to dense ``scipy.linalg.expm(-tau L) @ P`` for
    a small graph, or column-wise sparse ``expm_multiply`` if scipy is
    available.
    """
    n = P.shape[0]
    if eigvals is not None and eigvecs is not None:
        U = np.asarray(eigvecs, dtype=np.float64)         # (n, k)
        lam = np.asarray(eigvals, dtype=np.float64)       # (k,)
        if U.shape[0] != n:
            raise ValueError(
                f"eigvecs.shape[0]={U.shape[0]} mismatches n_nodes={n}"
            )
        d = np.exp(-float(tau) * lam)                     # (k,)
        P_smooth = U @ (d[:, None] * (U.T @ P))
    else:
        if L is None:
            raise ValueError(
                "_heat_smooth requires either (eigvals, eigvecs) or L"
            )
        sp_sparse, sp_splinalg = _try_import_scipy()
        if sp_splinalg is not None and _is_sparse(L):
            P_smooth = sp_splinalg.expm_multiply(-float(tau) * L, P)
            P_smooth = np.asarray(P_smooth)
        else:
            try:
                from scipy.linalg import expm  # type: ignore
                K = expm(-float(tau) * _to_dense(L))
                P_smooth = K @ P
            except Exception:
                # No scipy at all — degrade to Tikhonov.
                return _tikhonov_smooth(P, L if L is not None else np.zeros((n, n)), tau)

    P_smooth = np.maximum(P_smooth, 0.0)
    row_sums = P_smooth.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return P_smooth / row_sums


# ---------------------------------------------------------------------------
# Public smoothing API
# ---------------------------------------------------------------------------

def graph_smooth_posteriors(
    posteriors: np.ndarray,
    laplacian: Optional[LaplacianLike] = None,
    *,
    tau: float = 1.0,
    kernel: str = "tikhonov",
    eigvals: Optional[np.ndarray] = None,
    eigvecs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Smooth per-node category posteriors against a graph Laplacian.

    Parameters
    ----------
    posteriors
        ``(n_nodes, n_categories)`` array of per-node category
        posteriors. Each row should be (approximately) a probability
        vector. The smoother re-normalises rows on output, so input
        non-normalisation is tolerated.
    laplacian
        Graph Laplacian — dense or sparse. Required for
        ``kernel="tikhonov"`` and for ``kernel="heat"`` without
        precomputed eigendecomposition.
    tau
        Smoothing strength. Larger ``tau`` => more aggressive
        smoothing. Sensible defaults: 0.5 for "light denoising", 2.0
        for "trust the graph more than per-node MAP".
    kernel
        ``"tikhonov"`` (default; numerically robust) or ``"heat"``
        (closed-form when eigendecomposition is supplied).
    eigvals, eigvecs
        Optional precomputed eigendecomposition. If supplied with
        ``kernel="heat"``, the smoother uses the eigenbasis directly
        and the ``laplacian`` argument may be omitted.

    Returns
    -------
    np.ndarray
        ``(n_nodes, n_categories)`` smoothed posteriors with every
        row summing to 1 and entries non-negative.
    """
    P = np.asarray(posteriors, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"posteriors must be 2-D; got shape {P.shape}")
    if tau < 0:
        raise ValueError(f"tau must be non-negative; got {tau}")

    if tau == 0.0:
        # No-op shortcut — caller asked for the identity smoother.
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return P / row_sums

    if kernel == "tikhonov":
        if laplacian is None:
            raise ValueError("kernel='tikhonov' requires a laplacian")
        return _tikhonov_smooth(P, laplacian, tau)
    if kernel == "heat":
        return _heat_smooth(P, laplacian, tau=tau, eigvals=eigvals, eigvecs=eigvecs)
    raise ValueError(f"unknown kernel {kernel!r}; expected 'tikhonov' or 'heat'")


# ---------------------------------------------------------------------------
# Spectral consistency score
# ---------------------------------------------------------------------------

def spectral_consistency_score(
    posteriors: np.ndarray,
    laplacian: LaplacianLike,
    *,
    normalize: bool = True,
) -> float:
    """Dirichlet energy of per-node category posteriors on the graph.

    Computes

    .. math::

       E = \\frac{1}{n_{\\text{nodes}} \\, n_{\\text{categories}}}
           \\, \\mathrm{tr}(P^T L P)

    Lower is better — it measures the squared-difference of category
    posteriors across edges, weighted by edge weight (encoded in
    ``L``). The optional normalisation keeps the score comparable
    across tiles of different size and taxonomy width.

    A useful sanity check: a perfectly uninformative posterior
    (uniform on every node) has score 0 because constant signals lie
    in the Laplacian's null space. A posterior that flips category
    every other node along a tight road scores high.
    """
    P = np.asarray(posteriors, dtype=np.float64)
    if _is_sparse(laplacian):
        LP = laplacian @ P                       # (n, C)
    else:
        LP = np.asarray(laplacian, dtype=np.float64) @ P
    energy = float(np.sum(P * LP))
    if not normalize:
        return energy
    n, C = P.shape
    if n == 0 or C == 0:
        return 0.0
    return energy / (n * C)


# ---------------------------------------------------------------------------
# Utilities for SceneGraph integration
# ---------------------------------------------------------------------------

def cg_node_index_map(scene_graph_dict: Mapping[str, object]) -> Dict[int, int]:
    """Return ``{node_position_in_graph: cg_node_idx}`` for nodes that
    carry a CityGraph index (PR-2 Option A).

    Nodes without a ``cg_node_idx`` attribute are excluded.
    """
    nodes = scene_graph_dict.get("nodes") or []
    out: Dict[int, int] = {}
    for i, node in enumerate(nodes):
        attrs = node.get("attributes") or {}
        cg_idx = attrs.get("cg_node_idx")
        if cg_idx is None:
            cg_idx = node.get("cg_node_idx")
        if cg_idx is None:
            continue
        try:
            out[i] = int(cg_idx)
        except (TypeError, ValueError):
            continue
    return out


def posteriors_array_from_scene_graph(
    scene_graph_dict: Mapping[str, object],
    *,
    n_categories: Optional[int] = None,
) -> np.ndarray:
    """Stack per-node ``category_posterior`` into an ``(n_nodes, C)`` array.

    Each ``category_posterior`` entry is the schema-versioned list of
    ``{"category": ..., "p": ...}`` dicts; this helper just collects
    the ``p`` values in array form.
    """
    nodes = scene_graph_dict.get("nodes") or []
    if not nodes:
        return np.zeros((0, n_categories or 0))
    if n_categories is None:
        n_categories = len(nodes[0].get("category_posterior") or [])
    P = np.zeros((len(nodes), n_categories), dtype=np.float64)
    for i, node in enumerate(nodes):
        cp = node.get("category_posterior") or []
        for j, entry in enumerate(cp[:n_categories]):
            P[i, j] = float(entry.get("p", 0.0) or 0.0)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return P / row_sums


__all__ = [
    "graph_smooth_posteriors",
    "spectral_consistency_score",
    "cg_node_index_map",
    "posteriors_array_from_scene_graph",
]

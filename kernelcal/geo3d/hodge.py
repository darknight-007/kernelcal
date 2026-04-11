"""Hodge Laplacian complex for triangle meshes.

Builds the three Hodge Laplacians arising from the simplicial chain complex
of a triangle mesh:

    0-chains (vertices) →[B₁]→ 1-chains (edges) →[B₂]→ 2-chains (faces)

Laplacians:
    L₀ = B₁ᵀ B₁                        (vertex,  n_V × n_V)
    L₁ = B₁ B₁ᵀ + B₂ᵀ B₂              (edge,    n_E × n_E)
    L₂ = B₂ B₂ᵀ                        (face,    n_F × n_F)

Hodge decomposition (on edge signals):
    any f ∈ ℝ^n_E = grad(g) ⊕ curl(h) ⊕ harmonic
    where grad = image(B₁ᵀ), curl = image(B₂ᵀ), harmonic = ker(L₁).

Betti numbers (topological invariants):
    β₀ = dim ker L₀   (connected components)
    β₁ = dim ker L₁   (independent 1-cycles / "handles")
    β₂ = dim ker L₂   (enclosed voids)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Canonical edge and face orderings
# ---------------------------------------------------------------------------

def _build_edge_index(faces: np.ndarray) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    """Return (edge → index, edges_array) from triangle faces.

    Edges are stored as (i, j) with i < j, in first-appearance order.
    """
    f = np.asarray(faces, dtype=int)
    edge_idx: dict[tuple[int, int], int] = {}
    for tri in f[:, :3]:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            if (u, v) not in edge_idx:
                edge_idx[(u, v)] = len(edge_idx)
    edges = np.array(sorted(edge_idx.keys(), key=lambda e: edge_idx[e]), dtype=int)
    return edge_idx, edges


# ---------------------------------------------------------------------------
# Boundary operators (sparse for memory efficiency)
# ---------------------------------------------------------------------------

def boundary_1(n_vertices: int, faces: np.ndarray) -> sp.csr_matrix:
    """Boundary operator B₁: ℝ^n_E → ℝ^n_V.

    B₁[v, e] = +1 if v is the head of edge e (higher-index endpoint),
               −1 if v is the tail (lower-index endpoint).

    Shape: (n_V, n_E).
    """
    edge_idx, edges = _build_edge_index(faces)
    n_E = len(edges)
    rows, cols, data = [], [], []
    for (i, j), eid in edge_idx.items():
        rows += [i, j]
        cols += [eid, eid]
        data += [-1.0, +1.0]
    B1 = sp.csr_matrix(
        (data, (rows, cols)), shape=(n_vertices, n_E), dtype=float
    )
    return B1


def boundary_2(faces: np.ndarray) -> sp.csr_matrix:
    """Boundary operator B₂: ℝ^n_F → ℝ^n_E.

    For face (a, b, c) ordered a < b < c, the boundary consists of three
    oriented edges; the sign follows the orientation convention.

    Shape: (n_E, n_F).
    """
    edge_idx, edges = _build_edge_index(faces)
    n_E = len(edges)
    f = np.asarray(faces, dtype=int)[:, :3]
    n_F = f.shape[0]
    rows, cols, data = [], [], []
    for fid, tri in enumerate(f):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            sign = +1.0 if u < v else -1.0
            key = (min(u, v), max(u, v))
            eid = edge_idx[key]
            rows.append(eid)
            cols.append(fid)
            data.append(sign)
    B2 = sp.csr_matrix(
        (data, (rows, cols)), shape=(n_E, n_F), dtype=float
    )
    return B2


# ---------------------------------------------------------------------------
# Hodge Laplacians (dense for eigendecomposition; sparse for large meshes)
# ---------------------------------------------------------------------------

def hodge_laplacian_0(B1: sp.csr_matrix) -> np.ndarray:
    """L₀ = B₁ᵀ B₁  (vertex Laplacian).  Shape: (n_V, n_V)."""
    return (B1 @ B1.T).toarray()


def hodge_laplacian_1(B1: sp.csr_matrix, B2: sp.csr_matrix) -> np.ndarray:
    """L₁ = B₁ B₁ᵀ + B₂ᵀ B₂  (edge Laplacian).  Shape: (n_E, n_E)."""
    L1_down = (B1.T @ B1).toarray()
    L1_up = (B2 @ B2.T).toarray()
    return L1_down + L1_up


def hodge_laplacian_2(B2: sp.csr_matrix) -> np.ndarray:
    """L₂ = B₂ᵀ B₂  (face Laplacian).  Shape: (n_F, n_F)."""
    return (B2.T @ B2).toarray()


# ---------------------------------------------------------------------------
# Betti numbers
# ---------------------------------------------------------------------------

def _null_dim(M: np.ndarray, tol: float = 1e-10) -> int:
    """Dimension of null space of M via numerical rank."""
    if M.shape[0] == 0 or M.shape[1] == 0:
        return max(M.shape[1], 0)
    return int(M.shape[1]) - int(np.linalg.matrix_rank(M, tol=tol))


def betti_numbers(
    n_vertices: int,
    faces: np.ndarray,
    tol: float = 1e-10,
) -> tuple[int, int, int]:
    """Compute Betti numbers (β₀, β₁, β₂) of the triangle mesh.

    β₀  connected components
    β₁  independent 1-cycles (handles/loops)
    β₂  enclosed voids

    For a closed orientable surface of genus g:
        β₀ = 1, β₁ = 2g, β₂ = 1, Euler characteristic = 2 - 2g.
    """
    f = np.asarray(faces, dtype=int)
    B1 = boundary_1(n_vertices, f)
    B2 = boundary_2(f)
    L0 = hodge_laplacian_0(B1)
    L1 = hodge_laplacian_1(B1, B2)
    L2 = hodge_laplacian_2(B2)
    b0 = _null_dim(L0, tol)
    b1 = _null_dim(L1, tol)
    b2 = _null_dim(L2, tol)
    return b0, b1, b2


# ---------------------------------------------------------------------------
# Hodge spectral basis
# ---------------------------------------------------------------------------

@dataclass
class HodgeSpectralBasis:
    """Eigenpairs for all three Hodge Laplacians.

    Attributes
    ----------
    eigenvalues_0, eigenvectors_0 : L₀ spectrum (vertex signals)
    eigenvalues_1, eigenvectors_1 : L₁ spectrum (edge signals)
    eigenvalues_2, eigenvectors_2 : L₂ spectrum (face signals)
    betti : (β₀, β₁, β₂)
    meta  : provenance
    """

    eigenvalues_0: np.ndarray
    eigenvectors_0: np.ndarray
    eigenvalues_1: np.ndarray
    eigenvectors_1: np.ndarray
    eigenvalues_2: np.ndarray
    eigenvectors_2: np.ndarray
    betti: tuple[int, int, int]
    meta: dict[str, Any]

    def harmonic_vertex_modes(self) -> np.ndarray:
        """Eigenvectors of L₀ with eigenvalue ≈ 0 (one per component)."""
        tol = 1e-9 * float(self.eigenvalues_0.max() or 1.0)
        mask = self.eigenvalues_0 < tol
        return self.eigenvectors_0[:, mask]

    def harmonic_edge_modes(self) -> np.ndarray:
        """Eigenvectors of L₁ with eigenvalue ≈ 0 (topology of 1-cycles)."""
        tol = 1e-9 * float(self.eigenvalues_1.max() or 1.0)
        mask = self.eigenvalues_1 < tol
        return self.eigenvectors_1[:, mask]


def build_hodge_basis(
    n_vertices: int,
    faces: np.ndarray,
    n_modes_0: int | None = None,
    n_modes_1: int | None = None,
    n_modes_2: int | None = None,
    tol_betti: float = 1e-10,
) -> HodgeSpectralBasis:
    """Compute Hodge Laplacian eigenpairs for all three chain levels.

    Truncates to the first ``n_modes_l`` eigenpairs for level l if given,
    keeping all modes if ``None``.
    """
    f = np.asarray(faces, dtype=int)
    B1 = boundary_1(n_vertices, f)
    B2 = boundary_2(f)

    L0 = hodge_laplacian_0(B1)
    L1 = hodge_laplacian_1(B1, B2)
    L2 = hodge_laplacian_2(B2)

    def _eigh_trunc(L: np.ndarray, k: int | None) -> tuple[np.ndarray, np.ndarray]:
        L = (L + L.T) / 2
        vals, vecs = np.linalg.eigh(L)
        if k is not None:
            k = min(k, len(vals))
            vals, vecs = vals[:k], vecs[:, :k]
        return vals, vecs

    lam0, phi0 = _eigh_trunc(L0, n_modes_0)
    lam1, phi1 = _eigh_trunc(L1, n_modes_1)
    lam2, phi2 = _eigh_trunc(L2, n_modes_2)

    b0 = _null_dim(L0, tol_betti)
    b1 = _null_dim(L1, tol_betti)
    b2 = _null_dim(L2, tol_betti)

    n_E = B2.shape[0]
    n_F = f.shape[0]
    meta = {
        "n_vertices": n_vertices,
        "n_edges": n_E,
        "n_faces": n_F,
        "n_modes_0": len(lam0),
        "n_modes_1": len(lam1),
        "n_modes_2": len(lam2),
        "euler_characteristic": n_vertices - n_E + n_F,
    }
    return HodgeSpectralBasis(
        eigenvalues_0=lam0,
        eigenvectors_0=phi0,
        eigenvalues_1=lam1,
        eigenvectors_1=phi1,
        eigenvalues_2=lam2,
        eigenvectors_2=phi2,
        betti=(b0, b1, b2),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Hodge decomposition of an edge signal
# ---------------------------------------------------------------------------

def hodge_decompose(
    edge_signal: np.ndarray,
    B1: sp.csr_matrix,
    B2: sp.csr_matrix,
    rcond: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose edge signal f = grad + curl + harmonic.

    Gradient lives in image(B₁ᵀ), i.e. columns of B₁ᵀ  ∈ ℝ^{n_E × n_V}.
    Curl lives in image(B₂),       i.e. columns of B₂    ∈ ℝ^{n_E × n_F}.
    Harmonic = ker(L₁).

    Returns
    -------
    grad      : gradient component (n_E,)
    curl      : curl component     (n_E,)
    harmonic  : harmonic component (n_E,)
    """
    f = np.asarray(edge_signal, dtype=float)

    # B1 : (n_V, n_E)  →  B1.T : (n_E, n_V)  maps vertex scalars → edge scalars
    # Gradient component: project f onto image of B1.T
    B1T = B1.T.toarray()  # (n_E, n_V)
    g_coeffs, *_ = np.linalg.lstsq(B1T, f, rcond=rcond)
    grad = B1T @ g_coeffs

    # B2 : (n_E, n_F)  maps face scalars → edge scalars
    # Curl component: project residual onto image of B2
    B2_dense = B2.toarray()  # (n_E, n_F)
    residual = f - grad
    h_coeffs, *_ = np.linalg.lstsq(B2_dense, residual, rcond=rcond)
    curl = B2_dense @ h_coeffs

    harmonic = f - grad - curl
    return grad, curl, harmonic

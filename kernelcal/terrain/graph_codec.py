"""Graph spectral telemetry: encode / decode for TerrainGraph twins.

Lossless path: transmit sparse edges + node attributes, rebuild L exactly.
Spectral attachment: store k smallest Laplacian eigenpairs for diagnostics
(kernelcal.terrain.diagnostics) on the receiver without full dense eigh.

Topology guard (decoder-style, aligned with geo3d): require
``k >= beta0 + beta1`` for the spectral attachment to meet the same
information budget as the 3D twin decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import struct
from typing import Any, BinaryIO

import numpy as np

from .dem import TerrainGraph, terrain_graph_laplacian
from .diagnostics import fixed_point_kernel, spectral_entropy_from_laplacian


def combinatorial_betti(num_nodes: int, edges: np.ndarray) -> tuple[int, int]:
    """Return (beta0, beta1) for an undirected simple graph on nodes 0..n-1."""
    n = int(num_nodes)
    if n <= 0:
        return 0, 0
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return n, 0
    adj: list[set[int]] = [set() for _ in range(n)]
    for e in edges.reshape(-1, 2):
        a, b = int(e[0]), int(e[1])
        if a == b or not (0 <= a < n and 0 <= b < n):
            continue
        adj[a].add(b)
        adj[b].add(a)
    visited = [False] * n
    beta0 = 0
    for v in range(n):
        if visited[v]:
            continue
        beta0 += 1
        stack = [v]
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            for w in adj[u]:
                if not visited[w]:
                    stack.append(w)
    e_count = edges.shape[0]
    beta1 = max(0, e_count - n + beta0)
    return beta0, beta1


def topology_guard_k(k: int, beta0: int, beta1: int) -> tuple[bool, str]:
    """Return (ok, message) for spectral mode budget vs Betti sum."""
    need = int(beta0) + int(beta1)
    kk = int(k)
    if kk >= need:
        return True, f"k={kk} >= beta0+beta1={need} (guard satisfied)"
    return False, f"k={kk} < beta0+beta1={need} (guard violated; increase k)"


@dataclass
class GraphSpectralPacket:
    """Serializable graph twin: lossless topology + optional Laplacian modes."""

    version: int
    n_nodes: int
    edges: np.ndarray  # (E, 2) int32
    weights: np.ndarray  # (E,) float64
    positions: np.ndarray  # (N, 2) float64
    elevations: np.ndarray  # (N,) float64
    shape_rc: tuple[int, int]
    cell_index: np.ndarray | None  # optional (H, W) int32, -1 absent
    beta0: int
    beta1: int
    lam_k: np.ndarray  # (k,) smallest k eigenvalues (ascending)
    U_k: np.ndarray  # (N, k) matching columns

    def k_transmitted(self) -> int:
        return int(self.lam_k.shape[0])


# ---------------------------------------------------------------------------
# Streamable binary packet format (.kcg)
# ---------------------------------------------------------------------------

# Layout (little-endian):
#   magic(4) = b"KCG1"
#   flags(u16): bit0 -> has_cell_index
#   n_arrays(u16)
#   n_nodes(i32), shape_h(i32), shape_w(i32), beta0(i32), beta1(i32), k(i32)
# Then repeated array records:
#   name_len(u8), name(bytes ascii)
#   dtype_code(u8), ndim(u8), reserved(u16)
#   shape[i32] * ndim
#   nbytes(u64)
#   payload(raw bytes, C-order)

_KCG_MAGIC = b"KCG1"
_KCG_HEADER_STRUCT = struct.Struct("<4sHHiiiiii")
_KCG_ARRAY_PREFIX_STRUCT = struct.Struct("<BBBH")
_KCG_NBYTES_STRUCT = struct.Struct("<Q")

_DTYPE_CODE_TO_STR = {
    1: "<i4",
    2: "<f8",
    3: "<f4",
}
_DTYPE_STR_TO_CODE = {
    "<i4": 1,
    "<f8": 2,
    "<f4": 3,
}


def _read_exact(fp: BinaryIO, n: int) -> bytes:
    b = fp.read(n)
    if b is None or len(b) != n:
        raise ValueError(f"Unexpected EOF while reading {n} bytes.")
    return b


def _pack_array_record(fp: BinaryIO, name: str, arr: np.ndarray) -> None:
    if not name.isascii():
        raise ValueError(f"Array field name must be ASCII: {name!r}")
    nb = name.encode("ascii")
    if len(nb) > 255:
        raise ValueError(f"Array field name too long: {name!r}")

    a = np.ascontiguousarray(arr)
    dt = a.dtype.newbyteorder("<")
    dts = dt.str
    if dts not in _DTYPE_STR_TO_CODE:
        raise ValueError(f"Unsupported dtype for stream codec: {a.dtype!r} ({dts})")
    code = _DTYPE_STR_TO_CODE[dts]
    ndim = int(a.ndim)
    if ndim > 255:
        raise ValueError(f"Array ndim too large for stream codec: {ndim}")

    fp.write(_KCG_ARRAY_PREFIX_STRUCT.pack(len(nb), code, ndim, 0))
    fp.write(nb)
    if ndim > 0:
        fp.write(struct.pack("<" + "i" * ndim, *[int(x) for x in a.shape]))
    payload = a.astype(dt, copy=False).tobytes(order="C")
    fp.write(_KCG_NBYTES_STRUCT.pack(len(payload)))
    fp.write(payload)


def _unpack_array_record(fp: BinaryIO) -> tuple[str, np.ndarray]:
    name_len, code, ndim, _ = _KCG_ARRAY_PREFIX_STRUCT.unpack(_read_exact(fp, _KCG_ARRAY_PREFIX_STRUCT.size))
    name = _read_exact(fp, int(name_len)).decode("ascii")
    shape = ()
    if int(ndim) > 0:
        shape = struct.unpack("<" + "i" * int(ndim), _read_exact(fp, 4 * int(ndim)))
    nbytes = _KCG_NBYTES_STRUCT.unpack(_read_exact(fp, _KCG_NBYTES_STRUCT.size))[0]
    payload = _read_exact(fp, int(nbytes))
    dts = _DTYPE_CODE_TO_STR.get(int(code))
    if dts is None:
        raise ValueError(f"Unknown dtype code in stream codec: {code}")
    dt = np.dtype(dts)
    arr = np.frombuffer(payload, dtype=dt)
    exp = int(np.prod(shape, dtype=np.int64)) if len(shape) > 0 else 1
    if exp * dt.itemsize != int(nbytes):
        raise ValueError(
            f"Array byte-size mismatch for field {name!r}: expected {exp * dt.itemsize}, got {nbytes}"
        )
    arr = arr.reshape(shape if len(shape) > 0 else ())
    return name, arr.copy()


def packet_to_stream_bytes(pkt: GraphSpectralPacket) -> bytes:
    """Serialize packet to a compact stream-friendly binary blob (.kcg)."""
    bio = BytesIO()
    write_packet_stream(pkt, bio)
    return bio.getvalue()


def packet_from_stream_bytes(data: bytes) -> GraphSpectralPacket:
    """Deserialize packet from stream-friendly binary blob (.kcg)."""
    bio = BytesIO(data)
    return read_packet_stream(bio)


def write_packet_stream(pkt: GraphSpectralPacket, fp: BinaryIO) -> None:
    """Write packet to a binary stream in .kcg format."""
    arrays: list[tuple[str, np.ndarray]] = [
        ("edges", np.asarray(pkt.edges, dtype=np.int32)),
        ("weights", np.asarray(pkt.weights, dtype=np.float64)),
        ("positions", np.asarray(pkt.positions, dtype=np.float64)),
        ("elevations", np.asarray(pkt.elevations, dtype=np.float64)),
        ("lam_k", np.asarray(pkt.lam_k, dtype=np.float64)),
        ("U_k", np.asarray(pkt.U_k, dtype=np.float64)),
    ]
    has_ci = pkt.cell_index is not None
    if has_ci:
        arrays.append(("cell_index", np.asarray(pkt.cell_index, dtype=np.int32)))
    flags = 1 if has_ci else 0
    fp.write(
        _KCG_HEADER_STRUCT.pack(
            _KCG_MAGIC,
            int(flags),
            int(len(arrays)),
            int(pkt.n_nodes),
            int(pkt.shape_rc[0]),
            int(pkt.shape_rc[1]),
            int(pkt.beta0),
            int(pkt.beta1),
            int(pkt.k_transmitted()),
        )
    )
    for name, arr in arrays:
        _pack_array_record(fp, name, arr)


def read_packet_stream(fp: BinaryIO) -> GraphSpectralPacket:
    """Read packet from a binary stream in .kcg format."""
    hdr = _KCG_HEADER_STRUCT.unpack(_read_exact(fp, _KCG_HEADER_STRUCT.size))
    magic, flags, n_arrays, n_nodes, shape_h, shape_w, beta0, beta1, k = hdr
    if magic != _KCG_MAGIC:
        raise ValueError(f"Invalid KCG magic: {magic!r}")
    has_ci = bool(int(flags) & 1)
    arrs: dict[str, np.ndarray] = {}
    for _ in range(int(n_arrays)):
        name, arr = _unpack_array_record(fp)
        arrs[name] = arr
    required = {"edges", "weights", "positions", "elevations", "lam_k", "U_k"}
    miss = sorted(required - set(arrs.keys()))
    if miss:
        raise ValueError(f"Missing required arrays in KCG payload: {miss}")
    ci = arrs.get("cell_index")
    if has_ci and ci is None:
        raise ValueError("KCG header says cell_index is present, but array is missing.")
    lam_k = np.asarray(arrs["lam_k"], dtype=float).reshape(-1)
    if int(k) != int(lam_k.shape[0]):
        raise ValueError(f"KCG k mismatch: header k={k}, lam_k size={lam_k.shape[0]}")
    return GraphSpectralPacket(
        version=1,
        n_nodes=int(n_nodes),
        edges=np.asarray(arrs["edges"], dtype=np.int32).reshape(-1, 2),
        weights=np.asarray(arrs["weights"], dtype=float).reshape(-1),
        positions=np.asarray(arrs["positions"], dtype=float).reshape(int(n_nodes), 2),
        elevations=np.asarray(arrs["elevations"], dtype=float).reshape(int(n_nodes)),
        shape_rc=(int(shape_h), int(shape_w)),
        cell_index=np.asarray(ci, dtype=np.int32) if ci is not None else None,
        beta0=int(beta0),
        beta1=int(beta1),
        lam_k=lam_k,
        U_k=np.asarray(arrs["U_k"], dtype=float).reshape(int(n_nodes), int(k)),
    )


def write_packet_stream_file(pkt: GraphSpectralPacket, path: str) -> None:
    """Write stream packet to a file path (binary .kcg)."""
    p = path if isinstance(path, str) else str(path)
    import os
    parent = os.path.dirname(p)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(p, "wb") as f:
        write_packet_stream(pkt, f)


def read_packet_stream_file(path: str) -> GraphSpectralPacket:
    """Read stream packet from a file path (binary .kcg)."""
    with open(path, "rb") as f:
        return read_packet_stream(f)


def terrain_graph_from_packet(pkt: GraphSpectralPacket) -> TerrainGraph:
    """Rebuild TerrainGraph from lossless fields."""
    h, w = int(pkt.shape_rc[0]), int(pkt.shape_rc[1])
    if pkt.cell_index is not None and pkt.cell_index.shape == (h, w):
        ci = np.asarray(pkt.cell_index, dtype=np.int32)
    else:
        ci = np.full((h, w), -1, dtype=np.int32)
    return TerrainGraph(
        positions=np.asarray(pkt.positions, dtype=float),
        elevations=np.asarray(pkt.elevations, dtype=float),
        edges=np.asarray(pkt.edges, dtype=np.int32),
        weights=np.asarray(pkt.weights, dtype=float),
        shape=(h, w),
        cell_index=ci,
    )


def encode_graph_packet(
    tg: TerrainGraph,
    k_modes: int,
    *,
    store_cell_index: bool = False,
) -> GraphSpectralPacket:
    """Build packet: combinatorial Betti, full Laplacian eigh, keep k smallest modes."""
    edges = np.asarray(tg.edges, dtype=np.int32)
    w = np.asarray(tg.weights, dtype=float)
    pos = np.asarray(tg.positions, dtype=float)
    z = np.asarray(tg.elevations, dtype=float)
    n = int(pos.shape[0])
    beta0, beta1 = combinatorial_betti(n, edges)
    L = terrain_graph_laplacian(tg)
    lam_all, U_all = np.linalg.eigh(L)
    kk = max(1, min(int(k_modes), n))
    lam_k = lam_all[:kk].copy()
    U_k = U_all[:, :kk].copy()
    nrows, ncols = int(tg.shape[0]), int(tg.shape[1])
    ci = np.asarray(tg.cell_index, dtype=np.int32) if store_cell_index else None
    return GraphSpectralPacket(
        version=1,
        n_nodes=n,
        edges=edges,
        weights=w,
        positions=pos,
        elevations=z,
        shape_rc=(nrows, ncols),
        cell_index=ci,
        beta0=beta0,
        beta1=beta1,
        lam_k=lam_k,
        U_k=U_k,
    )


def decode_graph_packet(pkt: GraphSpectralPacket) -> TerrainGraph:
    """Lossless decode of graph structure and attributes."""
    return terrain_graph_from_packet(pkt)


def lossless_sparse_graph_npz_bytes(
    tg: TerrainGraph,
    *,
    compress: bool = True,
    store_cell_index: bool = False,
) -> int:
    """Compressed npz size for sparse graph data only (no eigenpairs, no Betti ints).

    Uses the same ``np.savez_compressed`` path as ``packet_to_npz_bytes`` so
    sizes are comparable to the full telemetry blob. Omits ``lam_k`` / ``U_k``
    and small scalar arrays present in the full packet — a practical **honest
    baseline** for "send the graph, rebuild ``L`` locally".

    Parameters
    ----------
    store_cell_index
        If True, include ``cell_index`` when it matches ``tg.shape`` (matches
        ``encode_graph_packet(..., store_cell_index=True)``).
    """
    bio = BytesIO()
    edges = np.asarray(tg.edges, dtype=np.int32)
    weights = np.asarray(tg.weights, dtype=float)
    positions = np.asarray(tg.positions, dtype=float)
    elevations = np.asarray(tg.elevations, dtype=float)
    nrows, ncols = int(tg.shape[0]), int(tg.shape[1])
    save_kw: dict[str, Any] = dict(
        n_nodes=np.array([int(elevations.shape[0])], dtype=np.int32),
        edges=edges,
        weights=weights,
        positions=positions,
        elevations=elevations,
        shape_h=np.array([nrows], dtype=np.int32),
        shape_w=np.array([ncols], dtype=np.int32),
    )
    if store_cell_index:
        ci = np.asarray(tg.cell_index, dtype=np.int32)
        if ci.shape == (nrows, ncols):
            save_kw["cell_index"] = ci
    if compress:
        np.savez_compressed(bio, **save_kw)
    else:
        np.savez(bio, **save_kw)
    return len(bio.getvalue())


def packet_to_npz_bytes(pkt: GraphSpectralPacket, *, compress: bool = True) -> bytes:
    """Serialize packet to compressed npz bytes (lossless + spectral attachment)."""
    bio = BytesIO()
    ci = pkt.cell_index
    save_kw: dict[str, Any] = dict(
        version=np.array([pkt.version], dtype=np.int32),
        n_nodes=np.array([pkt.n_nodes], dtype=np.int32),
        edges=pkt.edges,
        weights=pkt.weights,
        positions=pkt.positions,
        elevations=pkt.elevations,
        shape_h=np.array([pkt.shape_rc[0]], dtype=np.int32),
        shape_w=np.array([pkt.shape_rc[1]], dtype=np.int32),
        beta0=np.array([pkt.beta0], dtype=np.int32),
        beta1=np.array([pkt.beta1], dtype=np.int32),
        lam_k=pkt.lam_k,
        U_k=pkt.U_k,
    )
    if ci is not None:
        save_kw["cell_index"] = ci
    np.savez_compressed(bio, **save_kw) if compress else np.savez(bio, **save_kw)
    return bio.getvalue()


def packet_from_npz_bytes(data: bytes) -> GraphSpectralPacket:
    """Load packet from npz bytes."""
    bio = BytesIO(data)
    z = np.load(bio, allow_pickle=False)
    h = int(z["shape_h"][0])
    w = int(z["shape_w"][0])
    ci = z["cell_index"] if "cell_index" in z.files else None
    return GraphSpectralPacket(
        version=int(z["version"][0]),
        n_nodes=int(z["n_nodes"][0]),
        edges=z["edges"],
        weights=z["weights"],
        positions=z["positions"],
        elevations=z["elevations"],
        shape_rc=(h, w),
        cell_index=ci,
        beta0=int(z["beta0"][0]),
        beta1=int(z["beta1"][0]),
        lam_k=z["lam_k"],
        U_k=z["U_k"],
    )


def low_rank_laplacian_from_packet(pkt: GraphSpectralPacket) -> np.ndarray:
    """Reconstruct rank-k Gram approximation U_k diag(lam_k) U_k^T (not exact L)."""
    Uk = np.asarray(pkt.U_k, dtype=float)
    lk = np.asarray(pkt.lam_k, dtype=float)
    return (Uk * lk) @ Uk.T


def kernelcal_graph_diagnostics(L: np.ndarray) -> dict[str, float]:
    """Run kernelcal terrain diagnostics on a Laplacian matrix."""
    H = spectral_entropy_from_laplacian(L)
    _h_star, fp_info = fixed_point_kernel(L)
    return {
        "spectral_entropy": float(H),
        "fixed_point_residual": float(fp_info["residual"]),
    }


def reconstruction_frobenius(L_true: np.ndarray, L_hat: np.ndarray) -> float:
    d = L_true - L_hat
    return float(np.linalg.norm(d, ord="fro"))

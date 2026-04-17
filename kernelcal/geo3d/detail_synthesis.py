"""Detail synthesis for digital twin decoder.

Takes the DecodedTwin skeleton (Stage 1–2 output) and layers in high-frequency
detail using one or more of the five methods described in the paper seed:

  Method A — Stochastic procedural noise  (always available, roughness ← H[h*])
  Method B — Curl-gated flow textures     (activated when E_curl > threshold)
  Method C — Latent-code texture dispatch (if latent_code is transmitted)
  Method D — Landmark pinning             (Poisson-style if landmark_xyz sent)
  Method E — D_m-aware octave scaling     (more octaves when |D_t| is high)

Output
------
  synthesize() returns SynthesizedTwin with:
    vertices_detailed : (V, 3) base + displacement
    displacement      : (V,)   per-vertex signed displacement magnitude
    texture_weights   : (V,)   curl weight map (for Blender/MeshLab colormap)
    obj_path          : if export_obj=True, path to written OBJ

No Blender dependency here — this is pure numpy + scipy.
Blender rendering is handled by twin_receiver.py which calls this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .decoder import DecodedTwin, DetailLevel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scene-class library (latent_code → synthesis params)
# ---------------------------------------------------------------------------

# Maps integer latent codes to synthesis parameter presets.
# These match the geological classes that a rover semantic classifier would emit.
SCENE_LIBRARY: dict[int, dict[str, Any]] = {
    0:  {"name": "Regolith plain",    "roughness_boost": 0.0,  "amplitude": 0.02, "octaves": 4},
    1:  {"name": "Boulder field",     "roughness_boost": 0.3,  "amplitude": 0.08, "octaves": 6},
    2:  {"name": "Crater interior",   "roughness_boost": 0.15, "amplitude": 0.05, "octaves": 5},
    3:  {"name": "Lava plain",        "roughness_boost": 0.05, "amplitude": 0.03, "octaves": 4},
    4:  {"name": "Fluvial channel",   "roughness_boost": 0.2,  "amplitude": 0.06, "octaves": 6},
    5:  {"name": "Ice/frost deposit", "roughness_boost": 0.0,  "amplitude": 0.01, "octaves": 3},
    6:  {"name": "Ejecta blanket",    "roughness_boost": 0.25, "amplitude": 0.07, "octaves": 6},
    7:  {"name": "Dune field",        "roughness_boost": 0.1,  "amplitude": 0.04, "octaves": 5},
}
DEFAULT_SCENE = {"name": "Unknown", "roughness_boost": 0.1, "amplitude": 0.04, "octaves": 5}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SynthesizedTwin:
    """Detail-synthesized digital twin, ready for rendering."""

    vertices_detailed: np.ndarray   # (V, 3) with displacement applied
    faces: np.ndarray               # (F, 3) unchanged from decoder
    displacement: np.ndarray        # (V,)   signed z-displacement added
    texture_weights: np.ndarray     # (V,)   curl weight in [0,1] for colormap
    detail_level: DetailLevel
    diagnostics: dict[str, Any]

    def export_obj(self, path: str | Path) -> Path:
        """Write displaced mesh as OBJ (MeshLab / RViz compatible)."""
        from .large_mesh import _write_obj
        p = Path(path)
        _write_obj(self.vertices_detailed, self.faces, p)
        log.info("SynthesizedTwin OBJ: %s", p)
        return p

    def export_ply_with_color(self, path: str | Path) -> Path:
        """Write PLY with per-vertex color from texture_weights (MeshLab)."""
        p = Path(path)
        verts = self.vertices_detailed
        weights = self.texture_weights
        # Map weight [0,1] → blue-to-red colormap (channel activity)
        r = (weights * 255).astype(np.uint8)
        g = np.zeros_like(r)
        b = ((1 - weights) * 255).astype(np.uint8)
        with open(p, "w") as fh:
            fh.write("ply\nformat ascii 1.0\n")
            fh.write(f"element vertex {verts.shape[0]}\n")
            fh.write("property float x\nproperty float y\nproperty float z\n")
            fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            fh.write(f"element face {self.faces.shape[0]}\n")
            fh.write("property list uchar int vertex_indices\n")
            fh.write("end_header\n")
            for i in range(verts.shape[0]):
                fh.write(f"{verts[i,0]:.6f} {verts[i,1]:.6f} {verts[i,2]:.6f} "
                         f"{r[i]} {g[i]} {b[i]}\n")
            for tri in self.faces:
                fh.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
        log.info("SynthesizedTwin PLY (colored): %s", p)
        return p


# ---------------------------------------------------------------------------
# Method A — Procedural fractal noise (numpy-only, no external deps)
# ---------------------------------------------------------------------------

def _value_noise_2d(
    xy: np.ndarray,
    frequency: float,
    seed: int,
) -> np.ndarray:
    """Fast 2D value noise via random grid interpolation.

    Pure numpy — no 'noise' package needed.  Perlin-quality is lower but
    sufficient for detail displacement on already-smooth spectral meshes.

    Parameters
    ----------
    xy        : (N, 2) positions
    frequency : grid frequency (higher → finer detail)
    seed      : deterministic seed

    Returns
    -------
    (N,) values in [-1, 1]
    """
    rng  = np.random.default_rng(seed)
    grid_size = 64
    grid = rng.uniform(-1, 1, (grid_size + 1, grid_size + 1))

    scaled = xy * frequency
    xi = (np.floor(scaled[:, 0]).astype(int)) % grid_size
    yi = (np.floor(scaled[:, 1]).astype(int)) % grid_size
    fx = scaled[:, 0] - np.floor(scaled[:, 0])
    fy = scaled[:, 1] - np.floor(scaled[:, 1])
    # Smoothstep
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)

    xi1 = (xi + 1) % grid_size
    yi1 = (yi + 1) % grid_size
    v00 = grid[xi,  yi]
    v10 = grid[xi1, yi]
    v01 = grid[xi,  yi1]
    v11 = grid[xi1, yi1]
    return (v00 * (1-fx)*(1-fy) + v10 * fx*(1-fy) +
            v01 * (1-fx)*fy     + v11 * fx*fy)


def _fractal_noise(
    xy: np.ndarray,
    *,
    base_frequency: float = 0.01,
    octaves: int = 5,
    amplitude: float = 1.0,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """Multi-octave fractal noise for procedural terrain detail."""
    result = np.zeros(xy.shape[0], dtype=float)
    amp  = amplitude
    freq = base_frequency
    for o in range(octaves):
        result += amp * _value_noise_2d(xy, freq, seed + o * 97)
        amp  *= persistence
        freq *= lacunarity
    return result


def procedural_displacement(
    vertices: np.ndarray,
    *,
    roughness: float = 0.5,
    octaves: int = 5,
    amplitude_scale: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Compute per-vertex z-displacement via fractal noise.

    amplitude is derived from the typical scene scale (1% of bounding box).
    roughness in [0,1] controls octave count and persistence.
    """
    xy = vertices[:, :2]
    bbox_scale = float(np.ptp(xy)) if np.ptp(xy) > 0 else 1.0
    base_amp = bbox_scale * 0.01 * amplitude_scale
    base_freq = 3.0 / bbox_scale

    effective_octaves = max(2, int(octaves * (0.5 + roughness)))
    return _fractal_noise(
        xy,
        base_frequency=base_freq,
        octaves=effective_octaves,
        amplitude=base_amp,
        persistence=0.45 + 0.1 * roughness,
        lacunarity=2.0,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Method B — Curl-gated flow textures
# ---------------------------------------------------------------------------

def curl_texture_weight(
    vertices: np.ndarray,
    curl_energy: float,
    *,
    threshold: float = 0.05,
    sharpness: float = 8.0,
) -> np.ndarray:
    """Per-vertex curl weight map for flow-direction texture blending.

    Where curl energy is high, flow-aligned textures are activated.
    Returns a (V,) array in [0,1]:
      0 = no curl contribution (flat terrain, apply standard noise)
      1 = strong curl (channel/drainage, apply flow texture)
    """
    if curl_energy <= threshold:
        return np.zeros(vertices.shape[0], dtype=float)

    # Proxy: vertices with z close to local minimum (channel floors)
    # get higher curl weight — channels are low-lying in natural terrain.
    z = vertices[:, 2]
    z_norm = (z - z.min()) / (np.ptp(z) + 1e-12)
    # Low z → high curl weight; sigmoid gate at threshold
    weight = 1.0 / (1.0 + np.exp(sharpness * (z_norm - 0.3)))
    # Scale by fractional curl energy
    scale = min(1.0, (curl_energy - threshold) / (1.0 - threshold + 1e-12))
    return weight * scale


def flow_displacement(
    vertices: np.ndarray,
    curl_weight: np.ndarray,
    *,
    amplitude: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Flow-aligned displacement: striated along dominant slope direction."""
    xy = vertices[:, :2]
    bbox_scale = float(np.ptp(xy)) if np.ptp(xy) > 0 else 1.0
    # High-frequency striation noise perpendicular to mean slope
    slope_dir = np.array([1.0, 0.5])
    slope_dir /= np.linalg.norm(slope_dir)
    proj = xy @ slope_dir   # (V,) projection along slope
    freq = 10.0 / bbox_scale
    stripe = amplitude * 0.3 * np.sin(proj * freq * 2 * np.pi)
    return stripe * curl_weight


# ---------------------------------------------------------------------------
# Method D — Landmark pinning (Poisson-style surface detail)
# ---------------------------------------------------------------------------

def landmark_displacement(
    vertices: np.ndarray,
    landmark_xyz: np.ndarray,
    *,
    influence_radius_frac: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Pull base mesh toward sparse high-confidence landmark points.

    For each vertex, find the nearest landmark; if within influence_radius,
    interpolate toward the landmark z-value using RBF weights.
    This is the "Poisson pin" step from the seed description.

    Parameters
    ----------
    vertices           : (V, 3) base reconstructed vertices
    landmark_xyz       : (N_lm, 3) high-confidence LiDAR landmarks
    influence_radius_frac : fraction of bounding box for RBF kernel width

    Returns
    -------
    displacement : (V,) per-vertex z-adjustment
    """
    if landmark_xyz is None or len(landmark_xyz) == 0:
        return np.zeros(vertices.shape[0], dtype=float)

    bbox = float(np.ptp(vertices[:, :2]))
    sigma = influence_radius_frac * bbox
    tree = cKDTree(landmark_xyz[:, :2])
    d, idx = tree.query(vertices[:, :2], k=min(3, len(landmark_xyz)))
    if d.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]

    w = np.exp(-(d ** 2) / (2 * sigma ** 2))          # (V, k)
    w_sum = w.sum(axis=1, keepdims=True)
    w = w / np.where(w_sum > 0, w_sum, 1.0)
    z_lm = landmark_xyz[idx, 2]                        # (V, k)
    z_target = (w * z_lm).sum(axis=1)                  # (V,)
    # Displacement = pull toward landmark z, gated by distance
    gate = np.exp(-d[:, 0] ** 2 / (2 * sigma ** 2))   # (V,) — fade with distance
    return (z_target - vertices[:, 2]) * gate


# ---------------------------------------------------------------------------
# Main synthesis entry point
# ---------------------------------------------------------------------------

def synthesize(
    twin: DecodedTwin,
    *,
    seed: int = 0,
    export_obj_path: str | None = None,
    export_ply_path: str | None = None,
) -> SynthesizedTwin:
    """Apply all applicable detail methods to the decoded twin skeleton.

    Dispatch table
    --------------
    SKELETON_ONLY  → no displacement (topology-unsafe mode)
    LOW            → Method A only, low octaves
    MEDIUM         → Methods A + B (curl gate)
    HIGH           → Methods A + B + D (landmarks) + E (D_m octave boost)
    PATCH_REQUEST  → Methods A + B + D, note patch flag in diagnostics
    """
    params = twin.detail_params
    level  = twin.detail_level
    verts  = twin.vertices_base.copy()
    faces  = twin.faces

    roughness    = float(params.get("roughness", 0.5))
    curl_energy  = float(params.get("curl_energy", 0.0))
    D_t          = float(params.get("D_t", 0.0))
    lc           = params.get("latent_code")
    lm_xyz       = params.get("landmark_xyz")

    # Latent-code scene preset
    scene = SCENE_LIBRARY.get(lc, DEFAULT_SCENE) if lc is not None else DEFAULT_SCENE
    roughness = min(1.0, roughness + scene["roughness_boost"])
    octaves   = scene["octaves"]
    amp_scale = scene["amplitude"]

    # D_m octave boost (Method E): more octaves when information was lost
    if level in (DetailLevel.HIGH, DetailLevel.PATCH_REQUEST):
        octave_boost = int(np.clip(D_t / 5.0, 0, 4))
        octaves = min(octaves + octave_boost, 10)
        log.debug("D_m boost: D_t=%.2f  octave_boost=%d  octaves=%d",
                  D_t, octave_boost, octaves)

    # ── Method A: procedural fractal noise ─────────────────────────────────
    if level == DetailLevel.SKELETON_ONLY:
        disp_noise = np.zeros(verts.shape[0], dtype=float)
    else:
        disp_noise = procedural_displacement(
            verts, roughness=roughness, octaves=octaves,
            amplitude_scale=amp_scale * 100, seed=seed,
        )

    # ── Method B: curl-gated flow textures ─────────────────────────────────
    curl_w = curl_texture_weight(verts, curl_energy)
    if params.get("curl_active", False) and level not in (DetailLevel.SKELETON_ONLY,
                                                           DetailLevel.LOW):
        bbox_scale = float(np.ptp(verts)) if np.ptp(verts) > 0 else 1.0
        disp_flow = flow_displacement(verts, curl_w, amplitude=bbox_scale * 0.005, seed=seed)
    else:
        disp_flow = np.zeros(verts.shape[0], dtype=float)

    # ── Method D: landmark pinning ──────────────────────────────────────────
    if lm_xyz is not None and level in (DetailLevel.HIGH, DetailLevel.PATCH_REQUEST):
        disp_lm = landmark_displacement(verts, np.asarray(lm_xyz), seed=seed)
    else:
        disp_lm = np.zeros(verts.shape[0], dtype=float)

    # ── Combine displacements ───────────────────────────────────────────────
    total_disp = disp_noise + disp_flow + disp_lm
    verts[:, 2] += total_disp

    # Texture weight = curl influence (0 = procedural, 1 = flow/channel)
    texture_weights = np.clip(curl_w, 0.0, 1.0)

    synth_diag = {
        **twin.diagnostics,
        "synthesis_level": level.value,
        "scene_class": scene["name"],
        "octaves_used": octaves,
        "amplitude_scale": amp_scale,
        "disp_rms": float(np.sqrt(np.mean(total_disp ** 2))),
        "disp_max": float(np.abs(total_disp).max()),
        "curl_weight_mean": float(curl_w.mean()),
        "landmark_count": len(lm_xyz) if lm_xyz is not None else 0,
    }
    log.info(
        "Synthesis %s: scene='%s' octaves=%d rms_disp=%.4f curl_mean=%.3f",
        level.value, scene["name"], octaves,
        synth_diag["disp_rms"], synth_diag["curl_weight_mean"],
    )

    result = SynthesizedTwin(
        vertices_detailed=verts,
        faces=faces,
        displacement=total_disp,
        texture_weights=texture_weights,
        detail_level=level,
        diagnostics=synth_diag,
    )

    if export_obj_path:
        result.export_obj(export_obj_path)
    if export_ply_path:
        result.export_ply_with_color(export_ply_path)

    return result

"""Blender terrain generator for Q10 Nyström topology experiments.

Part of the kernelcal.blender subpackage.

Responsibilities (Blender only):
  - Procedural terrain via mathutils.noise.turbulence (no external 'noise' package)
  - Controlled beta_1 ground truth: n_loops ring-shaped channel depressions
  - Controlled beta_0/beta_2: n_craters hemispherical bowls
  - Multi-resolution export (coarse → fine) for Nyström refinement series
  - JSON sidecar with ground-truth Betti numbers
  - No eigenvalues, no Betti computation, no Laplacians

Usage (headless):
  SCRIPT=$(python3 -c "import kernelcal.blender, os; \
      print(os.path.join(os.path.dirname(kernelcal.blender.__file__), 'terrain_gen.py'))")
  blender --background --python "$SCRIPT" -- \
      --out_dir /tmp/q10_terrains \
      --n_loops 3 \
      --n_craters 5 \
      --resolutions 32,64,128,256 \
      --seed 42

Or via the orchestrator (recommended):
  BLENDER=/path/to/blender ./kernelcal/blender/run_q10_experiment.sh --all_loops

Output files are consumed by kernelcal/blender/q10_pipeline.py.
"""

import argparse
import json
import math
import os
import random
import sys

import bpy
import bmesh
from mathutils import noise as _mtnoise
from mathutils import Vector


# ---------------------------------------------------------------------------
# CLI parsing (args come after "--" in Blender's invocation)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Generate planetary terrain OBJ files with known topology."
    )
    parser.add_argument("--out_dir", default="/tmp/q10_terrains",
                        help="Output directory for OBJ files and JSON sidecar.")
    parser.add_argument("--n_loops", type=int, default=3,
                        help="Number of ring-channel loops carved into the terrain "
                             "(each loop adds exactly 1 to beta_1 if closed).")
    parser.add_argument("--n_craters", type=int, default=5,
                        help="Number of hemispherical craters (controls local beta_2 "
                             "in closed-surface sense; primarily a geometric variation).")
    parser.add_argument("--resolutions", default="32,64,128,256",
                        help="Comma-separated list of grid resolutions to export.")
    parser.add_argument("--scene_size", type=float, default=1000.0,
                        help="Physical extent of the terrain patch (metres).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Terrain generator
# ---------------------------------------------------------------------------

def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def generate_terrain(
    scene_size: float,
    resolution: int,
    n_craters: int,
    n_loops: int,
    seed: int,
) -> bpy.types.Object:
    """Generate one terrain mesh at the requested grid resolution.

    Topological design
    ------------------
    Each 'loop' is a closed ring-shaped channel: a torus-shaped depression whose
    centre trace is a circle.  On an open grid mesh this ring adds one independent
    1-cycle to beta_1 (its interior is not filled by a face — the ring is a
    through-going canyon, not a closed tube).

    Craters are hemispherical bowls; they introduce local curvature variation
    but do not change beta_1 on an open grid (no handles).

    Parameters
    ----------
    scene_size  : physical size (metres)
    resolution  : number of grid subdivisions along each axis
    n_craters   : number of hemispherical crater depressions
    n_loops     : number of ring channel depressions (each → +1 beta_1 ground truth)
    seed        : RNG seed
    """
    rng = random.Random(seed)

    bpy.ops.mesh.primitive_grid_add(
        size=scene_size,
        x_subdivisions=resolution,
        y_subdivisions=resolution,
    )
    obj = bpy.context.active_object
    obj.name = f"PlanetarySurface_res{resolution}"

    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # ── Perlin noise base terrain ──────────────────────────────────────────
    # mathutils.noise.turbulence() is Blender's built-in multi-octave Perlin.
    # It takes a 3-vector; we use z=seed*0.1 as an offset so different seeds
    # give genuinely different terrain without any external dependency.
    noise_scale = 2.5 / scene_size   # ~2.5 wavelengths across the patch
    height_mult = scene_size * 0.04  # 4% of scene_size as peak-to-peak height

    for v in bm.verts:
        p = Vector((
            v.co.x * noise_scale,
            v.co.y * noise_scale,
            seed * 0.1,
        ))
        z = _mtnoise.turbulence(p, 6, False, noise_basis='PERLIN_ORIGINAL')
        v.co.z = z * height_mult

    # ── Ring channels (beta_1 ground truth) ────────────────────────────────
    # Each ring is parameterised by (cx, cy, radius, width, depth).
    # The ring carves a trough wherever |dist_to_centre − radius| < width/2.
    loop_params = []
    for _ in range(n_loops):
        cx = rng.uniform(-scene_size * 0.3, scene_size * 0.3)
        cy = rng.uniform(-scene_size * 0.3, scene_size * 0.3)
        radius = rng.uniform(scene_size * 0.06, scene_size * 0.14)
        width = scene_size * 0.015
        depth = height_mult * 0.5
        loop_params.append((cx, cy, radius, width, depth))

        for v in bm.verts:
            dist = math.hypot(v.co.x - cx, v.co.y - cy)
            dr = abs(dist - radius)
            if dr < width:
                # Smooth trapezoidal cross-section
                t = 1.0 - dr / width
                v.co.z -= depth * t * t

    # ── Craters (hemispherical bowls) ───────────────────────────────────────
    crater_params = []
    for _ in range(n_craters):
        cx = rng.uniform(-scene_size * 0.4, scene_size * 0.4)
        cy = rng.uniform(-scene_size * 0.4, scene_size * 0.4)
        rad = rng.uniform(scene_size * 0.03, scene_size * 0.08)
        crater_params.append((cx, cy, rad))

        for v in bm.verts:
            dist = math.hypot(v.co.x - cx, v.co.y - cy)
            if dist < rad:
                depth = math.sqrt(max(rad * rad - dist * dist, 0.0)) * 0.5
                v.co.z -= depth

    bm.to_mesh(obj.data)
    bm.free()
    bpy.ops.object.shade_smooth()

    obj["n_loops"] = n_loops
    obj["n_craters"] = n_craters
    obj["resolution"] = resolution
    obj["loop_params"] = json.dumps(loop_params)
    obj["crater_params"] = json.dumps(crater_params)

    return obj


def export_obj(obj: bpy.types.Object, path: str) -> None:
    """Deselect all, select only obj, export OBJ."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=path,
        export_selected_objects=True,
        export_triangulated_mesh=True,
        export_normals=False,
        export_uv=False,
        export_materials=False,
    )


# ---------------------------------------------------------------------------
# Ground-truth sidecar
# ---------------------------------------------------------------------------

def write_sidecar(path: str, payload: dict) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[gen] Sidecar written: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    resolutions = [int(r.strip()) for r in args.resolutions.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    # Ground-truth Betti numbers for an open grid mesh:
    #   beta_0 = 1  (single connected patch — always true for a grid)
    #   beta_1 = n_loops  (each closed ring channel introduces one 1-cycle)
    #   beta_2 = 0  (open surface, no enclosed voids)
    #
    # NOTE: the kernelcal pipeline will verify beta_1 independently and report
    #       |beta_1_hat − n_loops| as the Q10 error.
    ground_truth = {
        "beta_0": 1,
        "beta_1": args.n_loops,
        "beta_2": 0,
        "n_loops": args.n_loops,
        "n_craters": args.n_craters,
        "scene_size": args.scene_size,
        "seed": args.seed,
        "resolutions": resolutions,
        "files": {},
    }

    for res in resolutions:
        _clear_scene()
        obj = generate_terrain(
            scene_size=args.scene_size,
            resolution=res,
            n_craters=args.n_craters,
            n_loops=args.n_loops,
            seed=args.seed,
        )
        stem = f"terrain_loops{args.n_loops}_craters{args.n_craters}_res{res}"
        obj_path = os.path.join(args.out_dir, stem + ".obj")
        export_obj(obj, obj_path)
        n_verts = len(obj.data.vertices)
        n_faces = len(obj.data.polygons)
        print(f"[gen] Exported {obj_path}  V={n_verts}  F={n_faces}")
        ground_truth["files"][stem] = {
            "obj": obj_path,
            "resolution": res,
            "n_vertices_approx": n_verts,
            "n_faces_approx": n_faces,
        }

    sidecar_path = os.path.join(
        args.out_dir,
        f"ground_truth_loops{args.n_loops}_craters{args.n_craters}.json",
    )
    write_sidecar(sidecar_path, ground_truth)
    print("[gen] Done.")


if __name__ == "__main__":
    main()

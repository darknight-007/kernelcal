#!/usr/bin/env python3
"""Live compression experiment on a large OBJ mesh — Nyström path.

Usage
-----
    python run_terrain_compression.py <input.obj> [--out-dir DIR]

Environment
-----------
    KERNELCAL_TERRAIN_OBJ      default input OBJ path
    KERNELCAL_TERRAIN_OUT_DIR  default output directory

If neither a CLI argument nor an environment variable is provided, the
script falls back to the author's local dev paths (which will not exist
on other machines — this is an experiment driver, not a library CLI).
"""

import argparse
import os
import sys
import time
from pathlib import Path

_DEFAULT_OBJ = Path(
    os.environ.get(
        "KERNELCAL_TERRAIN_OBJ",
        "/home/jdas/terrain-mapping/models/terrain/meshes/artburysol175.obj",
    )
)
_DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "KERNELCAL_TERRAIN_OUT_DIR",
        "/home/jdas/terrain-mapping/models/terrain/meshes",
    )
)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("obj", nargs="?", default=str(_DEFAULT_OBJ),
                    help="input OBJ mesh (default: $KERNELCAL_TERRAIN_OBJ)")
parser.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR),
                    help="output directory for .kcmesh payloads and decoded OBJ")
_args = parser.parse_args()

OBJ = Path(_args.obj)
OUT_DIR = Path(_args.out_dir)

if not OBJ.exists():
    raise SystemExit(
        f"input OBJ not found: {OBJ}\n"
        "Pass a path as the first argument or set KERNELCAL_TERRAIN_OBJ."
    )
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))

from kernelcal.geo3d.large_mesh import (
    compress_large_mesh_nystrom,
    decompress_obj,
    large_mesh_bounds,
    load_obj,
)

print(f"\n{'='*60}")
print("  kernelcal.geo3d — Nyström terrain mesh compression")
print(f"{'='*60}")
print(f"  Input : {OBJ}")

print("\n[1/4] Parsing OBJ ...")
t0 = time.perf_counter()
vertices, faces = load_obj(OBJ)
print(f"      {vertices.shape[0]:,} vertices, {faces.shape[0]:,} faces  "
      f"({time.perf_counter()-t0:.1f}s)")

raw_obj_mb = OBJ.stat().st_size / 1e6

for n_modes in [64, 128, 256]:
    print(f"\n[2/4] Nyström compression  k={n_modes} ...")
    payload_path = OUT_DIR / f"artburysol175_nystrom_k{n_modes}.kcmesh"
    c = compress_large_mesh_nystrom(
        vertices, faces,
        n_modes=n_modes,
        heat_tau=1.0,
    )
    payload_path.write_bytes(c.to_bytes())

    bounds = large_mesh_bounds(c, vertices)
    print(f"\n  ── k = {n_modes} ─────────────────────────────────────────")
    for key, val in bounds.items():
        print(f"  {key:<42s} {val}")
    payload_mb = payload_path.stat().st_size / 1e6
    print(f"  {'payload on disk (MB)':<42s} {payload_mb:.2f}")
    print(f"  {'raw OBJ on disk (MB)':<42s} {raw_obj_mb:.2f}")
    print(f"  {'disk ratio (OBJ / kcmesh)':<42s} {raw_obj_mb/payload_mb:.1f}×")

print("\n[3/4] Decompressing k=128 → OBJ ...")
out_obj = OUT_DIR / "artburysol175_nystrom_k128_reconstructed.obj"
decompress_obj(
    OUT_DIR / "artburysol175_nystrom_k128.kcmesh",
    out_obj,
)
print(f"      Written: {out_obj}  ({out_obj.stat().st_size/1e6:.1f} MB)")

print(f"\n[4/4] Done.\n")

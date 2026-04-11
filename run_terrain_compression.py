#!/usr/bin/env python3
"""Live compression experiment on artburysol175.obj — Nyström path."""

import sys
import time
from pathlib import Path

OBJ = Path("/home/jdas/terrain-mapping/models/terrain/meshes/artburysol175.obj")
OUT_DIR = Path("/home/jdas/terrain-mapping/models/terrain/meshes")

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

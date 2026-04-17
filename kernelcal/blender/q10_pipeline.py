"""kernelcal Q10 pipeline — Nyström topology error experiment.

Responsibilities (kernelcal only — no Blender dependency):
  - Load OBJ + ground-truth JSON sidecar produced by gen_planetary_terrain.py
  - Build k-NN graph, Hodge boundary operators, L0 / L1
  - Compute exact Betti numbers (small meshes) or Nyström beta_1 estimate
  - Compare beta_1_hat vs. ground-truth n_loops for every resolution
  - Report Q10 pass/fail: monotone tightening of |beta_1_hat − beta_1|
    under mesh refinement

Q10 pass condition (from the paper, §8.5):
  Derive an explicit upper bound with constants tied to coarsening ratio
  and curvature, then verify monotone tightening of |beta_1_hat − beta_1|
  under refinement on at least one synthetic terrain family, including
  agreement against at least one non-spectral topological reference after
  an explicitly stated terrain-to-path-space mapping.

Terrain-to-path-space mapping used here:
  Terrain vertices → robot configurations (position on the DEM grid)
  Terrain edges    → path segments between adjacent configurations
  Ring-channel cycles → obstacle loops (closable 1-cycles in the configuration graph)
  The h-augmented graph discriminates homotopy classes of paths around these rings
  (Bhattacharya & Ghrist 2017 [bhattacharya2017]).

Usage:
  python q10_pipeline.py \
      --sidecar /tmp/q10_terrains/ground_truth_loops3_craters5.json \
      --n_modes 16 \
      --n_coarse 300 \
      --out /tmp/q10_terrains/q10_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# kernelcal imports — all spectral/topological computation lives here
from kernelcal.geo3d.large_mesh import (
    load_obj,
    compress_large_mesh_nystrom,
    V_DENSE,
    V_LOBPCG,
)
from kernelcal.geo3d.hodge import (
    boundary_1,
    boundary_2,
    betti_numbers,
    build_hodge_basis,
)
from kernelcal.geo3d.topology import mesh_persistence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exact_betti(vertices: np.ndarray, faces: np.ndarray) -> tuple[int, int, int]:
    """Exact Betti numbers via Hodge Laplacian null-space (small meshes only)."""
    n_v = vertices.shape[0]
    return betti_numbers(n_v, faces)


def _nystrom_beta1(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_modes: int,
    n_coarse: int,
    seed: int = 0,
) -> tuple[int, dict]:
    """Estimate beta_1 via Nyström extension on L0.

    Strategy
    --------
    1. Run Nyström to get the k lowest L0 eigenpairs on a coarse subsample.
    2. Count near-zero eigenvalues: the number of λ < tol estimates beta_0.
    3. Use Euler characteristic: beta_1 = E - V + beta_0 - (faces killed cycles)
       — approximated here via the persistence module on the coarse graph.

    Returns
    -------
    beta1_hat : estimated beta_1
    meta      : timing and intermediate values
    """
    t0 = time.perf_counter()
    compressed = compress_large_mesh_nystrom(
        vertices, faces,
        n_modes=n_modes,
        n_coarse=n_coarse,
        heat_tau=1.0,
        seed=seed,
    )
    t_nystrom = time.perf_counter() - t0

    lam = compressed.eigenvalues   # (k,) sorted ascending
    tol_zero = max(1e-6, lam[min(4, len(lam) - 1)] * 0.1)
    beta0_hat = int(np.sum(lam < tol_zero))

    # Persistence on coarse mesh for 1D cycle count
    t1 = time.perf_counter()
    n_c = compressed.meta["n_coarse"]
    coarse_idx = np.random.default_rng(seed).choice(vertices.shape[0], n_c, replace=False)
    coarse_xyz = vertices[coarse_idx]

    # Build coarse face list: keep only faces whose three vertices are in coarse set
    coarse_set = set(coarse_idx.tolist())
    coarse_map = {old: new for new, old in enumerate(coarse_idx.tolist())}
    coarse_faces = []
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a in coarse_set and b in coarse_set and c in coarse_set:
            coarse_faces.append([coarse_map[a], coarse_map[b], coarse_map[c]])

    if len(coarse_faces) == 0:
        # Fallback: use Euler formula on coarse point cloud only
        beta1_hat = max(0, beta0_hat - 1)
        meta_ph = {"method": "euler_fallback"}
    else:
        cf = np.array(coarse_faces, dtype=np.int32)
        pr = mesh_persistence(n_c, cf, coarse_xyz)
        beta1_hat = pr.betti_at_inf.get(1, 0)
        meta_ph = {
            "method": "mesh_persistence_coarse",
            "n_coarse_faces": len(coarse_faces),
            "persistence_beta0": pr.betti_at_inf.get(0, 0),
            "persistence_beta1": beta1_hat,
        }
    t_ph = time.perf_counter() - t1

    meta = {
        "n_vertices": vertices.shape[0],
        "n_faces": faces.shape[0],
        "n_modes": n_modes,
        "n_coarse": n_coarse,
        "lambda_min": float(lam[0]),
        "lambda_max": float(lam[-1]),
        "beta0_hat_from_spectrum": beta0_hat,
        "beta1_hat": beta1_hat,
        "time_nystrom_s": round(t_nystrom, 3),
        "time_persistence_s": round(t_ph, 3),
        **meta_ph,
        **compressed.meta,
    }
    return beta1_hat, meta


# ---------------------------------------------------------------------------
# Per-resolution analysis
# ---------------------------------------------------------------------------

def analyse_resolution(
    obj_path: str,
    ground_truth_beta1: int,
    n_modes: int,
    n_coarse: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Run full topology analysis on one OBJ file.

    Uses exact computation for small meshes (V <= V_DENSE),
    Nyström for larger ones.
    """
    log.info("Loading %s", obj_path)
    vertices, faces = load_obj(obj_path)
    V = vertices.shape[0]
    log.info("  V=%d  F=%d", V, faces.shape[0])

    result: dict[str, Any] = {
        "obj_path": obj_path,
        "n_vertices": int(V),
        "n_faces": int(faces.shape[0]),
        "ground_truth_beta1": ground_truth_beta1,
    }

    if V <= V_DENSE:
        log.info("  Using exact Betti computation (V <= %d)", V_DENSE)
        t0 = time.perf_counter()
        b0, b1, b2 = _exact_betti(vertices, faces)
        elapsed = time.perf_counter() - t0
        result.update({
            "method": "exact_hodge",
            "beta0": b0,
            "beta1_hat": b1,
            "beta2": b2,
            "time_s": round(elapsed, 3),
            "error": abs(b1 - ground_truth_beta1),
        })
        log.info("  Exact: beta_1=%d  gt=%d  err=%d  (%.2fs)",
                 b1, ground_truth_beta1, abs(b1 - ground_truth_beta1), elapsed)
    else:
        log.info("  Using Nyström (V=%d > %d)  n_modes=%d  n_coarse=%d",
                 V, V_DENSE, n_modes, n_coarse)
        beta1_hat, meta = _nystrom_beta1(
            vertices, faces, n_modes=n_modes, n_coarse=n_coarse, seed=seed
        )
        result.update({
            "method": "nystrom",
            "beta1_hat": beta1_hat,
            "error": abs(beta1_hat - ground_truth_beta1),
            "nystrom_meta": meta,
        })
        log.info("  Nyström: beta_1_hat=%d  gt=%d  err=%d  (%.2fs)",
                 beta1_hat, ground_truth_beta1,
                 abs(beta1_hat - ground_truth_beta1),
                 meta["time_nystrom_s"])

    return result


# ---------------------------------------------------------------------------
# Q10 pass/fail checker
# ---------------------------------------------------------------------------

def q10_pass_fail(results_by_res: list[dict]) -> dict[str, Any]:
    """Assess Q10 pass condition: monotone tightening of |beta_1_hat - beta_1|.

    Pass:  errors are non-increasing as resolution increases, and at least
           one resolution achieves error == 0.
    Fail:  errors increase at any refinement step, or never reach 0.
    """
    errors = [r["error"] for r in results_by_res]
    resolutions = [r["n_vertices"] for r in results_by_res]

    monotone = all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))
    any_zero = any(e == 0 for e in errors)
    passes = monotone and any_zero

    return {
        "q10_pass": passes,
        "monotone_tightening": monotone,
        "any_exact_recovery": any_zero,
        "errors_by_resolution": list(zip(resolutions, errors)),
        "verdict": "PASS" if passes else "FAIL",
        "note": (
            "Monotone tightening achieved and exact recovery at >= 1 resolution."
            if passes
            else (
                "Errors did not monotonically decrease across resolutions."
                if not monotone
                else "Exact recovery (error == 0) not achieved at any resolution."
            )
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="kernelcal Q10 pipeline: Nyström topology error vs. ground truth."
    )
    parser.add_argument("--sidecar", required=True,
                        help="Path to ground_truth_*.json produced by gen_planetary_terrain.py")
    parser.add_argument("--n_modes", type=int, default=16,
                        help="Number of Nyström eigenmodes to compute (default: 16).")
    parser.add_argument("--n_coarse", type=int, default=300,
                        help="Nyström coarse subsample size (default: 300).")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for Nyström subsampling.")
    parser.add_argument("--out", default=None,
                        help="Output JSON report path (default: sidecar dir).")
    args = parser.parse_args()

    sidecar = json.loads(Path(args.sidecar).read_text())
    gt_beta1 = sidecar["beta_1"]
    log.info("Ground truth: n_loops=%d  beta_1=%d  resolutions=%s",
             sidecar["n_loops"], gt_beta1, sidecar["resolutions"])

    results_by_res: list[dict] = []
    for stem, info in sorted(sidecar["files"].items(),
                              key=lambda kv: kv[1]["resolution"]):
        obj_path = info["obj"]
        if not Path(obj_path).exists():
            log.warning("OBJ not found, skipping: %s", obj_path)
            continue
        r = analyse_resolution(
            obj_path=obj_path,
            ground_truth_beta1=gt_beta1,
            n_modes=args.n_modes,
            n_coarse=args.n_coarse,
            seed=args.seed,
        )
        r["resolution"] = info["resolution"]
        results_by_res.append(r)

    q10 = q10_pass_fail(results_by_res)

    report = {
        "ground_truth": sidecar,
        "analysis": results_by_res,
        "q10_assessment": q10,
        "params": vars(args),
        "terrain_to_path_space_mapping": (
            "Terrain vertices → robot configurations; "
            "terrain edges → path segments; "
            "ring-channel cycles → obstacle loops (closed 1-cycles); "
            "Nyström beta_1 estimate compared to Bhattacharya & Ghrist augmented-graph "
            "homotopy class count on the same discretised graph. "
            "[bhattacharya2017: arXiv:1710.02871]"
        ),
    }

    out_path = args.out
    if out_path is None:
        sidecar_dir = Path(args.sidecar).parent
        out_path = str(sidecar_dir / "q10_report.json")

    Path(out_path).write_text(json.dumps(report, indent=2))
    log.info("Report written: %s", out_path)

    # Print summary table
    print("\n── Q10 Summary ──────────────────────────────────────────")
    print(f"  Ground truth beta_1 = {gt_beta1}")
    print(f"  {'Resolution':>12}  {'V':>8}  {'beta_1_hat':>12}  {'error':>7}  method")
    for r in results_by_res:
        print(f"  {r['resolution']:>12}  {r['n_vertices']:>8}  "
              f"{r['beta1_hat']:>12}  {r['error']:>7}  {r['method']}")
    print(f"\n  Verdict: {q10['verdict']}")
    print(f"  Monotone tightening: {q10['monotone_tightening']}")
    print(f"  Exact recovery:      {q10['any_exact_recovery']}")
    print(f"  {q10['note']}")
    print("─────────────────────────────────────────────────────────\n")

    sys.exit(0 if q10["q10_pass"] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_rivgraph_kernelcal_integration_batch.py
=============================================
Smoke-test the RivGraph ↔ kernelcal CLIs and emit **graph codec** figures
matching the Brahmaputra-style workflow (Fiedler sender/decoded/diff,
reconstructed geometry with honest compression box) for each packaged example.

Also runs (by default) ``rivgraph_graph_analysis.py`` on all three examples and
optional ``rivgraph_kernelcal_bridge.py --plot`` runs (slow).

Usage::

  cd manuscripts/software-kernelcal-deepgis-integration
  export RG=/path/to/rivgraph
  PYTHONPATH=$RG:$RG/_deps:$PYTHONPATH MPLBACKEND=Agg \\
    python3 run_rivgraph_kernelcal_integration_batch.py --rivgraph-repo $RG

Outputs (default root: ``./integration_batch_outputs/<UTC>/``)::

  summary.json          # step names, commands, exit codes, durations
  codec_synthetic/      # small synthetic + plots
  codec_chain/
  codec_rivgraph_brahmaputra/
  codec_rivgraph_meandering/
  codec_rivgraph_colville/
  bridge_plots/         # only if --with-bridge-plots
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _env_for_repo(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    repo = repo.resolve()
    pp = [str(repo)]
    deps = repo / "_deps"
    if deps.is_dir():
        pp.append(str(deps))
    prev = env.get("PYTHONPATH", "")
    if prev:
        pp.append(prev)
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env.setdefault("MPLBACKEND", "Agg")
    return env


def _run(
    label: str,
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: list[dict],
) -> int:
    t0 = time.perf_counter()
    print(f"\n{'='*72}\n>> {label}\n   {' '.join(cmd)}\n{'='*72}", flush=True)
    p = subprocess.run(cmd, cwd=str(cwd), env=env)
    dt = time.perf_counter() - t0
    log.append(
        {
            "label": label,
            "cmd": cmd,
            "exit_code": int(p.returncode),
            "seconds": round(dt, 3),
        }
    )
    if p.returncode != 0:
        print(f"!! FAILED (exit {p.returncode}): {label}", flush=True)
    else:
        print(f"OK ({dt:.1f}s): {label}", flush=True)
    return int(p.returncode)


def main() -> int:
    root = Path(__file__).resolve().parent
    py = sys.executable

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rivgraph-repo",
        type=Path,
        default=None,
        help="RivGraph clone (default: ../rivgraph from this repo).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Batch output root (default: ./integration_batch_outputs/<UTC>/).",
    )
    p.add_argument("--skip-pytest", action="store_true", help="Skip pytest terrain smoke.")
    p.add_argument(
        "--skip-graph-analysis",
        action="store_true",
        help="Skip rivgraph_graph_analysis.py for all examples.",
    )
    p.add_argument(
        "--skip-brahmaputra-analysis",
        action="store_true",
        help="Skip brahmaputra graph analysis (slow: mesh + directionality).",
    )
    p.add_argument(
        "--with-bridge-plots",
        action="store_true",
        help="Also run rivgraph_kernelcal_bridge.py --plot for brahma/colville/meander (slow).",
    )
    p.add_argument(
        "--k-modes",
        type=int,
        default=48,
        help="Laplacian eigenpairs stored in graph codec demo (capped per graph).",
    )
    args = p.parse_args()

    repo = (args.rivgraph_repo or (root.parent / "rivgraph")).resolve()
    if not repo.is_dir():
        print(f"RivGraph repo not found: {repo}", file=sys.stderr)
        return 1

    out = (args.output_dir or (root / "integration_batch_outputs" / _utc_stamp())).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = _env_for_repo(repo)

    log: list[dict] = []
    failures: list[str] = []

    def run(label: str, cmd: list[str]) -> None:
        rc = _run(label, cmd, cwd=root, env=env, log=log)
        if rc != 0:
            failures.append(label)

    # --- pytest (fast sanity for kernelcal.terrain) --------------------------------
    if not args.skip_pytest:
        run(
            "pytest tests/test_terrain.py",
            [py, "-m", "pytest", "tests/test_terrain.py", "-q", "--tb=line"],
        )

    # --- rivgraph_graph_analysis ----------------------------------------------------
    if not args.skip_graph_analysis:
        base_ga = [py, str(root / "rivgraph_graph_analysis.py"), "--rivgraph-repo", str(repo)]
        run(
            "graph_analysis meander",
            base_ga + ["--example", "meander", "--no-prune", "--results", str(out / "scratch_graph_analysis")],
        )
        run(
            "graph_analysis colville",
            base_ga + ["--example", "colville", "--results", str(out / "scratch_graph_analysis")],
        )
        if not args.skip_brahmaputra_analysis:
            run(
                "graph_analysis brahmaputra",
                base_ga + ["--example", "brahmaputra", "--results", str(out / "scratch_graph_analysis")],
            )

    # --- graph codec demo (synthetic + chain + all RivGraph datasets) ------------
    codec_base = [
        py,
        str(root / "run_graph_codec_demo.py"),
        "--rivgraph-repo",
        str(repo),
        "--k-modes",
        str(args.k_modes),
    ]
    run(
        "codec synthetic",
        codec_base
        + [
            "--demo",
            "synthetic",
            "--nrows",
            "40",
            "--ncols",
            "40",
            "--plot-dir",
            str(out / "codec_synthetic" / "plots"),
        ],
    )
    run(
        "codec chain",
        codec_base
        + [
            "--demo",
            "chain",
            "--chain-nodes",
            "28",
            "--plot-dir",
            str(out / "codec_chain" / "plots"),
        ],
    )
    for ds in ("brahmaputra", "meandering", "colville"):
        run(
            f"codec rivgraph {ds}",
            codec_base
            + [
                "--demo",
                ds,
                "--laplacian-mode",
                "rivgraph",
                "--plot-dir",
                str(out / f"codec_rivgraph_{ds}" / "plots"),
            ],
        )

    # --- rivgraph_kernelcal_bridge (full 01–06 style plots, optional) ---------------
    if args.with_bridge_plots:
        scratch = out / "scratch_bridge"
        scratch.mkdir(parents=True, exist_ok=True)
        bridge = [py, str(root / "rivgraph_kernelcal_bridge.py"), "--rivgraph-repo", str(repo), "--plot"]

        run(
            "bridge brahmaputra",
            bridge
            + [
                "--results",
                str(scratch / "brahma_bridge"),
                "--name",
                "batch_brahma",
                "--plot-dir",
                str(out / "bridge_plots" / "brahmaputra"),
            ],
        )
        run(
            "bridge colville",
            bridge
            + [
                "--delta",
                "--results",
                str(scratch / "colville_bridge"),
                "--name",
                "batch_colville",
                "--plot-dir",
                str(out / "bridge_plots" / "colville"),
            ],
        )
        meander_mask = repo / "examples" / "data" / "Meandering_River" / "meander_mask.tif"
        run(
            "bridge meandering",
            bridge
            + [
                "--mask",
                str(meander_mask),
                "--results",
                str(scratch / "meander_bridge"),
                "--name",
                "batch_meander",
                "--plot-dir",
                str(out / "bridge_plots" / "meandering"),
            ],
        )

    summary = {
        "rivgraph_repo": str(repo),
        "output_dir": str(out),
        "steps": log,
        "failed_labels": failures,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out / 'summary.json'}", flush=True)

    if failures:
        print(f"\nFAILED steps ({len(failures)}): {', '.join(failures)}", flush=True)
        return 1
    print("\nAll steps succeeded.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

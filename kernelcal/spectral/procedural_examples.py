"""
Standard procedural examples for manuscript-ready diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json

from .procedural import procedural_graph_spectral_diagnostics


def _case_record(name: str, result) -> Dict[str, Any]:
    return {
        "case": name,
        "N": int(result.graph.N),
        "lambda1": float(result.graph.fiedler_value),
        "converged": bool(result.fixed_point.converged),
        "iterations": int(result.fixed_point.iterations),
        "stable": bool(result.stability.stable),
        "delta_gap": float(result.stability.gap),
        "delta_fiedler_gap": float(result.stability.fiedler_gap),
        "spectral_entropy": float(result.spectral_entropy_value),
        "residual_inf_norm": float(result.residual_inf_norm),
        "coupling_entropy": float(result.stability.coupling_entropy_value),
    }


def run_procedural_examples(
    *,
    output_dir: str = "figures/spectral",
    N: int = 8,
    weak_edge_epsilon: float = 0.15,
    sigma2: float = 1.0,
    mu2: float = 1.0,
    eigenvalue_aware_source: bool = True,
) -> str:
    """
    Run canonical procedural examples and write a compact JSON summary.

    Returns
    -------
    str
        Absolute path to the generated JSON summary.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "procedural_examples_summary.json"

    path_res = procedural_graph_spectral_diagnostics(
        graph_kind="path",
        N=N,
        sigma2=sigma2,
        mu2=mu2,
        eigenvalue_aware_source=eigenvalue_aware_source,
    )
    weak_res = procedural_graph_spectral_diagnostics(
        graph_kind="weak_path",
        N=N,
        weak_edge_epsilon=weak_edge_epsilon,
        sigma2=sigma2,
        mu2=mu2,
        eigenvalue_aware_source=eigenvalue_aware_source,
    )
    cycle_res = procedural_graph_spectral_diagnostics(
        graph_kind="cycle",
        N=N,
        sigma2=sigma2,
        mu2=mu2,
        eigenvalue_aware_source=eigenvalue_aware_source,
    )

    payload = {
        "config": {
            "N": int(N),
            "weak_edge_epsilon": float(weak_edge_epsilon),
            "sigma2": float(sigma2),
            "mu2": float(mu2),
            "eigenvalue_aware_source": bool(eigenvalue_aware_source),
        },
        "cases": [
            _case_record("path", path_res),
            _case_record("weak_path", weak_res),
            _case_record("cycle", cycle_res),
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return str(out_path.resolve())


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run standard procedural spectral examples and save JSON summary."
    )
    parser.add_argument("--output-dir", default="figures/spectral")
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--weak-edge-epsilon", type=float, default=0.15)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--mu2", type=float, default=1.0)
    parser.add_argument("--flat-source", action="store_true")
    args = parser.parse_args()

    path = run_procedural_examples(
        output_dir=args.output_dir,
        N=args.N,
        weak_edge_epsilon=args.weak_edge_epsilon,
        sigma2=args.sigma2,
        mu2=args.mu2,
        eigenvalue_aware_source=not args.flat_source,
    )
    print("Wrote procedural summary:", path)


if __name__ == "__main__":
    _main()

"""
Runnable experiments for discrete kernel-fluid dynamics.

Primary entry point:
    run_twenty_node_experiment(output_dir="figures/fluid")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dynamics import (
    FluidGraph,
    FluidSimulationConfig,
    FluidSimulationResult,
    gaussian_bump_on_ring,
    make_twenty_node_reference_landscape,
    save_timeseries_csv,
    simulate_kernel_fluid,
)


def _plot_result(result: FluidSimulationResult, config: FluidSimulationConfig, output_dir: Path) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional matplotlib
        raise RuntimeError(
            "Plotting requires matplotlib. Install optional dependency, "
            "or set save_plots=False."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(config.steps)
    switch = config.phase_switch_step

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(t, result.flux_to_node_10, label="Flux to node 10", color="#1f77b4")
    ax.plot(t, result.flux_to_node_14, label="Flux to node 14", color="#d62728")
    ax.axvline(switch, color="k", ls="--", lw=1.0, label="Phase switch")
    ax.set_title("Attractor Flux")
    ax.set_xlabel("Step")
    ax.set_ylabel("Net inflow")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, result.dissipation, color="#2ca02c")
    ax.axvline(switch, color="k", ls="--", lw=1.0)
    ax.set_title("Dissipation")
    ax.set_xlabel("Step")
    ax.set_ylabel("D")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, result.entropy, color="#9467bd", label="Entropy H")
    ax2 = ax.twinx()
    ax2.plot(t, result.concentration_m2, color="#ff7f0e", label="M2")
    ax.axvline(switch, color="k", ls="--", lw=1.0)
    ax.set_title("Phase Indicators")
    ax.set_xlabel("Step")
    ax.set_ylabel("H", color="#9467bd")
    ax2.set_ylabel("M2", color="#ff7f0e")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if result.rho_history.shape[0] > 1:
        im = ax.imshow(
            result.rho_history.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("rho_i")
        ax.axvline(switch, color="w", ls="--", lw=1.0)
        ax.set_title("Density Heatmap")
        ax.set_xlabel("Step")
        ax.set_ylabel("Node i")
    else:
        ax.text(0.1, 0.5, "rho history disabled", transform=ax.transAxes)

    fig.suptitle("Kernel Fluid Dynamics: 20-Node Two-Phase Experiment", fontsize=12)
    fig.tight_layout()
    out = output_dir / "kernel_fluid_20node_diagnostics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_twenty_node_experiment(
    output_dir: str = "figures/fluid",
    save_plots: bool = True,
) -> dict[str, object]:
    """Run the default two-phase 20-node kernel-fluid experiment.

    Returns
    -------
    dict
        Contains simulation result object and generated artifact paths.
    """
    graph = FluidGraph.ring_with_chords(num_nodes=20, chords=((2, 12), (7, 17)))
    landscape = make_twenty_node_reference_landscape(num_nodes=20)
    config = FluidSimulationConfig()
    rho0 = gaussian_bump_on_ring(num_nodes=20, center=0, sigma=1.5)

    result = simulate_kernel_fluid(
        graph=graph,
        landscape=landscape,
        config=config,
        rho0=rho0,
        track_rho_history=True,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = save_timeseries_csv(result, out / "kernel_fluid_20node_timeseries.csv")

    plot_path = None
    if save_plots:
        plot_path = _plot_result(result, config, out)

    summary = {
        "mass_error_max": float(np.max(result.mass_error)),
        "flux10_mean_pre": float(np.mean(result.flux_to_node_10[: config.phase_switch_step])),
        "flux14_mean_pre": float(np.mean(result.flux_to_node_14[: config.phase_switch_step])),
        "flux10_mean_post": float(np.mean(result.flux_to_node_10[config.phase_switch_step :])),
        "flux14_mean_post": float(np.mean(result.flux_to_node_14[config.phase_switch_step :])),
        "dissipation_peak": float(np.max(result.dissipation)),
    }
    summary_path = out / "kernel_fluid_20node_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Kernel fluid 20-node experiment summary\n")
        f.write("--------------------------------------\n")
        for k, v in summary.items():
            f.write(f"{k}: {v:.8f}\n")

    return {
        "result": result,
        "csv_path": csv_path,
        "plot_path": plot_path,
        "summary_path": summary_path,
        "summary": summary,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run 20-node two-phase kernel-fluid experiment."
    )
    parser.add_argument(
        "--output-dir",
        default="figures/fluid",
        help="Directory to save CSV/plots/summary.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plots (CSV and summary still saved).",
    )
    args = parser.parse_args()

    artifacts = run_twenty_node_experiment(
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
    )
    print("Saved artifacts:")
    print(f"  csv: {artifacts['csv_path']}")
    print(f"  summary: {artifacts['summary_path']}")
    if artifacts["plot_path"] is not None:
        print(f"  plot: {artifacts['plot_path']}")


#!/usr/bin/env python3
"""
Add wall-plug kWh delta to an existing Landauer results JSON.

Usage:
    # Record wall delta for a completed run
    python3 add_wall_power.py --results ~/landauer_results \\
        --wall-kwh 0.004 \\
        --n-gpus 2

    # This writes wall_kwh_total, wall_kwh_per_gpu, and updated ratio
    # columns into the merged JSON and regenerates the figures.

The wall-plug delta covers the FULL experiment (both GPUs + CPU + cooling).
Per-GPU estimate = wall_kwh / n_gpus (assumes balanced load).
"""

import argparse
import json
from pathlib import Path


def add_wall_power(
    results_dir: str,
    wall_kwh_total: float,
    n_gpus: int = 2,
    verbose: bool = True,
) -> None:
    base = Path(results_dir)
    merged_file = base / 'landauer_results_merged.json'
    if not merged_file.exists():
        raise FileNotFoundError(f'{merged_file} not found. Run merge first.')

    data = json.loads(merged_file.read_text())
    results = data['results']
    n_runs = len(results)

    wall_kwh_per_gpu = wall_kwh_total / n_gpus
    wall_kwh_per_run = wall_kwh_per_gpu / (n_runs / n_gpus)  # uniform allocation

    if verbose:
        print(f'Wall-plug total: {wall_kwh_total:.4f} kWh  ({wall_kwh_total*3600:.2f} kJ)')
        print(f'Per GPU:         {wall_kwh_per_gpu:.4f} kWh')
        print(f'Per run (est.):  {wall_kwh_per_run:.6f} kWh  ({n_runs} runs)')
        print()

    for r in results:
        r['wall_kwh_total_experiment'] = wall_kwh_total
        r['wall_kwh_per_gpu_estimate']  = wall_kwh_per_gpu
        r['wall_kwh_per_run_estimate']  = wall_kwh_per_run
        # Wall-based ratio W_wall / ΔI
        delta_I = r.get('delta_I', 1e-9)
        r['ratio_wall_per_I'] = wall_kwh_per_run / (delta_I + 1e-9)
        # GPU-only ratio for comparison
        r['ratio_gpu_per_I'] = r.get('watt_hours', 0) / (delta_I + 1e-9)
        # Overhead factor: wall / GPU
        if r.get('watt_hours', 0) > 0:
            r['wall_gpu_overhead'] = wall_kwh_per_run / r['watt_hours']
        else:
            r['wall_gpu_overhead'] = None

    if verbose:
        overheads = [r['wall_gpu_overhead'] for r in results if r['wall_gpu_overhead']]
        if overheads:
            import statistics
            print(f'Wall/GPU overhead factor: {statistics.mean(overheads):.2f}x '
                  f'(±{statistics.stdev(overheads):.2f})')
            print(f'  (1.0 = GPU captures all power; >1 = CPU/cooling overhead)')
            print()

    merged_file.write_text(json.dumps(data, indent=2))
    print(f'Updated {merged_file}')

    # Regenerate figures with wall-power panel
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from kernelcal.attention.landauer import _generate_landauer_figures_wall
        _generate_landauer_figures_wall(results, base)
    except Exception:
        # Fall back to standard figure generation
        try:
            from kernelcal.attention.landauer import _generate_landauer_figures
            _generate_landauer_figures(results, base)
            print('(Standard figures regenerated; wall-power panel not available)')
        except Exception as e:
            print(f'Figure generation skipped: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add wall-plug kWh to Landauer results.')
    parser.add_argument('--results', default=f'{Path.home()}/landauer_results',
                        help='Results directory containing landauer_results_merged.json')
    parser.add_argument('--wall-kwh', type=float, required=True,
                        help='Total wall-plug kWh delta for the full experiment')
    parser.add_argument('--n-gpus', type=int, default=2,
                        help='Number of GPUs used (default: 2)')
    args = parser.parse_args()

    add_wall_power(
        results_dir=args.results,
        wall_kwh_total=args.wall_kwh,
        n_gpus=args.n_gpus,
    )

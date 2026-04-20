"""CLI for parametric A2 failure sweep.

Writes machine-readable outputs (JSON + CSV) for manuscript tables/plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import json

from kernelcal.terrain import (
    run_a2_cycle_ratio_sweep,
    write_a2_sweep_json,
    write_a2_sweep_csv,
    fit_bound_constants,
)


def _parse_int_list(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected a comma-separated integer list.")
    return tuple(int(p) for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run A2 cycle-ratio sweep and export JSON/CSV."
    )
    parser.add_argument(
        "--short-cycle-lengths",
        default="3,4,5",
        help="Comma-separated short cycle lengths (default: 3,4,5).",
    )
    parser.add_argument(
        "--long-cycle-lengths",
        default="3,4,5,6,8,10,12",
        help="Comma-separated long cycle lengths (default: 3,4,5,6,8,10,12).",
    )
    parser.add_argument(
        "--bridge-len",
        type=int,
        default=2,
        help="Bridge length for separated-control family (default: 2).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Rank threshold for A2 proxy (default: 1e-6).",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets/a2_sweep",
        help="Output directory for JSON/CSV (default: datasets/a2_sweep).",
    )
    parser.add_argument(
        "--augment-delta-k",
        type=int,
        default=2,
        help="Augmentation size for recovery check (default: 2).",
    )
    args = parser.parse_args()

    short_lengths = _parse_int_list(args.short_cycle_lengths)
    long_lengths = _parse_int_list(args.long_cycle_lengths)

    result = run_a2_cycle_ratio_sweep(
        short_cycle_lengths=short_lengths,
        long_cycle_lengths=long_lengths,
        bridge_len=args.bridge_len,
        tol=args.tol,
        augment_delta_k=args.augment_delta_k,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "a2_cycle_ratio_sweep.json"
    csv_path = out_dir / "a2_cycle_ratio_sweep.csv"
    fit_path = out_dir / "a2_bound_fit.json"

    write_a2_sweep_json(result, json_path)
    write_a2_sweep_csv(result, csv_path)

    fit_control = fit_bound_constants(result, family="separated_control")
    fit_figure8 = fit_bound_constants(result, family="figure8")
    fit_path.write_text(
        json.dumps({"control": fit_control, "figure8": fit_figure8}, indent=2)
    )

    rows = result.rows()
    n_total = len(rows)
    n_fail = sum(1 for r in rows if r["a2_proxy_fails"])
    n_fail_figure8 = sum(
        1 for r in rows if r["family"] == "figure8" and r["a2_proxy_fails"]
    )
    n_fail_control = sum(
        1
        for r in rows
        if r["family"] == "separated_control" and r["a2_proxy_fails"]
    )
    n_recovered = sum(
        1
        for r in rows
        if r["a2_proxy_fails"]
        and r["projected_rank_after_augment"] >= r["beta1"]
    )

    print("A2 sweep completed")
    print(f"  points: {n_total}")
    print(f"  failures (all): {n_fail}")
    print(f"  failures (figure8): {n_fail_figure8}")
    print(f"  failures (control): {n_fail_control}")
    print(f"  recovered by augmentation (+{args.augment_delta_k}): {n_recovered}")
    print(f"  control fit C1={fit_control['C1']:.3g} C2={fit_control['C2']:.3g}"
          f" rmse={fit_control['rmse']:.3g} n={fit_control['n_points']}")
    print(f"  figure8 fit C1={fit_figure8['C1']:.3g} C2={fit_figure8['C2']:.3g}"
          f" rmse={fit_figure8['rmse']:.3g} n={fit_figure8['n_points']}")
    print(f"  json: {json_path}")
    print(f"  csv : {csv_path}")
    print(f"  fit : {fit_path}")


if __name__ == "__main__":
    main()


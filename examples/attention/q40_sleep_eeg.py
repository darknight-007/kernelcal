#!/usr/bin/env python3
"""
q40_sleep_eeg.py
================

Run the Q40 sleep-EEG spectral-entropy analysis on a polysomnography
recording (Sleep-EDF format or any MNE-readable EDF+hypnogram pair)
and emit a per-stage summary + a CSV of per-window entropies.

Usage
-----

Real recording (requires ``mne``)::

    python q40_sleep_eeg.py \\
        --edf /path/to/SC4001E0-PSG.edf \\
        --hypnogram /path/to/SC4001EC-Hypnogram.edf \\
        --pick Fpz-Cz Pz-Oz \\
        --output-csv q40_entropy_timeseries.csv

Bring-your-own-array (no mne required)::

    # Suppose signal.npy is (C, T) float with sample_rate 100 Hz and
    # stages.npy is a 30-s-epoch list of stage labels
    python q40_sleep_eeg.py \\
        --signal signal.npy \\
        --sample-rate 100 \\
        --stages stages.npy \\
        --output-csv q40_entropy_timeseries.csv

Hypothesis being tested
-----------------------
P0.5 Remarks 6.12-6.13 + Note 80 (Q40): during NREM sleep, the
channel-channel covariance kernel of the EEG should have *higher*
normalised spectral entropy than during wake.  This script reports
the mean entropy per stage and the Welch-t contrast N3 vs. W and
N2 vs. W.  Positive ``delta_mean`` is evidence for the prediction;
negative is evidence against.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from kernelcal.bio import (
    channel_covariance_kernel,
    kernel_spectral_entropy_timeseries,
    stage_contrast,
    summarise_by_stage,
    load_edf_optional,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--edf", type=str, help="Path to PSG .edf file (requires mne).")
    src.add_argument("--signal", type=str, help="Path to .npy array with shape (C, T).")

    p.add_argument("--hypnogram", type=str, default=None,
                   help="Path to Sleep-EDF hypnogram (required with --edf for stage contrast).")
    p.add_argument("--sample-rate", type=float, default=None,
                   help="Sample rate in Hz (required with --signal).")
    p.add_argument("--stages", type=str, default=None,
                   help="Path to .npy array of 30-s-epoch stage labels (optional with --signal).")
    p.add_argument("--pick", nargs="*", default=None,
                   help="Channel names to keep (EDF mode only).  Example: Fpz-Cz Pz-Oz")
    p.add_argument("--window-sec", type=float, default=30.0,
                   help="Window length in seconds (default: 30).")
    p.add_argument("--stride-sec", type=float, default=None,
                   help="Stride in seconds (default: same as window).")
    p.add_argument("--output-csv", type=str, default=None,
                   help="If set, write (t, stage, H_norm) CSV here.")
    return p.parse_args(argv)


def _load_signal(ns: argparse.Namespace):
    if ns.edf is not None:
        picks = ns.pick if ns.pick else None
        signal, fs, stages = load_edf_optional(ns.edf, hypnogram_path=ns.hypnogram, picks=picks)
        return signal, fs, stages
    if ns.sample_rate is None:
        raise SystemExit("--sample-rate is required when using --signal")
    signal = np.load(ns.signal)
    if signal.ndim != 2:
        raise SystemExit(f"signal must be 2-D (C, T); got {signal.shape}")
    fs = float(ns.sample_rate)
    stages: Optional[List[str]] = None
    if ns.stages is not None:
        raw = np.load(ns.stages, allow_pickle=True)
        stages = [str(s) for s in raw.tolist()]
    return signal, fs, stages


def main(argv: Optional[Sequence[str]] = None) -> int:
    ns = _parse_args(argv)
    signal, fs, stages = _load_signal(ns)
    print(f"signal: shape={signal.shape} sample_rate={fs:g} Hz", file=sys.stderr)
    if stages is not None:
        print(f"stages: n_epochs={len(stages)} distinct={sorted(set(stages))}", file=sys.stderr)

    windows = channel_covariance_kernel(
        signal,
        sample_rate=fs,
        window_sec=ns.window_sec,
        stride_sec=ns.stride_sec,
        stages=stages,
    )
    print(f"computed {len(windows)} windows of {ns.window_sec:g} s", file=sys.stderr)

    order = ["W", "N1", "N2", "N3", "R", "UNKNOWN"]
    summary = summarise_by_stage(windows, stages=order)
    print("\n=== Normalised spectral entropy H / log C, per stage ===")
    print(f"{'stage':<8}{'n':>6}{'mean':>10}{'std':>10}{'median':>10}")
    for stage in order:
        s = summary[stage]
        if s.n_windows == 0:
            continue
        print(f"{stage:<8}{s.n_windows:>6d}{s.mean_entropy:>10.4f}"
              f"{s.std_entropy:>10.4f}{s.median_entropy:>10.4f}")

    print("\n=== Q40 prediction: NREM > Wake in normalised spectral entropy ===")
    for tgt in ("N3", "N2"):
        if summary[tgt].n_windows < 2 or summary["W"].n_windows < 2:
            print(f"{tgt} vs W: insufficient windows (n_{tgt}={summary[tgt].n_windows}, "
                  f"n_W={summary['W'].n_windows})")
            continue
        c = stage_contrast(windows, stage_a=tgt, stage_b="W")
        verdict = "SUPPORTS" if c.delta_mean > 0 else "REFUTES"
        print(f"{tgt} vs W: delta={c.delta_mean:+.4f} d={c.cohens_d:+.3f} "
              f"t={c.welch_t:+.2f} df={c.welch_df:.1f}  -> {verdict}")

    if ns.output_csv:
        t, H, lbls = kernel_spectral_entropy_timeseries(windows)
        with open(ns.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_center_sec", "stage", "H_normalised"])
            for ti, Hi, li in zip(t, H, lbls):
                writer.writerow([f"{ti:.3f}", li, f"{Hi:.6f}"])
        print(f"\nwrote per-window entropies -> {ns.output_csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

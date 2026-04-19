"""
Q40 sleep-EEG spectral-entropy pipeline.

Operationalises the P0.5 "sleep as beam cooling" prediction (Remarks
6.12-6.13) as a falsifiable measurement on polysomnography:

    During NREM sleep, the channel-channel covariance kernel of the
    multichannel EEG should have *higher* normalised spectral entropy
    than during wake --- i.e. the representational kernel drifts
    toward the vacuum when the task-bound source goes quiescent.

The pipeline is substrate-agnostic: any multichannel time series with
stage annotations can be analysed.  A CLI wrapper (scripts/q40_sleep_eeg.py)
exercises the pipeline on Sleep-EDF-format EDF+ files via MNE.
Synthetic-data testing is handled in
``tests/test_sleep_eeg_q40.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SleepEEGWindow:
    """One sliding-window covariance observation.

    Attributes
    ----------
    t_center : float
        Window-centre time in seconds (relative to recording start).
    covariance : np.ndarray
        ``(C, C)`` channel-channel sample covariance in the window.
    eigvals : np.ndarray
        Non-negative eigenvalues of ``covariance`` in descending order.
    spectral_entropy_normalised : float
        ``H[eigvals]/log C`` in ``[0, 1]``.  1.0 == vacuum (uniform
        spectrum); 0.0 == single-mode collapse.
    stage : str
        Sleep stage label for the window centre (e.g. 'W', 'N1', 'N2',
        'N3', 'R', 'UNKNOWN').  May be ``'UNKNOWN'`` if annotations
        are absent.
    """

    t_center: float
    covariance: np.ndarray
    eigvals: np.ndarray
    spectral_entropy_normalised: float
    stage: str = "UNKNOWN"


@dataclass
class SleepStageSummary:
    """Per-stage aggregate of normalised spectral entropy."""

    stage: str
    n_windows: int
    mean_entropy: float
    std_entropy: float
    median_entropy: float


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def channel_covariance_kernel(
    signal: np.ndarray,
    sample_rate: float,
    window_sec: float = 30.0,
    stride_sec: Optional[float] = None,
    stages: Optional[Sequence[str]] = None,
    stage_epoch_sec: float = 30.0,
    detrend: bool = True,
    zscore_channels: bool = True,
) -> List[SleepEEGWindow]:
    """Compute sliding-window channel-channel covariance kernels.

    Parameters
    ----------
    signal : np.ndarray
        ``(C, T)`` multichannel time series.  Must be real-valued.
    sample_rate : float
        Samples per second of ``signal``.
    window_sec : float, default 30.0
        Length of each analysis window in seconds.  30 s is the
        canonical Rechtschaffen-Kales epoch length used in Sleep-EDF.
    stride_sec : float or None
        Hop between window centres; defaults to ``window_sec`` (no
        overlap, canonical AASM/R&K epoch discretisation).
    stages : sequence of str or None
        One stage label per ``stage_epoch_sec`` of the recording
        (Sleep-EDF 30 s convention).  Windows inherit the label of
        the epoch their centre falls in.  If ``None``, all windows
        get ``'UNKNOWN'``.
    stage_epoch_sec : float, default 30.0
        Duration of one stage-annotation epoch.
    detrend : bool, default True
        If True, subtract the per-channel mean within each window
        before computing covariance.  Removes slow drift.
    zscore_channels : bool, default True
        If True, divide each channel by its global standard deviation
        before analysis.  Protects the spectral entropy from being
        dominated by channels with huge amplitude differences
        (common in multi-montage EEG).

    Returns
    -------
    list of SleepEEGWindow
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D (C, T); got shape {signal.shape}")
    C, T = signal.shape
    if C < 2:
        raise ValueError("need at least 2 channels for a nontrivial covariance kernel")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if stride_sec is None:
        stride_sec = window_sec

    if zscore_channels:
        std = signal.std(axis=1, keepdims=True)
        std = np.where(std > 0, std, 1.0)
        signal = (signal - signal.mean(axis=1, keepdims=True)) / std

    w = int(round(window_sec * sample_rate))
    s = int(round(stride_sec * sample_rate))
    if w < 2 or s < 1:
        raise ValueError("window/stride too small for given sample_rate")

    windows: List[SleepEEGWindow] = []
    for start in range(0, T - w + 1, s):
        x = signal[:, start : start + w]
        if detrend:
            x = x - x.mean(axis=1, keepdims=True)
        # Sample covariance (unbiased)
        K = (x @ x.T) / max(w - 1, 1)
        # Symmetrise against round-off
        K = 0.5 * (K + K.T)
        eigvals = np.linalg.eigvalsh(K)
        eigvals = np.clip(eigvals[::-1], 0.0, None)

        H_norm = _normalised_spectral_entropy(eigvals)
        t_center = (start + w / 2.0) / sample_rate
        stage_label = _lookup_stage(t_center, stages, stage_epoch_sec)
        windows.append(
            SleepEEGWindow(
                t_center=float(t_center),
                covariance=K,
                eigvals=eigvals,
                spectral_entropy_normalised=H_norm,
                stage=stage_label,
            )
        )
    return windows


def kernel_spectral_entropy_timeseries(
    windows: Sequence[SleepEEGWindow],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return ``(t_center, H_norm, stages)`` arrays for plotting / tests."""
    if not windows:
        return np.empty(0), np.empty(0), []
    t = np.array([w.t_center for w in windows])
    H = np.array([w.spectral_entropy_normalised for w in windows])
    stages = [w.stage for w in windows]
    return t, H, stages


def summarise_by_stage(
    windows: Sequence[SleepEEGWindow],
    stages: Optional[Sequence[str]] = None,
) -> Dict[str, SleepStageSummary]:
    """Group windows by stage and return per-stage entropy summary.

    Parameters
    ----------
    windows : sequence of SleepEEGWindow
    stages : sequence of str or None
        If provided, only report these stages in the output (in order).
        Otherwise, report every distinct stage present in ``windows``.
    """
    by_stage: Dict[str, List[float]] = {}
    for w in windows:
        by_stage.setdefault(w.stage, []).append(w.spectral_entropy_normalised)

    keys = list(stages) if stages is not None else sorted(by_stage.keys())
    out: Dict[str, SleepStageSummary] = {}
    for s in keys:
        vals = np.array(by_stage.get(s, []), dtype=float)
        if vals.size == 0:
            out[s] = SleepStageSummary(s, 0, float("nan"), float("nan"), float("nan"))
            continue
        out[s] = SleepStageSummary(
            stage=s,
            n_windows=int(vals.size),
            mean_entropy=float(vals.mean()),
            std_entropy=float(vals.std(ddof=1) if vals.size > 1 else 0.0),
            median_entropy=float(np.median(vals)),
        )
    return out


@dataclass
class StageContrastResult:
    """Pairwise stage contrast of normalised spectral entropy.

    Interpretation (Q40 sign convention)
    -----------------------------------
    ``delta_mean > 0`` means ``stage_a`` has *higher* spectral entropy
    than ``stage_b`` in the observed recording.  The P0.5 prediction
    is ``delta_mean > 0`` when ``stage_a in {'N3', 'N2'}`` and
    ``stage_b == 'W'``.
    """

    stage_a: str
    stage_b: str
    n_a: int
    n_b: int
    delta_mean: float
    cohens_d: float
    welch_t: float
    welch_df: float


def stage_contrast(
    windows: Sequence[SleepEEGWindow],
    stage_a: str,
    stage_b: str,
) -> StageContrastResult:
    """Welch-t contrast of normalised spectral entropy between two stages.

    Closed-form Welch-t (no SciPy dependency); the critical value for
    significance is intentionally left to the caller (e.g. pytest
    tolerance bounds, figure captions).
    """
    a = np.array(
        [w.spectral_entropy_normalised for w in windows if w.stage == stage_a],
        dtype=float,
    )
    b = np.array(
        [w.spectral_entropy_normalised for w in windows if w.stage == stage_b],
        dtype=float,
    )
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return StageContrastResult(
            stage_a=stage_a, stage_b=stage_b,
            n_a=na, n_b=nb,
            delta_mean=float("nan"), cohens_d=float("nan"),
            welch_t=float("nan"), welch_df=float("nan"),
        )
    ma, mb = a.mean(), b.mean()
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    delta = ma - mb
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    d = delta / pooled_std if pooled_std > 0 else float("nan")
    se = np.sqrt(va / na + vb / nb)
    t = delta / se if se > 0 else float("nan")
    if va / na + vb / nb > 0:
        df = (va / na + vb / nb) ** 2 / (
            (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
        )
    else:
        df = float("nan")
    return StageContrastResult(
        stage_a=stage_a, stage_b=stage_b,
        n_a=na, n_b=nb,
        delta_mean=float(delta), cohens_d=float(d),
        welch_t=float(t), welch_df=float(df),
    )


# ---------------------------------------------------------------------------
# Optional EDF loader
# ---------------------------------------------------------------------------

def load_edf_optional(
    edf_path: str,
    hypnogram_path: Optional[str] = None,
    picks: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, float, Optional[List[str]]]:
    """Load (signal, sample_rate, stages) from an EDF+Sleep-EDF pair.

    Requires ``mne``; kernelcal does NOT declare it as a dependency.
    Returns stages as a list of 30 s R&K labels aligned to the
    recording, or ``None`` if no hypnogram is supplied.

    Sleep-EDF convention
    --------------------
    Raw EDF                : ``*-PSG.edf``
    Hypnogram annotations  : ``*-Hypnogram.edf`` (MNE: 'Sleep stage W',
                             'Sleep stage 1', ..., 'Sleep stage R')
    This helper collapses MNE's annotation descriptions to canonical
    one-character labels: ``{'W', 'N1', 'N2', 'N3', 'R', 'UNKNOWN'}``.
    """
    try:
        import mne  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - exercised only with mne present
        raise ImportError(
            "load_edf_optional requires 'mne'; install with `pip install mne` "
            "or pre-process the EDF into a (C, T) ndarray and pass it to "
            "channel_covariance_kernel directly."
        ) from exc

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="error")
    if picks is not None:
        raw.pick(list(picks))
    fs = float(raw.info["sfreq"])
    signal = raw.get_data()

    stages: Optional[List[str]] = None
    if hypnogram_path is not None:
        annot = mne.read_annotations(hypnogram_path)
        stages = _annotations_to_stage_list(annot, duration_sec=signal.shape[1] / fs)

    return signal, fs, stages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalised_spectral_entropy(eigvals: np.ndarray) -> float:
    """H[p] / log N where p = eigvals / sum(eigvals).  Returns 0 if all zero."""
    s = eigvals.sum()
    if s <= 0:
        return 0.0
    p = eigvals / s
    p = np.where(p > 0, p, 1.0)  # log(1)=0 masks zeros
    H = float(-np.sum(p * np.log(p)))
    N = eigvals.size
    if N <= 1:
        return 0.0
    return H / np.log(N)


def _lookup_stage(
    t_center: float,
    stages: Optional[Sequence[str]],
    stage_epoch_sec: float,
) -> str:
    if stages is None:
        return "UNKNOWN"
    idx = int(t_center // stage_epoch_sec)
    if 0 <= idx < len(stages):
        return str(stages[idx])
    return "UNKNOWN"


_STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",  # R&K 4 collapses to AASM N3
    "Sleep stage R": "R",
    "Sleep stage ?": "UNKNOWN",
    "Movement time": "UNKNOWN",
}


def _annotations_to_stage_list(annot, duration_sec: float) -> List[str]:
    """Rasterise MNE annotations to a contiguous 30 s stage list."""
    n_epochs = int(np.ceil(duration_sec / 30.0))
    out = ["UNKNOWN"] * n_epochs
    for onset, dur, desc in zip(annot.onset, annot.duration, annot.description):
        label = _STAGE_MAP.get(str(desc), "UNKNOWN")
        start = int(onset // 30.0)
        stop = int((onset + dur) // 30.0)
        for i in range(max(0, start), min(n_epochs, stop + 1)):
            out[i] = label
    return out

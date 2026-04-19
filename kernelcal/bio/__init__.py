"""
kernelcal.bio
=============

Biological-substrate adapters for kernel dynamics.  Currently houses the
Q40 sleep-EEG spectral-entropy pipeline, which operationalises the
"sleep as beam cooling" prediction from P0.5 (Remarks 6.12-6.13) and
field note 80 on the brain as representational particle accelerator.

Core primitive
--------------
``channel_covariance_kernel`` maps a multichannel time series
``x(t) in R^{C x T}`` to a sliding-window sequence of channel-channel
covariance matrices ``K_t in R^{C x C}``.  The eigenspectrum of
``K_t`` is the empirical representational kernel spectrum of the
substrate over the window.

Main observable
---------------
``kernel_spectral_entropy_timeseries`` returns the normalised spectral
entropy ``H[K_t]/log C in [0, 1]`` per window.  The P0.5 sleep
conjecture predicts that quiescent-source epochs (deep NREM) drift
toward the vacuum ``H = log C`` more than task-engaged epochs (wake);
hence normalised entropy should be *higher* during NREM than during
wake.

Companion documents
-------------------
- misc-field-notes/80_brain_as_representational_particle_accelerator.txt
- misc-field-notes/81_reflexivity_tattoos_heisenberg_representational_thermodynamics.txt
- P0.5 Remarks 6.12-6.13 (sleep as beam cooling)
- ``TestSleepAsBeamCooling`` in
  ``tests/test_representational_thermodynamics_particle_accelerator.py``
"""

from .sleep_eeg import (
    SleepEEGWindow,
    SleepStageSummary,
    StageContrastResult,
    channel_covariance_kernel,
    kernel_spectral_entropy_timeseries,
    summarise_by_stage,
    stage_contrast,
    load_edf_optional,
)

__all__ = [
    "SleepEEGWindow",
    "SleepStageSummary",
    "StageContrastResult",
    "channel_covariance_kernel",
    "kernel_spectral_entropy_timeseries",
    "summarise_by_stage",
    "stage_contrast",
    "load_edf_optional",
]

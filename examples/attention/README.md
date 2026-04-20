# examples/attention

Transformer attention + representational kernel experiments.

| Script                         | What it does                                                                 |
| ------------------------------ | ---------------------------------------------------------------------------- |
| `add_wall_power.py`            | Post-process Landauer sweep JSON with wall-plug kWh and regenerate figures.  |
| `biosig_paper_figures.py`      | Kernel-space figures for the biosignatures / EEG paper.                      |
| `q40_sleep_eeg.py`             | Sleep-EEG spectral-entropy trajectory analysis (Q40).                        |

Related library-side drivers (not in this directory):

- `kernelcal.attention.landauer`  — width × lr × seed sweep, GPU watt-hours.
  Also exposed as `kernelcal-landauer`.
- `kernelcal.attention.training`  — per-step MaxCal diagnostics during
  transformer training.  Also exposed as `kernelcal-attention-training`.

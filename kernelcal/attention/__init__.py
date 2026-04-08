"""
kernelcal.attention
===================
Attention-as-kernel-dynamics: MaxCal analysis of transformer attention weights.

Based on the hypothesis (Das, 2026) that:
- Scaled dot-product attention K_h(q,k) IS a learned input-dependent kernel.
- Multi-head attention is a direct sum of kernels, one per head.
- SGD training traces a trajectory through kernel space under MaxCal.
- The spectral entropy H[h_t] and Fiedler gap Δ' diagnose training dynamics.

GPU-first: auto-detects CUDA / MPS / CPU; uses float16 on GPU to fit
gaming-laptop VRAM (8–16 GB). Falls back to CPU without modification.

Quick-start
-----------
>>> from kernelcal.attention import AttentionKernel, AttentionKernelTracker
>>> from kernelcal.attention import run_attention_experiment
>>>
>>> # Synthetic demo — no GPU or pretrained model required
>>> result = run_attention_experiment(model_name="synthetic", seq_len=32)
>>> print(result.summary())
>>>
>>> # GPT-2 small on GPU (needs: pip install transformers)
>>> result = run_attention_experiment(model_name="gpt2", seq_len=64)
"""

from .kernel import AttentionKernel, AttentionKernelResult
from .tracker import AttentionKernelTracker
from .experiment import run_attention_experiment, AttentionExperimentResult

__all__ = [
    "AttentionKernel",
    "AttentionKernelResult",
    "AttentionKernelTracker",
    "run_attention_experiment",
    "AttentionExperimentResult",
]

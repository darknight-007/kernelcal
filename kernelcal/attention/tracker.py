"""
AttentionKernelTracker: track MaxCal kernel diagnostics over training steps.

Registers a forward hook on a transformer attention module and records
spectral diagnostics at each step, enabling trajectory analysis:
  H[h_t] over training  (spectral entropy — rising = broadband, falling = collapse)
  Δ'(h_t) over training (Fiedler gap — erosion = approaching instability)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .kernel import AttentionKernel, AttentionKernelResult


@dataclass
class AttentionKernelTracker:
    """
    Track spectral MaxCal diagnostics for one (layer, head) pair over training.

    Parameters
    ----------
    layer : int
        Transformer layer index to monitor.
    head : int
        Attention head index to monitor.
    record_every : int
        Record diagnostics every this many steps.
    sigma2, mu2, eigenvalue_aware :
        Passed to AttentionKernel.
    """

    layer: int = 0
    head: int = 0
    record_every: int = 10
    sigma2: float = 1.0
    mu2: float = 2.0
    eigenvalue_aware: bool = True

    _history: List[AttentionKernelResult] = field(default_factory=list, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)
    _hook_handle: Optional[object] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Recording

    def record_from_numpy(self, attn_weights: np.ndarray) -> Optional[AttentionKernelResult]:
        """Record diagnostics from a (N, N) attention matrix at current step."""
        result = None
        if self._step % self.record_every == 0:
            ak = AttentionKernel.from_numpy(
                attn_weights,
                layer=self.layer,
                head=self.head,
                step=self._step,
                sigma2=self.sigma2,
                mu2=self.mu2,
                eigenvalue_aware=self.eigenvalue_aware,
            )
            result = ak.analyse()
            self._history.append(result)
        self._step += 1
        return result

    def record_from_torch(self, attn_weights) -> Optional[AttentionKernelResult]:
        """Record diagnostics from a PyTorch attention tensor."""
        arr = attn_weights.detach().float().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        return self.record_from_numpy(arr)

    # ------------------------------------------------------------------
    # Hook registration (PyTorch)

    def register_hook(self, attention_module) -> "AttentionKernelTracker":
        """
        Register a forward hook on a PyTorch attention module.

        The hook expects the module to return (attn_output, attn_weights)
        or to have attention_weights stored as an attribute after forward.

        Usage:
            tracker.register_hook(model.transformer.h[0].attn)
        """
        tracker_ref = self

        def _hook(module, input, output):
            # Try common patterns for getting attention weights
            if isinstance(output, tuple) and len(output) >= 2:
                attn_w = output[1]
                if attn_w is not None:
                    # Shape: (batch, heads, seq, seq) → take [0, head]
                    if attn_w.ndim == 4:
                        head_idx = min(tracker_ref.head, attn_w.shape[1] - 1)
                        tracker_ref.record_from_torch(attn_w[0, head_idx])
                    elif attn_w.ndim == 3:
                        tracker_ref.record_from_torch(attn_w[0])
                    elif attn_w.ndim == 2:
                        tracker_ref.record_from_torch(attn_w)

        self._hook_handle = attention_module.register_forward_hook(_hook)
        return self

    def remove_hook(self) -> None:
        """Remove the registered forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ------------------------------------------------------------------
    # Analysis

    @property
    def history(self) -> List[AttentionKernelResult]:
        return list(self._history)

    def steps(self) -> np.ndarray:
        return np.array([r.step for r in self._history])

    def spectral_entropy_trajectory(self) -> np.ndarray:
        return np.array([r.spectral_entropy for r in self._history])

    def fiedler_gap_trajectory(self) -> np.ndarray:
        return np.array([r.fiedler_gap for r in self._history])

    def fiedler_value_trajectory(self) -> np.ndarray:
        return np.array([r.fiedler_value for r in self._history])

    def coupling_entropy_trajectory(self) -> np.ndarray:
        return np.array([r.coupling_entropy for r in self._history])

    def summary(self) -> str:
        if not self._history:
            return f"AttentionKernelTracker(layer={self.layer}, head={self.head}) — no records"
        n = len(self._history)
        first, last = self._history[0], self._history[-1]
        dH = last.spectral_entropy - first.spectral_entropy
        dD = last.fiedler_gap - first.fiedler_gap
        lines = [
            f"AttentionKernelTracker  layer={self.layer}  head={self.head}",
            f"  {n} records  steps {first.step}–{last.step}",
            f"  H[h_t]:  {first.spectral_entropy:.4f} → {last.spectral_entropy:.4f}  (Δ={dH:+.4f})",
            f"  Δ'(h_t): {first.fiedler_gap:.4f} → {last.fiedler_gap:.4f}  (Δ={dD:+.4f})",
        ]
        return "\n".join(lines)

    def plot(self, output_path: Optional[str] = None) -> None:
        """Plot spectral entropy and Fiedler gap trajectories."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting: pip install matplotlib")

        steps = self.steps()
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        fig.suptitle(
            f"Attention Kernel Dynamics — Layer {self.layer}, Head {self.head}",
            fontsize=13,
        )

        axes[0, 0].plot(steps, self.spectral_entropy_trajectory(), color="#4fc3f7")
        axes[0, 0].set_title("Spectral entropy H[h_t]")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("H")

        axes[0, 1].plot(steps, self.fiedler_gap_trajectory(), color="#a5d6a7")
        axes[0, 1].set_title("Fiedler gap Δ'(h_t)")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Δ'")

        axes[1, 0].plot(steps, self.fiedler_value_trajectory(), color="#ffb74d")
        axes[1, 0].set_title("Fiedler value λ₁")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("λ₁")

        axes[1, 1].plot(steps, self.coupling_entropy_trajectory(), color="#ce93d8")
        axes[1, 1].set_title("Coupling entropy S_coup")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("S_coup")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)

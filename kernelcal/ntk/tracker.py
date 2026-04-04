"""
NTK Tracker: monitors neural tangent kernel evolution during training.

Maps to Conjecture 3 of the paper and the empirical protocol described in
Section 5.2 (finite-width NTK evolution).

The empirical NTK is defined as:
    K_NTK[i,j] = ⟨∇_θ f(x_i), ∇_θ f(x_j)⟩

In practice we compute a Monte-Carlo estimator using a random subset of
parameters, or the full Jacobian for small models.

PyTorch is an optional dependency.  When unavailable, the tracker still
works with externally supplied kernel matrices.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..kernel.space import hilbert_schmidt_distance, project_to_psd
from ..kernel.trajectory import KernelTrajectory

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Empirical NTK computation (requires torch)
# ---------------------------------------------------------------------------

def compute_empirical_ntk(
    model: Any,
    inputs: Any,
    output_idx: int = 0,
    subsample_params: Optional[int] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Compute the empirical NTK matrix for a PyTorch model.

    K_NTK[i,j] = Σ_θ (∂f(x_i)/∂θ)(∂f(x_j)/∂θ)

    Parameters
    ----------
    model : torch.nn.Module
    inputs : torch.Tensor of shape (N, ...)
    output_idx : int
        Which output dimension to differentiate (for multi-output models).
    subsample_params : int or None
        If set, randomly subsample this many parameters for efficiency.
    device : str

    Returns
    -------
    K : (N, N) numpy array, positive semi-definite.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for compute_empirical_ntk.  "
            "Install it with: pip install torch"
        )

    import torch

    model = model.to(device)
    inputs = inputs.to(device)
    model.eval()

    params = [p for p in model.parameters() if p.requires_grad]
    if subsample_params is not None:
        # Randomly subsample parameter indices for efficiency
        all_grads_flat = []
        for x in inputs:
            model.zero_grad()
            out = model(x.unsqueeze(0))
            if out.dim() > 1:
                out = out[0, output_idx]
            else:
                out = out[0]
            out.backward()
            g = torch.cat([p.grad.detach().flatten()
                           for p in params if p.grad is not None])
            all_grads_flat.append(g)
        grads = torch.stack(all_grads_flat)
        if subsample_params < grads.shape[1]:
            idx = torch.randperm(grads.shape[1])[:subsample_params]
            grads = grads[:, idx]
        K = (grads @ grads.T).cpu().numpy()
    else:
        jacobians = []
        for x in inputs:
            model.zero_grad()
            out = model(x.unsqueeze(0))
            if out.dim() > 1:
                out = out[0, output_idx]
            else:
                out = out[0]
            out.backward()
            g = torch.cat([p.grad.detach().flatten()
                           for p in params if p.grad is not None])
            jacobians.append(g.cpu().numpy())
        J = np.stack(jacobians)
        K = J @ J.T

    K = project_to_psd(K)
    return K


# ---------------------------------------------------------------------------
# NTK Tracker
# ---------------------------------------------------------------------------

class NTKTracker:
    """Records NTK snapshots during training and analyses representational drift.

    Usage with PyTorch
    ------------------
    tracker = NTKTracker(probe_inputs=X_probe)
    for step, (X, y) in enumerate(loader):
        loss = train_step(model, X, y)
        if step % record_every == 0:
            tracker.record(step, model=model)

    Usage without torch (supply external kernel matrices)
    ------------------------------------------------------
    tracker = NTKTracker()
    tracker.record_matrix(step=0, K=K0)
    tracker.record_matrix(step=100, K=K1)

    Parameters
    ----------
    probe_inputs : optional torch.Tensor
        Fixed set of inputs used to compute the NTK at each snapshot.
    record_every : int
        Ignored here; caller controls when to call record().
    subsample_params : int or None
        Passed to compute_empirical_ntk for efficiency.
    """

    def __init__(
        self,
        probe_inputs: Any = None,
        subsample_params: Optional[int] = None,
        device: str = "cpu",
    ):
        self.probe_inputs = probe_inputs
        self.subsample_params = subsample_params
        self.device = device

        self._trajectory = KernelTrajectory(name="NTK evolution")
        self._wall_times: List[float] = []
        self._steps: List[int] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, step: int, model: Any,
               probe_inputs: Any = None) -> "NTKTracker":
        """Compute and store the empirical NTK at the given training step."""
        inputs = probe_inputs if probe_inputs is not None else self.probe_inputs
        if inputs is None:
            raise ValueError("probe_inputs must be provided either at init or record().")
        K = compute_empirical_ntk(
            model, inputs,
            subsample_params=self.subsample_params,
            device=self.device,
        )
        return self.record_matrix(step, K)

    def record_matrix(self, step: int, K: np.ndarray) -> "NTKTracker":
        """Store a pre-computed kernel matrix at the given step."""
        self._trajectory.add(t=float(step), K=np.asarray(K, dtype=float))
        self._steps.append(step)
        self._wall_times.append(time.time())
        return self

    # ------------------------------------------------------------------
    # Trajectory access
    # ------------------------------------------------------------------

    @property
    def trajectory(self) -> KernelTrajectory:
        return self._trajectory

    def hs_distances(self) -> np.ndarray:
        """Sequential HS distances between consecutive NTK snapshots."""
        return self._trajectory.segment_distances()

    def cumulative_drift(self) -> np.ndarray:
        """Cumulative HS distance from the initial NTK."""
        return self._trajectory.cumulative_length()

    # ------------------------------------------------------------------
    # Convergence analysis
    # ------------------------------------------------------------------

    def convergence_rate(self) -> float:
        """Exponential decay exponent λ of ‖K(t) − K(t−1)‖_HS ≈ A·e^{−λt}.

        Positive value means the NTK is converging.
        """
        return self._trajectory.decay_rate()

    def is_converged(self, tol: float = 1e-3, window: int = 5) -> bool:
        return self._trajectory.is_convergent(tol=tol, window=window)

    def convergence_step(self, tol: float = 1e-3, window: int = 5) -> Optional[int]:
        """Training step at which the NTK first converged."""
        t = self._trajectory.convergence_time(tol=tol, window=window)
        return int(t) if t is not None else None

    # ------------------------------------------------------------------
    # Normalised kernel at a given step
    # ------------------------------------------------------------------

    def kernel_at(self, step: int) -> np.ndarray:
        """Interpolated kernel matrix at a given training step."""
        return self._trajectory.interpolate(float(step))

    def initial_kernel(self) -> Optional[np.ndarray]:
        if len(self._trajectory) == 0:
            return None
        _, K = self._trajectory[0]
        return K

    def final_kernel(self) -> Optional[np.ndarray]:
        if len(self._trajectory) == 0:
            return None
        _, K = self._trajectory[-1]
        return K

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        dists = self.hs_distances()
        return {
            "n_snapshots": len(self._trajectory),
            "total_drift_hs": float(self._trajectory.path_length()),
            "mean_step_distance": float(np.mean(dists)) if len(dists) > 0 else 0.0,
            "max_step_distance": float(np.max(dists)) if len(dists) > 0 else 0.0,
            "convergence_rate": self.convergence_rate(),
            "is_converged": self.is_converged(),
            "convergence_step": self.convergence_step(),
        }

    def __repr__(self) -> str:
        return (f"NTKTracker(snapshots={len(self._trajectory)}, "
                f"drift={self._trajectory.path_length():.4f}, "
                f"converged={self.is_converged()})")

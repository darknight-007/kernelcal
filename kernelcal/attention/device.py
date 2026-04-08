"""
Device detection and tensor utilities.
Auto-selects CUDA > MPS (Apple Silicon) > CPU.
Uses float16 on GPU for gaming-laptop VRAM efficiency.
"""

from __future__ import annotations
from typing import Optional


def best_device(prefer: Optional[str] = None) -> "torch.device":
    """Return the best available device, or the specified one."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for kernelcal.attention. "
            "Install with: pip install torch"
        )
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def best_dtype(device: "torch.device") -> "torch.dtype":
    """float16 on GPU/MPS (fits 8-GB gaming VRAM), float32 on CPU."""
    import torch
    if device.type in ("cuda", "mps"):
        return torch.float16
    return torch.float32


def device_info(device: "torch.device") -> str:
    """Human-readable device info."""
    try:
        import torch
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            vram_gb = props.total_memory / 1e9
            return f"CUDA — {props.name} ({vram_gb:.1f} GB VRAM)"
        if device.type == "mps":
            return "Apple Silicon MPS"
        return "CPU"
    except Exception:
        return str(device)

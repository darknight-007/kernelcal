"""
run_attention_experiment: end-to-end MaxCal attention kernel experiment.

Three modes, all runnable on a gaming laptop (RTX 3060/4060, 8 GB VRAM):

  "synthetic"   — no PyTorch required; pure numpy synthetic softmax weights.
                  Runs instantly on CPU. Tests all kernelcal.attention machinery.

  "gpt2"        — GPT-2 small (117M params) via HuggingFace Transformers.
                  Uses float16 on GPU. ~1.5 GB VRAM. Runs a few forward passes
                  and records spectral diagnostics across layers/heads.

  "distilgpt2"  — DistilGPT-2 (82M params). ~0.9 GB VRAM. Fastest GPU mode.

  "custom"      — supply your own PyTorch model; register hooks manually.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .kernel import AttentionKernel, AttentionKernelResult
from .tracker import AttentionKernelTracker
from .device import best_device, best_dtype, device_info


@dataclass
class AttentionExperimentResult:
    """Collected results from an attention kernel experiment."""

    model_name: str
    device_name: str
    n_layers: int
    n_heads: int
    seq_len: int
    elapsed_s: float
    # layer → head → AttentionKernelResult
    results: Dict[int, Dict[int, AttentionKernelResult]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"AttentionExperiment  model={self.model_name}  device={self.device_name}",
            f"  layers={self.n_layers}  heads={self.n_heads}  seq_len={self.seq_len}",
            f"  elapsed={self.elapsed_s:.2f}s",
        ]
        for layer_idx, heads in sorted(self.results.items()):
            for head_idx, res in sorted(heads.items()):
                lines.append(
                    f"  L{layer_idx:02d}H{head_idx:02d}  "
                    f"λ₁={res.fiedler_value:.4f}  "
                    f"H={res.spectral_entropy:.3f}  "
                    f"Δ'={res.fiedler_gap:.3f}  "
                    f"S_coup={res.coupling_entropy:.3f}  "
                    f"{'✓' if res.converged else '✗'}"
                )
        return "\n".join(lines)

    def to_json(self) -> dict:
        out = {
            "model_name": self.model_name,
            "device_name": self.device_name,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "seq_len": self.seq_len,
            "elapsed_s": self.elapsed_s,
            "results": {},
        }
        for layer_idx, heads in self.results.items():
            out["results"][str(layer_idx)] = {}
            for head_idx, res in heads.items():
                out["results"][str(layer_idx)][str(head_idx)] = {
                    "fiedler_value": res.fiedler_value,
                    "spectral_entropy": res.spectral_entropy,
                    "hessian_gap": res.hessian_gap,
                    "fiedler_gap": res.fiedler_gap,
                    "coupling_entropy": res.coupling_entropy,
                    "residual_inf_norm": res.residual_inf_norm,
                    "converged": res.converged,
                }
        return out

    def save_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))


# ──────────────────────────────────────────────────────────────────────────────

def run_attention_experiment(
    model_name: str = "synthetic",
    seq_len: int = 32,
    n_prompts: int = 4,
    layers: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
    sigma2: float = 1.0,
    mu2: float = 2.0,
    eigenvalue_aware: bool = True,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> AttentionExperimentResult:
    """
    Run MaxCal spectral diagnostics on transformer attention kernels.

    Parameters
    ----------
    model_name : "synthetic" | "gpt2" | "distilgpt2" | "gpt2-medium"
        "synthetic" — pure numpy, no PyTorch, runs instantly on CPU.
        Others — HuggingFace Transformers (requires: pip install transformers).
    seq_len : int
        Token sequence length. 32–128 is fine for gaming laptop.
    n_prompts : int
        Number of forward passes to average over (transformer modes).
    layers : list[int] or None
        Which layers to analyse. None = all.
    heads : list[int] or None
        Which heads to analyse. None = all.
    sigma2, mu2, eigenvalue_aware :
        MaxCal source parameters.
    device : str or None
        "cuda" | "mps" | "cpu" | None (auto-detect).
    output_dir : str or None
        Directory to save summary JSON and trajectory plots.
    verbose : bool
        Print progress.

    Returns
    -------
    AttentionExperimentResult
    """
    t0 = time.time()
    dev = best_device(device)
    dev_str = device_info(dev)

    if verbose:
        print(f"[kernelcal.attention] model={model_name}  device={dev_str}")

    if model_name == "synthetic":
        result = _run_synthetic(
            seq_len=seq_len,
            n_prompts=n_prompts,
            layers=layers,
            heads=heads,
            sigma2=sigma2,
            mu2=mu2,
            eigenvalue_aware=eigenvalue_aware,
            verbose=verbose,
        )
    else:
        result = _run_hf_model(
            model_name=model_name,
            seq_len=seq_len,
            n_prompts=n_prompts,
            layers=layers,
            heads=heads,
            sigma2=sigma2,
            mu2=mu2,
            eigenvalue_aware=eigenvalue_aware,
            device=dev,
            verbose=verbose,
        )

    result.elapsed_s = time.time() - t0
    result.device_name = dev_str

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"{model_name.replace('/', '_')}_attention_diagnostics.json"
        result.save_json(str(json_path))
        if verbose:
            print(f"[kernelcal.attention] saved → {json_path}")

    if verbose:
        print(result.summary())

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Backends

def _run_synthetic(
    seq_len: int,
    n_prompts: int,
    layers: Optional[List[int]],
    heads: Optional[List[int]],
    sigma2: float,
    mu2: float,
    eigenvalue_aware: bool,
    verbose: bool,
) -> "AttentionExperimentResult":
    """Pure numpy synthetic attention — no GPU or transformers needed."""
    n_layers_sim = len(layers) if layers else 3
    n_heads_sim = len(heads) if heads else 4
    layer_idxs = layers if layers else list(range(n_layers_sim))
    head_idxs = heads if heads else list(range(n_heads_sim))

    results: Dict[int, Dict[int, AttentionKernelResult]] = {}

    for li, layer in enumerate(layer_idxs):
        results[layer] = {}
        for hi, head in enumerate(head_idxs):
            # Average attention weights over n_prompts runs
            attn_sum = np.zeros((seq_len, seq_len))
            for p in range(n_prompts):
                ak_tmp = AttentionKernel.synthetic(
                    seq_len=seq_len,
                    temperature=1.0 + 0.5 * li,   # deeper layers = sharper
                    seed=li * 100 + hi * 10 + p,
                )
                attn_sum += ak_tmp._K
            attn_mean = attn_sum / n_prompts

            ak = AttentionKernel(
                attn_mean, layer=layer, head=head, step=0,
                sigma2=sigma2, mu2=mu2, eigenvalue_aware=eigenvalue_aware,
            )
            res = ak.analyse()
            results[layer][head] = res

            if verbose:
                print(
                    f"  [synthetic] L{layer:02d}H{head:02d}  "
                    f"λ₁={res.fiedler_value:.4f}  H={res.spectral_entropy:.3f}  "
                    f"Δ'={res.fiedler_gap:.3f}  {'✓' if res.converged else '✗'}"
                )

    return AttentionExperimentResult(
        model_name="synthetic",
        device_name="CPU",
        n_layers=len(layer_idxs),
        n_heads=len(head_idxs),
        seq_len=seq_len,
        elapsed_s=0.0,
        results=results,
    )


def _run_hf_model(
    model_name: str,
    seq_len: int,
    n_prompts: int,
    layers: Optional[List[int]],
    heads: Optional[List[int]],
    sigma2: float,
    mu2: float,
    eigenvalue_aware: bool,
    device: "torch.device",
    verbose: bool,
) -> "AttentionExperimentResult":
    """HuggingFace Transformers backend — GPU-accelerated."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for non-synthetic experiments.\n"
            "Install with: pip install transformers torch"
        ) from exc

    dtype = best_dtype(device)
    if verbose:
        print(f"  Loading {model_name} (dtype={dtype}) ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        output_attentions=True,
    ).to(device).eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Model loaded  {n_params:.0f}M params")

    # Probe prompts — short sentences ensuring seq_len is respected
    probe_texts = [
        "The kernel of a linear map encodes what the transformation destroys.",
        "Entropy measures our ignorance of the microstate of a physical system.",
        "A transformer learns to attend to relevant context across a sequence.",
        "Information is a thermodynamic resource subject to Landauer's principle.",
        "The fixed point of a dynamical system is where the derivative vanishes.",
        "Self-consistent kernels are determined by the MaxCal field equation.",
        "Photosystem I uses entropy as fuel for uphill electron transfer.",
        "The grapevine says attention was always a kernel.",
    ][:n_prompts]

    cfg = model.config
    n_layers_total = cfg.n_layer if hasattr(cfg, "n_layer") else cfg.num_hidden_layers
    n_heads_total = cfg.n_head if hasattr(cfg, "n_head") else cfg.num_attention_heads

    layer_idxs = layers if layers else list(range(n_layers_total))
    head_idxs = heads if heads else list(range(n_heads_total))

    # Accumulate attention weights across prompts: layer → head → sum
    attn_accum: Dict[int, Dict[int, np.ndarray]] = {
        li: {hi: None for hi in head_idxs} for li in layer_idxs
    }

    with torch.inference_mode():
        for pi, text in enumerate(probe_texts):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            input_ids = enc["input_ids"].to(device)
            actual_seq = input_ids.shape[-1]

            outputs = model(input_ids, output_attentions=True)
            all_attn = outputs.attentions  # tuple of (1, n_heads, seq, seq)

            for li in layer_idxs:
                if li >= len(all_attn):
                    continue
                layer_attn = all_attn[li][0].float().cpu().numpy()  # (n_heads, seq, seq)
                for hi in head_idxs:
                    if hi >= layer_attn.shape[0]:
                        continue
                    A = layer_attn[hi]  # (seq, seq)
                    if attn_accum[li][hi] is None:
                        attn_accum[li][hi] = np.zeros((actual_seq, actual_seq))
                    # handle varying seq length by padding/trimming
                    s = min(A.shape[0], attn_accum[li][hi].shape[0])
                    attn_accum[li][hi][:s, :s] += A[:s, :s]

    # Run MaxCal analysis on averaged kernels
    results: Dict[int, Dict[int, AttentionKernelResult]] = {}
    for li in layer_idxs:
        results[li] = {}
        for hi in head_idxs:
            A_sum = attn_accum[li][hi]
            if A_sum is None:
                continue
            A_mean = A_sum / n_prompts
            ak = AttentionKernel(
                A_mean, layer=li, head=hi, step=0,
                sigma2=sigma2, mu2=mu2, eigenvalue_aware=eigenvalue_aware,
            )
            res = ak.analyse()
            results[li][hi] = res
            if verbose:
                print(
                    f"  L{li:02d}H{hi:02d}  "
                    f"λ₁={res.fiedler_value:.4f}  H={res.spectral_entropy:.3f}  "
                    f"Δ'={res.fiedler_gap:.3f}  S_coup={res.coupling_entropy:.3f}  "
                    f"{'✓' if res.converged else '✗'}"
                )

    return AttentionExperimentResult(
        model_name=model_name,
        device_name="",  # filled by caller
        n_layers=len(layer_idxs),
        n_heads=len(head_idxs),
        seq_len=seq_len,
        elapsed_s=0.0,
        results=results,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="MaxCal spectral diagnostics on transformer attention kernels."
    )
    parser.add_argument("--model", default="synthetic",
                        help="Model name: synthetic | gpt2 | distilgpt2 | gpt2-medium")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--n-prompts", type=int, default=4)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--heads", type=int, nargs="*", default=None)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--mu2", type=float, default=2.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_attention_experiment(
        model_name=args.model,
        seq_len=args.seq_len,
        n_prompts=args.n_prompts,
        layers=args.layers,
        heads=args.heads,
        sigma2=args.sigma2,
        mu2=args.mu2,
        device=args.device,
        output_dir=args.output_dir,
        verbose=True,
    )


if __name__ == "__main__":
    _main()

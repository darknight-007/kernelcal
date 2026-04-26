"""Compressed per-superquadric spectrum sidecar.

The OceanOptics UV-VIS-NIR spectrometer on the earth-rover delivers a
1024-2048 channel reading at ~10 Hz through a collimating lens (~5-10 cm
spot at the camera bore-sight).  The naive payload (2048 * float32 *
10 Hz = 80 kB/s = 640 kbps) by itself blows the 100 kbps link budget.

This module provides a compact binary sidecar -- the
:class:`SpectrumPacket` -- that compresses each per-SQ accumulated mean
spectrum into **96 bytes** using a hybrid DCT-32 + PCA-K representation:

* The first 32 DCT-II coefficients of the mean spectrum, quantized to
  ``int16`` log-scale: captures the smooth envelope (continuum) and
  broad absorption features.

* An optional 8-component PCA projection onto a globally-trained
  basis (fit once over a reference vegetation/soil library), stored
  as ``float16``: captures the residual fine structure orthogonal
  to the smooth DCT envelope.

* Metadata: ``n_samples`` (Welford count), wavelength range, quality
  score, flags.

96 bytes per spectrum at 1 spectrum / SQ at ~50 SQs/s = 4.8 kB/s =
~38 kbps -- comfortable inside the 100 kbps trike->server link.

Wire layout
-----------

::

    offset  size  field            encoding
    ------  ----  ---------------  ------------------------------------
    0       1     version          uint8, currently 1
    1       1     flags            uint8 bitfield
                                    bit 0: has_pca_projection
                                    bit 1: log_quantized_dct (reserved)
                                    bit 2: spectrum_normalized (reserved)
    2       2     n_samples        uint16, Welford count (sat at 65535)
    4       1     quality_score    uint8, [0, 1] *255
    5       1     reserved         uint8
    6       2     lambda_lo_dnm    uint16, 0.1 nm units (cap 65535 = 6553.5 nm)
    8       2     lambda_hi_dnm    uint16, 0.1 nm units
    10      2     spectrum_norm    float16, max-abs normalization of input
    12      2     coeff_max        float16, max |DCT coeff| (int16 scale)
    14      64    dct32            32 x int16 big-endian, linearly-quantized
                                   coefficients on ``[-coeff_max, +coeff_max]``
    78      16    pca8             8 x float16 big-endian PCA components
                                   (zero if has_pca_projection=0)
    94      2     n_channels       uint16, encode-time spectrum length

Total: **96 bytes** per posed spectrum.

Why DCT first, PCA second
-------------------------

* DCT captures *band-agnostic* smooth structure (chlorophyll-red-edge
  curvature, water-band depth) without requiring training data.
* PCA captures dataset-specific deviations (e.g. PHX-mesquite vs PHX-
  palo-verde vs concrete-roof) once you have a reference library.
* Either alone works; together they're sub-additive in MSE.

If the basis isn't available (cold start), the DCT-only mode is still
informative and the ``has_pca_projection`` flag stays 0.

Public API
----------

::

    from kernelcal.distinction_game.geometry.spectrum import (
        SpectrumAccumulator,
        SpectrumPacket,
        compress_spectrum,
        decompress_spectrum,
    )
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.fft import dct as _scipy_dct, idct as _scipy_idct  # type: ignore[import]

    _HAS_SCIPY_FFT = True
except ImportError:  # pragma: no cover -- scipy is a hard kernelcal dep
    _HAS_SCIPY_FFT = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PACKET_VERSION: int = 1

#: Total wire bytes per spectrum sidecar.
SPECTRUM_PACKET_BYTES: int = 96

#: Number of DCT-II coefficients retained.
DCT_RETAINED: int = 32

#: Number of PCA components retained (when basis available).
PCA_RETAINED: int = 8

#: Wavelength quantization step in nm (0.1 nm = 1 deci-nm).
LAMBDA_STEP_NM: float = 0.1

FLAG_HAS_PCA: int = 1 << 0
FLAG_LOG_DCT: int = 1 << 1
FLAG_NORMALIZED: int = 1 << 2

_INT16_MAX: int = 32767


# ---------------------------------------------------------------------------
# DCT helpers (no scipy hard-dep -- use numpy FFT for Type-II DCT)
# ---------------------------------------------------------------------------


def _dct2_orthonormal(x: np.ndarray) -> np.ndarray:
    """Type-II DCT, orthonormalized.

    Uses ``scipy.fft.dct`` when available; otherwise falls back to a
    direct cosine-basis implementation (slow but correct).  Output
    length equals input length; we only retain the first K coefficients
    downstream.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N == 0:
        return np.zeros(0)
    if _HAS_SCIPY_FFT:
        return _scipy_dct(x, type=2, norm="ortho")
    # Fallback: direct sum (O(N^2))
    n = np.arange(N)
    k = np.arange(N).reshape(-1, 1)
    basis = np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    out = (basis @ x)
    out[0] *= math.sqrt(1.0 / N)
    out[1:] *= math.sqrt(2.0 / N)
    return out


def _idct2_orthonormal(c: np.ndarray, N: int) -> np.ndarray:
    """Inverse orthonormal Type-II DCT (= DCT-III, ortho-normalized).

    ``c`` may be shorter than ``N`` -- missing high-freq coefficients
    are zero-padded.  This is the lossy reconstruction used by
    :func:`decompress_spectrum`.
    """
    c = np.asarray(c, dtype=float).ravel()
    if c.size < N:
        cf = np.zeros(N)
        cf[: c.size] = c
        c = cf
    elif c.size > N:
        c = c[:N]
    if _HAS_SCIPY_FFT:
        return _scipy_idct(c, type=2, norm="ortho")
    # Fallback: direct sum
    n = np.arange(N).reshape(-1, 1)
    k = np.arange(N)
    basis = np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    cs = c.copy()
    cs[0] *= math.sqrt(1.0 / N)
    cs[1:] *= math.sqrt(2.0 / N)
    return (basis @ cs).ravel()


# ---------------------------------------------------------------------------
# Coefficient quantization (linear int16 with header scale)
# ---------------------------------------------------------------------------


def _quant_linear_int16(values: np.ndarray) -> Tuple[np.ndarray, float]:
    """Sign-preserving linear quantization to int16.

    Returns ``(codes, coeff_max)`` where ``codes`` is shape ``values``
    in ``int16`` mapped over ``[-coeff_max, +coeff_max] -> [-32767, 32767]``,
    and ``coeff_max`` is recorded in the packet header so the decoder
    can recover the original magnitudes.

    Tiny / zero arrays map to zero codes and ``coeff_max = 0``.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return np.zeros(0, dtype=np.int16), 0.0
    abs_max = float(np.max(np.abs(v)))
    if not math.isfinite(abs_max) or abs_max < 1e-12:
        return np.zeros_like(v, dtype=np.int16), 0.0
    codes = np.clip(np.round(v / abs_max * _INT16_MAX), -_INT16_MAX, _INT16_MAX)
    return codes.astype(np.int16), abs_max


def _dequant_linear_int16(codes: np.ndarray, coeff_max: float) -> np.ndarray:
    """Inverse of :func:`_quant_linear_int16`."""
    c = np.asarray(codes, dtype=float).ravel()
    if c.size == 0 or coeff_max == 0.0:
        return np.zeros_like(c)
    return c / _INT16_MAX * coeff_max


# ---------------------------------------------------------------------------
# SpectrumPacket dataclass and codec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectrumPacket:
    """Decoded representation of a 96-byte spectrum sidecar.

    Attributes
    ----------
    n_samples
        Welford sample count saturating at 65535.
    quality_score
        0..1; producer-side quality estimate (e.g. integration time
        OK, no saturation, sensor temp OK).
    lambda_lo_nm, lambda_hi_nm
        Wavelength range (nm) covered by the original spectrum at
        encode time.  The decoder uses these to grid-resample.
    n_channels
        Encode-time spectrum length (so the receiver runs an IDCT of
        the matching length before any optional resampling).
    spectrum_norm
        Max-abs normalization factor applied to the spectrum before
        DCT, so the absolute scale can be recovered on the receiver.
    dct_coeffs
        Length-32 ``float64`` array, first 32 DCT-II coefficients of
        the (normalized) mean spectrum.
    pca_proj
        Optional length-8 PCA projection coefficients, or None if the
        producer didn't have a basis.
    flags
        Bitfield (``FLAG_*`` constants).
    """

    n_samples: int = 0
    quality_score: float = 1.0
    lambda_lo_nm: float = 200.0
    lambda_hi_nm: float = 1100.0
    n_channels: int = 2048
    spectrum_norm: float = 1.0
    dct_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(DCT_RETAINED))
    pca_proj: Optional[np.ndarray] = None
    flags: int = FLAG_NORMALIZED

    # ---- Wire format ------------------------------------------------

    def to_bytes(self) -> bytes:
        return _pack_spectrum_packet(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> "SpectrumPacket":
        return _unpack_spectrum_packet(data)

    # ---- Decompression to a spectrum --------------------------------

    def reconstruct(
        self,
        n_channels: int = 2048,
        *,
        pca_basis: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reconstruct an ``n_channels``-length spectrum.

        Parameters
        ----------
        n_channels
            Output spectrum length (e.g. 2048 to match the OceanOptics
            native CCD pitch).
        pca_basis
            Optional ``(PCA_RETAINED, n_channels)`` PCA basis matrix
            used to add the residual fine structure.  If None and
            ``flags & FLAG_HAS_PCA`` is set, only the DCT envelope is
            reconstructed.
        """
        return _reconstruct_from_packet(self, n_channels, pca_basis=pca_basis)


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------

# Header: version, flags, n_samples, quality, reserved,
#         lambda_lo, lambda_hi, spectrum_norm (f16), coeff_max (f16)
# = 1+1+2+1+1+2+2+2+2 = 14 bytes
_HEADER_STRUCT = struct.Struct(">BBHBBHHee")
_DCT_STRUCT = struct.Struct(">32h")          # 64 bytes
_PCA_STRUCT = struct.Struct(">8e")           # 16 bytes (8 x float16)
_TAIL_STRUCT = struct.Struct(">H")           # 2 bytes: n_channels (uint16)


def _pack_spectrum_packet(pkt: SpectrumPacket) -> bytes:
    flags = int(pkt.flags) & 0xFF
    coeffs = np.asarray(pkt.dct_coeffs, dtype=float).ravel()
    if coeffs.size < DCT_RETAINED:
        padded = np.zeros(DCT_RETAINED)
        padded[: coeffs.size] = coeffs
        coeffs = padded
    elif coeffs.size > DCT_RETAINED:
        coeffs = coeffs[:DCT_RETAINED]

    codes, coeff_max = _quant_linear_int16(coeffs)

    quality = int(np.clip(round(float(pkt.quality_score) * 255.0), 0, 255))
    n_s = int(np.clip(int(pkt.n_samples), 0, 65535))
    lo = int(np.clip(round(float(pkt.lambda_lo_nm) / LAMBDA_STEP_NM), 0, 65535))
    hi = int(np.clip(round(float(pkt.lambda_hi_nm) / LAMBDA_STEP_NM), 0, 65535))
    n_ch = int(np.clip(int(pkt.n_channels), 0, 65535))

    header = _HEADER_STRUCT.pack(
        PACKET_VERSION,
        flags,
        n_s,
        quality,
        0,
        lo,
        hi,
        float(np.float16(pkt.spectrum_norm)),
        float(np.float16(coeff_max)),
    )
    dct_blob = _DCT_STRUCT.pack(*[int(c) for c in codes])

    if pkt.pca_proj is not None and (flags & FLAG_HAS_PCA):
        pca = np.asarray(pkt.pca_proj, dtype=np.float16).ravel()
        if pca.size < PCA_RETAINED:
            padded = np.zeros(PCA_RETAINED, dtype=np.float16)
            padded[: pca.size] = pca
            pca = padded
        elif pca.size > PCA_RETAINED:
            pca = pca[:PCA_RETAINED]
        pca_blob = _PCA_STRUCT.pack(*[float(c) for c in pca])
    else:
        pca_blob = b"\x00" * 16

    tail = _TAIL_STRUCT.pack(n_ch)

    out = header + dct_blob + pca_blob + tail
    if len(out) != SPECTRUM_PACKET_BYTES:
        raise AssertionError(
            f"SpectrumPacket pack: produced {len(out)} bytes, "
            f"expected {SPECTRUM_PACKET_BYTES}."
        )
    return out


def _unpack_spectrum_packet(data: bytes) -> SpectrumPacket:
    if len(data) < SPECTRUM_PACKET_BYTES:
        raise ValueError(
            f"SpectrumPacket.unpack: need {SPECTRUM_PACKET_BYTES} bytes, "
            f"got {len(data)}"
        )
    (version, flags, n_s, quality, _reserved, lo, hi, spectrum_norm_f16,
     coeff_max_f16) = _HEADER_STRUCT.unpack_from(data, 0)
    if version != PACKET_VERSION:
        raise ValueError(f"Unsupported SpectrumPacket version {version}")

    dct_codes = np.array(_DCT_STRUCT.unpack_from(data, 14), dtype=np.int16)
    dct_coeffs = _dequant_linear_int16(dct_codes, float(coeff_max_f16))

    pca_proj: Optional[np.ndarray] = None
    if flags & FLAG_HAS_PCA:
        pca_proj = np.array(_PCA_STRUCT.unpack_from(data, 78), dtype=float)

    (n_ch,) = _TAIL_STRUCT.unpack_from(data, 94)

    return SpectrumPacket(
        n_samples=int(n_s),
        quality_score=float(quality) / 255.0,
        lambda_lo_nm=float(lo) * LAMBDA_STEP_NM,
        lambda_hi_nm=float(hi) * LAMBDA_STEP_NM,
        n_channels=int(n_ch) if n_ch > 0 else 2048,
        spectrum_norm=float(spectrum_norm_f16),
        dct_coeffs=dct_coeffs,
        pca_proj=pca_proj,
        flags=int(flags),
    )


# ---------------------------------------------------------------------------
# Compression / decompression of a raw spectrum
# ---------------------------------------------------------------------------


def compress_spectrum(
    spectrum: np.ndarray,
    *,
    lambda_lo_nm: float,
    lambda_hi_nm: float,
    n_samples: int = 1,
    quality_score: float = 1.0,
    pca_basis: Optional[np.ndarray] = None,
) -> SpectrumPacket:
    """Compress a raw spectrum to a :class:`SpectrumPacket`.

    Pipeline:

    1. Normalize the input by its max-absolute value (records the norm
       in the header so absolute scale is recovered on the receiver).
    2. Run an orthonormal DCT-II.
    3. Keep the first :data:`DCT_RETAINED` coefficients.
    4. Linearly quantize them to int16 with a header-stored max.
    5. Optionally project the residual onto a PCA basis trained
       against the same spectrum length.

    Parameters
    ----------
    spectrum
        ``(N,)`` float spectrum (any sign / scale).
    lambda_lo_nm, lambda_hi_nm
        Wavelength range covered by ``spectrum``.
    n_samples
        Welford sample count contributing to this mean spectrum.
    quality_score
        Producer-side quality in ``[0, 1]``.
    pca_basis
        Optional ``(>=PCA_RETAINED, N)`` basis matrix (rows are basis
        vectors).  When provided the residual after DCT-32 reconstruction
        is projected onto the first :data:`PCA_RETAINED` rows.
    """
    s = np.asarray(spectrum, dtype=float).ravel()
    if s.size < 2:
        raise ValueError(
            f"compress_spectrum: spectrum must have >= 2 channels, got {s.size}"
        )
    if not (lambda_hi_nm > lambda_lo_nm):
        raise ValueError(
            f"compress_spectrum: lambda_hi_nm ({lambda_hi_nm}) must exceed "
            f"lambda_lo_nm ({lambda_lo_nm})"
        )

    abs_max = float(np.max(np.abs(s)))
    flags = FLAG_NORMALIZED
    if abs_max > 1e-12:
        s_n = s / abs_max
        spectrum_norm = abs_max
    else:
        s_n = s.copy()
        spectrum_norm = 1.0

    coeffs_full = _dct2_orthonormal(s_n)
    coeffs_keep = coeffs_full[:DCT_RETAINED].copy()

    pca_proj: Optional[np.ndarray] = None
    if pca_basis is not None:
        basis = np.asarray(pca_basis, dtype=float)
        if basis.ndim != 2 or basis.shape[0] < PCA_RETAINED:
            raise ValueError(
                f"compress_spectrum: pca_basis must be (>={PCA_RETAINED}, N); "
                f"got {basis.shape}"
            )
        if basis.shape[1] != s.size:
            raise ValueError(
                f"compress_spectrum: pca_basis last dim ({basis.shape[1]}) "
                f"!= spectrum length ({s.size})"
            )
        # Residual is computed in *normalized* space, then projected.
        recon_n = _idct2_orthonormal(coeffs_keep, s.size)
        residual_n = s_n - recon_n
        pca_proj = basis[:PCA_RETAINED] @ residual_n
        flags |= FLAG_HAS_PCA

    return SpectrumPacket(
        n_samples=int(n_samples),
        quality_score=float(quality_score),
        lambda_lo_nm=float(lambda_lo_nm),
        lambda_hi_nm=float(lambda_hi_nm),
        n_channels=int(s.size),
        spectrum_norm=float(spectrum_norm),
        dct_coeffs=coeffs_keep,
        pca_proj=pca_proj,
        flags=flags,
    )


def decompress_spectrum(
    packet: SpectrumPacket,
    *,
    n_channels: int = 2048,
    pca_basis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reconstruct an ``n_channels``-length spectrum from a packet.

    See :meth:`SpectrumPacket.reconstruct`.  Optional ``pca_basis`` adds
    the residual fine structure when ``FLAG_HAS_PCA`` is set.
    """
    return _reconstruct_from_packet(packet, n_channels, pca_basis=pca_basis)


def _reconstruct_from_packet(
    packet: SpectrumPacket,
    n_channels: int,
    *,
    pca_basis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reconstruct a spectrum from a packet.

    The IDCT is run at the encode-time length ``packet.n_channels`` so
    coefficient semantics are preserved; the result is then linearly
    interpolated to the requested ``n_channels``.  Finally the
    ``spectrum_norm`` is applied to recover absolute scale.
    """
    encode_len = max(int(packet.n_channels), len(packet.dct_coeffs))
    recon_n = _idct2_orthonormal(packet.dct_coeffs, encode_len)

    if (
        (packet.flags & FLAG_HAS_PCA)
        and packet.pca_proj is not None
        and pca_basis is not None
    ):
        basis = np.asarray(pca_basis, dtype=float)
        if basis.ndim == 2 and basis.shape[0] >= PCA_RETAINED and basis.shape[1] == encode_len:
            recon_n = recon_n + basis[:PCA_RETAINED].T @ packet.pca_proj

    # Resample to requested length.
    if encode_len != n_channels:
        x_old = np.linspace(0.0, 1.0, encode_len)
        x_new = np.linspace(0.0, 1.0, n_channels)
        recon_n = np.interp(x_new, x_old, recon_n)

    return recon_n * float(packet.spectrum_norm)


# ---------------------------------------------------------------------------
# SpectrumAccumulator -- per-SQ running-mean spectrum
# ---------------------------------------------------------------------------


@dataclass
class SpectrumAccumulator:
    """Per-SQ running mean of UV-VIS-NIR spectra.

    The earth-rover side accumulates a single mean spectrum per SQ
    over many bore-sight ray hits.  At publish time the mean is
    compressed via :func:`compress_spectrum` into a 96-byte
    :class:`SpectrumPacket` and shipped as a sidecar to the SQ's
    32-byte primitive packet.

    Notes
    -----
    Memory is O(n_channels) per SQ, not O(n_samples).  For 2048
    channels that's 16 kB per SQ -- fine for hundreds of SQs on a
    Jetson / NUC class host.
    """

    n_channels: int = 2048
    lambda_lo_nm: float = 200.0
    lambda_hi_nm: float = 1100.0
    n_samples: int = 0
    weight_sum: float = 0.0
    mean_spectrum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.n_channels < 2:
            raise ValueError(
                f"SpectrumAccumulator: n_channels must be >= 2 (got {self.n_channels})"
            )
        self.mean_spectrum = np.zeros(int(self.n_channels), dtype=float)

    def update(self, spectrum: np.ndarray, *, weight: float = 1.0) -> None:
        """Incorporate a new spectrum (resampled to the accumulator grid)."""
        s = np.asarray(spectrum, dtype=float).ravel()
        if s.size == 0 or weight <= 0.0 or not math.isfinite(weight):
            return
        if not np.all(np.isfinite(s)):
            return
        if s.size != self.n_channels:
            # Linear-interp resample to accumulator grid.
            x_old = np.linspace(0.0, 1.0, s.size)
            x_new = np.linspace(0.0, 1.0, self.n_channels)
            s = np.interp(x_new, x_old, s)
        new_w = self.weight_sum + weight
        delta = s - self.mean_spectrum
        self.mean_spectrum = self.mean_spectrum + (weight / new_w) * delta
        self.weight_sum = new_w
        self.n_samples += 1

    def merge(self, other: "SpectrumAccumulator") -> None:
        """In-place merge with another accumulator (must share grid)."""
        if other.weight_sum <= 0.0:
            return
        if self.n_channels != other.n_channels:
            # Resample other to our grid before merging.
            x_old = np.linspace(0.0, 1.0, other.n_channels)
            x_new = np.linspace(0.0, 1.0, self.n_channels)
            other_mean = np.interp(x_new, x_old, other.mean_spectrum)
        else:
            other_mean = other.mean_spectrum

        if self.weight_sum <= 0.0:
            self.mean_spectrum = other_mean.copy()
            self.weight_sum = other.weight_sum
            self.n_samples = other.n_samples
            return

        new_w = self.weight_sum + other.weight_sum
        delta = other_mean - self.mean_spectrum
        self.mean_spectrum = self.mean_spectrum + (other.weight_sum / new_w) * delta
        self.weight_sum = new_w
        self.n_samples += other.n_samples

    def reset(self) -> None:
        self.mean_spectrum.fill(0.0)
        self.n_samples = 0
        self.weight_sum = 0.0

    def to_packet(
        self,
        *,
        quality_score: float = 1.0,
        pca_basis: Optional[np.ndarray] = None,
    ) -> SpectrumPacket:
        """Compress the running mean into a 96-byte packet."""
        return compress_spectrum(
            self.mean_spectrum,
            lambda_lo_nm=self.lambda_lo_nm,
            lambda_hi_nm=self.lambda_hi_nm,
            n_samples=self.n_samples,
            quality_score=quality_score,
            pca_basis=pca_basis,
        )


__all__ = [
    "SPECTRUM_PACKET_BYTES",
    "DCT_RETAINED",
    "PCA_RETAINED",
    "FLAG_HAS_PCA",
    "FLAG_LOG_DCT",
    "FLAG_NORMALIZED",
    "SpectrumPacket",
    "SpectrumAccumulator",
    "compress_spectrum",
    "decompress_spectrum",
]

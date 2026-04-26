"""Quantized binary codec for posed superquadrics.

Wire format (fixed 32 bytes per primitive + optional trailers)
==============================================================

::

    offset  size  field          encoding
    ------  ----  -------------  --------------------------------------
    0       1     version        uint8, currently ``1``
    1       1     flags          uint8 bitfield, see ``FLAG_*`` constants
    2       1     class_idx      uint8, semantic-class lookup (0 = none)
    3       1     reserved       uint8, must be zero on encode, ignored
    4       6     log_a          3x int16 big-endian, quantized log(scale)
                                 over ``[log(A_MIN), log(A_MAX)]``
                                 -> ~0.04 % multiplicative precision on
                                 each semi-axis.
    10      2     epsilon        2x uint8, quantized log(eps) over
                                 ``[log(EPS_MIN), log(EPS_MAX)]``
                                 -> ~1.4 % multiplicative precision.
    12      8     orientation    4x int16 big-endian, quaternion
                                 components in ``[-1, 1]`` scaled by
                                 ``32767``.  Always stored as
                                 ``[qx, qy, qz, qw]``; sign is
                                 normalized so ``qw >= 0`` to avoid
                                 quaternion double-cover.
    20      12    translation    3x int32 big-endian, millimetre units.
                                 Range ``+/- ~2.1e9 mm`` = +/- 2,000 km.

Total: **32 bytes** per posed primitive header.

Optional trailers (presence indicated in ``flags``)
---------------------------------------------------

Order on the wire (after the 32-byte header), each section omitted when
its flag is zero:

1. ``FLAG_HAS_PARENT`` (bit 0): 8 trailing bytes -- big-endian
   int64 hash of a parent id (caller-supplied; receivers can leave
   it as opaque association key).

2. ``FLAG_HAS_PROPERTIES`` (bit 1): variable-length property trailer
   from :func:`kernelcal.distinction_game.geometry.properties.encode_property_trailer`.
   Format: 1 byte ``n_props``, then for each property a 1-byte
   :class:`PropertyId` followed by a 1/2/4-byte quantized value
   per the registry.  Receivers must consult the global property
   registry to know each value width -- unknown IDs abort decoding.

3. ``FLAG_HAS_SPECTRUM`` (bit 2): 96 bytes -- a single
   :class:`SpectrumPacket` (DCT-32 + optional PCA-8 compressed
   UV-VIS-NIR spectrum).

The codec deliberately drops the SQ ``id`` and ``attributes`` (these
travel through the JSON envelope of ``POST /api/observe``).  Pack/unpack
is therefore *lossy* on identity but *lossless on geometry* up to
quantization precision.

Link-budget context
-------------------

* Bare SQ header: ``32 B``  -- 25 SQs/window @ 1 Hz = 6.4 kbps
* + parent_hash trailer:    +``8 B`` per SQ that has a parent
* + 5-property trailer:     +``11 B`` per SQ (1 length + 5 * (1 ID + 1-2 B))
* + spectrum sidecar:       +``96 B`` per SQ that has a spectrum
* Worst case (all flags):   ``147 B`` per SQ

At 100 SQs/sec with the full appearance trailer, the wire rate is
~120 kbps.  For 64 kbps links the producer should drop spectrum
sidecars (or batch them at 1/N rate); the SQ header + properties
together stay under 32 kbps.

See also :mod:`.properties` and :mod:`.spectrum`.
"""

from __future__ import annotations

import math
import struct
from typing import Mapping, Optional, Tuple, Union

import numpy as np

from .properties import (
    MAX_PROPERTIES_PER_TRAILER,
    PropertyId,
    decode_property_trailer,
    encode_property_trailer,
    encoded_size as property_trailer_size,
)
from .spectrum import SPECTRUM_PACKET_BYTES, SpectrumPacket
from .superquadric import (
    A_MAX,
    A_MIN,
    EPS_MAX,
    EPS_MIN,
    Superquadric,
    _axis_angle_from_so3,
    _so3_from_axis_angle,
)

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

_VERSION: int = 1

#: Total bytes in a packed posed superquadric (no optional trailers).
PACKED_BYTES: int = 32

#: Trailing bytes when ``flags & FLAG_HAS_PARENT`` is set.
PACKED_PARENT_BYTES: int = 8

#: Trailing bytes when ``flags & FLAG_HAS_SPECTRUM`` is set.
PACKED_SPECTRUM_BYTES: int = SPECTRUM_PACKET_BYTES

FLAG_HAS_PARENT: int = 1 << 0
FLAG_HAS_PROPERTIES: int = 1 << 1
FLAG_HAS_SPECTRUM: int = 1 << 2

# Quantization grids ---------------------------------------------------------

_LOG_A_LO: float = math.log(A_MIN)
_LOG_A_HI: float = math.log(A_MAX)
_LOG_A_INT16_MIN: int = -32767
_LOG_A_INT16_MAX: int = 32767

_LOG_EPS_LO: float = math.log(EPS_MIN)
_LOG_EPS_HI: float = math.log(EPS_MAX)
_LOG_EPS_UINT8_MAX: int = 255

#: Translation resolution: 1 mm per int32 unit.  Range is 32-bit signed
#: integer mm = +/- ~2.1 million km, so this never clips for any
#: realistic local frame.  For nanometre-scale lab work, override the
#: ``t_resolution_m`` argument to :func:`pack_superquadric`.
DEFAULT_T_RESOLUTION_M: float = 1.0e-3

#: Quaternion components are stored as int16 in [-1, 1].
_QUAT_INT16_SCALE: int = 32767


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def _quant_log_a(a: float) -> int:
    """Quantize a single semi-axis length to a 16-bit signed log code."""
    a = float(np.clip(a, A_MIN, A_MAX))
    log_a = math.log(a)
    frac = (log_a - _LOG_A_LO) / (_LOG_A_HI - _LOG_A_LO)
    code = int(round(frac * (_LOG_A_INT16_MAX - _LOG_A_INT16_MIN) + _LOG_A_INT16_MIN))
    return int(np.clip(code, _LOG_A_INT16_MIN, _LOG_A_INT16_MAX))


def _dequant_log_a(code: int) -> float:
    frac = (int(code) - _LOG_A_INT16_MIN) / (_LOG_A_INT16_MAX - _LOG_A_INT16_MIN)
    log_a = _LOG_A_LO + frac * (_LOG_A_HI - _LOG_A_LO)
    return float(np.clip(math.exp(log_a), A_MIN, A_MAX))


def _quant_eps(e: float) -> int:
    e = float(np.clip(e, EPS_MIN, EPS_MAX))
    log_e = math.log(e)
    frac = (log_e - _LOG_EPS_LO) / (_LOG_EPS_HI - _LOG_EPS_LO)
    code = int(round(frac * _LOG_EPS_UINT8_MAX))
    return int(np.clip(code, 0, _LOG_EPS_UINT8_MAX))


def _dequant_eps(code: int) -> float:
    frac = int(code) / _LOG_EPS_UINT8_MAX
    log_e = _LOG_EPS_LO + frac * (_LOG_EPS_HI - _LOG_EPS_LO)
    return float(np.clip(math.exp(log_e), EPS_MIN, EPS_MAX))


def _quat_from_so3(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion ``[x, y, z, w]`` with ``w >= 0``."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    if q[3] < 0.0:
        q = -q
    n = float(np.linalg.norm(q))
    if n > 0.0:
        q = q / n
    return q


def _so3_from_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = q / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _quant_translation(t: np.ndarray, resolution_m: float) -> np.ndarray:
    units = np.round(np.asarray(t, dtype=float).reshape(3) / resolution_m).astype(np.int64)
    return np.clip(units, -(2 ** 31 - 1), 2 ** 31 - 1).astype(np.int32)


def _dequant_translation(code: np.ndarray, resolution_m: float) -> np.ndarray:
    return np.asarray(code, dtype=np.int64).astype(float) * resolution_m


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------


_HEADER_STRUCT = struct.Struct(">BBBB")  # version, flags, class_idx, reserved
_LOGA_STRUCT = struct.Struct(">3h")
_EPS_STRUCT = struct.Struct(">2B")
_QUAT_STRUCT = struct.Struct(">4h")
_T_STRUCT = struct.Struct(">3i")
_PARENT_STRUCT = struct.Struct(">q")  # int64 hash


def pack_superquadric(
    sq: Superquadric,
    *,
    class_idx: int = 0,
    parent_hash: Optional[int] = None,
    properties: Optional[Mapping[Union[PropertyId, int, str], float]] = None,
    spectrum: Optional[SpectrumPacket] = None,
    t_resolution_m: float = DEFAULT_T_RESOLUTION_M,
) -> bytes:
    """Pack a posed :class:`Superquadric` to ~32 bytes plus optional trailers.

    Parameters
    ----------
    sq
        Source primitive.
    class_idx
        Optional semantic-class index in ``[0, 255]``.  ``0`` is the
        sentinel "unspecified" — caller must agree on the lookup table
        out-of-band.  See ``kernelcal.distinction_game.taxonomy`` for
        candidate enumerations.
    parent_hash
        Optional 64-bit signed integer linking this primitive to a
        parent (e.g. ``hash(parent_sq.id) & 0x7FFFFFFFFFFFFFFF``).
        When provided, ``flags & FLAG_HAS_PARENT`` is set and 8 trailer
        bytes are appended.
    properties
        Optional ``{PropertyId | int | str: float}`` map of derived
        material/health properties.  When non-empty, ``flags &
        FLAG_HAS_PROPERTIES`` is set and a variable-length property
        trailer is appended (see :mod:`.properties`).
    spectrum
        Optional :class:`SpectrumPacket` (96-byte UV-VIS-NIR sidecar).
        When provided, ``flags & FLAG_HAS_SPECTRUM`` is set and 96
        trailing bytes are appended.
    t_resolution_m
        Translation quantization step.  Default 1 mm.
    """
    if class_idx < 0 or class_idx > 255:
        raise ValueError(f"class_idx must fit in uint8; got {class_idx}.")

    flags = 0
    if parent_hash is not None:
        flags |= FLAG_HAS_PARENT

    has_props = bool(properties)
    if has_props:
        flags |= FLAG_HAS_PROPERTIES

    has_spectrum = spectrum is not None
    if has_spectrum:
        flags |= FLAG_HAS_SPECTRUM

    header = _HEADER_STRUCT.pack(_VERSION, flags, int(class_idx), 0)
    log_a = _LOGA_STRUCT.pack(*(_quant_log_a(a) for a in sq.scale))
    eps = _EPS_STRUCT.pack(_quant_eps(sq.epsilon[0]), _quant_eps(sq.epsilon[1]))

    q = _quat_from_so3(sq.R)
    quat_codes = tuple(int(np.clip(round(c * _QUAT_INT16_SCALE), -_QUAT_INT16_SCALE, _QUAT_INT16_SCALE)) for c in q)
    quat = _QUAT_STRUCT.pack(*quat_codes)

    t_codes = _quant_translation(sq.t, t_resolution_m)
    t_blob = _T_STRUCT.pack(int(t_codes[0]), int(t_codes[1]), int(t_codes[2]))

    body = header + log_a + eps + quat + t_blob
    assert len(body) == PACKED_BYTES, f"packed body is {len(body)} bytes, expected {PACKED_BYTES}"

    if parent_hash is not None:
        # Saturate to int64 range and pack.
        ph = int(parent_hash)
        if ph < -(2 ** 63) or ph > (2 ** 63 - 1):
            ph = ph & ((1 << 63) - 1)
        body = body + _PARENT_STRUCT.pack(ph)

    if has_props:
        body = body + encode_property_trailer(properties)

    if has_spectrum:
        spec_bytes = spectrum.to_bytes()
        if len(spec_bytes) != PACKED_SPECTRUM_BYTES:
            raise ValueError(
                f"pack_superquadric: spectrum packet is {len(spec_bytes)} bytes; "
                f"expected {PACKED_SPECTRUM_BYTES}."
            )
        body = body + spec_bytes

    return body


def unpack_superquadric(
    data: bytes,
    *,
    t_resolution_m: float = DEFAULT_T_RESOLUTION_M,
) -> Tuple[Superquadric, dict]:
    """Inverse of :func:`pack_superquadric`.

    Returns ``(sq, meta)`` where ``meta`` carries::

        {
            "version":       int,
            "flags":         int,
            "class_idx":     int,
            "parent_hash":   Optional[int],
            "properties":    Optional[Dict[PropertyId, float]],
            "spectrum":      Optional[SpectrumPacket],
            "bytes_consumed": int,    # total bytes read incl. trailers
        }

    The recovered SQ has a fresh auto-generated ``id`` and empty
    ``attributes``; the caller is responsible for repopulating semantic
    metadata from the surrounding envelope.
    """
    if len(data) < PACKED_BYTES:
        raise ValueError(
            f"unpack_superquadric: need >= {PACKED_BYTES} bytes, got {len(data)}."
        )
    version, flags, class_idx, _reserved = _HEADER_STRUCT.unpack_from(data, 0)
    if version != _VERSION:
        raise ValueError(f"Unsupported codec version: {version}")
    log_a = _LOGA_STRUCT.unpack_from(data, 4)
    eps_codes = _EPS_STRUCT.unpack_from(data, 10)
    quat_codes = _QUAT_STRUCT.unpack_from(data, 12)
    t_codes = _T_STRUCT.unpack_from(data, 20)

    scale = np.array([_dequant_log_a(c) for c in log_a], dtype=float)
    epsilon = np.array([_dequant_eps(eps_codes[0]), _dequant_eps(eps_codes[1])], dtype=float)
    quat = np.array([c / _QUAT_INT16_SCALE for c in quat_codes], dtype=float)
    R = _so3_from_quat(quat)
    t = _dequant_translation(np.asarray(t_codes, dtype=np.int32), t_resolution_m)

    sq = Superquadric(scale=scale, epsilon=epsilon, R=R, t=t)

    cursor = PACKED_BYTES

    parent_hash: Optional[int] = None
    if flags & FLAG_HAS_PARENT:
        if len(data) < cursor + PACKED_PARENT_BYTES:
            raise ValueError(
                f"unpack_superquadric: parent flag set but only {len(data)} bytes "
                f"available; need {cursor + PACKED_PARENT_BYTES}."
            )
        (parent_hash,) = _PARENT_STRUCT.unpack_from(data, cursor)
        cursor += PACKED_PARENT_BYTES

    properties: Optional[dict] = None
    if flags & FLAG_HAS_PROPERTIES:
        decoded, consumed = decode_property_trailer(data, offset=cursor)
        properties = decoded
        cursor += consumed

    spectrum_packet: Optional[SpectrumPacket] = None
    if flags & FLAG_HAS_SPECTRUM:
        if len(data) < cursor + PACKED_SPECTRUM_BYTES:
            raise ValueError(
                f"unpack_superquadric: spectrum flag set but only "
                f"{len(data)} bytes available; need {cursor + PACKED_SPECTRUM_BYTES}."
            )
        spectrum_packet = SpectrumPacket.from_bytes(
            bytes(data[cursor : cursor + PACKED_SPECTRUM_BYTES])
        )
        cursor += PACKED_SPECTRUM_BYTES

    meta = {
        "version": int(version),
        "flags": int(flags),
        "class_idx": int(class_idx),
        "parent_hash": parent_hash,
        "properties": properties,
        "spectrum": spectrum_packet,
        "bytes_consumed": int(cursor),
    }
    return sq, meta


def packed_size(
    *,
    has_parent: bool = False,
    properties: Optional[Mapping[Union[PropertyId, int, str], float]] = None,
    has_spectrum: bool = False,
) -> int:
    """Return the total wire size for a SQ with the given trailers."""
    size = PACKED_BYTES
    if has_parent:
        size += PACKED_PARENT_BYTES
    if properties:
        size += property_trailer_size(properties)
    if has_spectrum:
        size += PACKED_SPECTRUM_BYTES
    return size


__all__ = [
    "PACKED_BYTES",
    "PACKED_PARENT_BYTES",
    "PACKED_SPECTRUM_BYTES",
    "FLAG_HAS_PARENT",
    "FLAG_HAS_PROPERTIES",
    "FLAG_HAS_SPECTRUM",
    "DEFAULT_T_RESOLUTION_M",
    "MAX_PROPERTIES_PER_TRAILER",
    "pack_superquadric",
    "unpack_superquadric",
    "packed_size",
]

"""
adaptq/reference/packed_kv.py
============================
Issue #230: Reference packed KV cache implementation.

This is the REFERENCE implementation — intentionally simple and readable.
Its purpose is correctness, research experiments, debugging, and
contributor onboarding.

The C++ implementation (KVFlatBuffer / AttentionHead) is the optimized
production path. This Python code should produce identical results.

Architecture
------------
The cache stores, per token, per (layer, head):
  - packed quantized values (K and V separately)
  - float32 scale (one per token per head for K, one for V)

Memory breakdown is explicit (see PackedKVCache.memory_bytes()).

Quantization pipeline (matching quantizer.cpp, issue #229 corrected):
  1. L2-normalise: x_hat = x / ||x||
  2. Zero-pad to next_pow2(head_dim)
  3. Apply Rademacher sign flip D (deterministic per-head seed)
  4. FWHT: (1/sqrt(p)) * H * x_hat_padded
  5. Scale by sqrt(p) → values ≈ N(0,1)
  6. Soft-clip to ±3σ
  7. Codebook lookup in N(0,1) domain → packed bits
  8. Store: packed_bytes + float32_scale

Scale = ||x|| (L2 norm of original input, NOT norm * clip)

Dequantization:
  unpack → cb[idx] / sqrt(p) → IFWHT → multiply by scale

Usage
-----
    cache = PackedKVCache(
        bits=4,
        head_dim=128,
        n_heads=8,
        n_kv_heads=8,
        n_layers=12,
        max_seq_len=2048,
    )
    cache.append(layer=0, k=k_tensor, v=v_tensor)  # k,v: [n_kv_heads, head_dim]
    k_recon, v_recon = cache.get(layer=0)           # returns [seq_len, n_kv_heads, head_dim]
    print(cache.memory_bytes())

Note on GPU
-----------
This reference implementation is CPU-only. GPU support is not yet implemented.
All tensors are processed on CPU (converted if necessary).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Lloyd-Max N(0,1) codebooks (issue #229 corrected symmetric values)
# Source: Max (1960) / Jayant & Noll (1984) Table 4.2
# ---------------------------------------------------------------------------

_CB2 = np.array([-1.5104, -0.4528, 0.4528, 1.5104], dtype=np.float32)

_CB3 = np.array([
    -2.1529, -1.3439, -0.7560, -0.2451,
     0.2451,  0.7560,  1.3439,  2.1529
], dtype=np.float32)

_CB4 = np.array([
    # Symmetric: CB4[i] == -CB4[15-i]
    -2.4008, -1.8944, -1.5105, -1.2016,
    -0.9138, -0.6402, -0.3795, -0.1258,
     0.1258,  0.3795,  0.6402,  0.9138,
     1.2016,  1.5105,  1.8944,  2.4008
], dtype=np.float32)

_CODEBOOKS = {2: _CB2, 3: _CB3, 4: _CB4}


def get_codebook(bits: int) -> np.ndarray:
    """Return the Lloyd-Max codebook for the given bit width."""
    if bits not in _CODEBOOKS:
        raise ValueError(f"Unsupported bit width {bits}. Must be 2, 3, or 4.")
    return _CODEBOOKS[bits]


# ---------------------------------------------------------------------------
# FWHT utilities
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def _make_rademacher(d: int, seed: int) -> np.ndarray:
    """Generate a Rademacher ±1 sign vector deterministically from seed.
    Must match the XOR-shift generator in fwht.cpp:gen_rademacher().
    """
    D = np.zeros(d, dtype=np.float32)
    s = seed if seed != 0 else 0xDEADBEEFCAFE1234
    # Match the C++ XOR-shift: s ^= s<<13; s ^= s>>7; s ^= s<<17
    # Use 64-bit arithmetic
    s = int(s) & 0xFFFFFFFFFFFFFFFF
    for i in range(d):
        s ^= (s << 13) & 0xFFFFFFFFFFFFFFFF
        s ^= (s >> 7)  & 0xFFFFFFFFFFFFFFFF
        s ^= (s << 17) & 0xFFFFFFFFFFFFFFFF
        D[i] = 1.0 if (s & 1) else -1.0
    return D


def _fwht(x: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform (in-place), returns (1/sqrt(n))*H*x.
    Input must have length = power of 2.
    """
    n = len(x)
    assert n & (n - 1) == 0, "FWHT requires power-of-2 length"
    x = x.copy()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a, b = x[j], x[j + h]
                x[j], x[j + h] = a + b, a - b
        h *= 2
    return x / math.sqrt(n)


def fwht_forward(x_padded: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Apply D then FWHT: output = (1/sqrt(p)) * H * D * x."""
    return _fwht(x_padded * D)


def fwht_inverse(y: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Inverse of fwht_forward: recover D * x from y.
    Inverse of (1/sqrt(p))*H*D*x is D * (1/sqrt(p)) * H * y.
    """
    return D * _fwht(y)


# ---------------------------------------------------------------------------
# Bit packing / unpacking
# ---------------------------------------------------------------------------

def pack_indices(indices: np.ndarray, bits: int) -> bytes:
    """Pack integer indices (values in [0, 2^bits-1]) into a byte array.
    Matches quantizer.cpp:pack_indices().
    """
    n = len(indices)
    if bits == 4:
        assert n % 2 == 0
        out = bytearray(n // 2)
        for i in range(n // 2):
            out[i] = ((indices[2 * i] & 0xF) << 4) | (indices[2 * i + 1] & 0xF)
        return bytes(out)
    elif bits == 2:
        assert n % 4 == 0
        out = bytearray(n // 4)
        for i in range(n // 4):
            out[i] = (
                ((indices[4 * i]     & 3) << 6) |
                ((indices[4 * i + 1] & 3) << 4) |
                ((indices[4 * i + 2] & 3) << 2) |
                (indices[4 * i + 3]  & 3)
            )
        return bytes(out)
    elif bits == 3:
        assert n % 8 == 0
        out = bytearray(n * 3 // 8)
        g = n // 8
        for i in range(g):
            s = indices[i * 8: i * 8 + 8]
            p = out[i * 3: i * 3 + 3]
            p[0] = ((s[0] << 5) | (s[1] << 2) | (s[2] >> 1)) & 0xFF
            p[1] = ((s[2] << 7) | (s[3] << 4) | (s[4] << 1) | (s[5] >> 2)) & 0xFF
            p[2] = ((s[5] << 6) | (s[6] << 3) | s[7]) & 0xFF
            out[i * 3], out[i * 3 + 1], out[i * 3 + 2] = p[0], p[1], p[2]
        return bytes(out)
    else:
        raise ValueError(f"Unsupported bits={bits}")


def unpack_indices(packed: bytes, n: int, bits: int) -> np.ndarray:
    """Unpack byte array back to integer index array."""
    out = np.zeros(n, dtype=np.int32)
    if bits == 4:
        for i in range(n // 2):
            out[2 * i]     = (packed[i] >> 4) & 0xF
            out[2 * i + 1] = packed[i] & 0xF
    elif bits == 2:
        for i in range(n // 4):
            out[4 * i]     = (packed[i] >> 6) & 3
            out[4 * i + 1] = (packed[i] >> 4) & 3
            out[4 * i + 2] = (packed[i] >> 2) & 3
            out[4 * i + 3] = packed[i] & 3
    elif bits == 3:
        g = n // 8
        for i in range(g):
            p = packed[i * 3: i * 3 + 3]
            j = i * 8
            out[j]     = (p[0] >> 5) & 7
            out[j + 1] = (p[0] >> 2) & 7
            out[j + 2] = ((p[0] & 3) << 1) | (p[1] >> 7)
            out[j + 3] = (p[1] >> 4) & 7
            out[j + 4] = (p[1] >> 1) & 7
            out[j + 5] = ((p[1] & 1) << 2) | (p[2] >> 6)
            out[j + 6] = (p[2] >> 3) & 7
            out[j + 7] = p[2] & 7
    else:
        raise ValueError(f"Unsupported bits={bits}")
    return out


# ---------------------------------------------------------------------------
# Core quantize / dequantize
# ---------------------------------------------------------------------------

def quantize_vector(x: np.ndarray, D: np.ndarray, bits: int, skip_fwht: bool = False, skip_rademacher: bool = False) -> Tuple[bytes, float]:
    """Quantize a float vector x using the corrected pipeline (issue #229)."""
    dim = len(x)
    padded = len(D)
    cb = get_codebook(bits)

    # Step 1: L2-normalise
    norm = float(np.sqrt(np.dot(x, x) + 1e-12))
    x_hat = x / norm

    # Step 2: zero-pad
    x_padded = np.zeros(padded, dtype=np.float32)
    x_padded[:dim] = x_hat

    # Rademacher only?
    if skip_rademacher:
        D_eff = np.ones_like(D)
    else:
        D_eff = D

    if skip_fwht:
        y = x_padded * D_eff
        y = y * math.sqrt(padded)
    else:
        # Step 3: FWHT + scale by √p
        y = fwht_forward(x_padded, D_eff)   # values are O(1/√p)
        y = y * math.sqrt(padded)       # now ≈ N(0,1)

    # Step 4: soft-clip to ±3σ
    sigma = float(np.std(y)) + 1e-12
    clip = 3.0 * sigma
    y = np.clip(y, -clip, clip)
    # NOTE: do NOT divide by clip — codebook is N(0,1) domain

    # Step 5: codebook lookup
    indices = np.searchsorted(cb, y - 1e-9).astype(np.int32)
    # Nearest centroid (handle boundary: ensure index in [0, len(cb)-1])
    indices = np.clip(indices, 0, len(cb) - 1)
    # Adjust for midpoint tie-breaking (match branchless search in C++)
    for i in range(len(y)):
        c = indices[i]
        if c > 0:
            d_lo = abs(y[i] - cb[c - 1])
            d_hi = abs(y[i] - cb[c])
            if d_lo < d_hi:
                indices[i] = c - 1

    packed = pack_indices(indices, bits)
    return packed, norm


def dequantize_vector(packed: bytes, scale: float, D: np.ndarray, bits: int, dim: int, skip_fwht: bool = False, skip_rademacher: bool = False) -> np.ndarray:
    """Reconstruct float vector from packed representation."""
    padded = len(D)
    cb = get_codebook(bits)

    # Unpack
    indices = unpack_indices(packed, padded, bits)

    if skip_rademacher:
        D_eff = np.ones_like(D)
    else:
        D_eff = D

    # cb[idx[i]] / √p → IFWHT → × scale
    y = cb[indices] / math.sqrt(padded)  # shape (padded,)
    
    if skip_fwht:
        x_hat_padded = y * D_eff
    else:
        x_hat_padded = fwht_inverse(y, D_eff)    # shape (padded,)
        
    x_hat = x_hat_padded[:dim]           # trim padding

    return (x_hat * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-head state
# ---------------------------------------------------------------------------

@dataclass
class HeadCache:
    """Packed quantized KV storage for a single (layer, head)."""
    bits: int
    head_dim: int
    padded_dim: int
    D: np.ndarray        # Rademacher sign vector, shape (padded_dim,)
    skip_fwht: bool = False
    skip_rademacher: bool = False

    # Per-token storage (lists, appended as generation proceeds)
    k_packed: List[bytes] = field(default_factory=list)
    v_packed: List[bytes] = field(default_factory=list)
    k_scale:  List[float] = field(default_factory=list)
    v_scale:  List[float] = field(default_factory=list)

    @property
    def seq_len(self) -> int:
        return len(self.k_packed)

    @property
    def packed_bytes_per_token(self) -> int:
        return (self.padded_dim * self.bits + 7) // 8

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        """Quantize and append one token's K and V."""
        k_bytes, k_s = quantize_vector(k.astype(np.float32), self.D, self.bits, self.skip_fwht, self.skip_rademacher)
        v_bytes, v_s = quantize_vector(v.astype(np.float32), self.D, self.bits, self.skip_fwht, self.skip_rademacher)
        self.k_packed.append(k_bytes)
        self.v_packed.append(v_bytes)
        self.k_scale.append(k_s)
        self.v_scale.append(v_s)

    def get_k(self) -> np.ndarray:
        """Dequantize all K vectors. Returns shape (seq_len, head_dim)."""
        out = np.zeros((self.seq_len, self.head_dim), dtype=np.float32)
        for t in range(self.seq_len):
            out[t] = dequantize_vector(
                self.k_packed[t], self.k_scale[t], self.D, self.bits, self.head_dim, self.skip_fwht, self.skip_rademacher
            )
        return out

    def get_v(self) -> np.ndarray:
        """Dequantize all V vectors. Returns shape (seq_len, head_dim)."""
        out = np.zeros((self.seq_len, self.head_dim), dtype=np.float32)
        for t in range(self.seq_len):
            out[t] = dequantize_vector(
                self.v_packed[t], self.v_scale[t], self.D, self.bits, self.head_dim, self.skip_fwht, self.skip_rademacher
            )
        return out

    def reset(self) -> None:
        self.k_packed.clear()
        self.v_packed.clear()
        self.k_scale.clear()
        self.v_scale.clear()

    def memory_bytes(self) -> dict:
        """Return detailed memory breakdown in bytes."""
        n = self.seq_len
        payload = n * 2 * self.packed_bytes_per_token  # K + V packed bits
        scale   = n * 2 * 4  # float32 scale per token per head, K + V
        # Rademacher sign vector: int8 in C++, float32 here for simplicity
        meta    = self.padded_dim * 4  # D vector (float32)
        return {
            "payload_bytes": payload,
            "scale_bytes": scale,
            "metadata_bytes": meta,
            "padding_bytes": (self.padded_dim - self.head_dim) * self.bits // 8 * n * 2,
            "total_bytes": payload + scale + meta,
        }


# ---------------------------------------------------------------------------
# PackedKVCache — the main public API
# ---------------------------------------------------------------------------

class PackedKVCache:
    """Reference packed KV cache for AdapTQ.

    This is the authoritative compressed-KV storage. No dense FP16/FP32 copy
    is maintained — the packed representation IS the cache.

    Memory is explicitly tracked and reported. See memory_bytes() for the
    full breakdown including payload, scales, metadata, and padding waste.

    Usage:
        cache = PackedKVCache(bits=4, head_dim=128, n_kv_heads=8, n_layers=12)
        cache.append(layer=0, k=k, v=v)   # k,v: [n_kv_heads, head_dim]
        k_r, v_r = cache.get(layer=0)     # [seq_len, n_kv_heads, head_dim]
        print(cache.memory_bytes())

    Args:
        bits:       quantization bits (2, 3, or 4)
        head_dim:   attention head dimension
        n_kv_heads: number of KV heads (may differ from Q heads in GQA/MQA)
        n_layers:   number of transformer layers
        base_seed:  base seed for Rademacher vectors (unique per layer+head)
        dense_ref:  if True, also maintain a dense FP32 reference for comparison
                    (only useful for debugging; doubles memory)
    """

    def __init__(
        self,
        bits: int,
        head_dim: int,
        n_kv_heads: int,
        n_layers: int,
        base_seed: int = 0xDEADBEEFCAFE,
        dense_ref: bool = False,
        skip_fwht: bool = False,
        skip_rademacher: bool = False,
    ):
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4; got {bits}")
        if head_dim <= 0 or n_kv_heads <= 0 or n_layers <= 0:
            raise ValueError("head_dim, n_kv_heads, n_layers must be positive")

        self.bits       = bits
        self.head_dim   = head_dim
        self.padded_dim = _next_pow2(head_dim)
        self.n_kv_heads = n_kv_heads
        self.n_layers   = n_layers
        self.dense_ref  = dense_ref
        self.skip_fwht  = skip_fwht
        self.skip_rademacher = skip_rademacher

        # One HeadCache per (layer, head)
        self._caches: Dict[Tuple[int, int], HeadCache] = {}
        for layer in range(n_layers):
            for head in range(n_kv_heads):
                seed = (base_seed ^ (layer * 1000 + head) * 0xDEADBEEF) & 0xFFFFFFFFFFFFFFFF
                D = _make_rademacher(self.padded_dim, seed)
                self._caches[(layer, head)] = HeadCache(
                    bits=bits,
                    head_dim=head_dim,
                    padded_dim=self.padded_dim,
                    D=D,
                    skip_fwht=self.skip_fwht,
                    skip_rademacher=self.skip_rademacher
                )

        # Optional dense reference (for correctness comparison only)
        self._dense_k: Dict[int, List[np.ndarray]] = {l: [] for l in range(n_layers)} if dense_ref else {}
        self._dense_v: Dict[int, List[np.ndarray]] = {l: [] for l in range(n_layers)} if dense_ref else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Quantize and append K/V for one token.

        Args:
            layer: layer index [0, n_layers)
            k:     float array of shape (n_kv_heads, head_dim)
            v:     float array of shape (n_kv_heads, head_dim)
        """
        self._check_layer(layer)
        if k.shape != (self.n_kv_heads, self.head_dim):
            raise ValueError(
                f"k shape {k.shape} != expected ({self.n_kv_heads}, {self.head_dim})"
            )
        if v.shape != (self.n_kv_heads, self.head_dim):
            raise ValueError(
                f"v shape {v.shape} != expected ({self.n_kv_heads}, {self.head_dim})"
            )

        for h in range(self.n_kv_heads):
            self._caches[(layer, h)].append(k[h], v[h])

        if self.dense_ref:
            self._dense_k[layer].append(k.copy())
            self._dense_v[layer].append(v.copy())

    def get(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """Dequantize and return all K/V for a layer.

        Returns:
            (k, v) each of shape (seq_len, n_kv_heads, head_dim)
        """
        self._check_layer(layer)
        seq_len = self._caches[(layer, 0)].seq_len
        k_out = np.zeros((seq_len, self.n_kv_heads, self.head_dim), dtype=np.float32)
        v_out = np.zeros((seq_len, self.n_kv_heads, self.head_dim), dtype=np.float32)
        for h in range(self.n_kv_heads):
            cache = self._caches[(layer, h)]
            k_out[:, h, :] = cache.get_k()
            v_out[:, h, :] = cache.get_v()
        return k_out, v_out

    def get_dense_ref(self, layer: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return the dense FP32 reference (only available if dense_ref=True)."""
        if not self.dense_ref:
            return None, None
        k_list = self._dense_k[layer]
        v_list = self._dense_v[layer]
        if not k_list:
            return None, None
        return np.stack(k_list, axis=0), np.stack(v_list, axis=0)

    def reset(self, layer: Optional[int] = None) -> None:
        """Reset cache for one layer or all layers."""
        if layer is not None:
            self._check_layer(layer)
            for h in range(self.n_kv_heads):
                self._caches[(layer, h)].reset()
            if self.dense_ref:
                self._dense_k[layer].clear()
                self._dense_v[layer].clear()
        else:
            for c in self._caches.values():
                c.reset()
            if self.dense_ref:
                for l in range(self.n_layers):
                    self._dense_k[l].clear()
                    self._dense_v[l].clear()

    def sequence_length(self, layer: int = 0) -> int:
        """Number of tokens currently cached."""
        return self._caches[(layer, 0)].seq_len

    def memory_bytes(self) -> dict:
        """Return detailed memory breakdown across all layers and heads.

        Returns a dict with:
            payload_bytes:  packed quantized K+V data
            scale_bytes:    float32 K+V scales (n_tokens × n_kv_heads × 2 layers × 4 bytes)
            metadata_bytes: Rademacher sign vectors and config
            padding_bytes:  wasted bits from next_pow2 padding (if head_dim != padded_dim)
            total_bytes:    payload + scale + metadata
            fp16_equivalent_bytes: what the same data would cost in FP16
            compression_ratio: fp16_equivalent / total
        """
        total_payload = 0
        total_scale = 0
        total_meta = 0
        total_padding = 0

        for c in self._caches.values():
            mb = c.memory_bytes()
            total_payload += mb["payload_bytes"]
            total_scale   += mb["scale_bytes"]
            total_meta    += mb["metadata_bytes"]
            total_padding += mb["padding_bytes"]

        # FP16 equivalent: n_tokens × n_kv_heads × head_dim × 2 (K+V) × 2 bytes (FP16) × n_layers
        n_tokens = self.sequence_length()
        fp16_equiv = (
            n_tokens * self.n_kv_heads * self.head_dim * 2 * 2 * self.n_layers
        )
        total = total_payload + total_scale + total_meta
        ratio = (fp16_equiv / total) if total > 0 else 0.0

        return {
            "payload_bytes":           total_payload,
            "scale_bytes":             total_scale,
            "metadata_bytes":          total_meta,
            "padding_bytes":           total_padding,
            "total_bytes":             total,
            "fp16_equivalent_bytes":   fp16_equiv,
            "fp32_equivalent_bytes":   fp16_equiv * 2,
            "compression_ratio_vs_fp16": ratio,
            "head_dim":                self.head_dim,
            "padded_dim":              self.padded_dim,
            "padding_overhead_pct":    (self.padded_dim - self.head_dim) / self.head_dim * 100,
            "bits":                    self.bits,
            "n_tokens":                n_tokens,
            "n_kv_heads":              self.n_kv_heads,
            "n_layers":                self.n_layers,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_layer(self, layer: int) -> None:
        if layer < 0 or layer >= self.n_layers:
            raise ValueError(f"layer {layer} out of range [0, {self.n_layers})")

    def __repr__(self) -> str:
        mb = self.memory_bytes()
        return (
            f"PackedKVCache(bits={self.bits}, head_dim={self.head_dim}, "
            f"padded={self.padded_dim}, n_kv_heads={self.n_kv_heads}, "
            f"n_layers={self.n_layers}, seq_len={self.sequence_length()}, "
            f"total={mb['total_bytes']/1024:.1f}KB, "
            f"ratio={mb['compression_ratio_vs_fp16']:.2f}x vs FP16)"
        )

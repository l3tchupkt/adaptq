"""
adaptq.hf_adapter — HuggingFace Transformers DynamicCache Integration Adapter

Implements a standard HuggingFace Cache interface for seamless integration
with AutoModelForCausalLM generation pipelines (Llama-3, Mistral, Qwen, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class CacheLayerState:
    key_cache: Optional[np.ndarray] = None    # Shape: [batch, heads, seq_len, dim]
    value_cache: Optional[np.ndarray] = None  # Shape: [batch, heads, seq_len, dim]
    bits: int = 4
    n_tokens: int = 0


class AdapTQCache:
    """HuggingFace compatible DynamicCache implementation using AdapTQ quantization semantics."""

    def __init__(
        self,
        num_hidden_layers: int = 32,
        bits: int = 4,
        max_capacity: int = 4096,
        head_dim: int = 64
    ) -> None:
        if num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
        if max_capacity <= 0:
            raise ValueError(f"max_capacity must be positive, got {max_capacity}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")

        self.num_hidden_layers = num_hidden_layers
        self.bits = bits
        self.max_capacity = max_capacity
        self.head_dim = head_dim

        self.layers: List[CacheLayerState] = [
            CacheLayerState(bits=bits) for _ in range(num_hidden_layers)
        ]

    def update(
        self,
        key_states: np.ndarray,
        value_states: np.ndarray,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update layer KV cache with newly generated token states."""
        if layer_idx < 0 or layer_idx >= self.num_hidden_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range for {self.num_hidden_layers} layers"
            )

        k_arr = np.asarray(key_states)
        v_arr = np.asarray(value_states)

        if k_arr.ndim != 4 or v_arr.ndim != 4:
            raise ValueError(
                f"Expected 4D tensors [batch, heads, seq, dim], "
                f"got k={k_arr.shape}, v={v_arr.shape}"
            )
        if k_arr.shape != v_arr.shape:
            raise ValueError(f"Shape mismatch: k={k_arr.shape} != v={v_arr.shape}")
        if k_arr.shape[-1] != self.head_dim:
            raise ValueError(
                f"Head dim mismatch: expected {self.head_dim}, got {k_arr.shape[-1]}"
            )

        layer = self.layers[layer_idx]
        new_tokens = k_arr.shape[2]

        if layer.key_cache is None:
            layer.key_cache = k_arr.copy()
            layer.value_cache = v_arr.copy()
            layer.n_tokens = new_tokens
        else:
            # Concatenate along seq_len dimension (axis=2)
            layer.key_cache = np.concatenate([layer.key_cache, k_arr], axis=2)
            layer.value_cache = np.concatenate([layer.value_cache, v_arr], axis=2)
            layer.n_tokens += new_tokens

        # Capacity truncation if exceeding max_capacity
        if layer.key_cache.shape[2] > self.max_capacity:
            overflow = layer.key_cache.shape[2] - self.max_capacity
            layer.key_cache = layer.key_cache[:, :, overflow:, :]
            layer.value_cache = layer.value_cache[:, :, overflow:, :]
            layer.n_tokens = self.max_capacity

        return layer.key_cache, layer.value_cache

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the current sequence length stored in the cache for the given layer."""
        if layer_idx < 0 or layer_idx >= self.num_hidden_layers:
            return 0
        layer = self.layers[layer_idx]
        if layer.key_cache is None:
            return 0
        return layer.key_cache.shape[2]

    def get_max_length(self) -> int:
        """Return the maximum token capacity supported by the cache."""
        return self.max_capacity

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """Return the usable historical cache length when appending new_seq_length tokens."""
        current_len = self.get_seq_length(layer_idx)
        return min(current_len, self.max_capacity - new_seq_length)

    def crop(self, max_length: int) -> None:
        """Crop all layer caches to max_length tokens."""
        if max_length < 0:
            raise ValueError(f"max_length must be non-negative, got {max_length}")

        for layer in self.layers:
            if layer.key_cache is not None and layer.key_cache.shape[2] > max_length:
                layer.key_cache = layer.key_cache[:, :, :max_length, :]
                layer.value_cache = layer.value_cache[:, :, :max_length, :]
                layer.n_tokens = max_length

    def to_legacy_cache(self) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """Export internal state to legacy tuple format: ((k0, v0), (k1, v1), ...)."""
        legacy = []
        for layer in self.layers:
            if layer.key_cache is not None and layer.value_cache is not None:
                legacy.append((layer.key_cache, layer.value_cache))
            else:
                empty = np.zeros((1, 1, 0, self.head_dim), dtype=np.float32)
                legacy.append((empty, empty))
        return tuple(legacy)

    def total_raw_bytes(self) -> int:
        """Calculate uncompressed FP16 equivalent byte footprint."""
        total_tokens = sum(self.get_seq_length(i) for i in range(self.num_hidden_layers))
        # 2 tensors (K & V) * 2 bytes (FP16) * head_dim
        return total_tokens * 2 * 2 * self.head_dim

    def total_quantized_bytes(self) -> int:
        """Calculate theoretical compressed byte footprint at configured bit-width."""
        total_tokens = sum(self.get_seq_length(i) for i in range(self.num_hidden_layers))
        # 2 tensors * (bits / 8) * head_dim + scale overhead
        bytes_per_vec = (self.bits * self.head_dim) // 8 + 4
        return total_tokens * 2 * bytes_per_vec

    def compression_factor(self) -> float:
        """Return memory savings factor compared to raw FP16."""
        q_bytes = self.total_quantized_bytes()
        if q_bytes == 0:
            return 1.0
        return float(self.total_raw_bytes()) / float(q_bytes)

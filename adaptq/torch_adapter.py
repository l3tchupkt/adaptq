try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None    # type: ignore
    nn = None       # type: ignore
    _TORCH_AVAILABLE = False

try:
    from .core import Engine, _ADAPTQ_PY_AVAILABLE
    _ENGINE_AVAILABLE = _ADAPTQ_PY_AVAILABLE
except ImportError:
    Engine = None   # type: ignore[assignment,misc]
    _ENGINE_AVAILABLE = False



if _TORCH_AVAILABLE and _ENGINE_AVAILABLE:
    class AdaptQAttention(nn.Module):
        """
        PyTorch wrapper for AdapTQ KV cache attention.

        Integrates with PyTorch's nn.Module API for drop-in use during
        autoregressive generation (one token at a time per sequence).

        Constraints:
            - CPU-only: the AdapTQ C++ backend is a CPU runtime.
              Tensors are moved to CPU before processing.
            - batch_size == 1 only: AdapTQ maintains a single KV cache
              per Engine instance. Multi-sequence batches require one Engine
              per sequence. Use a list of AdaptQAttention modules or call
              reset_cache() between sequences.
            - Input shape: (1, heads, dim) — one token per forward() call.
              For prompt prefill across multiple positions, call forward()
              once per token in sequence order.
        """

        def __init__(
            self,
            dim: int,
            heads: int,
            bits: int = 4,
            capacity: int = 4096,
            seed: int = 42,
        ):
            super().__init__()
            self.dim = dim
            self.heads = heads
            self.bits = bits
            self.engine = Engine(
                dim=dim, heads=heads, bits=bits, capacity=capacity, seed=seed
            )

        def forward(
            self, q: "torch.Tensor", k: "torch.Tensor", v: "torch.Tensor"
        ) -> "torch.Tensor":
            """
            Args:
                q: (1, heads, dim) — query for the current token
                k: (1, heads, dim) — key for the current token (appended to cache)
                v: (1, heads, dim) — value for the current token (appended to cache)

            Returns:
                (1, heads, dim) attention output tensor on the same device as q.
            """
            if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
                raise ValueError(
                    f"AdaptQAttention.forward() expects 3D tensors (batch, heads, dim). "
                    f"Got shapes q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
                )

            batch_size = q.size(0)
            if batch_size != 1:
                raise ValueError(
                    f"AdaptQAttention does not support batch_size > 1 (got {batch_size}). "
                    "AdapTQ maintains a single KV cache per Engine instance. "
                    "For multi-sequence batches, create one AdaptQAttention per sequence "
                    "and process each independently."
                )

            if q.size(1) != self.heads or q.size(2) != self.dim:
                raise ValueError(
                    f"Expected q shape (1, {self.heads}, {self.dim}), "
                    f"got {tuple(q.shape)}."
                )

            if k.shape != q.shape:
                raise ValueError(
                    f"Expected k shape {tuple(q.shape)}, got {tuple(k.shape)}."
                )

            if v.shape != q.shape:
                raise ValueError(
                    f"Expected v shape {tuple(q.shape)}, got {tuple(v.shape)}."
                )

            # Move to CPU float32 numpy — AdapTQ backend is pure CPU C++ (float32).
            k_np = k.detach().float().cpu().numpy()[0]  # (heads, dim)
            v_np = v.detach().float().cpu().numpy()[0]
            q_np = q.detach().float().cpu().numpy()[0]

            self.engine.append(k_np, v_np)
            out_np = self.engine.compute(q_np)  # (heads, dim)

            return torch.from_numpy(out_np).unsqueeze(0).to(device=q.device, dtype=q.dtype)  # (1, heads, dim)

        def reset_cache(self) -> None:
            """Clear the KV cache. Call between independent sequences."""
            self.engine.reset()

else:
    class AdaptQAttention:  # type: ignore[no-redef]
        """Stub: PyTorch not installed. pip install torch to use AdaptQAttention."""
        def __init__(self, *args, **kwargs):
            if not _TORCH_AVAILABLE:
                raise ImportError(
                    "AdaptQAttention requires PyTorch.\n"
                    "Install with: pip install torch\n"
                    "or: pip install 'adaptq[torch]'"
                )
            if not _ENGINE_AVAILABLE:
                raise ImportError(
                    "AdaptQAttention requires the AdapTQ C++ runtime engine.\n"
                    "Please compile the C++ shared library before using AdaptQAttention."
                )

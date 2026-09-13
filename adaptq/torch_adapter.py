try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("PyTorch is required to use the AdaptQ PyTorch wrapper.")


from .core import Engine


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
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
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

        # Move to CPU numpy — AdapTQ backend is pure CPU C++.
        k_np = k.detach().cpu().numpy()[0]  # (heads, dim)
        v_np = v.detach().cpu().numpy()[0]
        q_np = q.detach().cpu().numpy()[0]

        self.engine.append(k_np, v_np)
        out_np = self.engine.compute(q_np)  # (heads, dim)

        return torch.from_numpy(out_np).unsqueeze(0).to(q.device)  # (1, heads, dim)

    def reset_cache(self) -> None:
        """Clear the KV cache. Call between independent sequences."""
        self.engine.reset()

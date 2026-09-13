try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("PyTorch is required to use the AdaptQ PyTorch wrapper.")

import numpy as np

from .core import Engine


class AdaptQAttention(nn.Module):
    """
    PyTorch Wrapper for AdapTQ.
    Integrates directly with PyTorch's nn.Module API.
    """
    def __init__(self, dim: int, heads: int, bits: int = 4, capacity: int = 4096, seed: int = 42):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.bits = bits
        self.engine = Engine(dim=dim, heads=heads, bits=bits, capacity=capacity, seed=seed)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Expects inputs of shape (batch_size, heads, dim).
        Note: Currently AdapTQ handles a batch_size of 1 for append_kv internally,
        but we can loop over the batch size or just assert batch_size == 1 for generation step.
        """
        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, "Expected 3D tensors: (batch, heads, dim)"
        batch_size = q.size(0)
        
        # Move to CPU numpy for backend (AdapTQ backend is pure CPU C++)
        k_np = k.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        q_np = q.detach().cpu().numpy()
        
        out = np.zeros_like(q_np)
        
        # Currently, AdapTQ C++ context is single-instance per `Engine`.
        # For batch inference, context is strictly sequential in Generation step.
        for b in range(batch_size):
            self.engine.append(k_np[b], v_np[b])
            out[b] = self.engine.compute(q_np[b])
            
        # Returning tensor on the same device as query
        return torch.from_numpy(out).to(q.device)

    def reset_cache(self):
        self.engine.reset()

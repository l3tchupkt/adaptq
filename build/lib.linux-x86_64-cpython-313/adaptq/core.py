import numpy as np

try:
    import adaptq_py
except ImportError:
    raise ImportError("AdapTQ C++ extension not built. Run 'pip install .' to build from source.")

class Engine:
    """
    API for Adaptive Streaming Vector Quantization.
    Usage:
        engine = adaptq.Engine(dim=128, heads=4, bits=4)
        engine.append(k, v)
        out = engine.compute(q)
    """
    def __init__(self, dim: int, heads: int, bits: int = 4, 
                 capacity: int = 4096, seed: int = 42, 
                 v_mass: float = 0.95, hybrid_thresh: int = 512):
        self.dim = dim
        self.heads = heads
        self.bits = bits
        self.capacity = capacity
        
        # Internal state
        self._ctx = adaptq_py.MHAContext(
            n_heads=heads,
            head_dim=dim,
            bits=bits,
            capacity=capacity,
            seed=seed,
            v_mass=v_mass,
            hybrid_thresh=hybrid_thresh
        )
        self.pos = 0

    def append(self, k: np.ndarray, v: np.ndarray):
        """
        Append k, v for the current step.
        Supports inputs of shape (heads, dim).
        """
        if k.shape != (self.heads, self.dim) or v.shape != (self.heads, self.dim):
            raise ValueError(f"k and v must be shape ({self.heads}, {self.dim})")
            
        for h in range(self.heads):
            # Using astype to guarantee contiguous float32 alignment to C++ backend
            k_h = np.ascontiguousarray(k[h], dtype=np.float32)
            v_h = np.ascontiguousarray(v[h], dtype=np.float32)
            self._ctx.append(h, k_h, v_h, self.pos)
            
        self.pos += 1

    def compute(self, q: np.ndarray) -> np.ndarray:
        """
        Compute attention output for query q.
        Supports inputs of shape (heads, dim). Returns shape (heads, dim).
        """
        if q.shape != (self.heads, self.dim):
            raise ValueError(f"q must be shape ({self.heads}, {self.dim})")
            
        out = np.zeros((self.heads, self.dim), dtype=np.float32)
        for h in range(self.heads):
            q_h = np.ascontiguousarray(q[h], dtype=np.float32)
            out[h] = self._ctx.compute(h, q_h)
            
        return out

    def reset(self):
        """Clear the KV cache."""
        self._ctx.reset()
        self.pos = 0

    @property
    def kv_bytes(self) -> int:
        return self._ctx.kv_bytes()

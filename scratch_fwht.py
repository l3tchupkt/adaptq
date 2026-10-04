import numpy as np
import torch
import math
from adaptq.research.packed_kv import _fwht

def fwht_pt(x: torch.Tensor) -> torch.Tensor:
    n = x.size(-1)
    h = 1
    out = x.clone()
    while h < n:
        out = out.view(-1, n // (2 * h), 2, h)
        a = out[:, :, 0, :].clone()
        b = out[:, :, 1, :].clone()
        out[:, :, 0, :] = a + b
        out[:, :, 1, :] = a - b
    return out.view(x.shape) / math.sqrt(n)

x_np = np.random.randn(8).astype(np.float32)
y_np = _fwht(x_np)

x_pt = torch.from_numpy(x_np)
y_pt = fwht_pt(x_pt).numpy()

print("Numpy:", y_np)
print("Torch:", y_pt)
print("Max Diff:", np.max(np.abs(y_np - y_pt)))

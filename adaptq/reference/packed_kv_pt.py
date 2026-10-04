import math
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

# Lloyd-Max N(0,1) codebooks
_CB2 = [-1.5104, -0.4528, 0.4528, 1.5104]
_CB3 = [-2.1529, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1529]
_CB4 = [-2.4008, -1.8944, -1.5105, -1.2016, -0.9138, -0.6402, -0.3795, -0.1258,
         0.1258,  0.3795,  0.6402,  0.9138,  1.2016,  1.5105,  1.8944,  2.4008]

def get_codebook_pt(bits: int, device: torch.device) -> torch.Tensor:
    if bits == 2: return torch.tensor(_CB2, dtype=torch.float32, device=device)
    if bits == 3: return torch.tensor(_CB3, dtype=torch.float32, device=device)
    if bits == 4: return torch.tensor(_CB4, dtype=torch.float32, device=device)
    raise ValueError()

def _next_pow2(n: int) -> int:
    if n <= 1: return 1
    p = 1
    while p < n: p <<= 1
    return p

def _make_rademacher_pt(d: int, seed: int, device: torch.device) -> torch.Tensor:
    D = torch.zeros(d, dtype=torch.float32, device=device)
    s = seed if seed != 0 else 0xDEADBEEFCAFE1234
    s = int(s) & 0xFFFFFFFFFFFFFFFF
    for i in range(d):
        s ^= (s << 13) & 0xFFFFFFFFFFFFFFFF
        s ^= (s >> 7)  & 0xFFFFFFFFFFFFFFFF
        s ^= (s << 17) & 0xFFFFFFFFFFFFFFFF
        D[i] = 1.0 if (s & 1) else -1.0
    return D

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
        h *= 2
    return out.view(x.shape) / math.sqrt(n)

def quantize_vector_pt(x: torch.Tensor, D: torch.Tensor, bits: int, cb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # x shape: (batch, n_kv_heads, head_dim)
    norm = torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-12
    x_hat = x / norm
    
    padded = D.size(-1)
    if padded > x.size(-1):
        x_padded = torch.zeros(x.shape[:-1] + (padded,), dtype=x.dtype, device=x.device)
        x_padded[..., :x.size(-1)] = x_hat
    else:
        x_padded = x_hat
        
    y = fwht_pt(x_padded * D) * math.sqrt(padded)
    
    sigma = y.std(dim=-1, keepdim=True) + 1e-12
    clip = 3.0 * sigma
    y = torch.clamp(y, min=-clip, max=clip)
    
    # Nearest centroid broadcast
    y_expanded = y.unsqueeze(-1)
    dists = torch.abs(y_expanded - cb)
    indices = torch.argmin(dists, dim=-1).to(torch.uint8)
    
    return indices, norm.squeeze(-1)

def dequantize_vector_pt(indices: torch.Tensor, scale: torch.Tensor, D: torch.Tensor, cb: torch.Tensor, head_dim: int) -> torch.Tensor:
    y = cb[indices.long()] / math.sqrt(D.size(-1))
    x_hat_padded = fwht_pt(y) * D
    x_hat = x_hat_padded[..., :head_dim]
    return x_hat * scale.unsqueeze(-1)

class PackedKVCachePT:
    def __init__(self, bits: int, head_dim: int, n_kv_heads: int, n_layers: int, base_seed: int = 0xDEADBEEFCAFE, dense_ref: bool = False, device: torch.device = torch.device('cpu')):
        self.bits = bits
        self.head_dim = head_dim
        self.padded_dim = _next_pow2(head_dim)
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.device = device
        self.dense_ref = dense_ref
        
        self.cb = get_codebook_pt(bits, device)
        
        # Rademacher vectors: shape (n_layers, n_kv_heads, padded_dim)
        self.D = torch.zeros((n_layers, n_kv_heads, self.padded_dim), dtype=torch.float32, device=device)
        for l in range(n_layers):
            for h in range(n_kv_heads):
                seed = (base_seed ^ (l * 1000 + h) * 0xDEADBEEF) & 0xFFFFFFFFFFFFFFFF
                self.D[l, h] = _make_rademacher_pt(self.padded_dim, seed, device)
                
        # Cache storage
        self.k_indices: Dict[int, List[torch.Tensor]] = {l: [] for l in range(n_layers)}
        self.v_indices: Dict[int, List[torch.Tensor]] = {l: [] for l in range(n_layers)}
        self.k_scales: Dict[int, List[torch.Tensor]] = {l: [] for l in range(n_layers)}
        self.v_scales: Dict[int, List[torch.Tensor]] = {l: [] for l in range(n_layers)}
        
        if self.dense_ref:
            self._dense_k = {l: [] for l in range(n_layers)}
            self._dense_v = {l: [] for l in range(n_layers)}

    def append_batch(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        # k, v shape: (seq_len, n_kv_heads, head_dim)
        ki, ks = quantize_vector_pt(k, self.D[layer], self.bits, self.cb)
        vi, vs = quantize_vector_pt(v, self.D[layer], self.bits, self.cb)
        
        self.k_indices[layer].append(ki)
        self.v_indices[layer].append(vi)
        self.k_scales[layer].append(ks)
        self.v_scales[layer].append(vs)
        
        if self.dense_ref:
            self._dense_k[layer].extend(list(k.cpu().numpy()))
            self._dense_v[layer].extend(list(v.cpu().numpy()))

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        k_pt = torch.from_numpy(k).to(self.device).float().unsqueeze(0) # (1, n_kv_heads, head_dim)
        v_pt = torch.from_numpy(v).to(self.device).float().unsqueeze(0)
        
        ki, ks = quantize_vector_pt(k_pt, self.D[layer], self.bits, self.cb)
        vi, vs = quantize_vector_pt(v_pt, self.D[layer], self.bits, self.cb)
        
        self.k_indices[layer].append(ki)
        self.v_indices[layer].append(vi)
        self.k_scales[layer].append(ks)
        self.v_scales[layer].append(vs)
        
        if self.dense_ref:
            self._dense_k[layer].append(k.copy())
            self._dense_v[layer].append(v.copy())

    def get(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        ki = torch.cat(self.k_indices[layer], dim=0) # (seq_len, n_kv_heads, padded_dim)
        vi = torch.cat(self.v_indices[layer], dim=0)
        ks = torch.cat(self.k_scales[layer], dim=0)  # (seq_len, n_kv_heads)
        vs = torch.cat(self.v_scales[layer], dim=0)
        
        k_recon = dequantize_vector_pt(ki, ks, self.D[layer], self.cb, self.head_dim)
        v_recon = dequantize_vector_pt(vi, vs, self.D[layer], self.cb, self.head_dim)
        
        return k_recon.cpu().numpy(), v_recon.cpu().numpy()

    def get_dense_ref(self, layer: int):
        if not self.dense_ref: return None, None
        if not self._dense_k[layer]: return None, None
        return np.stack(self._dense_k[layer], axis=0), np.stack(self._dense_v[layer], axis=0)

    def sequence_length(self, layer: int = 0) -> int:
        if not self.k_indices[layer]: return 0
        return sum(t.size(0) for t in self.k_indices[layer])

    def reset(self, layer=None):
        if layer is not None:
            self.k_indices[layer].clear()
            self.v_indices[layer].clear()
            self.k_scales[layer].clear()
            self.v_scales[layer].clear()
            if self.dense_ref:
                self._dense_k[layer].clear()
                self._dense_v[layer].clear()
        else:
            for l in range(self.n_layers):
                self.reset(l)

    def memory_bytes(self) -> dict:
        n = self.sequence_length()
        packed_bytes_per_token = (self.padded_dim * self.bits + 7) // 8
        payload = n * 2 * packed_bytes_per_token * self.n_kv_heads * self.n_layers
        scale = n * 2 * 4 * self.n_kv_heads * self.n_layers
        meta = self.padded_dim * 4 * self.n_kv_heads * self.n_layers
        padding_bytes = (self.padded_dim - self.head_dim) * self.bits // 8 * n * 2 * self.n_kv_heads * self.n_layers
        total = payload + scale + meta
        fp16_equiv = n * self.n_kv_heads * self.head_dim * 2 * 2 * self.n_layers
        
        return {
            "payload_bytes": payload,
            "scale_bytes": scale,
            "metadata_bytes": meta,
            "padding_bytes": padding_bytes,
            "total_bytes": total,
            "fp16_equivalent_bytes": fp16_equiv,
            "fp32_equivalent_bytes": fp16_equiv * 2,
            "compression_ratio_vs_fp16": (fp16_equiv / total) if total > 0 else 0.0,
            "head_dim": self.head_dim,
            "padded_dim": self.padded_dim,
            "padding_overhead_pct": (self.padded_dim - self.head_dim) / self.head_dim * 100,
            "bits": self.bits,
            "n_tokens": n,
            "n_kv_heads": self.n_kv_heads,
            "n_layers": self.n_layers,
        }

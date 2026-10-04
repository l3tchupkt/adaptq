import math
import numpy as np
import pytest

from adaptq.reference.packed_kv import (
    PackedKVCache,
    fwht_forward,
    fwht_inverse,
    get_codebook,
    pack_indices,
    unpack_indices,
    _make_rademacher,
)

def test_fwht_roundtrip():
    dim = 128
    D = _make_rademacher(dim, 42)
    x = np.random.normal(0, 1, size=dim).astype(np.float32)
    
    y = fwht_forward(x, D)
    x_recon = fwht_inverse(y, D)
    
    assert np.allclose(x, x_recon, atol=1e-5), "FWHT inverse did not recover original"

def test_pack_unpack():
    n = 128
    for bits in (2, 3, 4):
        cb = get_codebook(bits)
        max_idx = len(cb) - 1
        indices = np.random.randint(0, max_idx + 1, size=n, dtype=np.int32)
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, n, bits)
        assert np.array_equal(indices, unpacked), f"pack/unpack failed for {bits}-bit"

def test_cache_init():
    cache = PackedKVCache(bits=4, head_dim=128, n_kv_heads=2, n_layers=2)
    assert cache.sequence_length() == 0
    assert cache.n_kv_heads == 2
    assert cache.n_layers == 2
    assert cache.bits == 4
    
    with pytest.raises(ValueError):
        PackedKVCache(bits=5, head_dim=128, n_kv_heads=2, n_layers=2)

def test_cache_append_get():
    head_dim = 64
    n_kv_heads = 4
    cache = PackedKVCache(bits=4, head_dim=head_dim, n_kv_heads=n_kv_heads, n_layers=2, dense_ref=True)
    
    # Append token 0
    k0 = np.random.normal(0, 1, size=(n_kv_heads, head_dim)).astype(np.float32)
    v0 = np.random.normal(0, 1, size=(n_kv_heads, head_dim)).astype(np.float32)
    cache.append(layer=0, k=k0, v=v0)
    
    assert cache.sequence_length(layer=0) == 1
    
    # Append token 1
    k1 = np.random.normal(0, 1, size=(n_kv_heads, head_dim)).astype(np.float32)
    v1 = np.random.normal(0, 1, size=(n_kv_heads, head_dim)).astype(np.float32)
    cache.append(layer=0, k=k1, v=v1)
    
    assert cache.sequence_length(layer=0) == 2
    
    k_out, v_out = cache.get(layer=0)
    assert k_out.shape == (2, n_kv_heads, head_dim)
    assert v_out.shape == (2, n_kv_heads, head_dim)
    
    dense_k, dense_v = cache.get_dense_ref(layer=0)
    assert dense_k.shape == (2, n_kv_heads, head_dim)
    assert dense_v.shape == (2, n_kv_heads, head_dim)
    
    # Check that reconstructed K/V has high cosine similarity with dense K/V
    for h in range(n_kv_heads):
        for t in range(2):
            orig_k = dense_k[t, h, :]
            recon_k = k_out[t, h, :]
            cos_sim_k = np.dot(orig_k, recon_k) / (np.linalg.norm(orig_k) * np.linalg.norm(recon_k))
            assert cos_sim_k > 0.8, f"K Cosine sim too low: {cos_sim_k}"
            
            orig_v = dense_v[t, h, :]
            recon_v = v_out[t, h, :]
            cos_sim_v = np.dot(orig_v, recon_v) / (np.linalg.norm(orig_v) * np.linalg.norm(recon_v))
            assert cos_sim_v > 0.8, f"V Cosine sim too low: {cos_sim_v}"

def test_cache_memory():
    cache = PackedKVCache(bits=3, head_dim=128, n_kv_heads=2, n_layers=1)
    
    # Append 10 tokens
    for _ in range(10):
        k = np.random.normal(0, 1, size=(2, 128)).astype(np.float32)
        v = np.random.normal(0, 1, size=(2, 128)).astype(np.float32)
        cache.append(layer=0, k=k, v=v)
        
    mb = cache.memory_bytes()
    # 10 tokens, 2 heads, K+V -> 40 vectors
    # payload: 128 values * 3 bits / 8 bytes = 48 bytes per vector
    # total payload = 40 * 48 = 1920 bytes
    assert mb["payload_bytes"] == 1920
    # scale bytes: 40 vectors * 4 bytes = 160 bytes
    assert mb["scale_bytes"] == 160
    assert mb["padding_bytes"] == 0 # 128 is power of 2

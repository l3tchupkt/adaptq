import pytest
try:
    import torch
    from adaptq.torch_adapter import AdaptQAttention, _TORCH_AVAILABLE, _ENGINE_AVAILABLE
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False
    _ENGINE_AVAILABLE = False

def test_torch_adapter_shapes_and_dtypes():
    if not _TORCH_AVAILABLE or not _ENGINE_AVAILABLE:
        pytest.skip("PyTorch or Engine not available")

    dim = 64
    heads = 2
    layer = AdaptQAttention(dim=dim, heads=heads, bits=4, capacity=32)

    # 1. Test invalid dimension (2D instead of 3D)
    with pytest.raises(ValueError, match="expects 3D tensors"):
        layer(torch.randn(1, heads), torch.randn(1, heads), torch.randn(1, heads))

    # 2. Test batch size > 1
    with pytest.raises(ValueError, match="does not support batch_size > 1"):
        layer(torch.randn(2, heads, dim), torch.randn(2, heads, dim), torch.randn(2, heads, dim))

    # 3. Test mismatched k and v shapes
    q = torch.randn(1, heads, dim)
    bad_k = torch.randn(1, heads + 1, dim)
    bad_v = torch.randn(1, heads, dim + 1)
    with pytest.raises(ValueError, match="Expected k shape"):
        layer(q, bad_k, q)
    with pytest.raises(ValueError, match="Expected v shape"):
        layer(q, q, bad_v)

    # 4. Test float16 dtype preservation
    q_f16 = torch.randn(1, heads, dim, dtype=torch.float16)
    k_f16 = torch.randn(1, heads, dim, dtype=torch.float16)
    v_f16 = torch.randn(1, heads, dim, dtype=torch.float16)
    out_f16 = layer(q_f16, k_f16, v_f16)
    assert out_f16.dtype == torch.float16
    assert out_f16.shape == (1, heads, dim)

    # 5. Test float32
    layer.reset_cache()
    q_f32 = torch.randn(1, heads, dim, dtype=torch.float32)
    out_f32 = layer(q_f32, q_f32, q_f32)
    assert out_f32.dtype == torch.float32
    assert out_f32.shape == (1, heads, dim)

if __name__ == "__main__":
    if _TORCH_AVAILABLE and _ENGINE_AVAILABLE:
        test_torch_adapter_shapes_and_dtypes()
        print("All torch_adapter tests passed!")
    else:
        print("PyTorch or Engine not available, skipping.")

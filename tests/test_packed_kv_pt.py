import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adaptq.reference.packed_kv_pt import PackedKVCachePT


def test_get_on_empty_layer_returns_empty_arrays():
    cache = PackedKVCachePT(bits=4, head_dim=64, n_kv_heads=2, n_layers=3)

    for layer in range(3):
        k, v = cache.get(layer)
        assert k.shape == (0, 2, 64)
        assert v.shape == (0, 2, 64)
        assert k.dtype == np.float32
        assert v.dtype == np.float32


def test_get_preserves_shape_after_append():
    cache = PackedKVCachePT(bits=4, head_dim=64, n_kv_heads=2, n_layers=1)
    k = np.random.RandomState(0).normal(size=(2, 64)).astype(np.float32)
    v = np.random.RandomState(1).normal(size=(2, 64)).astype(np.float32)

    cache.append(0, k, v)
    k_out, v_out = cache.get(0)

    assert cache.sequence_length(0) == 1
    assert k_out.shape == (1, 2, 64)
    assert v_out.shape == (1, 2, 64)


def test_invalid_layer_is_rejected_consistently():
    cache = PackedKVCachePT(bits=4, head_dim=64, n_kv_heads=2, n_layers=2)
    k = np.zeros((2, 64), dtype=np.float32)
    v = np.zeros((2, 64), dtype=np.float32)

    for layer in (-1, 2, 99):
        with pytest.raises(ValueError):
            cache.append(layer, k, v)
        with pytest.raises(ValueError):
            cache.get(layer)
        with pytest.raises(ValueError):
            cache.sequence_length(layer)
        with pytest.raises(ValueError):
            cache.get_dense_ref(layer)
        with pytest.raises(ValueError):
            cache.reset(layer)


def test_append_validates_k_and_v_shape():
    cache = PackedKVCachePT(bits=4, head_dim=64, n_kv_heads=2, n_layers=1)

    with pytest.raises(ValueError):
        cache.append(0, np.zeros((2, 32), dtype=np.float32),
                     np.zeros((2, 64), dtype=np.float32))
    with pytest.raises(ValueError):
        cache.append(0, np.zeros((2, 64), dtype=np.float32),
                     np.zeros((2, 32), dtype=np.float32))


def test_append_batch_validates_shape_and_sequence_length():
    cache = PackedKVCachePT(bits=4, head_dim=64, n_kv_heads=2, n_layers=1)

    with pytest.raises(ValueError):
        cache.append_batch(0, torch.zeros(1, 2, 32), torch.zeros(1, 2, 64))
    with pytest.raises(ValueError):
        cache.append_batch(0, torch.zeros(1, 2, 64), torch.zeros(2, 2, 64))
    with pytest.raises(TypeError):
        cache.append_batch(0, np.zeros((1, 2, 64), dtype=np.float32),
                           torch.zeros(1, 2, 64))


def test_constructor_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        PackedKVCachePT(bits=5, head_dim=64, n_kv_heads=2, n_layers=1)
    with pytest.raises(ValueError):
        PackedKVCachePT(bits=4, head_dim=0, n_kv_heads=2, n_layers=1)
    with pytest.raises(ValueError):
        PackedKVCachePT(bits=4, head_dim=64, n_kv_heads=0, n_layers=1)
    with pytest.raises(ValueError):
        PackedKVCachePT(bits=4, head_dim=64, n_kv_heads=2, n_layers=0)

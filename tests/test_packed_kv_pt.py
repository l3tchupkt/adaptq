import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adaptq.reference.packed_kv_pt import PackedKVCachePT


def test_memory_bytes_with_uneven_layer_occupancy():
    cache = PackedKVCachePT(bits=4, head_dim=32, n_kv_heads=2, n_layers=3)

    for layer, count in ((0, 1), (1, 3)):
        for _ in range(count):
            k = np.random.RandomState(count).normal(size=(2, 32)).astype(np.float32)
            v = np.random.RandomState(count + 1).normal(size=(2, 32)).astype(np.float32)
            cache.append(layer, k, v)

    mb = cache.memory_bytes()

    total_slots = (1 + 3 + 0) * 2
    packed = (32 * 4 + 7) // 8
    assert mb["payload_bytes"] == total_slots * 2 * packed
    assert mb["scale_bytes"] == total_slots * 2 * 4
    assert mb["fp16_equivalent_bytes"] == total_slots * 32 * 2 * 2
    assert mb["compression_ratio_vs_fp16"] == pytest.approx(
        mb["fp16_equivalent_bytes"] / mb["total_bytes"]
    )
    assert mb["n_tokens"] == 3

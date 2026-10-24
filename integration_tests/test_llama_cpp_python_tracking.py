"""Unit coverage for llama-cpp-python KV tracking without serializing state."""
from __future__ import annotations

from adaptq.runtime_py.backends.llama_cpp_python import LlamaCppPythonAdapter
from adaptq.runtime_py.metadata import GenerationResult, RuntimeMetadata


def test_decode_next_does_not_serialize_llama_state():
    """Per-token KV tracking must not call llama-cpp-python save_state()."""

    class FakeModel:
        def __init__(self):
            self.save_state_calls = 0

        def sample(self):
            return 42

        def token_eos(self):
            return 2

        def eval(self, tokens):
            return None

        def save_state(self):
            self.save_state_calls += 1
            raise AssertionError("save_state() must not be called for KV metrics")

    adapter = LlamaCppPythonAdapter.__new__(LlamaCppPythonAdapter)
    adapter._model = FakeModel()
    adapter._done = False
    adapter._generated_ids = []
    adapter._prompt_tokens = [1, 2, 3]
    adapter._last_result = GenerationResult()
    adapter._meta = RuntimeMetadata(
        n_layers=2,
        n_heads=4,
        head_dim=8,
        adaptq_bits=4,
    )

    assert adapter.decode_next() == 42
    assert adapter._model.save_state_calls == 0

    stats = adapter.get_kv_stats()
    assert stats.n_tokens_cached == 4
    assert stats.kv_bytes_fp16 == 1024
    # 4 tokens * 2 layers * 4 heads * 8 dims * 2 (K+V) = 512 values.
    # At 4 bits each that is 256 bytes, so the ratio is 16 / 4 = 4.
    assert stats.kv_bytes_adaptq == 256
    assert stats.compression_ratio == 4.0


def test_kv_byte_accounting_scales_with_bit_width():
    n_layers, n_heads, head_dim = 2, 4, 8
    n_tokens = 4
    n_values = n_tokens * n_layers * n_heads * head_dim * 2
    fp16_bytes = n_values * 2

    for bits in (2, 3, 4):
        adapter = LlamaCppPythonAdapter.__new__(LlamaCppPythonAdapter)
        adapter._model = object()
        adapter._prompt_tokens = [1, 2, 3]
        adapter._generated_ids = [42]
        adapter._meta = RuntimeMetadata(
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            adaptq_bits=bits,
        )

        adapter._update_adaptq_tracking()

        expected = (n_values * bits + 7) // 8
        assert adapter._kv_bytes_fp16 == fp16_bytes
        assert adapter._kv_bytes_adaptq == expected
        assert adapter._kv_bytes_adaptq < fp16_bytes

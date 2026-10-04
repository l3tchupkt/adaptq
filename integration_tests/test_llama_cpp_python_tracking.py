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
    assert stats.kv_bytes_adaptq == 128
    assert stats.compression_ratio == 8.0

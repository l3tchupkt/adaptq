"""
tests/test_base_streaming.py
============================
Unit tests for IRuntimeAdapter.generate_streaming() and generation_result() metrics tracking.

Verifies that the base adapter contract correctly captures timing, token counts,
text accumulation, error handling, and session lifecycle during streaming.
"""
from __future__ import annotations

import unittest
from typing import List, Optional

from adaptq.runtime_py.base import IRuntimeAdapter
from adaptq.runtime_py.metadata import (
    GenerationResult,
    KVStats,
    ModelConfig,
    RuntimeMetadata,
    SessionConfig,
)


class DummyAdapter(IRuntimeAdapter):
    """Concrete mock adapter for testing IRuntimeAdapter base behavior."""

    def __init__(self, vocab_map=None, tokens_to_emit=None):
        self.vocab = vocab_map or {1: "Hello", 2: " ", 3: "world", 4: "!"}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.tokens_to_emit = list(tokens_to_emit if tokens_to_emit is not None else [1, 2, 3, 4])
        self.emit_idx = 0
        self.session_active = False
        self.session_ended = False
        self.begin_session_should_fail = False
        self.prefill_should_fail = False
        self.prefilled_tokens: List[int] = []
        self._error_msg = ""
        self._cached_tokens = 42

    def load_model(self, cfg: ModelConfig) -> bool:
        return True

    def load_tokenizer(self, tokenizer_path: str) -> bool:
        return True

    def begin_session(self, cfg: SessionConfig) -> bool:
        if self.begin_session_should_fail:
            self._error_msg = "simulated begin_session failure"
            return False
        self.session_active = True
        self.session_ended = False
        self.emit_idx = 0
        return True

    def end_session(self) -> bool:
        self.session_active = False
        self.session_ended = True
        return True

    def prefill(self, tokens: List[int]) -> bool:
        if self.prefill_should_fail:
            self._error_msg = "simulated prefill failure"
            return False
        self.prefilled_tokens = list(tokens)
        return True

    def decode_next(self) -> Optional[int]:
        if self.emit_idx < len(self.tokens_to_emit):
            tok = self.tokens_to_emit[self.emit_idx]
            self.emit_idx += 1
            return tok
        return None

    def get_kv_stats(self) -> KVStats:
        return KVStats(
            kv_bytes_adaptq=1024,
            kv_bytes_fp16=4096,
            n_tokens_cached=self._cached_tokens,
        )

    def clear_kv_cache(self) -> bool:
        self.prefilled_tokens = []
        return True

    def metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(backend_name="dummy", model_name="dummy-model")

    def tokenize(self, text: str) -> List[int]:
        return [self.reverse_vocab[w] for w in text.split() if w in self.reverse_vocab]

    def detokenize(self, token_ids: List[int]) -> str:
        return "".join(self.vocab.get(t, f"<{t}>") for t in token_ids)

    def last_error(self) -> str:
        return self._error_msg


class DummyAdapterNoTokenizer(DummyAdapter):
    """Adapter where tokenize/detokenize are not implemented (raise NotImplementedError)."""

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError("tokenize not implemented")

    def detokenize(self, token_ids: List[int]) -> str:
        raise NotImplementedError("detokenize not implemented")


class TestBaseStreaming(unittest.TestCase):
    """Test suite for base generate_streaming() and metrics capture."""

    def test_default_generation_result_before_call(self):
        adapter = DummyAdapter()
        res = adapter.generation_result()
        self.assertIsInstance(res, GenerationResult)
        self.assertEqual(res.n_generated_tokens, 0)
        self.assertEqual(res.text, "")

    def test_streaming_yields_all_chunks_in_order(self):
        adapter = DummyAdapter()
        chunks = list(adapter.generate_streaming("Hello world", max_new_tokens=10))
        self.assertEqual(chunks, ["Hello", " ", "world", "!"])
        self.assertTrue(adapter.session_ended)

    def test_metrics_populated_after_stream_completion(self):
        adapter = DummyAdapter()
        chunks = list(adapter.generate_streaming("Hello world", max_new_tokens=10))
        self.assertEqual("".join(chunks), "Hello world!")

        res = adapter.generation_result()
        self.assertEqual(res.n_generated_tokens, 4)
        self.assertEqual(res.token_ids, [1, 2, 3, 4])
        self.assertEqual(res.text, "Hello world!")
        self.assertGreater(res.wall_time_ms, 0)
        self.assertGreaterEqual(res.tokens_per_sec, 0)
        self.assertEqual(res.kv_stats.n_tokens_cached, 42)
        self.assertFalse(res.error)
        self.assertTrue(res.success)

    def test_early_break_records_partial_metrics_and_ends_session(self):
        adapter = DummyAdapter(tokens_to_emit=[1, 2, 3, 4, 1, 2])
        yielded = []
        for chunk in adapter.generate_streaming("Hello", max_new_tokens=10):
            yielded.append(chunk)
            if len(yielded) == 2:
                break

        self.assertEqual(len(yielded), 2)
        self.assertTrue(adapter.session_ended)

        res = adapter.generation_result()
        self.assertEqual(res.n_generated_tokens, 2)
        self.assertEqual(res.token_ids, [1, 2])
        self.assertGreater(res.wall_time_ms, 0)

    def test_begin_session_failure_records_error(self):
        adapter = DummyAdapter()
        adapter.begin_session_should_fail = True
        chunks = list(adapter.generate_streaming("Hello"))
        self.assertEqual(chunks, [])

        res = adapter.generation_result()
        self.assertTrue(res.error)
        self.assertIn("begin_session failed", res.error)
        self.assertFalse(res.success)

    def test_prefill_failure_records_error_and_ends_session(self):
        adapter = DummyAdapter()
        adapter.prefill_should_fail = True
        chunks = list(adapter.generate_streaming("Hello"))
        self.assertEqual(chunks, [])
        self.assertTrue(adapter.session_ended)

        res = adapter.generation_result()
        self.assertTrue(res.error)
        self.assertIn("prefill failed", res.error)

    def test_detokenize_fallback_when_not_implemented(self):
        adapter = DummyAdapterNoTokenizer(tokens_to_emit=[10, 20])
        chunks = list(adapter.generate_streaming("test", max_new_tokens=5))
        self.assertEqual(chunks, ["<10>", "<20>"])

        res = adapter.generation_result()
        self.assertEqual(res.n_generated_tokens, 2)
        self.assertEqual(res.text, "<10><20>")

    def test_generate_also_updates_generation_result(self):
        adapter = DummyAdapter()
        res_direct = adapter.generate("Hello world", max_new_tokens=5)
        res_stored = adapter.generation_result()
        self.assertIs(res_direct, res_stored)
        self.assertEqual(res_stored.n_generated_tokens, 4)
        self.assertEqual(res_stored.text, "Hello world!")


if __name__ == "__main__":
    unittest.main()

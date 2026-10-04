"""
tests/test_ollama_streaming.py
==============================
Unit tests for OllamaAdapter.generate_streaming().

These tests do NOT require a running Ollama server.  All HTTP calls are mocked
using unittest.mock so the suite runs fully offline.

Run with:
    pytest tests/test_ollama_streaming.py -v
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from adaptq.runtime_py.backends.ollama import OllamaAdapter
from adaptq.runtime_py.metadata import GenerationResult, KVStats, ModelConfig, SessionConfig


def _make_ndjson_lines(*fragments, eval_count=5, prompt_eval_count=3):
    """Build a list of raw NDJSON byte-lines as Ollama would stream them."""
    lines = []
    for frag in fragments:
        lines.append(json.dumps({"response": frag, "done": False}).encode())
    lines.append(
        json.dumps(
            {
                "response": "",
                "done": True,
                "eval_count": eval_count,
                "prompt_eval_count": prompt_eval_count,
            }
        ).encode()
    )
    return lines


def _mock_streaming_response(lines, status_code=200):
    """Create a mock requests.Response that streams the given NDJSON lines."""
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests import HTTPError
        mock_resp.raise_for_status.side_effect = HTTPError(f"HTTP {status_code}")
    mock_resp.iter_lines.return_value = iter(lines)
    return mock_resp


class TestOllamaAdapterStreamingInit(unittest.TestCase):
    """Verify that OllamaAdapter raises ImportError when requests is missing."""

    @patch("adaptq.runtime_py.backends.ollama._REQUESTS_AVAILABLE", False)
    def test_raises_import_error_when_requests_missing(self):
        with self.assertRaises(ImportError) as ctx:
            OllamaAdapter()
        self.assertIn("pip install requests", str(ctx.exception))


class TestOllamaGenerateStreaming(unittest.TestCase):
    """Tests for OllamaAdapter.generate_streaming()."""

    def setUp(self):
        with patch("adaptq.runtime_py.backends.ollama._REQUESTS_AVAILABLE", True):
            self.adapter = OllamaAdapter()
        self.adapter._model_name = "tinyllama"
        self.adapter._model_cfg = ModelConfig(
            model_path="tinyllama",
            temperature=0.7,
            seed=99,
        )

    @patch("adaptq.runtime_py.backends.ollama.requests")
    def test_yields_text_fragments_in_order(self, mock_requests):
        fragments = ["Hello", ", ", "world", "!"]
        mock_requests.post.return_value = _mock_streaming_response(
            _make_ndjson_lines(*fragments, eval_count=4, prompt_eval_count=2)
        )
        result_chunks = list(self.adapter.generate_streaming("Say hello", max_new_tokens=10))
        self.assertEqual(result_chunks, fragments)

    @patch("adaptq.runtime_py.backends.ollama.requests")
    def test_joined_text_matches_concatenated_chunks(self, mock_requests):
        fragments = ["The", " sky", " is", " blue", "."]
        mock_requests.post.return_value = _mock_streaming_response(
            _make_ndjson_lines(*fragments, eval_count=5, prompt_eval_count=3)
        )
        chunks = list(self.adapter.generate_streaming("Describe the sky"))
        self.assertEqual("".join(chunks), "The sky is blue.")

    @patch("adaptq.runtime_py.backends.ollama.requests")
    def test_empty_lines_are_skipped(self, mock_requests):
        lines = [
            b"",
            json.dumps({"response": "Hi", "done": False}).encode(),
            json.dumps({"response": "!", "done": False}).encode(),
            json.dumps({"response": "", "done": True, "eval_count": 2, "prompt_eval_count": 1}).encode(),
        ]
        mock_requests.post.return_value = _mock_streaming_response(lines)
        chunks = list(self.adapter.generate_streaming("Greet"))
        self.assertEqual(chunks, ["Hi", "!"])

    @patch("adaptq.runtime_py.backends.ollama.requests")
    def test_malformed_json_lines_are_skipped(self, mock_requests):
        lines = [
            b"not-json",
            json.dumps({"response": "OK", "done": False}).encode(),
            json.dumps({"response": "", "done": True, "eval_count": 1, "prompt_eval_count": 1}).encode(),
        ]
        mock_requests.post.return_value = _mock_streaming_response(lines)
        chunks = list(self.adapter.generate_streaming("Test"))
        self.assertEqual(chunks, ["OK"])

    @patch("adaptq.runtime_py.backends.ollama.requests")
    def test_generation_result_populated_after_exhaustion(self, mock_requests):
        fragments = ["token1", " token2"]
        mock_requests.post.return_value = _mock_streaming_response(
            _make_ndjson_lines(*fragments, eval_count=7, prompt_eval_count=4)
        )
        list(self.adapter.generate_streaming("Prompt", max_new_tokens=20))
        res = self.adapter.generation_result()
        self.assertIsInstance(res, GenerationResult)
        self.assertEqual(res.n_generated_tokens, 7)
        self.assertEqual(res.n_prompt_tokens, 4)
        self.assertEqual(res.text, "token1 token2")
        self.assertGreater(res.wall_time_ms, 0)
        self.assertFalse(res.error)

    @patch("adaptq.runtime_py.backends.ollama.requests")
    def test_kv_stats_available_after_exhaustion(self, mock_requests):
        mock_requests.post.return_value = _mock_streaming_response(
            _make_ndjson_lines("hello", eval_count=10, prompt_eval_count=2)
        )
        list(self.adapter.generate_streaming("Prompt"))
        stats = self.adapter.get_kv_stats()
        self.assertIsInstance(stats, KVStats)
        self.assertEqual(stats.n_tokens_cached, 10)

    @patch("adaptq.runtime_py.backends.ollama.requests")
    def test_http_error_sets_result_error_field(self, mock_requests):
        mock_requests.post.return_value = _mock_streaming_response([], status_code=500)
        chunks = list(self.adapter.generate_streaming("Will fail"))
        self.assertEqual(chunks, [])
        res = self.adapter.generation_result()
        self.assertTrue(res.error)

    @patch("adaptq.runtime_py.backends.ollama.requests")
    def test_custom_session_cfg_max_new_tokens_forwarded(self, mock_requests):
        mock_requests.post.return_value = _mock_streaming_response(
            _make_ndjson_lines("ok", eval_count=1, prompt_eval_count=1)
        )
        cfg = SessionConfig(prompt="ignored", max_new_tokens=5)
        list(self.adapter.generate_streaming("Real prompt", max_new_tokens=5, session_cfg=cfg))
        call_kwargs = mock_requests.post.call_args
        payload = call_kwargs[1]["json"]
        self.assertEqual(payload["options"]["num_predict"], 5)

    @patch("adaptq.runtime_py.backends.ollama.requests")
    def test_chunks_without_response_key_not_yielded(self, mock_requests):
        lines = [
            json.dumps({"status": "loading", "done": False}).encode(),
            json.dumps({"response": "A", "done": False}).encode(),
            json.dumps({"done": True, "eval_count": 1, "prompt_eval_count": 0}).encode(),
        ]
        mock_requests.post.return_value = _mock_streaming_response(lines)
        chunks = list(self.adapter.generate_streaming("Prompt"))
        self.assertEqual(chunks, ["A"])


if __name__ == "__main__":
    unittest.main()

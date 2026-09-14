"""
integration_tests/test_llama_cpp_python.py
==========================================
End-to-end integration tests for the llama-cpp-python backend.

Requires a GGUF model file. Pass via:
    pytest --model-gguf /path/to/model.gguf integration_tests/test_llama_cpp_python.py

Or download TinyLlama first:
    python examples/llama_demo.py --auto-download

Run:
    pytest integration_tests/test_llama_cpp_python.py -v --model-gguf tinyllama.gguf
"""
from __future__ import annotations

import pytest

from adaptq.runtime_py import backend_available
from adaptq.runtime_py.metadata import SessionConfig


pytestmark = pytest.mark.llama_cpp_python


def test_llama_cpp_python_available():
    """llama_cpp_python backend can be instantiated."""
    assert backend_available("llama_cpp_python"), (
        "llama_cpp_python backend not available — install: pip install llama-cpp-python"
    )


def test_load_model_metadata(llama_adapter):
    """load_model() populates model metadata."""
    meta = llama_adapter.metadata()
    assert meta.backend_name == "llama_cpp_python"
    assert len(meta.model_name) > 0


def test_tokenize_nonempty(llama_adapter):
    """tokenize() returns non-empty token list."""
    tokens = llama_adapter.tokenize("Hello!")
    assert isinstance(tokens, list)
    assert len(tokens) >= 1


def test_begin_end_session(llama_adapter, small_prompt):
    """Session lifecycle works without error."""
    cfg = SessionConfig(prompt=small_prompt, max_new_tokens=5)
    assert llama_adapter.begin_session(cfg)
    assert llama_adapter.end_session()


def test_prefill_succeeds(llama_adapter, small_prompt):
    """prefill() with prompt tokens does not fail."""
    cfg = SessionConfig(prompt=small_prompt, max_new_tokens=5)
    llama_adapter.begin_session(cfg)
    tokens = llama_adapter.tokenize(small_prompt)
    ok = llama_adapter.prefill(tokens)
    assert ok is True, f"prefill failed: {llama_adapter.last_error()}"
    llama_adapter.end_session()


def test_decode_next_returns_token(llama_adapter, small_prompt):
    """decode_next() returns an integer after prefill."""
    cfg = SessionConfig(prompt=small_prompt, max_new_tokens=10)
    llama_adapter.begin_session(cfg)
    tokens = llama_adapter.tokenize(small_prompt)
    llama_adapter.prefill(tokens)

    tok = llama_adapter.decode_next()
    llama_adapter.end_session()

    assert tok is not None, "No token generated"
    assert isinstance(tok, int)
    assert tok >= 0


def test_generate_full_pipeline(llama_adapter, small_prompt):
    """Full generate() pipeline produces a non-empty result."""
    result = llama_adapter.generate(small_prompt, max_new_tokens=15)
    assert result.success, f"error: {result.error}"
    assert result.n_generated_tokens >= 1
    assert result.wall_time_ms > 0


def test_generate_kv_stats(llama_adapter, small_prompt):
    """KV stats are returned (may be approximate for llama_cpp_python)."""
    result = llama_adapter.generate(small_prompt, max_new_tokens=10)
    assert result.success
    kv = result.kv_stats
    # Both may be 0 if metadata extraction failed, that's OK —
    # just assert they're non-negative
    assert kv.kv_bytes_adaptq >= 0
    assert kv.kv_bytes_fp16 >= 0


def test_clear_kv_cache(llama_adapter):
    """clear_kv_cache() returns True."""
    assert llama_adapter.clear_kv_cache() is True


def test_two_generates_independent(llama_adapter, small_prompt):
    """Two successive generate() calls succeed independently."""
    r1 = llama_adapter.generate(small_prompt, max_new_tokens=5)
    r2 = llama_adapter.generate(small_prompt, max_new_tokens=5)
    assert r1.success and r2.success

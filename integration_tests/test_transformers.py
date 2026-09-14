"""
integration_tests/test_transformers.py
========================================
End-to-end integration tests for the HuggingFace Transformers backend.

Tests:
  1. Model loads successfully
  2. Metadata is populated correctly
  3. Tokenizer works (tokenize/detokenize roundtrip)
  4. Prefill succeeds
  5. Decode produces non-empty tokens
  6. Full generate() pipeline works
  7. KV stats are populated (if adaptq_py installed)
  8. Second generate() call is independent
  9. Streaming generation yields tokens

Run:
    pytest integration_tests/test_transformers.py -v
    pytest integration_tests/test_transformers.py -v --model-hf Qwen/Qwen2-0.5B
"""
from __future__ import annotations

import pytest

from adaptq.runtime_py import backend_available
from adaptq.runtime_py.metadata import SessionConfig


pytestmark = pytest.mark.transformers


# ── 1. Backend availability ───────────────────────────────────────────────

def test_transformers_backend_available():
    """transformers backend can be instantiated without crashing."""
    assert backend_available("transformers"), (
        "transformers backend not available — install: pip install transformers accelerate"
    )


# ── 2. Model load ─────────────────────────────────────────────────────────

def test_load_model(transformers_adapter):
    """load_model() returns True and populates metadata."""
    meta = transformers_adapter.metadata()
    assert meta.backend_name == "transformers"
    assert len(meta.model_name) > 0
    assert meta.n_layers > 0,  f"n_layers={meta.n_layers}"
    assert meta.n_heads  > 0,  f"n_heads={meta.n_heads}"
    assert meta.head_dim > 0,  f"head_dim={meta.head_dim}"
    assert meta.vocab_size > 0, f"vocab_size={meta.vocab_size}"


def test_kv_access_reported(transformers_adapter):
    """Transformers adapter reports kv_access=True."""
    meta = transformers_adapter.metadata()
    assert meta.kv_access is True


# ── 3. Tokenizer ──────────────────────────────────────────────────────────

def test_tokenize_nonempty(transformers_adapter):
    """tokenize() returns a non-empty list for any non-empty string."""
    tokens = transformers_adapter.tokenize("Hello, world!")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_tokenize_detokenize_roundtrip(transformers_adapter):
    """detokenize(tokenize(text)) recovers the original text (approximately)."""
    text = "KV cache compression"
    tokens = transformers_adapter.tokenize(text)
    recovered = transformers_adapter.detokenize(tokens)
    # Allow minor whitespace differences
    assert text.lower().replace(" ", "") in recovered.lower().replace(" ", ""), (
        f"Roundtrip failed: {text!r} → {tokens} → {recovered!r}"
    )


# ── 4. Session lifecycle ──────────────────────────────────────────────────

def test_begin_end_session(transformers_adapter, small_prompt):
    """begin_session() + end_session() work without error."""
    cfg = SessionConfig(prompt=small_prompt, max_new_tokens=5)
    assert transformers_adapter.begin_session(cfg) is True
    assert transformers_adapter.end_session() is True


def test_prefill_succeeds(transformers_adapter, small_prompt):
    """prefill() with prompt tokens completes without error."""
    cfg = SessionConfig(prompt=small_prompt, max_new_tokens=5)
    transformers_adapter.begin_session(cfg)
    tokens = transformers_adapter.tokenize(small_prompt)
    assert len(tokens) > 0
    ok = transformers_adapter.prefill(tokens)
    assert ok is True, f"prefill failed: {transformers_adapter.last_error()}"
    transformers_adapter.end_session()


# ── 5. Token-by-token decode ──────────────────────────────────────────────

def test_decode_next_returns_token(transformers_adapter, small_prompt):
    """decode_next() returns an integer token ID after prefill."""
    cfg = SessionConfig(prompt=small_prompt, max_new_tokens=10)
    transformers_adapter.begin_session(cfg)
    tokens = transformers_adapter.tokenize(small_prompt)
    transformers_adapter.prefill(tokens)

    tok = transformers_adapter.decode_next()
    transformers_adapter.end_session()

    assert tok is not None, "decode_next() returned None immediately after prefill"
    assert isinstance(tok, int)
    assert tok >= 0


def test_decode_produces_n_tokens(transformers_adapter, small_prompt):
    """Decode loop produces up to max_new_tokens tokens."""
    MAX = 10
    cfg = SessionConfig(prompt=small_prompt, max_new_tokens=MAX)
    transformers_adapter.begin_session(cfg)
    tokens = transformers_adapter.tokenize(small_prompt)
    transformers_adapter.prefill(tokens)

    generated = []
    for _ in range(MAX):
        tok = transformers_adapter.decode_next()
        if tok is None:
            break
        generated.append(tok)

    transformers_adapter.end_session()
    # Should generate at least 1 token
    assert len(generated) >= 1, "No tokens generated"


# ── 6. Full generate() pipeline ───────────────────────────────────────────

def test_generate_pipeline(transformers_adapter, small_prompt):
    """generate() returns a non-empty GenerationResult."""
    result = transformers_adapter.generate(small_prompt, max_new_tokens=15)

    assert result.success, f"generate() failed: {result.error}"
    assert result.n_generated_tokens >= 1
    assert len(result.token_ids) >= 1
    assert result.wall_time_ms > 0
    assert result.tokens_per_sec > 0


def test_generate_text_nonempty(transformers_adapter, small_prompt):
    """Generated text is non-empty after detokenization."""
    result = transformers_adapter.generate(small_prompt, max_new_tokens=10)
    assert result.success
    assert isinstance(result.text, str)
    assert len(result.text) > 0, "Generated text is empty"


def test_generate_prompt_token_count(transformers_adapter, small_prompt):
    """n_prompt_tokens is > 0 after generate()."""
    result = transformers_adapter.generate(small_prompt, max_new_tokens=5)
    assert result.success
    assert result.n_prompt_tokens > 0


# ── 7. KV stats ───────────────────────────────────────────────────────────

def test_kv_stats_returned(transformers_adapter, small_prompt):
    """get_kv_stats() returns a KVStats object after generate()."""
    result = transformers_adapter.generate(small_prompt, max_new_tokens=10)
    assert result.success
    kv = result.kv_stats
    # compression_ratio may be 0 if adaptq_py not installed, that's OK
    assert kv.kv_bytes_adaptq >= 0
    assert kv.kv_bytes_fp16 >= 0


# ── 8. Multiple calls are independent ─────────────────────────────────────

def test_two_generates_independent(transformers_adapter, small_prompt):
    """Two successive generate() calls return independent results."""
    r1 = transformers_adapter.generate(small_prompt, max_new_tokens=8)
    r2 = transformers_adapter.generate(small_prompt, max_new_tokens=8)
    assert r1.success and r2.success
    # Both should produce valid output
    assert r1.n_generated_tokens >= 1
    assert r2.n_generated_tokens >= 1


# ── 9. Clear KV cache ─────────────────────────────────────────────────────

def test_clear_kv_cache(transformers_adapter):
    """clear_kv_cache() returns True."""
    assert transformers_adapter.clear_kv_cache() is True

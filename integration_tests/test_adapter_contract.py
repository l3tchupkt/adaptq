"""
integration_tests/test_adapter_contract.py
==========================================
Backend-agnostic contract tests for IRuntimeAdapter.

Every backend adapter must satisfy these tests. Run against any
available backend to verify conformance with the IRuntimeAdapter contract.

Tests verify:
  1. create_adapter() returns an IRuntimeAdapter subclass
  2. metadata() returns RuntimeMetadata before load_model()
  3. list_available_backends() returns a list
  4. backend_available() returns bool
  5. Adapter repr() does not crash
  6. Stubs raise appropriate errors

Run:
    pytest integration_tests/test_adapter_contract.py -v
"""
from __future__ import annotations

import pytest

from adaptq.runtime_py import (
    IRuntimeAdapter,
    create_adapter,
    backend_available,
    list_available_backends,
    SUPPORTED_BACKENDS,
)
from adaptq.runtime_py.metadata import (
    GenerationResult,
    KVStats,
    ModelConfig,
    RuntimeMetadata,
    SessionConfig,
)


# ── Registry contract ─────────────────────────────────────────────────────

def test_supported_backends_nonempty():
    """SUPPORTED_BACKENDS is a non-empty list of strings."""
    assert isinstance(SUPPORTED_BACKENDS, list)
    assert len(SUPPORTED_BACKENDS) >= 6
    for name in SUPPORTED_BACKENDS:
        assert isinstance(name, str)


def test_list_available_backends_returns_list():
    """list_available_backends() returns a list (may be empty on fresh env)."""
    avail = list_available_backends()
    assert isinstance(avail, list)
    # All returned names must be in SUPPORTED_BACKENDS
    for name in avail:
        assert name in SUPPORTED_BACKENDS


def test_backend_available_returns_bool():
    """backend_available() returns bool for all supported backends."""
    for name in SUPPORTED_BACKENDS:
        result = backend_available(name)
        assert isinstance(result, bool)


def test_unknown_backend_raises():
    """create_adapter() raises ValueError for unknown backends."""
    with pytest.raises(ValueError, match="Unknown backend"):
        create_adapter("not_a_real_backend_12345")


# ── Stub backends raise appropriate errors ────────────────────────────────

def test_vllm_stub_raises_on_no_cuda():
    """VLLMAdapter raises RuntimeError or ImportError on CPU-only systems."""
    try:
        import torch
        if torch.cuda.is_available():
            pytest.skip("CUDA available — vLLM stub behavior not tested on GPU")
    except ImportError:
        pass
    with pytest.raises((RuntimeError, ImportError)):
        create_adapter("vllm")


def test_mlx_stub_raises_on_non_apple():
    """MLXAdapter raises RuntimeError on non-Apple Silicon."""
    import platform
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        pytest.skip("Running on Apple Silicon — MLX may actually be available")
    with pytest.raises((RuntimeError, ImportError, NotImplementedError)):
        create_adapter("mlx")


def test_llama_cpp_stub_raises():
    """LlamaCppAdapter (C++ direct) raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="llama_cpp C\\+\\+ adapter"):
        create_adapter("llama_cpp")


# ── Available backend contract ────────────────────────────────────────────

@pytest.mark.parametrize("backend", list_available_backends())
def test_adapter_is_iruntimeadapter_subclass(backend):
    """Each available adapter is an IRuntimeAdapter subclass."""
    from adaptq.runtime_py.backends import get_adapter_class
    cls = get_adapter_class(backend)
    assert issubclass(cls, IRuntimeAdapter), (
        f"{cls.__name__} does not inherit from IRuntimeAdapter"
    )


@pytest.mark.parametrize("backend", list_available_backends())
def test_adapter_metadata_before_load(backend):
    """metadata() returns a RuntimeMetadata even before load_model()."""
    adapter = create_adapter(backend)
    meta = adapter.metadata()
    assert isinstance(meta, RuntimeMetadata)
    assert meta.backend_name == backend or len(meta.backend_name) > 0


@pytest.mark.parametrize("backend", list_available_backends())
def test_adapter_repr_doesnt_crash(backend):
    """repr(adapter) does not raise."""
    adapter = create_adapter(backend)
    r = repr(adapter)
    assert isinstance(r, str)


@pytest.mark.parametrize("backend", list_available_backends())
def test_last_error_returns_string(backend):
    """last_error() returns a string (empty if no error)."""
    adapter = create_adapter(backend)
    err = adapter.last_error()
    assert isinstance(err, str)


# ── Dataclass contracts ───────────────────────────────────────────────────

def test_model_config_defaults():
    """ModelConfig has sensible defaults."""
    cfg = ModelConfig()
    assert cfg.n_ctx > 0
    assert cfg.n_threads > 0
    assert cfg.adaptq_bits in (2, 3, 4)


def test_session_config_defaults():
    """SessionConfig has sensible defaults."""
    cfg = SessionConfig()
    assert cfg.max_new_tokens > 0
    assert isinstance(cfg.log_tokens, bool)


def test_kv_stats_compression_ratio():
    """KVStats.compression_ratio is computed correctly."""
    kv = KVStats(kv_bytes_adaptq=1000, kv_bytes_fp16=8000)
    assert kv.compression_ratio == pytest.approx(8.0)


def test_generation_result_success():
    """GenerationResult.success is True when error is empty."""
    r = GenerationResult(text="hello", n_generated_tokens=5)
    assert r.success is True


def test_generation_result_failure():
    """GenerationResult.success is False when error is set."""
    r = GenerationResult(error="something went wrong")
    assert r.success is False


def test_generation_result_summary_nocrash():
    """GenerationResult.summary() does not raise."""
    r = GenerationResult(
        text="hello world",
        n_generated_tokens=2,
        wall_time_ms=100.0,
        tokens_per_sec=20.0,
        kv_stats=KVStats(kv_bytes_adaptq=500, kv_bytes_fp16=4000),
    )
    s = r.summary()
    assert isinstance(s, str)
    assert len(s) > 0

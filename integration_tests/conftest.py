"""
integration_tests/conftest.py
==============================
Shared pytest fixtures and markers for AdapTQ V2.1 integration tests.

Markers:
    @pytest.mark.transformers      — requires transformers backend
    @pytest.mark.llama_cpp_python  — requires llama_cpp_python backend + model
    @pytest.mark.ollama            — requires Ollama server running
    @pytest.mark.gpu_required      — requires CUDA GPU
    @pytest.mark.slow              — long-running test (skipped in CI by default)

Fixtures:
    transformers_adapter   — loaded TransformersAdapter (Qwen2-0.5B)
    llama_adapter          — loaded LlamaCppPythonAdapter (TinyLlama GGUF)
    small_prompt           — short test prompt
    snap_dir               — temporary directory for snapshot files
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Markers ───────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers",
        "transformers: requires transformers + accelerate + a HF model")
    config.addinivalue_line("markers",
        "llama_cpp_python: requires llama-cpp-python and a GGUF model file")
    config.addinivalue_line("markers",
        "ollama: requires Ollama server running at localhost:11434")
    config.addinivalue_line("markers",
        "gpu_required: requires CUDA GPU")
    config.addinivalue_line("markers",
        "slow: long-running test, skipped unless --run-slow is passed")


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true",
                     help="Include slow integration tests")
    parser.addoption("--model-hf", default="Qwen/Qwen2-0.5B",
                     help="HuggingFace model ID for transformers tests")
    parser.addoption("--model-gguf", default="",
                     help="Path to GGUF model file for llama_cpp_python tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow", default=False):
        skip_slow = pytest.mark.skip(reason="Pass --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def hf_model_path(request):
    return request.config.getoption("--model-hf", default="Qwen/Qwen2-0.5B")


@pytest.fixture(scope="session")
def gguf_model_path(request):
    path = request.config.getoption("--model-gguf", default="")
    if not path:
        # Look for any .gguf file in the project root
        for candidate in ROOT.glob("*.gguf"):
            return str(candidate)
        for candidate in ROOT.glob("**/*.gguf"):
            return str(candidate)
        pytest.skip("No GGUF model file found. "
                    "Pass --model-gguf=/path/to/model.gguf or "
                    "download with: python examples/llama_demo.py --auto-download")
    return path


@pytest.fixture(scope="session")
def transformers_adapter(hf_model_path):
    """Loaded TransformersAdapter, shared across the test session."""
    from adaptq.runtime_py import create_adapter, backend_available
    from adaptq.runtime_py.metadata import ModelConfig

    if not backend_available("transformers"):
        pytest.skip("transformers backend not available")

    adapter = create_adapter("transformers")
    cfg = ModelConfig(
        model_path=hf_model_path,
        adaptq_bits=4,
        adaptq_capacity=512,
        n_threads=2,
    )
    if not adapter.load_model(cfg):
        pytest.skip(f"Could not load model {hf_model_path}: {adapter.last_error()}")
    return adapter


@pytest.fixture(scope="session")
def llama_adapter(gguf_model_path):
    """Loaded LlamaCppPythonAdapter, shared across the test session."""
    from adaptq.runtime_py import create_adapter, backend_available
    from adaptq.runtime_py.metadata import ModelConfig

    if not backend_available("llama_cpp_python"):
        pytest.skip("llama_cpp_python backend not available")

    adapter = create_adapter("llama_cpp_python")
    cfg = ModelConfig(
        model_path=gguf_model_path,
        n_ctx=512,
        n_threads=2,
        adaptq_bits=4,
        adaptq_capacity=512,
    )
    if not adapter.load_model(cfg):
        pytest.skip(f"Could not load GGUF model: {adapter.last_error()}")
    return adapter


@pytest.fixture
def small_prompt():
    return "KV cache is"


@pytest.fixture
def snap_dir():
    """Temporary directory for snapshot files, cleaned up after each test."""
    with tempfile.TemporaryDirectory(prefix="adaptq_test_") as d:
        yield Path(d)

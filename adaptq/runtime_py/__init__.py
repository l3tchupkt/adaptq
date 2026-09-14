"""
adaptq.runtime_py
=================
Backend-agnostic runtime adapter layer for AdapTQ V2.1.

Provides IRuntimeAdapter and a factory for creating adapters for
different LLM inference backends.

Supported backends:
    "transformers"     HuggingFace Transformers (CPU/GPU, pip install transformers)
    "llama_cpp_python" llama-cpp-python (CPU, pip install llama-cpp-python)
    "ollama"           Ollama REST API (requires ollama server running)
    "vllm"             vLLM (stub, requires CUDA GPU)
    "mlx"              MLX (stub, requires Apple Silicon)
    "llama_cpp"        llama.cpp C++ direct (stub, requires source build)

Quick start:
    from adaptq.runtime_py import create_adapter
    from adaptq.runtime_py.metadata import ModelConfig

    # HuggingFace backend (no model file needed for small models)
    adapter = create_adapter("transformers")
    adapter.load_model(ModelConfig(model_path="Qwen/Qwen2-0.5B"))
    result = adapter.generate("What is AdapTQ?", max_new_tokens=50)
    print(result.summary())

    # llama-cpp-python backend (requires GGUF file)
    adapter = create_adapter("llama_cpp_python")
    adapter.load_model(ModelConfig(model_path="tinyllama.gguf"))
    result = adapter.generate("Hello!")
    print(result.summary())
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .base import IRuntimeAdapter
from .metadata import (
    GenerationResult,
    KVStats,
    ModelConfig,
    RuntimeMetadata,
    SessionConfig,
)
from .backends import SUPPORTED_BACKENDS, get_adapter_class

if TYPE_CHECKING:
    pass


def create_adapter(backend: str, **kwargs) -> IRuntimeAdapter:
    """
    Create an IRuntimeAdapter for the given backend.

    Args:
        backend: Backend name. One of: "transformers", "llama_cpp_python",
                 "ollama", "vllm", "mlx", "llama_cpp".
        **kwargs: Passed to the adapter constructor.

    Returns:
        An IRuntimeAdapter instance.

    Raises:
        ValueError: Unknown backend name.
        ImportError: Required package for the backend is not installed.
        RuntimeError: Backend preconditions not met (e.g. no GPU for vLLM).

    Example:
        adapter = create_adapter("transformers")
        adapter.load_model(ModelConfig(model_path="Qwen/Qwen2-0.5B"))
        result = adapter.generate("Tell me about KV cache quantization")
        print(result.text)
    """
    cls = get_adapter_class(backend)
    return cls(**kwargs)


def backend_available(backend: str) -> bool:
    """
    Check if a backend can be instantiated without errors.

    Returns True if the backend is available, False otherwise.
    Does NOT load a model — only checks that the adapter can be created.

    Example:
        if backend_available("transformers"):
            adapter = create_adapter("transformers")
    """
    try:
        cls = get_adapter_class(backend)
        # Try instantiating — this will raise ImportError/RuntimeError if
        # the required package is missing or preconditions are not met.
        obj = cls()
        return True
    except (ImportError, RuntimeError, NotImplementedError):
        return False
    except Exception:
        return False


def list_available_backends() -> list[str]:
    """
    Return a list of backend names that can be instantiated.

    Example:
        print(list_available_backends())
        # → ['transformers', 'llama_cpp_python']
    """
    return [b for b in SUPPORTED_BACKENDS if backend_available(b)]


__all__ = [
    "IRuntimeAdapter",
    "ModelConfig",
    "SessionConfig",
    "RuntimeMetadata",
    "GenerationResult",
    "KVStats",
    "SUPPORTED_BACKENDS",
    "create_adapter",
    "backend_available",
    "list_available_backends",
]

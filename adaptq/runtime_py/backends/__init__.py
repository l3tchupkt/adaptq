"""
adaptq.runtime_py.backends.__init__
=====================================
Backend registry — maps backend name strings to adapter classes.
Imports are lazy to avoid hard-failing when optional packages are missing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Type

if TYPE_CHECKING:
    from ..base import IRuntimeAdapter

_REGISTRY: Dict[str, str] = {
    "transformers":      "adaptq.runtime_py.backends.transformers_hf.TransformersAdapter",
    "llama_cpp_python":  "adaptq.runtime_py.backends.llama_cpp_python.LlamaCppPythonAdapter",
    "ollama":            "adaptq.runtime_py.backends.ollama.OllamaAdapter",
    "vllm":              "adaptq.runtime_py.backends.vllm.VLLMAdapter",
    "mlx":               "adaptq.runtime_py.backends.mlx.MLXAdapter",
    "llama_cpp":         "adaptq.runtime_py.backends.llama_cpp.LlamaCppAdapter",
}

SUPPORTED_BACKENDS = list(_REGISTRY.keys())


def get_adapter_class(backend: str) -> "Type[IRuntimeAdapter]":
    """
    Return the adapter class for the given backend name.

    Raises ValueError for unknown backends, ImportError if the
    backend's required package is not installed.
    """
    if backend not in _REGISTRY:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            f"Supported: {SUPPORTED_BACKENDS}"
        )

    dotted = _REGISTRY[backend]
    module_path, cls_name = dotted.rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


__all__ = ["SUPPORTED_BACKENDS", "get_adapter_class"]

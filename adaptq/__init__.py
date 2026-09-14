"""
adaptq - Adaptive Streaming Vector Quantization KV Cache for LLMs
==================================================================
V2.1: Multi-backend runtime adapter layer + Replay + Compare

Quick start:
    from adaptq import create_adapter
    adapter = create_adapter("transformers")
    result = adapter.generate("Hello!", max_new_tokens=50)
    print(result.text)

Optional extras:
    pip install 'adaptq[transformers]'  # HuggingFace Transformers backend
    pip install 'adaptq[llama]'         # llama-cpp-python backend
    pip install 'adaptq[torch]'         # AdaptQAttention PyTorch module
"""
from __future__ import annotations

__version__: str = "0.2.2"


# -- C extension (adaptq_py) --------------------------------------------------
# Installed by: pip install .  (builds via setup.py / pyproject.toml)
try:
    from .core import Engine
    _ENGINE_AVAILABLE = True
except ImportError:
    Engine = None            # type: ignore[assignment,misc]
    _ENGINE_AVAILABLE = False

# -- PyTorch wrapper (optional) -----------------------------------------------
# Installed by: pip install 'adaptq[torch]'
try:
    from .torch_adapter import AdaptQAttention
except ImportError:
    class AdaptQAttention:   # type: ignore[no-redef]
        """Stub: pip install 'adaptq[torch]' to use AdaptQAttention."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "AdaptQAttention requires PyTorch and the adaptq C extension.\n"
                "Install: pip install 'adaptq[torch]'"
            )

# -- Replay / Compare (uses CLI subprocess, optional) -------------------------
try:
    from .replay import ReplayEngine, ReplayResult, CompareResult, snapshot_info
    _REPLAY_AVAILABLE = True
except ImportError:
    ReplayEngine = CompareResult = ReplayResult = snapshot_info = None  # type: ignore
    _REPLAY_AVAILABLE = False


# -- V2.1: Runtime adapter layer (lazy) ---------------------------------------
def create_adapter(backend: str, **kwargs):
    """
    Create a backend adapter for AdapTQ inference.

    Available backends (install as needed):
        "transformers"      pip install transformers accelerate
        "llama_cpp_python"  pip install llama-cpp-python
        "ollama"            Requires Ollama server at localhost:11434
        "vllm"              pip install vllm  (CUDA GPU required)
        "mlx"               pip install mlx mlx-lm  (Apple Silicon only)

    Example:
        from adaptq import create_adapter
        from adaptq.runtime_py.metadata import ModelConfig
        adapter = create_adapter("transformers")
        adapter.load_model(ModelConfig(model_path="Qwen/Qwen2-0.5B"))
        result = adapter.generate("Hello!", max_new_tokens=50)
        print(result.text)
    """
    from adaptq.runtime_py import create_adapter as _create
    return _create(backend, **kwargs)


def list_available_backends() -> list:
    """Return a list of backend names currently importable in this environment."""
    from adaptq.runtime_py import list_available_backends as _list
    return _list()


__all__ = [
    # Version
    "__version__",
    # C extension
    "Engine",
    # PyTorch (optional)
    "AdaptQAttention",
    # Replay (optional)
    "ReplayEngine",
    "ReplayResult",
    "CompareResult",
    "snapshot_info",
    # V2.1 runtime adapters
    "create_adapter",
    "list_available_backends",
]

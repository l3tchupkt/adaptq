"""
adaptq.runtime_py.backends.mlx
================================
MLX (Apple Silicon) backend adapter stub for AdapTQ.

Status: STUB — Apple Silicon only.

MLX is Apple's ML framework optimized for Apple Silicon (M1/M2/M3).
It is not available on x86 Linux or Windows.

Full integration would subclass mlx_lm's model classes to intercept
K/V tensors in the attention layers, routing them through AdapTQ.

Requires:
  - Apple Silicon Mac (M1/M2/M3/M4)
  - pip install mlx mlx-lm

See: https://github.com/ml-explore/mlx-lm
"""
from __future__ import annotations

import platform

from ..base import IRuntimeAdapter
from ..metadata import GenerationResult, KVStats, ModelConfig, RuntimeMetadata, SessionConfig


def _check_apple_silicon():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "MLX backend requires Apple Silicon (M1/M2/M3/M4 Mac).\n"
            f"Detected: {platform.system()} {platform.machine()}\n"
            "For x86 Linux, use 'transformers' or 'llama_cpp_python' backends."
        )


class MLXAdapter(IRuntimeAdapter):
    """
    Stub adapter for MLX (Apple Silicon). Raises RuntimeError on non-Apple systems.
    """

    BACKEND_NAME = "mlx"

    def __init__(self):
        _check_apple_silicon()
        try:
            import mlx  # noqa: F401
        except ImportError:
            raise ImportError("MLX backend requires: pip install mlx mlx-lm")

    def _not_impl(self):
        raise NotImplementedError("MLXAdapter is a stub. Full implementation in V3.")

    def load_model(self, cfg: ModelConfig) -> bool:        self._not_impl(); return False
    def load_tokenizer(self, path: str) -> bool:           self._not_impl(); return False
    def begin_session(self, cfg: SessionConfig) -> bool:   self._not_impl(); return False
    def end_session(self) -> bool:                         self._not_impl(); return False
    def prefill(self, tokens) -> bool:                     self._not_impl(); return False
    def decode_next(self):                                 self._not_impl(); return None
    def get_kv_stats(self) -> KVStats:                     self._not_impl(); return KVStats()
    def clear_kv_cache(self) -> bool:                      self._not_impl(); return False
    def metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(backend_name=self.BACKEND_NAME)
    def generation_result(self) -> GenerationResult:
        return GenerationResult()

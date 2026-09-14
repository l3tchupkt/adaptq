"""
adaptq.runtime_py.backends.vllm
================================
vLLM backend adapter stub for AdapTQ.

Status: STUB — requires CUDA GPU.

vLLM exposes an `attention_backend` plugin interface that allows
custom KV cache implementations. Full AdapTQ integration is planned
for V3 when vLLM stable GPU support is available in this environment.

To implement the full adapter:
  1. Subclass vllm.attention.backends.abstract.AttentionBackend
  2. Override compute_kv_cache_shape() and forward()
  3. Feed K/V through adaptq.RuntimeContext in forward()
  4. See: https://docs.vllm.ai/en/latest/design/kernel/attention.html

Requires:
  - NVIDIA GPU with CUDA
  - pip install vllm
"""
from __future__ import annotations

from ..base import IRuntimeAdapter
from ..metadata import GenerationResult, KVStats, ModelConfig, RuntimeMetadata, SessionConfig


class VLLMAdapter(IRuntimeAdapter):
    """
    Stub adapter for vLLM. Raises RuntimeError on all calls.
    Full implementation requires CUDA GPU and vLLM installation.
    """

    BACKEND_NAME = "vllm"

    def __init__(self):
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "vLLM backend requires a CUDA-capable GPU.\n"
                    "Detected: CUDA not available.\n"
                    "For CPU-only inference, use 'transformers' or 'llama_cpp_python' backends."
                )
        except ImportError:
            raise RuntimeError("vLLM backend requires torch and CUDA GPU.")

        try:
            import vllm  # noqa: F401
        except ImportError:
            raise ImportError(
                "vLLM backend requires: pip install vllm\n"
                "(NVIDIA GPU + CUDA required)"
            )

    def _not_impl(self):
        raise NotImplementedError("VLLMAdapter is a stub. Full implementation in V3.")

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

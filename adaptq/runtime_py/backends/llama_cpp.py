"""
adaptq.runtime_py.backends.llama_cpp
======================================
llama.cpp C++ direct backend stub for AdapTQ.

Status: STUB — requires building and patching llama.cpp.

The C++ integration patches llama.cpp at the source level to intercept
KV tensors after ggml_compute(). This gives the most direct, lowest-overhead
integration but requires compiling llama.cpp with AdapTQ linked.

For a pip-installable integration without C++ compilation, use the
'llama_cpp_python' backend instead.

Integration guide: see integration/llama_cpp_patch.md
"""
from __future__ import annotations

from ..base import IRuntimeAdapter
from ..metadata import GenerationResult, KVStats, ModelConfig, RuntimeMetadata, SessionConfig


class LlamaCppAdapter(IRuntimeAdapter):
    """
    Stub adapter for llama.cpp C++ direct integration.

    Use this class as a guide for the C++ integration; see:
      integration/llama_cpp_patch.md
      runtime/adapters/adapter.h

    For Python-level integration, use LlamaCppPythonAdapter instead.
    """

    BACKEND_NAME = "llama_cpp"

    def __init__(self):
        raise NotImplementedError(
            "llama_cpp C++ adapter requires building AdapTQ into llama.cpp.\n"
            "See: integration/llama_cpp_patch.md\n\n"
            "For pip-installable Python integration, use:\n"
            "  create_adapter('llama_cpp_python')\n"
        )

    def load_model(self, cfg: ModelConfig) -> bool:        return False
    def load_tokenizer(self, path: str) -> bool:           return False
    def begin_session(self, cfg: SessionConfig) -> bool:   return False
    def end_session(self) -> bool:                         return False
    def prefill(self, tokens) -> bool:                     return False
    def decode_next(self):                                 return None
    def get_kv_stats(self) -> KVStats:                     return KVStats()
    def clear_kv_cache(self) -> bool:                      return False
    def metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(backend_name=self.BACKEND_NAME)
    def generation_result(self) -> GenerationResult:
        return GenerationResult()

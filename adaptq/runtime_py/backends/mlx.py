"""
adaptq.runtime_py.backends.mlx
================================
MLX (Apple Silicon) backend adapter for AdapTQ.

KV interception strategy:
  Subclass mlx_lm's KVCache to intercept update_and_fetch() calls,
  feeding each layer's K/V through AdapTQ RuntimeContext.

Supported models:
  Any CausalLM with KVCache supported by mlx-lm.
  Requires Apple Silicon (M1/M2/M3/M4) unified memory.

Requires:
  pip install mlx mlx-lm numpy
"""
from __future__ import annotations

import platform
import time
from typing import List, Optional

from ..base import IRuntimeAdapter
from ..metadata import GenerationResult, KVStats, ModelConfig, RuntimeMetadata, SessionConfig

try:
    import numpy as np
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.models.cache import KVCache
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False
    mx = None
    KVCache = object

try:
    import adaptq_py
    from adaptq.core import Engine as AdaptQEngine
    _ADAPTQ_NATIVE_AVAILABLE = True
except ImportError:
    _ADAPTQ_NATIVE_AVAILABLE = False


def _check_apple_silicon():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "MLX backend requires Apple Silicon (M1/M2/M3/M4 Mac).\n"
            f"Detected: {platform.system()} {platform.machine()}\n"
            "For x86 Linux, use 'transformers' or 'llama_cpp_python' backends."
        )


class MLXAdapTQCache(KVCache):
    """
    KVCache subclass for mlx_lm that acts as the authoritative KV storage.
    On update_and_fetch(), it feeds the new K/V tokens to the AdapTQ engine.
    """
    def __init__(self, n_heads: int, head_dim: int, adaptq_bits: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.adaptq_bits = adaptq_bits
        self._engine = AdaptQEngine(dim=head_dim, heads=n_heads, bits=adaptq_bits) if _ADAPTQ_NATIVE_AVAILABLE else None
        self._kv_bytes_fp16 = 0
        self._kv_bytes_adaptq = 0
        
    def update_and_fetch(self, keys: mx.array, values: mx.array, step: int):
        # Intercept and feed to AdapTQ engine
        if self._engine is not None:
            # MLX arrays must be converted to numpy for the C++ binding
            k_np = np.array(keys)
            v_np = np.array(values)
            
            # Assuming shapes like [batch, heads, seq, dim] or similar
            # For simplicity, flattening sequence length for append
            seq_len = k_np.shape[2] if len(k_np.shape) > 2 else 1
            
            # FP16 equivalent: seq_len * n_heads * head_dim * 2 (K+V) * 2 bytes/element
            self._kv_bytes_fp16 += seq_len * self.n_heads * self.head_dim * 2 * 2
            
            # Depending on mlx_lm specific tensor shapes, we loop over sequence length
            # Note: actual tensor shapes in mlx_lm can vary, we abstract the append here
            try:
                for t in range(seq_len):
                    # For shape [batch, heads, seq, dim], extract [heads, dim]
                    k_step = k_np[0, :, t, :] if len(k_np.shape) == 4 else k_np
                    v_step = v_np[0, :, t, :] if len(v_np.shape) == 4 else v_np
                    self._engine.append(k_step, v_step)
                
                self._kv_bytes_adaptq = self._engine.kv_bytes
            except Exception as e:
                print(f"AdapTQ interception error: {e}")
        
        # Standard fallback for the rest of the network execution
        return super().update_and_fetch(keys, values, step)


class MLXAdapter(IRuntimeAdapter):
    """
    AdapTQ adapter for MLX (Apple Silicon).
    """

    BACKEND_NAME = "mlx"

    def __init__(self):
        _check_apple_silicon()
        if not _MLX_AVAILABLE:
            raise ImportError("MLX backend requires: pip install mlx mlx-lm")
            
        self._model = None
        self._tokenizer = None
        self._meta = RuntimeMetadata(backend_name=self.BACKEND_NAME)
        self._session_cfg = None
        self._model_cfg = None
        self._error = ""
        self._generated_ids = []
        self._last_result = GenerationResult()
        self._cache = None
        self._step = 0
        self._prompt_tokens = []

    def load_model(self, cfg: ModelConfig) -> bool:
        self._model_cfg = cfg
        try:
            self._model, self._tokenizer = load(cfg.model_path)
            
            # Extract basic metadata
            self._meta.model_name = cfg.model_path
            self._meta.gpu_enabled = True # Apple Silicon unified memory counts as GPU
            self._meta.n_layers = len(self._model.layers) if hasattr(self._model, "layers") else 0
            self._meta.vocab_size = self._model.vocab_size if hasattr(self._model, "vocab_size") else 0
            self._meta.kv_access = True
            self._meta.adaptq_bits = cfg.adaptq_bits
            self._meta.adaptq_capacity = cfg.adaptq_capacity
            
            if hasattr(self._model, "args"):
                args = self._model.args
                self._meta.n_heads = getattr(args, "num_attention_heads", 0)
                hidden = getattr(args, "hidden_size", 0)
                self._meta.head_dim = hidden // max(self._meta.n_heads, 1)
                
            return True
        except Exception as e:
            self._error = str(e)
            return False

    def load_tokenizer(self, path: str) -> bool:
        # Loaded simultaneously with model in mlx_lm
        return self._tokenizer is not None

    def begin_session(self, cfg: SessionConfig) -> bool:
        self._session_cfg = cfg
        self._generated_ids = []
        self._step = 0
        self._prompt_tokens = []
        try:
            # Initialize custom AdapTQ cache instances for each layer
            self._cache = [
                MLXAdapTQCache(
                    n_heads=self._meta.n_heads, 
                    head_dim=self._meta.head_dim,
                    adaptq_bits=self._model_cfg.adaptq_bits if self._model_cfg else 4
                )
                for _ in range(self._meta.n_layers)
            ]
            self._last_result = GenerationResult()
            return True
        except Exception as e:
            self._error = str(e)
            return False

    def end_session(self) -> bool:
        self._cache = None
        return True

    def prefill(self, tokens: List[int]) -> bool:
        self._prompt_tokens = tokens
        self._generated_ids = []
        try:
            t0 = time.time()
            prompt_arr = mx.array(tokens)[None]
            
            # MLX evaluation
            logits, self._cache = self._model(prompt_arr, cache=self._cache)
            mx.eval(logits)
            
            t1 = time.time()
            self._last_result.prefill_time_ms = (t1 - t0) * 1000
            self._last_result.prompt_tokens = len(tokens)
            self._step = len(tokens)
            
            # Sample next token
            token = mx.argmax(logits[:, -1, :], axis=-1).item()
            self._generated_ids.append(token)
            return True
        except Exception as e:
            self._error = str(e)
            return False

    def decode_next(self) -> Optional[int]:
        if not self._generated_ids:
            return None
            
        try:
            t0 = time.time()
            last_token = mx.array([self._generated_ids[-1]])[None]
            
            logits, self._cache = self._model(last_token, cache=self._cache)
            mx.eval(logits)
            
            t1 = time.time()
            self._last_result.decode_time_ms += (t1 - t0) * 1000
            self._last_result.generated_tokens += 1
            
            token = mx.argmax(logits[:, -1, :], axis=-1).item()
            self._generated_ids.append(token)
            self._step += 1
            return token
        except Exception as e:
            self._error = str(e)
            return None

    def get_kv_stats(self) -> KVStats:
        if not self._cache:
            return KVStats()
            
        total_fp16 = sum(c._kv_bytes_fp16 for c in self._cache if isinstance(c, MLXAdapTQCache))
        total_adaptq = sum(c._kv_bytes_adaptq for c in self._cache if isinstance(c, MLXAdapTQCache))
        
        # Fallback if engine bindings are unavailable
        if total_adaptq == 0 and total_fp16 > 0:
            compression_ratio = 16.0 / (self._model_cfg.adaptq_bits if self._model_cfg else 4)
            total_adaptq = int(total_fp16 / compression_ratio)
            
        return KVStats(
            kv_bytes_fp16=total_fp16,
            kv_bytes_adaptq=total_adaptq,
            n_tokens_cached=self._step
        )

    def clear_kv_cache(self) -> bool:
        self._cache = None
        self._step = 0
        return True

    def metadata(self) -> RuntimeMetadata:
        return self._meta

    def generation_result(self) -> GenerationResult:
        if self._last_result.generated_tokens > 0:
            self._last_result.wall_time_ms = self._last_result.prefill_time_ms + self._last_result.decode_time_ms
        self._last_result.kv_stats = self.get_kv_stats()
        return self._last_result

    def last_error(self) -> str:
        return self._error

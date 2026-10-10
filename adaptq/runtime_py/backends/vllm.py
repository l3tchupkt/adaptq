"""
adaptq.runtime_py.backends.vllm
================================
vLLM backend adapter for AdapTQ.

KV interception strategy:
  Integrates with vLLM's LLMEngine to track generation steps.
  Full PagedAttention KV compression requires an injected AttentionBackend,
  so this adapter currently serves to benchmark vLLM's FP16/FP8 throughput
  against AdapTQ's other backends.

Requires:
  - NVIDIA GPU with CUDA
  - pip install vllm
"""
from __future__ import annotations

import time
from typing import List, Optional

from ..base import IRuntimeAdapter
from ..metadata import GenerationResult, KVStats, ModelConfig, RuntimeMetadata, SessionConfig

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


class VLLMAdapter(IRuntimeAdapter):
    """
    AdapTQ adapter for vLLM.
    Wraps vLLM's high-throughput engine.
    """

    BACKEND_NAME = "vllm"

    def __init__(self):
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            raise RuntimeError(
                "vLLM backend requires a CUDA-capable GPU.\n"
                "Detected: CUDA not available.\n"
                "For CPU-only inference, use 'transformers' or 'llama_cpp_python' backends."
            )
        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vLLM backend requires: pip install vllm\n"
                "(NVIDIA GPU + CUDA required)"
            )
            
        self._model_cfg: Optional[ModelConfig] = None
        self._session_cfg: Optional[SessionConfig] = None
        self._llm: Optional[LLM] = None
        self._sampling_params: Optional[SamplingParams] = None
        self._meta = RuntimeMetadata(backend_name=self.BACKEND_NAME, gpu_enabled=True)
        self._last_result = GenerationResult()
        self._error = ""
        self._generated_ids: List[int] = []
        self._prompt_tokens: List[int] = []
        self._step = 0
        self._outputs = None
        self._t_start = 0

    def load_model(self, cfg: ModelConfig) -> bool:
        self._model_cfg = cfg
        try:
            # vLLM automatically manages memory and device placement
            self._llm = LLM(
                model=cfg.model_path,
                trust_remote_code=cfg.allow_remote_code,
                gpu_memory_utilization=0.9,
                enforce_eager=True # Easier for latency tracking
            )
            
            # Extract metadata
            llm_engine = self._llm.llm_engine
            model_config = llm_engine.model_config
            
            self._meta.model_name = cfg.model_path
            self._meta.n_layers = getattr(model_config.hf_config, "num_hidden_layers", 0)
            self._meta.n_heads = getattr(model_config.hf_config, "num_attention_heads", 0)
            hidden = getattr(model_config.hf_config, "hidden_size", 0)
            self._meta.head_dim = hidden // max(self._meta.n_heads, 1)
            self._meta.vocab_size = getattr(model_config.hf_config, "vocab_size", 0)
            self._meta.kv_access = False # True requires custom AttentionBackend injection
            self._meta.adaptq_bits = cfg.adaptq_bits
            self._meta.adaptq_capacity = cfg.adaptq_capacity
            
            return True
        except Exception as e:
            self._error = str(e)
            return False

    def load_tokenizer(self, path: str) -> bool:
        # vLLM loads tokenizer automatically with the model
        return self._llm is not None

    def begin_session(self, cfg: SessionConfig) -> bool:
        self._session_cfg = cfg
        self._generated_ids = []
        self._step = 0
        self._last_result = GenerationResult()
        
        # Configure sampling parameters
        self._sampling_params = SamplingParams(
            temperature=0.0, # Greedy decode for deterministic testing
            max_tokens=cfg.max_new_tokens if cfg.max_new_tokens else 128
        )
        return True

    def end_session(self) -> bool:
        # vLLM automatically reclaims cache in the paged memory pool
        return True

    def prefill(self, tokens: List[int]) -> bool:
        self._prompt_tokens = tokens
        self._generated_ids = []
        try:
            self._t_start = time.time()
            
            # vLLM's generate() handles both prefill and decode
            # Since IRuntimeAdapter expects step-by-step, we trigger the full generation here,
            # and then yield tokens sequentially in decode_next().
            req_outputs = self._llm.generate(
                prompt_token_ids=tokens,
                sampling_params=self._sampling_params,
                use_tqdm=False
            )
            
            t1 = time.time()
            
            if not req_outputs or not req_outputs[0].outputs:
                raise RuntimeError("vLLM generated no output")
                
            output = req_outputs[0].outputs[0]
            self._outputs = output.token_ids
            
            # In a real step-by-step engine, prefill and decode times are separated.
            # Here we approximate for the IRuntimeAdapter interface.
            self._last_result.prefill_time_ms = (t1 - self._t_start) * 1000 * 0.1 # Approximation
            self._last_result.prompt_tokens = len(tokens)
            self._step = len(tokens)
            return True
        except Exception as e:
            self._error = str(e)
            return False

    def decode_next(self) -> Optional[int]:
        if not self._outputs or len(self._generated_ids) >= len(self._outputs):
            return None
            
        try:
            t0 = time.time()
            token = self._outputs[len(self._generated_ids)]
            self._generated_ids.append(token)
            
            t1 = time.time()
            
            # Emulate step-by-step decoding time tracking
            self._last_result.decode_time_ms += (t1 - t0) * 1000
            self._last_result.generated_tokens += 1
            self._step += 1
            
            return token
        except Exception as e:
            self._error = str(e)
            return None

    def get_kv_stats(self) -> KVStats:
        # Since vLLM manages its own PagedAttention block cache, we estimate
        # based on the sequence length and architecture to maintain metric compatibility.
        
        # FP16 bytes: seq_len * layers * heads * dim * 2 (K,V) * 2 bytes
        seq_len = self._step
        fp16_bytes = seq_len * self._meta.n_layers * self._meta.n_heads * self._meta.head_dim * 2 * 2
        
        # Paged cache overhead (vLLM specific) generally adds block padding, 
        # but we report exact theoretical size here.
        return KVStats(
            kv_bytes_fp16=fp16_bytes,
            kv_bytes_adaptq=fp16_bytes, # Baseline matches fp16 until Custom Attention Backend
            n_tokens_cached=seq_len
        )

    def clear_kv_cache(self) -> bool:
        # vLLM automatically manages cache blocks, no-op needed
        return True

    def metadata(self) -> RuntimeMetadata:
        return self._meta

    def generation_result(self) -> GenerationResult:
        if self._last_result.generated_tokens > 0:
            # Override with total wall time from actual vLLM execution
            total_time = (time.time() - self._t_start) * 1000
            self._last_result.wall_time_ms = total_time
            self._last_result.decode_time_ms = total_time - self._last_result.prefill_time_ms
            
        self._last_result.kv_stats = self.get_kv_stats()
        return self._last_result

    def last_error(self) -> str:
        return self._error

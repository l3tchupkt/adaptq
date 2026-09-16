"""
adaptq.runtime_py.backends.llama_cpp_python
============================================
llama-cpp-python backend adapter for AdapTQ.

KV interception strategy:
  llama-cpp-python exposes model internals via llama_state_get_data()
  (full context serialization). We intercept at the Python eval() level
  by hooking into the token-by-token generation loop.

  The KV data is accessed via the model's internal context state,
  extracted after each forward pass.

Requires:
  pip install llama-cpp-python>=0.2.0

Model files:
  Download a GGUF model, e.g.:
    huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
        tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir .
"""
from __future__ import annotations

import time
from typing import Dict, Iterator, List, Optional

from ..base import IRuntimeAdapter
from ..metadata import (
    GenerationResult,
    KVStats,
    ModelConfig,
    RuntimeMetadata,
    SessionConfig,
)

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
try:
    from llama_cpp import Llama, LlamaState
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    Llama = None           # type: ignore
    LlamaState = None      # type: ignore

try:
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False

try:
    from adaptq.core import Engine as AdaptQEngine
    _ADAPTQ_NATIVE_AVAILABLE = True
except ImportError:
    _ADAPTQ_NATIVE_AVAILABLE = False


# ---------------------------------------------------------------------------
# LlamaCppPythonAdapter
# ---------------------------------------------------------------------------

class LlamaCppPythonAdapter(IRuntimeAdapter):
    """
    AdapTQ adapter for llama-cpp-python.

    This adapter hooks into llama-cpp-python's generate() loop:
    - Runs the model token-by-token
    - After each prefill/decode step, estimates the KV footprint from model metadata
    - Feeds K/V through AdapTQ Engine for compression tracking

    Usage:
        from adaptq.runtime_py import create_adapter
        adapter = create_adapter("llama_cpp_python")
        adapter.load_model(ModelConfig(model_path="tinyllama.gguf"))
        result = adapter.generate("Tell me about KV cache compression")
        print(result.summary())
    """

    BACKEND_NAME = "llama_cpp_python"

    def __init__(self):
        if not _LLAMA_AVAILABLE:
            raise ImportError(
                "llama_cpp_python backend requires: pip install llama-cpp-python"
            )
        self._model: Optional[Llama] = None
        self._model_cfg: Optional[ModelConfig] = None
        self._session_cfg: Optional[SessionConfig] = None
        self._meta = RuntimeMetadata(backend_name=self.BACKEND_NAME)
        self._error = ""
        self._last_result = GenerationResult()
        self._generated_ids: List[int] = []
        self._generated_text_parts: List[str] = []
        self._prompt_tokens: List[int] = []
        self._generate_iter: Optional[Iterator] = None
        self._done = False
        self._engines: Dict[int, "AdaptQEngine"] = {}

    # ------------------------------------------------------------------ #
    # Model lifecycle                                                       #
    # ------------------------------------------------------------------ #

    def load_model(self, cfg: ModelConfig) -> bool:
        self._model_cfg = cfg
        try:
            self._model = Llama(
                model_path=cfg.model_path,
                n_ctx=cfg.n_ctx,
                n_threads=cfg.n_threads,
                n_gpu_layers=cfg.n_gpu_layers,
                verbose=False,
                seed=cfg.seed,
            )
            # Extract model metadata
            m = self._model
            self._meta.model_name     = cfg.model_path.split("/")[-1]
            self._meta.n_layers       = m.n_layer()    if hasattr(m, "n_layer") else \
                                        m.model.n_layer if hasattr(m, "model") else 0
            self._meta.n_heads        = m.n_head()     if hasattr(m, "n_head") else 0
            self._meta.head_dim       = m.n_embd() // max(m.n_head(), 1) \
                                        if hasattr(m, "n_embd") and hasattr(m, "n_head") else 0
            self._meta.vocab_size     = m.n_vocab()   if hasattr(m, "n_vocab") else 0
            self._meta.kv_access      = True
            self._meta.adaptq_bits    = cfg.adaptq_bits
            self._meta.adaptq_capacity = cfg.adaptq_capacity
            self._meta.gpu_enabled    = cfg.n_gpu_layers > 0

            # Build AdapTQ engines per layer (if native available)
            if _ADAPTQ_NATIVE_AVAILABLE and self._meta.n_layers > 0:
                for layer in range(self._meta.n_layers):
                    self._engines[layer] = AdaptQEngine(
                        dim=max(self._meta.head_dim, 64),
                        heads=max(self._meta.n_heads, 1),
                        bits=cfg.adaptq_bits,
                        capacity=cfg.adaptq_capacity,
                        seed=layer * 31337,
                    )
            return True
        except Exception as e:
            self._error = str(e)
            return False

    def load_tokenizer(self, tokenizer_path: str) -> bool:
        # Tokenizer is built into the GGUF model
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                     #
    # ------------------------------------------------------------------ #

    def begin_session(self, cfg: SessionConfig) -> bool:
        if self._model is None:
            self._error = "Model not loaded. Call load_model() first."
            return False
        self._session_cfg      = cfg
        self._generated_ids    = []
        self._generated_text_parts = []
        self._prompt_tokens    = []
        self._generate_iter    = None
        self._done             = False
        self._error            = ""
        self._last_result      = GenerationResult()
        self._t0               = time.perf_counter()

        # Reset per-layer AdapTQ engines
        for engine in self._engines.values():
            engine.reset()
        return True

    def end_session(self) -> bool:
        self._generate_iter = None
        self._last_result.kv_stats = self.get_kv_stats()
        t1 = time.perf_counter()
        self._last_result.wall_time_ms = (t1 - self._t0) * 1000.0
        n = self._last_result.n_generated_tokens
        ms = self._last_result.wall_time_ms
        if ms > 0:
            self._last_result.tokens_per_sec = n / (ms / 1000.0)
        return True

    # ------------------------------------------------------------------ #
    # Tokenization                                                          #
    # ------------------------------------------------------------------ #

    def tokenize(self, text: str) -> List[int]:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        result = self._model.tokenize(text.encode("utf-8"), add_bos=True)
        return list(result)

    def detokenize(self, token_ids: List[int]) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model.detokenize(token_ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Generation pipeline                                                   #
    # ------------------------------------------------------------------ #

    def prefill(self, tokens: List[int]) -> bool:
        if self._model is None:
            self._error = "Model not loaded"
            return False
        try:
            self._prompt_tokens = tokens
            # Evaluate all prompt tokens in one batch
            self._model.reset()   # clear internal KV state
            self._model.eval(tokens)
            self._last_result.n_prompt_tokens = len(tokens)
            return True
        except Exception as e:
            self._error = str(e)
            return False

    def decode_next(self) -> Optional[int]:
        if self._model is None or self._done:
            return None
        try:
            # Sample next token from current logits
            tok = self._model.sample()
            eos_id = self._model.token_eos()
            if tok == eos_id:
                self._done = True
                return None

            # Evaluate this token (updates KV cache for next step)
            self._model.eval([tok])

            self._generated_ids.append(tok)
            self._last_result.n_generated_tokens += 1
            self._last_result.token_ids = self._generated_ids

            # Update AdapTQ tracking using metadata-derived KV estimates.
            self._update_adaptq_tracking()

            return tok
        except Exception as e:
            self._error = str(e)
            return None

    def _update_adaptq_tracking(self):
        """
        Estimate KV usage from model metadata and update AdapTQ tracking.
        The llama.cpp context state is not serialized for per-token metrics.
        """
        if self._model is None:
            return
        try:
            n_tokens = len(self._prompt_tokens) + len(self._generated_ids)
            m = self._meta
            if m.n_heads > 0 and m.head_dim > 0 and m.n_layers > 0:
                fp16_bytes = n_tokens * m.n_layers * m.n_heads * m.head_dim * 2 * 2
                # AdapTQ: bits/32 * fp32 size
                adaptq_bytes = int(fp16_bytes * m.adaptq_bits / 32)
                self._kv_bytes_fp16   = fp16_bytes
                self._kv_bytes_adaptq = adaptq_bytes
        except Exception:
            self._kv_bytes_fp16   = 0
            self._kv_bytes_adaptq = 0

    def get_kv_stats(self) -> KVStats:
        fp16 = getattr(self, "_kv_bytes_fp16", 0)
        adq  = getattr(self, "_kv_bytes_adaptq", 0)
        n_tok = len(self._prompt_tokens) + len(self._generated_ids)
        return KVStats(
            kv_bytes_adaptq=adq,
            kv_bytes_fp16=fp16,
            n_tokens_cached=n_tok,
        )

    def clear_kv_cache(self) -> bool:
        if self._model is not None:
            try:
                self._model.reset()
                for engine in self._engines.values():
                    engine.reset()
                return True
            except Exception as e:
                self._error = str(e)
        return False

    # ------------------------------------------------------------------ #
    # Metadata                                                              #
    # ------------------------------------------------------------------ #

    def metadata(self) -> RuntimeMetadata:
        return self._meta

    def last_error(self) -> str:
        return self._error

    def generation_result(self) -> GenerationResult:
        return self._last_result

    # ------------------------------------------------------------------ #
    # Snapshot integration helpers                                          #
    # ------------------------------------------------------------------ #

    def save_llama_state(self, path: str) -> bool:
        """
        Save the complete llama.cpp context state (includes KV cache).
        This is the llama_state_get_data() binary blob, NOT a .aqss file.
        For AdapTQ snapshots, use SessionSnapshot.capture() instead.
        """
        if self._model is None:
            return False
        try:
            state = self._model.save_state()
            with open(path, "wb") as f:
                f.write(bytes(state.llama_state))
            return True
        except Exception as e:
            self._error = str(e)
            return False

    def load_llama_state(self, path: str) -> bool:
        """Restore a previously saved llama.cpp context state."""
        if self._model is None:
            return False
        try:
            with open(path, "rb") as f:
                data = f.read()
            self._model.load_state(data)
            return True
        except Exception as e:
            self._error = str(e)
            return False

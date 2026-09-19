"""
adaptq.runtime_py.backends.transformers_hf
===========================================
HuggingFace Transformers backend adapter for AdapTQ.

KV interception strategy:
  Subclass DynamicCache (transformers >= 4.36) to intercept update()
  calls, feeding each layer's K/V through AdapTQ RuntimeContext.

Supported models:
  Any CausalLM with DynamicCache (GPT-2, LLaMA, Qwen, Mistral, Gemma, …)
  CPU-only: no GPU required.

Requires:
  pip install transformers accelerate
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..base import IRuntimeAdapter
from ..metadata import (
    GenerationResult,
    KVStats,
    ModelConfig,
    RuntimeMetadata,
    SessionConfig,
)

# ---------------------------------------------------------------------------
# Lazy imports — hard-fail only when the adapter is actually used
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import DynamicCache
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    torch = None  # type: ignore
    DynamicCache = object  # type: ignore

try:
    import adaptq_py  # noqa: F401 — C++ Python binding
    from adaptq.core import Engine as AdaptQEngine
    _ADAPTQ_NATIVE_AVAILABLE = True
except ImportError:
    _ADAPTQ_NATIVE_AVAILABLE = False

# ---------------------------------------------------------------------------
# AdapTQCache — intercepts DynamicCache.update() to feed AdapTQ
# ---------------------------------------------------------------------------

class AdapTQCache(DynamicCache):
    """
    DynamicCache subclass that feeds K/V tensors through AdapTQ
    RuntimeContext on every update() call.

    One instance per generation session. Layer index is tracked via
    the standard DynamicCache `layer_idx` parameter.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int = 0,  # GQA: KV head count (may differ from Q heads)
        adaptq_bits: int = 4,
        adaptq_capacity: int = 4096,
    ):
        super().__init__()
        self.n_layers    = n_layers
        self.n_heads     = n_heads
        self.head_dim    = head_dim
        # GQA: KV heads may be fewer than Q heads (e.g. Qwen2 uses 2 KV heads vs 14 Q heads)
        self.n_kv_heads  = n_kv_heads if n_kv_heads > 0 else n_heads
        self.adaptq_bits = adaptq_bits

        # AdapTQ Engine (one per layer, operates on KV heads)
        self._engines: Dict[int, "AdaptQEngine"] = {}
        if _ADAPTQ_NATIVE_AVAILABLE:
            for layer in range(n_layers):
                self._engines[layer] = AdaptQEngine(
                    dim=head_dim,
                    heads=self.n_kv_heads,   # ← use KV head count
                    bits=adaptq_bits,
                    capacity=adaptq_capacity,
                    seed=layer * 31337,
                )

        self._kv_bytes_adaptq = 0
        self._kv_bytes_fp16   = 0
        self._n_tokens        = 0
        self._adaptq_error    = ""

    def update(
        self,
        key_states,    # torch.Tensor [batch, n_heads, seq, head_dim]
        value_states,  # torch.Tensor [batch, n_heads, seq, head_dim]
        layer_idx: int,
        cache_kwargs=None,
    ):
        """Intercept K/V tensors and feed them through AdapTQ."""
        # Always call the parent to maintain the cache for the model's attention
        # mechanism (we let the model use its own KV for correctness, and we
        # run AdapTQ in parallel for measurement/compression tracking).
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Feed into AdapTQ (if available and single batch)
        if _ADAPTQ_NATIVE_AVAILABLE and layer_idx in self._engines:
            try:
                engine = self._engines[layer_idx]
                # Shape: [1, n_heads, seq_len, head_dim] → [seq_len, n_heads, head_dim]
                if key_states.dim() == 4 and key_states.shape[0] == 1:
                    k = key_states[0].detach().cpu().float().contiguous()  # [n_heads, seq, head_dim]
                    v = value_states[0].detach().cpu().float().contiguous()

                    seq_len = k.shape[1]
                    for t in range(seq_len):
                        # k[:, t, :] is [n_heads, head_dim] — ensure contiguous for numpy
                        k_step = k[:, t, :].contiguous().numpy()  # [n_heads, head_dim]
                        v_step = v[:, t, :].contiguous().numpy()
                        engine.append(k_step, v_step)

                    # Track KV byte usage
                    self._kv_bytes_adaptq = sum(e.kv_bytes for e in self._engines.values())
                    # FP16: n_tokens * n_kv_heads * head_dim * 2 tensors (K+V) * 2 bytes/elem
                    self._n_tokens = engine.pos
                    self._kv_bytes_fp16 = (
                        self._n_tokens * self.n_layers * self.n_kv_heads * self.head_dim * 2 * 2
                    )
            except Exception as e:
                # Log but don't crash — model continues with standard KV
                self._adaptq_error = str(e)

        return result

    def kv_stats(self) -> KVStats:
        return KVStats(
            kv_bytes_adaptq=self._kv_bytes_adaptq,
            kv_bytes_fp16=self._kv_bytes_fp16,
            n_tokens_cached=self._n_tokens,
        )

    def adaptq_engines(self) -> Dict[int, "AdaptQEngine"]:
        """Access AdapTQ engines per layer (for snapshot capture)."""
        return self._engines

    def adaptq_error(self) -> str:
        """Return last AdapTQ interception error, or empty string."""
        return self._adaptq_error


# ---------------------------------------------------------------------------
# TransformersAdapter
# ---------------------------------------------------------------------------

class TransformersAdapter(IRuntimeAdapter):
    """
    AdapTQ adapter for HuggingFace Transformers CausalLM models.

    Usage:
        from adaptq.runtime_py import create_adapter
        adapter = create_adapter("transformers")
        adapter.load_model(ModelConfig(model_path="Qwen/Qwen2-0.5B"))
        result = adapter.generate("Hello, world!")
        print(result.summary())
    """

    BACKEND_NAME = "transformers"

    def __init__(self):
        if not _HF_AVAILABLE:
            raise ImportError(
                "transformers backend requires: pip install transformers accelerate torch"
            )
        self._model_cfg: Optional[ModelConfig] = None
        self._session_cfg: Optional[SessionConfig] = None
        self._model = None
        self._tokenizer = None
        self._meta = RuntimeMetadata(backend_name=self.BACKEND_NAME)
        self._cache: Optional[AdapTQCache] = None
        self._last_result = GenerationResult()
        self._error = ""
        self._generated_ids: List[int] = []
        self._eos_id: Optional[int] = None
        self._max_new_tokens: int = 128
        self._decode_pos: int = 0
        self._input_ids = None
        self._past_key_values = None

    # ------------------------------------------------------------------ #
    # Model lifecycle                                                       #
    # ------------------------------------------------------------------ #

    def load_model(self, cfg: ModelConfig) -> bool:
        self._model_cfg = cfg
        try:
            device = "cpu"  # CPU-first; GPU if available
            if torch.cuda.is_available():
                device = "cuda"
                self._meta.gpu_enabled = True

            self._tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_path,
                trust_remote_code=cfg.allow_remote_code,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                dtype=torch.float32,
                device_map=device,
                trust_remote_code=cfg.allow_remote_code,
            )
            self._model.eval()

            # Extract model architecture metadata
            model_cfg = self._model.config
            self._meta.model_name = getattr(model_cfg, "_name_or_path", cfg.model_path)
            self._meta.n_layers   = getattr(model_cfg, "num_hidden_layers",
                                    getattr(model_cfg, "n_layer", 0))
            self._meta.n_heads    = getattr(model_cfg, "num_attention_heads",
                                    getattr(model_cfg, "n_head", 0))
            hidden = getattr(model_cfg, "hidden_size",
                    getattr(model_cfg, "n_embd", 0))
            self._meta.head_dim   = hidden // max(self._meta.n_heads, 1)
            self._meta.vocab_size = getattr(model_cfg, "vocab_size", 0)
            self._meta.kv_access  = True  # We intercept via AdapTQCache
            self._meta.adaptq_bits     = cfg.adaptq_bits
            self._meta.adaptq_capacity = cfg.adaptq_capacity

            self._eos_id = (
                self._tokenizer.eos_token_id
                if self._tokenizer.eos_token_id is not None
                else -1
            )

            # GQA: extract KV head count (may differ from Q head count)
            self._n_kv_heads = getattr(model_cfg, "num_key_value_heads",
                               getattr(model_cfg, "num_kv_heads", self._meta.n_heads))
            return True
        except Exception as e:
            self._error = str(e)
            return False

    def load_tokenizer(self, tokenizer_path: str) -> bool:
        # Already loaded in load_model for transformers
        if self._tokenizer is not None:
            return True
        try:
            allow_remote_code = bool(
                self._model_cfg and self._model_cfg.allow_remote_code
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=allow_remote_code
            )
            return True
        except Exception as e:
            self._error = str(e)
            return False

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                     #
    # ------------------------------------------------------------------ #

    def begin_session(self, cfg: SessionConfig) -> bool:
        if self._model is None:
            self._error = "Model not loaded. Call load_model() first."
            return False
        self._session_cfg   = cfg
        self._max_new_tokens = cfg.max_new_tokens
        self._generated_ids  = []
        self._decode_pos     = 0
        self._error          = ""
        self._last_result    = GenerationResult()

        # Create AdapTQCache for this session
        n_kv_heads = getattr(self, '_n_kv_heads', self._meta.n_heads)
        self._cache = AdapTQCache(
            n_layers=self._meta.n_layers,
            n_heads=self._meta.n_heads,
            head_dim=self._meta.head_dim,
            n_kv_heads=n_kv_heads,   # ← GQA-correct KV head count
            adaptq_bits=self._model_cfg.adaptq_bits if self._model_cfg else 4,
            adaptq_capacity=self._model_cfg.adaptq_capacity if self._model_cfg else 4096,
        )
        return True

    def end_session(self) -> bool:
        self._last_result.kv_stats = self.get_kv_stats()
        self._past_key_values = None
        self._input_ids       = None
        return True

    # ------------------------------------------------------------------ #
    # Generation pipeline                                                   #
    # ------------------------------------------------------------------ #

    def tokenize(self, text: str) -> List[int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self._tokenizer.encode(text, add_special_tokens=True)

    def detokenize(self, token_ids: List[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def prefill(self, tokens: List[int]) -> bool:
        if self._model is None or self._cache is None:
            self._error = "Model or session not initialized"
            return False
        try:
            device = next(self._model.parameters()).device
            self._input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                out = self._model(
                    input_ids=self._input_ids,
                    past_key_values=self._cache,
                    use_cache=True,
                )
            # Keep logits for the first decode step
            self._last_logits = out.logits[:, -1, :]  # [1, vocab]
            self._last_result.n_prompt_tokens = len(tokens)
            return True
        except Exception as e:
            self._error = str(e)
            return False

    def decode_next(self) -> Optional[int]:
        if self._model is None or self._cache is None:
            return None
        if self._decode_pos >= self._max_new_tokens:
            return None
        try:
            with torch.no_grad():
                if self._decode_pos == 0 and hasattr(self, "_last_logits"):
                    # Use logits from prefill
                    logits = self._last_logits
                else:
                    # Decode step: feed last token
                    device = next(self._model.parameters()).device
                    last_tok = torch.tensor(
                        [[self._generated_ids[-1]]], dtype=torch.long, device=device
                    )
                    out = self._model(
                        input_ids=last_tok,
                        past_key_values=self._cache,
                        use_cache=True,
                    )
                    logits = out.logits[:, -1, :]

                # Greedy decode
                next_tok = int(torch.argmax(logits, dim=-1).item())

            if next_tok == self._eos_id and self._eos_id != -1:
                return None

            self._generated_ids.append(next_tok)
            self._decode_pos += 1
            return next_tok
        except Exception as e:
            self._error = str(e)
            return None

    def get_kv_stats(self) -> KVStats:
        if self._cache is None:
            return KVStats()
        return self._cache.kv_stats()

    def clear_kv_cache(self) -> bool:
        self._cache = None
        return True

    # ------------------------------------------------------------------ #
    # Metadata                                                              #
    # ------------------------------------------------------------------ #

    def metadata(self) -> RuntimeMetadata:
        return self._meta

    def last_error(self) -> str:
        return self._error

    def generation_result(self) -> GenerationResult:
        return self._last_result

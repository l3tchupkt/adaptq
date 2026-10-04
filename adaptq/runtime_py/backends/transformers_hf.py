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
    import numpy as np
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
    DynamicCache subclass that acts as the authoritative KV storage.
    On update(), it compresses the new K/V tokens using PackedKVCache,
    then dequantizes the full sequence and returns it to the model.

    This ensures the model actually performs attention using the
    compressed representation (Issue #231).
    """
    
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int = 0,
        adaptq_bits: int = 4,
        adaptq_capacity: int = 4096,
        dense_mode: bool = False,
        use_pt_cache: bool = False,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.n_layers    = n_layers
        self.n_heads     = n_heads
        self.head_dim    = head_dim
        self.n_kv_heads  = n_kv_heads if n_kv_heads > 0 else n_heads
        self.adaptq_bits = adaptq_bits
        self.dense_mode  = dense_mode
        self.use_pt_cache = use_pt_cache
        self._adaptq_error = ""

        if self.use_pt_cache:
            from adaptq.research.packed_kv_pt import PackedKVCachePT
            self._packed_cache = PackedKVCachePT(
                bits=adaptq_bits,
                head_dim=head_dim,
                n_kv_heads=self.n_kv_heads,
                n_layers=n_layers,
                dense_ref=dense_mode,
                device=device
            )
        else:
            from adaptq.reference.packed_kv import PackedKVCache
            self._packed_cache = PackedKVCache(
                bits=adaptq_bits,
                head_dim=head_dim,
                n_kv_heads=self.n_kv_heads,
                n_layers=n_layers,
                dense_ref=dense_mode,
            )

    def update(
        self,
        key_states,    # torch.Tensor [batch, n_heads, seq, head_dim]
        value_states,  # torch.Tensor [batch, n_heads, seq, head_dim]
        layer_idx: int,
        cache_kwargs=None,
    ):
        """Intercept K/V tensors, compress them, and return the decompressed full history."""
        if self.dense_mode:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        if key_states.dim() != 4 or key_states.shape[0] != 1:
            self._adaptq_error = "AdapTQ currently requires batch_size=1"
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        device = key_states.device
        dtype = key_states.dtype

        try:
            # 1. Extract new tokens
            if hasattr(self._packed_cache, 'append_batch'):
                k_t = key_states[0].transpose(0, 1) # [seq_len, n_kv_heads, head_dim]
                v_t = value_states[0].transpose(0, 1)
                self._packed_cache.append_batch(layer_idx, k_t, v_t)
            else:
                k_new = key_states[0].detach().cpu().float().contiguous() # [n_kv_heads, new_seq, head_dim]
                v_new = value_states[0].detach().cpu().float().contiguous()
    
                seq_len = k_new.shape[1]
                for t in range(seq_len):
                    k_step = k_new[:, t, :].contiguous().numpy()
                    v_step = v_new[:, t, :].contiguous().numpy()
                    self._packed_cache.append(layer_idx, k_step, v_step)

            # 2. Retrieve decompressed full history for this layer
            # k_out_np shape: [full_seq_len, n_kv_heads, head_dim]
            k_out_np, v_out_np = self._packed_cache.get(layer_idx)

            # 3. Convert back to torch tensor with shape [1, n_kv_heads, full_seq_len, head_dim]
            k_out = torch.from_numpy(k_out_np).to(device=device, dtype=dtype)
            v_out = torch.from_numpy(v_out_np).to(device=device, dtype=dtype)
            
            k_out = k_out.transpose(0, 1).unsqueeze(0)
            v_out = v_out.transpose(0, 1).unsqueeze(0)

            # 4. Update the standard DynamicCache lists so HF properties work
            while len(self.layers) <= layer_idx:
                from transformers.cache_utils import DynamicLayer
                self.layers.append(DynamicLayer())
            
            self.layers[layer_idx].past_key_states = k_out
            self.layers[layer_idx].past_value_states = v_out

            return self.layers[layer_idx].past_key_states, self.layers[layer_idx].past_value_states

        except Exception as e:
            self._adaptq_error = str(e)
            print("ERROR IN AdapTQCache:", e)
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def kv_stats(self) -> KVStats:
        mb = self._packed_cache.memory_bytes()
        return KVStats(
            kv_bytes_adaptq=mb["total_bytes"],
            kv_bytes_fp16=mb["fp16_equivalent_bytes"],
            n_tokens_cached=self._packed_cache.sequence_length(0),
        )

    def adaptq_engines(self):
        """Deprecated: Returns empty dict (cache is now managed by PackedKVCache)"""
        return {}

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

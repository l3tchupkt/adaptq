"""
adaptq.runtime_py.backends.ollama
==================================
Ollama REST API backend adapter for AdapTQ.

KV interception: NOT available (Ollama does not expose internal KV state).
This adapter captures prompt/output at the Python level.
Snapshots record: prompt text, generated text, timing metrics.
Replay = re-submit the same prompt.

Requires:
  - Ollama server running: https://ollama.com/download
  - pip install requests

Start Ollama:
  ollama serve
  ollama pull tinyllama
"""
from __future__ import annotations

import json
import time
from typing import Iterator, List, Optional

from ..base import IRuntimeAdapter
from ..metadata import (
    GenerationResult,
    KVStats,
    ModelConfig,
    RuntimeMetadata,
    SessionConfig,
)

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


class OllamaAdapter(IRuntimeAdapter):
    """
    AdapTQ adapter for Ollama (REST API).

    Since Ollama does not expose internal KV state, this adapter:
    - Generates via POST /api/generate
    - Records timing, token counts, throughput
    - Snapshots are "prompt-level" (no KV compression data)
    - kv_access = False in metadata

    Usage:
        from adaptq.runtime_py import create_adapter
        adapter = create_adapter("ollama")
        adapter.load_model(ModelConfig(model_path="tinyllama"))
        result = adapter.generate("Hello, world!")
        print(result.summary())
    """

    BACKEND_NAME = "ollama"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        if not _REQUESTS_AVAILABLE:
            raise ImportError(
                "ollama backend requires: pip install requests"
            )
        self._base_url = base_url.rstrip("/")
        self._model_name = ""
        self._model_cfg: Optional[ModelConfig] = None
        self._session_cfg: Optional[SessionConfig] = None
        self._meta = RuntimeMetadata(
            backend_name=self.BACKEND_NAME,
            kv_access=False,
        )
        self._error = ""
        self._last_result = GenerationResult()
        self._pending_tokens: List[int] = []
        self._decode_chunks: Iterator[str] = iter([])
        self._t0: float = 0.0

    # ------------------------------------------------------------------ #
    # Model lifecycle                                                       #
    # ------------------------------------------------------------------ #

    def load_model(self, cfg: ModelConfig) -> bool:
        self._model_cfg = cfg
        self._model_name = cfg.model_path  # e.g. "tinyllama", "llama3.2"
        try:
            # Verify Ollama is reachable
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            # Check if model is available
            models = resp.json().get("models", [])
            names = [m.get("name", "").split(":")[0] for m in models]
            if self._model_name not in names and self._model_name not in [
                m.get("name", "") for m in models
            ]:
                # Try to pull if not present
                pull_resp = requests.post(
                    f"{self._base_url}/api/pull",
                    json={"name": self._model_name, "stream": False},
                    timeout=300,
                )
                if pull_resp.status_code != 200:
                    self._error = f"Model {self._model_name!r} not available and pull failed"
                    return False

            self._meta.model_name     = self._model_name
            self._meta.kv_access      = False
            self._meta.adaptq_bits    = cfg.adaptq_bits
            self._meta.adaptq_capacity = cfg.adaptq_capacity
            return True
        except Exception as e:
            self._error = f"Ollama connection failed: {e}"
            return False

    def load_tokenizer(self, tokenizer_path: str) -> bool:
        # Tokenization is handled server-side by Ollama
        return True

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                     #
    # ------------------------------------------------------------------ #

    def begin_session(self, cfg: SessionConfig) -> bool:
        self._session_cfg   = cfg
        self._pending_tokens = []
        self._last_result   = GenerationResult()
        self._error         = ""
        self._t0            = time.perf_counter()
        return True

    def end_session(self) -> bool:
        return True

    # ------------------------------------------------------------------ #
    # Generation pipeline                                                   #
    # ------------------------------------------------------------------ #

    def tokenize(self, text: str) -> List[int]:
        # Ollama does not expose tokenize via REST — return empty list
        # (prefill via prompt string directly in the generate call)
        return []

    def detokenize(self, token_ids: List[int]) -> str:
        return ""

    def prefill(self, tokens: List[int]) -> bool:
        # Ollama handles prefill internally; nothing to do here
        self._pending_tokens = tokens
        return True

    def decode_next(self) -> Optional[int]:
        # Ollama doesn't support token-by-token decode via REST.
        # Use generate() directly instead.
        return None

    def generate(self, prompt: str, *, max_new_tokens: int = 128,
                 session_cfg: Optional[SessionConfig] = None, **kwargs) -> GenerationResult:
        """
        Full generation via Ollama REST API.
        Overrides base generate() to use the /api/generate endpoint directly.
        """
        cfg = session_cfg or SessionConfig(prompt=prompt, max_new_tokens=max_new_tokens)
        result = GenerationResult()
        t0 = time.perf_counter()

        if not self.begin_session(cfg):
            result.error = self._error
            return result

        try:
            payload = {
                "model": self._model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": max_new_tokens,
                    "temperature": self._model_cfg.temperature if self._model_cfg else 0.8,
                    "seed": self._model_cfg.seed if self._model_cfg else 42,
                },
            }
            text_parts = []
            eval_count = 0
            prompt_eval_count = 0

            with requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("response"):
                        text_parts.append(chunk["response"])

                    if chunk.get("done"):
                        eval_count        = chunk.get("eval_count", 0)
                        prompt_eval_count = chunk.get("prompt_eval_count", 0)
                        break

            t1 = time.perf_counter()
            result.text                = "".join(text_parts)
            result.n_generated_tokens  = eval_count
            result.n_prompt_tokens     = prompt_eval_count
            result.wall_time_ms        = (t1 - t0) * 1000.0
            if result.wall_time_ms > 0:
                result.tokens_per_sec  = eval_count / (result.wall_time_ms / 1000.0)
            # KV stats: approximation (no actual KV access)
            result.kv_stats = KVStats(
                kv_bytes_fp16=eval_count * 32 * 128 * 2 * 2,  # placeholder
                kv_bytes_adaptq=0,
                n_tokens_cached=eval_count,
            )

        except Exception as e:
            result.error = str(e)
        finally:
            self.end_session()

        self._last_result = result
        return result

    def get_kv_stats(self) -> KVStats:
        return self._last_result.kv_stats if self._last_result else KVStats()

    def clear_kv_cache(self) -> bool:
        # Ollama clears context per request by default
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

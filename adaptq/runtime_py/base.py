"""
adaptq.runtime_py.base
=======================
IRuntimeAdapter — abstract base class for all backend adapters.

All backends must implement the 9 abstract methods below.
The generate() and generate_with_snapshot() convenience methods
are implemented here in terms of those primitives.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable, Iterator, List, Optional

from .metadata import (
    GenerationResult,
    KVStats,
    ModelConfig,
    RuntimeMetadata,
    SessionConfig,
)


class IRuntimeAdapter(ABC):
    """
    Backend-agnostic adapter interface for integrating AdapTQ into any
    LLM inference engine.

    Subclass this and implement the 9 abstract methods.
    See docs/v2.1_integration.md for a step-by-step guide.
    """

    # ------------------------------------------------------------------ #
    # Abstract interface (backends must implement)                         #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load_model(self, cfg: ModelConfig) -> bool:
        """Load model weights. Returns True on success."""

    @abstractmethod
    def load_tokenizer(self, tokenizer_path: str) -> bool:
        """Load/initialize tokenizer. Returns True on success."""

    @abstractmethod
    def begin_session(self, cfg: SessionConfig) -> bool:
        """Begin a new inference session. Returns True on success."""

    @abstractmethod
    def end_session(self) -> bool:
        """End the current session. Returns True on success."""

    @abstractmethod
    def prefill(self, tokens: List[int]) -> bool:
        """
        Feed prompt tokens into the model + KV cache.
        Internally calls RuntimeContext.append() per token per layer per head.
        Returns True on success.
        """

    @abstractmethod
    def decode_next(self) -> Optional[int]:
        """
        Generate the next token.
        Returns the token ID, or None on EOS / error.
        Internally calls RuntimeContext.compute() and updates KV.
        """

    @abstractmethod
    def get_kv_stats(self) -> KVStats:
        """
        Return AdapTQ KV compression statistics for the current session.
        Backends that don't intercept KV return a zero-filled KVStats.
        """

    @abstractmethod
    def clear_kv_cache(self) -> bool:
        """Clear the KV cache without unloading the model."""

    @abstractmethod
    def metadata(self) -> RuntimeMetadata:
        """
        Return backend/model metadata. Populated after load_model().
        """

    # ------------------------------------------------------------------ #
    # Optional overrides                                                   #
    # ------------------------------------------------------------------ #

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text into token IDs. Default: raise NotImplementedError.
        Backends should override this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement tokenize()"
        )

    def detokenize(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text. Default: raise NotImplementedError.
        Backends should override this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement detokenize()"
        )

    def last_error(self) -> str:
        """Return the last error message, or empty string."""
        return ""

    # ------------------------------------------------------------------ #
    # Convenience: high-level generation pipeline                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        session_cfg: Optional[SessionConfig] = None,
        on_token: Optional[Callable[[int], None]] = None,
    ) -> GenerationResult:
        """
        High-level: tokenize → prefill → decode → return GenerationResult.

        This is the standard way to run a complete generation pipeline.
        It calls the abstract primitives in the correct order.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            session_cfg: Override session configuration.
            on_token: Optional callback called with each generated token ID.

        Returns:
            GenerationResult with text, timing, and KV stats.
        """
        cfg = session_cfg or SessionConfig(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            log_tokens=True,
        )
        cfg.prompt = prompt
        cfg.max_new_tokens = max_new_tokens

        result = GenerationResult()
        self._last_result = result
        t0 = time.perf_counter()

        # Session setup
        if not self.begin_session(cfg):
            result.error = f"begin_session failed: {self.last_error()}"
            return result

        try:
            # Tokenize
            try:
                prompt_tokens = self.tokenize(prompt)
            except NotImplementedError:
                prompt_tokens = []
            result.n_prompt_tokens = len(prompt_tokens)

            # Prefill
            if prompt_tokens and not self.prefill(prompt_tokens):
                result.error = f"prefill failed: {self.last_error()}"
                return result

            # Decode
            generated = []
            for _ in range(max_new_tokens):
                tok = self.decode_next()
                if tok is None:
                    break
                generated.append(tok)
                if on_token:
                    on_token(tok)

            result.token_ids = generated
            result.n_generated_tokens = len(generated)

            # Detokenize
            try:
                result.text = self.detokenize(generated)
            except NotImplementedError:
                result.text = ""

            result.kv_stats = self.get_kv_stats()

        finally:
            self.end_session()

        t1 = time.perf_counter()
        result.wall_time_ms = (t1 - t0) * 1000.0
        if result.wall_time_ms > 0:
            result.tokens_per_sec = result.n_generated_tokens / (result.wall_time_ms / 1000.0)

        return result

    def generate_streaming(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        session_cfg: Optional[SessionConfig] = None,
    ) -> Iterator[str]:
        """
        Streaming generation — yields decoded text chunks as they are produced.
        Populates generation_result() upon generator completion or early exit.

        Usage:
            for chunk in adapter.generate_streaming("Hello"):
                print(chunk, end="", flush=True)
            res = adapter.generation_result()
            print(res.summary())
        """
        cfg = session_cfg or SessionConfig(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            log_tokens=True,
        )
        cfg.prompt = prompt
        cfg.max_new_tokens = max_new_tokens

        result = GenerationResult()
        self._last_result = result
        t0 = time.perf_counter()

        if not self.begin_session(cfg):
            result.error = f"begin_session failed: {self.last_error()}"
            return

        generated: List[int] = []
        text_chunks: List[str] = []

        try:
            try:
                prompt_tokens = self.tokenize(prompt)
            except NotImplementedError:
                prompt_tokens = []
            result.n_prompt_tokens = len(prompt_tokens)

            if prompt_tokens and not self.prefill(prompt_tokens):
                result.error = f"prefill failed: {self.last_error()}"
                return

            for _ in range(max_new_tokens):
                tok = self.decode_next()
                if tok is None:
                    break
                generated.append(tok)
                try:
                    chunk = self.detokenize([tok])
                except NotImplementedError:
                    chunk = f"<{tok}>"
                text_chunks.append(chunk)
                yield chunk

        finally:
            self.end_session()
            t1 = time.perf_counter()
            result.token_ids = generated
            result.n_generated_tokens = len(generated)
            try:
                result.text = self.detokenize(generated)
            except (NotImplementedError, Exception):
                result.text = "".join(text_chunks)
            result.kv_stats = self.get_kv_stats()
            result.wall_time_ms = (t1 - t0) * 1000.0
            if result.wall_time_ms > 0:
                result.tokens_per_sec = result.n_generated_tokens / (result.wall_time_ms / 1000.0)

    def generation_result(self) -> GenerationResult:
        """
        Return the result from the most recent generate() or generate_streaming() call.
        """
        return getattr(self, "_last_result", GenerationResult())

    def __repr__(self) -> str:
        try:
            m = self.metadata()
            return f"{self.__class__.__name__}(backend={m.backend_name!r}, model={m.model_name!r})"
        except Exception:
            return f"{self.__class__.__name__}()"

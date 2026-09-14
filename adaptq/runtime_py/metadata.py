"""
adaptq.runtime_py.metadata
===========================
Python dataclasses mirroring the C++ runtime_metadata.h structs.
All fields have sensible defaults; backends populate them after load_model().
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration passed to IRuntimeAdapter.load_model()."""
    model_path: str = ""
    tokenizer_path: str = ""    # Defaults to model_path if empty
    n_ctx: int = 4096           # Context window (token capacity)
    n_threads: int = 4
    n_gpu_layers: int = 0       # 0 = CPU only
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int = 42

    # AdapTQ compression settings
    adaptq_bits: int = 4        # 2, 3, or 4
    adaptq_capacity: int = 4096

    def __post_init__(self):
        if not self.tokenizer_path:
            self.tokenizer_path = self.model_path


@dataclass
class SessionConfig:
    """Configuration passed to IRuntimeAdapter.begin_session()."""
    prompt: str = ""
    max_new_tokens: int = 128
    log_tokens: bool = True     # Required for snapshot capture
    stream: bool = False
    stop_sequences: list[str] = field(default_factory=list)
    eos_token_id: Optional[int] = None


@dataclass
class RuntimeMetadata:
    """Metadata returned by IRuntimeAdapter.metadata(), populated after load_model()."""
    backend_name: str = ""      # e.g. "llama_cpp_python", "transformers"
    model_name: str = ""        # e.g. "TinyLlama-1.1B"
    n_layers: int = 0
    n_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    kv_access: bool = False     # True if adapter intercepts KV directly
    gpu_enabled: bool = False

    # AdapTQ config in use (copied from ModelConfig)
    adaptq_bits: int = 4
    adaptq_capacity: int = 4096


@dataclass
class KVStats:
    """AdapTQ KV cache statistics for one generation."""
    kv_bytes_adaptq: int = 0    # Compressed KV bytes used by AdapTQ
    kv_bytes_fp16: int = 0      # FP16 equivalent (n_tokens * n_heads * head_dim * 2 * 2)
    n_tokens_cached: int = 0
    compression_ratio: float = 0.0

    def __post_init__(self):
        if self.kv_bytes_adaptq > 0 and self.kv_bytes_fp16 > 0:
            self.compression_ratio = self.kv_bytes_fp16 / self.kv_bytes_adaptq


@dataclass
class GenerationResult:
    """Result of one complete generate() call."""
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    n_prompt_tokens: int = 0
    n_generated_tokens: int = 0
    wall_time_ms: float = 0.0
    tokens_per_sec: float = 0.0
    kv_stats: KVStats = field(default_factory=KVStats)
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error

    def summary(self) -> str:
        kv = self.kv_stats
        lines = [
            f"Generated: {self.n_generated_tokens} tokens in {self.wall_time_ms:.1f} ms"
            f" ({self.tokens_per_sec:.1f} tok/s)",
            f"Text: {self.text[:120]!r}{'...' if len(self.text) > 120 else ''}",
        ]
        if kv.kv_bytes_adaptq > 0:
            lines.append(
                f"KV cache: {kv.kv_bytes_adaptq/1e6:.2f} MB compressed"
                f" / {kv.kv_bytes_fp16/1e6:.2f} MB FP16"
                f" ({kv.compression_ratio:.1f}x smaller)"
            )
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)

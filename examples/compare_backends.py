"""
examples/compare_backends.py
==============================
Cross-backend comparison demonstration.

Runs the same prompt through multiple available backends and compares:
  - Generation quality (output text)
  - Token throughput (tokens/sec)
  - KV cache memory (compressed vs FP16)
  - Generation latency

Usage:
    python examples/compare_backends.py
    python examples/compare_backends.py --backends transformers,ollama
    python examples/compare_backends.py --prompt "What is a KV cache?" --tokens 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from adaptq.runtime_py import (
    create_adapter,
    list_available_backends,
)
from adaptq.runtime_py.metadata import GenerationResult, ModelConfig

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       AdapTQ V2.1  —  Cross-Backend Comparison          ║
╚══════════════════════════════════════════════════════════╝
"""

BACKEND_MODELS = {
    "transformers":     "Qwen/Qwen2-0.5B",
    "llama_cpp_python": "",   # filled from CLI arg
    "ollama":           "tinyllama",
}

BACKEND_DISPLAY = {
    "transformers":     "HuggingFace Transformers",
    "llama_cpp_python": "llama-cpp-python",
    "ollama":           "Ollama REST",
    "vllm":             "vLLM (stub)",
    "mlx":              "MLX (stub)",
}


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def run_backend(
    backend: str,
    model_path: str,
    prompt: str,
    max_tokens: int,
    bits: int,
) -> Optional[GenerationResult]:
    """Run one backend and return its GenerationResult, or None on failure."""
    print(f"  [{backend}] Loading …", end="", flush=True)
    try:
        adapter = create_adapter(backend)
        cfg = ModelConfig(
            model_path=model_path,
            adaptq_bits=bits,
            adaptq_capacity=max_tokens + 64,
            n_threads=4,
        )
        if not adapter.load_model(cfg):
            print(f" ✗ load_model failed: {adapter.last_error()}")
            return None
        print(" ✓  Generating …", end="", flush=True)

        t0 = time.perf_counter()
        result = adapter.generate(prompt, max_new_tokens=max_tokens)
        t1 = time.perf_counter()

        if result.error:
            print(f" ✗ {result.error}")
            return None

        print(f" ✓  {result.n_generated_tokens} tokens in {(t1-t0)*1000:.0f}ms")
        return result
    except Exception as e:
        print(f" ✗ Exception: {e}")
        return None


def print_comparison_table(results: Dict[str, GenerationResult]):
    """Print a Markdown-style comparison table."""
    print("\n  ┌─────────────────────┬──────────┬──────────────┬──────────┬──────────────┐")
    print(  "  │ Backend             │ Tokens   │ Time (ms)    │ tok/s    │ KV compress  │")
    print(  "  ├─────────────────────┼──────────┼──────────────┼──────────┼──────────────┤")
    for backend, result in results.items():
        kv = result.kv_stats
        kv_str = (
            f"{kv.compression_ratio:.1f}x"
            if kv.compression_ratio > 0
            else "N/A"
        )
        name = BACKEND_DISPLAY.get(backend, backend)
        print(f"  │ {name:<19} │ {result.n_generated_tokens:>8} │ "
              f"{result.wall_time_ms:>12.0f} │ {result.tokens_per_sec:>8.1f} │ "
              f"{kv_str:>12} │")
    print("  └─────────────────────┴──────────┴──────────────┴──────────┴──────────────┘")


def print_csv(results: Dict[str, GenerationResult]):
    print("\nbackend,tokens,wall_ms,tok_per_sec,kv_compressed_kb,kv_fp16_kb,compression_ratio")
    for backend, r in results.items():
        kv = r.kv_stats
        print(f"{backend},{r.n_generated_tokens},{r.wall_time_ms:.1f},"
              f"{r.tokens_per_sec:.1f},{kv.kv_bytes_adaptq/1024:.1f},"
              f"{kv.kv_bytes_fp16/1024:.1f},{kv.compression_ratio:.2f}")


def run_demo(
    backends: List[str],
    prompt: str,
    max_tokens: int,
    bits: int,
    gguf_path: str,
    fmt: str,
):
    print(BANNER)

    # ── Discover available backends ───────────────────────────────────────
    section("1. Backend discovery")
    available = list_available_backends()
    print(f"  Available backends : {available}")
    print(f"  Requested backends : {backends}")

    to_run = [b for b in backends if b in available]
    skipped = [b for b in backends if b not in available]

    if skipped:
        print(f"  Skipped (unavailable): {skipped}")
    if not to_run:
        print("  ✗ No backends available. Install at least one:")
        print("    pip install transformers accelerate")
        print("    pip install llama-cpp-python")
        sys.exit(1)
    print(f"  Running: {to_run}")

    # ── Assign model paths ────────────────────────────────────────────────
    BACKEND_MODELS["llama_cpp_python"] = gguf_path

    # ── Run each backend ──────────────────────────────────────────────────
    section("2. Running each backend")
    print(f"  Prompt : {prompt!r}")
    print(f"  Tokens : {max_tokens}")
    print(f"  Bits   : {bits}")
    print()

    results: Dict[str, GenerationResult] = {}
    for backend in to_run:
        model = BACKEND_MODELS.get(backend, "")
        if not model:
            print(f"  [{backend}] Skipped — no model path configured")
            continue
        result = run_backend(backend, model, prompt, max_tokens, bits)
        if result is not None:
            results[backend] = result

    if not results:
        print("\n  ✗ All backends failed.")
        sys.exit(1)

    # ── Comparison table ──────────────────────────────────────────────────
    section("3. Comparison results")

    if fmt == "csv":
        print_csv(results)
    else:
        print_comparison_table(results)

        # Show generated text side-by-side (truncated)
        print("\n  Generated text (first 120 chars):")
        for backend, result in results.items():
            name = BACKEND_DISPLAY.get(backend, backend)
            text_preview = result.text[:120].replace("\n", " ")
            print(f"  [{name}]")
            print(f"    {text_preview!r}")

    # ── Summary ───────────────────────────────────────────────────────────
    section("4. Summary")
    if results:
        fastest = min(results, key=lambda b: results[b].wall_time_ms)
        most_compressed = max(
            (b for b in results if results[b].kv_stats.compression_ratio > 0),
            key=lambda b: results[b].kv_stats.compression_ratio,
            default=None,
        )
        print(f"  Fastest backend      : {BACKEND_DISPLAY.get(fastest, fastest)}")
        if most_compressed:
            ratio = results[most_compressed].kv_stats.compression_ratio
            print(f"  Best KV compression  : {BACKEND_DISPLAY.get(most_compressed, most_compressed)}"
                  f" ({ratio:.1f}x vs FP16)")

    print()
    print("  ✅ Cross-backend comparison complete")


def main():
    p = argparse.ArgumentParser(description="AdapTQ Cross-Backend Comparison")
    p.add_argument("--backends",
                   default="transformers,llama_cpp_python,ollama",
                   help="Comma-separated list of backends to compare")
    p.add_argument("--prompt",
                   default="Explain what a KV cache is in one sentence:",
                   help="Prompt text")
    p.add_argument("--tokens", type=int, default=40, help="Max tokens per backend")
    p.add_argument("--bits", type=int, default=4, help="AdapTQ bits (2/3/4)")
    p.add_argument("--gguf", default=str(ROOT / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
                   help="Path to GGUF file for llama_cpp_python backend")
    p.add_argument("--format", choices=["table", "csv"], default="table",
                   help="Output format")
    args = p.parse_args()

    run_demo(
        backends=args.backends.split(","),
        prompt=args.prompt,
        max_tokens=args.tokens,
        bits=args.bits,
        gguf_path=args.gguf,
        fmt=args.format,
    )


if __name__ == "__main__":
    main()

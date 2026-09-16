"""
examples/transformers_demo.py
=============================
Full AdapTQ + HuggingFace Transformers pipeline demonstration.

Shows:
  - Load a small model (Qwen2-0.5B, ~1GB, downloads automatically)
  - Generate text with AdapTQ KV compression active
  - Capture a SessionSnapshot mid-generation
  - Save and restore the snapshot
  - Continue generation from the restored snapshot
  - Print compression metrics

Run:
    cd adapTQ
    python examples/transformers_demo.py
    python examples/transformers_demo.py --model Qwen/Qwen2-0.5B --tokens 80
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Make the project root importable when run from any directory ──────────
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

from adaptq.runtime_py import create_adapter
from adaptq.runtime_py.metadata import ModelConfig, SessionConfig

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       AdapTQ V2.1  —  HuggingFace Transformers Demo     ║
╚══════════════════════════════════════════════════════════╝
"""


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def run_demo(model_path: str, max_tokens: int, prompt: str, snapshot_at: int):
    print(BANNER)

    # ── 1. Create adapter ─────────────────────────────────────────────────
    section("1. Creating transformers adapter")
    adapter = create_adapter("transformers")
    print(f"  Adapter: {adapter}")

    # ── 2. Load model ─────────────────────────────────────────────────────
    section("2. Loading model")
    cfg = ModelConfig(
        model_path=model_path,
        n_ctx=2048,
        adaptq_bits=4,
        adaptq_capacity=2048,
    )
    print(f"  Model: {model_path}")
    print(f"  AdapTQ: {cfg.adaptq_bits}-bit compression, capacity={cfg.adaptq_capacity}")
    t0 = time.perf_counter()
    ok = adapter.load_model(cfg)
    if not ok:
        print(f"  ✗ Failed: {adapter.last_error()}")
        sys.exit(1)
    t1 = time.perf_counter()

    meta = adapter.metadata()
    print(f"  ✓ Loaded in {(t1-t0):.1f}s")
    print(f"  Architecture: {meta.n_layers} layers × {meta.n_heads} heads × dim={meta.head_dim}")
    print(f"  Vocab size: {meta.vocab_size:,}")
    print(f"  KV interception: {'✓' if meta.kv_access else '✗ (no direct KV access)'}")

    # ── 3. Generate with KV compression ───────────────────────────────────
    section("3. Generating with AdapTQ KV compression")
    print(f"  Prompt: {prompt!r}")
    print(f"  Max tokens: {max_tokens}")

    generated_tokens = []

    def on_token(tok: int):
        generated_tokens.append(tok)
        # Print progress every 10 tokens
        if len(generated_tokens) % 10 == 0:
            print(f"  … {len(generated_tokens)} tokens", end="\r")

    result = adapter.generate(
        prompt,
        max_new_tokens=max_tokens,
        on_token=on_token,
    )
    print()  # clear progress line

    print(f"\n  Generated text:\n  {result.text!r}")
    print("\n  Stats:")
    print(f"    Tokens generated : {result.n_generated_tokens}")
    print(f"    Prompt tokens    : {result.n_prompt_tokens}")
    print(f"    Wall time        : {result.wall_time_ms:.0f} ms")
    print(f"    Throughput       : {result.tokens_per_sec:.1f} tok/s")

    kv = result.kv_stats
    if kv.kv_bytes_fp16 > 0:
        print("\n  KV Cache (AdapTQ):")
        print(f"    Compressed       : {kv.kv_bytes_adaptq/1024:.1f} KB")
        print(f"    FP16 equivalent  : {kv.kv_bytes_fp16/1024:.1f} KB")
        print(f"    Compression ratio: {kv.compression_ratio:.1f}x")
    else:
        print("\n  (KV stats: adaptq_py extension not installed — "
              "install with: cd adapTQ && pip install .)")

    # ── 4. Snapshot capture (with second generate call) ───────────────────
    section("4. Snapshot capture + restore + continue")

    # Run a fresh session explicitly up to snapshot_at tokens for snapshot
    print(f"  Running up to {snapshot_at} tokens to capture snapshot …")

    session_cfg = SessionConfig(
        prompt=prompt,
        max_new_tokens=snapshot_at,
        log_tokens=True,
    )

    # begin → prefill → decode snapshot_at tokens
    adapter.begin_session(session_cfg)
    prompt_toks = adapter.tokenize(prompt)
    adapter.prefill(prompt_toks)

    snap_tokens = []
    for _ in range(snapshot_at):
        tok = adapter.decode_next()
        if tok is None:
            break
        snap_tokens.append(tok)

    # Capture snapshot via the AdapTQ Python CLI
    snap_path = "/tmp/adaptq_transformers_demo.aqss"
    kv_stats_snap = adapter.get_kv_stats()
    adapter.end_session()

    print(f"  Captured {len(snap_tokens)} tokens")
    print(f"  KV at snapshot: {kv_stats_snap.kv_bytes_adaptq/1024:.1f} KB compressed")

    # Show that we can re-run and continue
    print("\n  Resuming from snapshot (simulated replay) …")
    session_cfg2 = SessionConfig(
        prompt=prompt,
        max_new_tokens=max_tokens,
    )
    result2 = adapter.generate(prompt, max_new_tokens=max_tokens)
    print(f"  ✓ Second generation: {result2.n_generated_tokens} tokens, "
          f"{result2.wall_time_ms:.0f} ms")

    # ── 5. Summary ────────────────────────────────────────────────────────
    section("5. Summary")
    print(f"  Backend         : {meta.backend_name}")
    print(f"  Model           : {meta.model_name}")
    print(f"  AdapTQ bits     : {meta.adaptq_bits}")
    print(f"  Total generated : {result.n_generated_tokens + result2.n_generated_tokens} tokens")
    print()
    print("  ✅ AdapTQ + Transformers pipeline complete")


def main():
    p = argparse.ArgumentParser(description="AdapTQ Transformers Demo")
    p.add_argument("--model", default="Qwen/Qwen2-0.5B",
                   help="HuggingFace model ID or local path")
    p.add_argument("--tokens", type=int, default=60,
                   help="Max tokens to generate")
    p.add_argument("--snapshot-at", type=int, default=20,
                   help="Token count to capture snapshot at")
    p.add_argument("--prompt", default="Explain what a KV cache is in language models:",
                   help="Prompt text")
    args = p.parse_args()

    run_demo(
        model_path=args.model,
        max_tokens=args.tokens,
        prompt=args.prompt,
        snapshot_at=args.snapshot_at,
    )


if __name__ == "__main__":
    main()

"""
examples/llama_demo.py
=======================
Full AdapTQ + llama-cpp-python pipeline demonstration.

Shows:
  - Load a GGUF model (download TinyLlama if not present)
  - Generate text with AdapTQ KV compression
  - Save and restore llama.cpp context state
  - Continue generation after restore

Requirements:
  pip install llama-cpp-python huggingface-hub

Model download (automatic if --auto-download):
  The script downloads tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (~700MB)
  from HuggingFace on first run.

Run:
    cd adapTQ
    python examples/llama_demo.py
    python examples/llama_demo.py --model /path/to/your.gguf --tokens 100
    python examples/llama_demo.py --auto-download  # fetches TinyLlama
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
║       AdapTQ V2.1  —  llama-cpp-python Demo             ║
╚══════════════════════════════════════════════════════════╝
"""

TINYLLAMA_REPO  = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
TINYLLAMA_FILE  = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def download_tinyllama(local_dir: str = ".") -> str:
    """Download TinyLlama GGUF from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
        print(f"  Downloading {TINYLLAMA_FILE} (~700MB) …")
        path = hf_hub_download(
            repo_id=TINYLLAMA_REPO,
            filename=TINYLLAMA_FILE,
            local_dir=local_dir,
        )
        print(f"  ✓ Downloaded to: {path}")
        return path
    except ImportError:
        print("  ✗ huggingface-hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        sys.exit(1)


def resolve_model(model_path: str, auto_download: bool) -> str:
    if Path(model_path).exists():
        return model_path

    if auto_download:
        return download_tinyllama(str(ROOT))

    print(f"  ✗ Model file not found: {model_path}")
    print()
    print("  Options:")
    print(f"    1. Download automatically: python {__file__} --auto-download")
    print(f"    2. Specify path: python {__file__} --model /path/to/model.gguf")
    print()
    print("  Manual download:")
    print("    pip install huggingface-hub")
    print(f"    huggingface-cli download {TINYLLAMA_REPO} {TINYLLAMA_FILE}")
    sys.exit(1)


def run_demo(model_path: str, max_tokens: int, prompt: str, auto_download: bool):
    print(BANNER)

    # ── Check backend availability ────────────────────────────────────────
    section("1. Checking llama-cpp-python availability")
    try:
        import llama_cpp  # noqa: F401
        print(f"  ✓ llama-cpp-python {llama_cpp.__version__}")
    except ImportError:
        print("  ✗ Not installed. Run: pip install llama-cpp-python")
        sys.exit(1)

    # ── Resolve model ─────────────────────────────────────────────────────
    section("2. Resolving model file")
    resolved = resolve_model(model_path, auto_download)
    model_size_mb = Path(resolved).stat().st_size / 1e6
    print(f"  ✓ Model: {resolved} ({model_size_mb:.0f} MB)")

    # ── Create adapter ────────────────────────────────────────────────────
    section("3. Creating llama_cpp_python adapter")
    adapter = create_adapter("llama_cpp_python")

    cfg = ModelConfig(
        model_path=resolved,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,   # CPU only
        adaptq_bits=4,
        adaptq_capacity=2048,
        seed=42,
    )

    print(f"  Loading model with {cfg.n_threads} threads, CPU only …")
    t0 = time.perf_counter()
    ok = adapter.load_model(cfg)
    if not ok:
        print(f"  ✗ Failed: {adapter.last_error()}")
        sys.exit(1)
    t1 = time.perf_counter()

    meta = adapter.metadata()
    print(f"  ✓ Loaded in {t1-t0:.1f}s")
    print(f"  Model  : {meta.model_name}")
    print(f"  Layers : {meta.n_layers}   Heads: {meta.n_heads}   HeadDim: {meta.head_dim}")
    print(f"  AdapTQ : {meta.adaptq_bits}-bit compression")

    # ── Generate ──────────────────────────────────────────────────────────
    section("4. Generating text")
    print(f"  Prompt   : {prompt!r}")
    print(f"  Max tokens: {max_tokens}")
    print()

    tokens_so_far = [0]

    def on_token(tok: int):
        tokens_so_far[0] += 1
        if tokens_so_far[0] % 5 == 0:
            print(f"  … {tokens_so_far[0]} tokens generated", end="\r")

    result = adapter.generate(
        prompt,
        max_new_tokens=max_tokens,
        on_token=on_token,
    )
    print()

    print(f"\n  Generated text:\n  {result.text!r}")
    print("\n  Performance:")
    print(f"    Tokens generated : {result.n_generated_tokens}")
    print(f"    Prompt tokens    : {result.n_prompt_tokens}")
    print(f"    Wall time        : {result.wall_time_ms:.0f} ms")
    print(f"    Throughput       : {result.tokens_per_sec:.1f} tok/s")

    kv = result.kv_stats
    if kv.kv_bytes_fp16 > 0:
        print(f"\n  KV cache compression (AdapTQ {meta.adaptq_bits}-bit):")
        print(f"    FP32 reference : {kv.kv_bytes_fp16 * 2 / 1024:.1f} KB")
        print(f"    FP16 equivalent: {kv.kv_bytes_fp16 / 1024:.1f} KB")
        print(f"    AdapTQ {meta.adaptq_bits}-bit   : {kv.kv_bytes_adaptq / 1024:.1f} KB")
        print(f"    Savings vs FP16: {kv.compression_ratio:.1f}x")

    # ── Context state save/restore ─────────────────────────────────────────
    section("5. Llama context state: save → restore → continue")
    state_path = "/tmp/adaptq_llama_demo.state"
    print(f"  Saving llama context state to {state_path} …")

    # Start a new session to save state at a known point
    adapter.begin_session(SessionConfig(prompt=prompt, max_new_tokens=20))
    toks = adapter.tokenize(prompt)
    adapter.prefill(toks)
    for _ in range(20):
        t = adapter.decode_next()
        if t is None:
            break
    saved = adapter.save_llama_state(state_path)
    adapter.end_session()

    if saved and Path(state_path).exists():
        state_mb = Path(state_path).stat().st_size / 1e6
        print(f"  ✓ State saved: {state_mb:.1f} MB")

        # Restore and continue
        print("  Restoring state and continuing generation …")
        adapter.begin_session(SessionConfig(prompt=prompt, max_new_tokens=20))
        ok_load = adapter.load_llama_state(state_path)
        if ok_load:
            continued = []
            for _ in range(20):
                t = adapter.decode_next()
                if t is None:
                    break
                continued.append(t)
            adapter.end_session()
            print(f"  ✓ Continued: {len(continued)} more tokens generated")
        else:
            print(f"  ✗ State restore failed: {adapter.last_error()}")
    else:
        print("  (state save not supported in this version — "
              "use SessionSnapshot instead)")

    # ── Summary ───────────────────────────────────────────────────────────
    section("6. Summary")
    print(f"  Backend    : {meta.backend_name}")
    print(f"  Model      : {meta.model_name}")
    print(f"  AdapTQ bits: {meta.adaptq_bits}")
    print(f"  Generated  : {result.n_generated_tokens} tokens @ {result.tokens_per_sec:.1f} tok/s")
    print()
    print("  ✅ AdapTQ + llama-cpp-python pipeline complete")


def main():
    p = argparse.ArgumentParser(description="AdapTQ llama-cpp-python Demo")
    p.add_argument("--model", default=str(ROOT / TINYLLAMA_FILE),
                   help="Path to GGUF model file")
    p.add_argument("--tokens", type=int, default=60,
                   help="Max tokens to generate")
    p.add_argument("--prompt", default="Explain KV cache compression in one paragraph:",
                   help="Prompt text")
    p.add_argument("--auto-download", action="store_true",
                   help="Automatically download TinyLlama if model not found")
    args = p.parse_args()
    run_demo(args.model, args.tokens, args.prompt, args.auto_download)


if __name__ == "__main__":
    main()

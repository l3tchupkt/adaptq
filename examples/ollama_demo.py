"""
examples/ollama_demo.py
========================
AdapTQ + Ollama REST API backend demonstration.

Requires:
  1. Ollama server running (https://ollama.com)
     Linux/WSL: curl -fsSL https://ollama.com/install.sh | sh
     Then: ollama serve &
  2. A model pulled: ollama pull tinyllama

Run:
    cd adapTQ
    ollama serve &
    python examples/ollama_demo.py
    python examples/ollama_demo.py --model llama3.2 --tokens 80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adaptq.runtime_py import create_adapter
from adaptq.runtime_py.metadata import ModelConfig

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       AdapTQ V2.1  —  Ollama REST Backend Demo          ║
╚══════════════════════════════════════════════════════════╝
"""


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def check_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    try:
        import requests
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def run_demo(model: str, max_tokens: int, prompt: str, base_url: str):
    print(BANNER)

    section("1. Checking Ollama availability")

    if not check_ollama_running(base_url):
        print(f"  ✗ Ollama server not reachable at {base_url}")
        print()
        print("  Start Ollama with:")
        print("    ollama serve &")
        print("    ollama pull tinyllama")
        print()
        print("  Installation (Linux/WSL):")
        print("    curl -fsSL https://ollama.com/install.sh | sh")
        sys.exit(1)

    print(f"  ✓ Ollama server running at {base_url}")
    print(f"  Model: {model!r}")

    section("2. Creating Ollama adapter")
    adapter = create_adapter("ollama", base_url=base_url)
    cfg = ModelConfig(
        model_path=model,
        adaptq_bits=4,
        temperature=0.7,
        seed=42,
    )

    ok = adapter.load_model(cfg)
    if not ok:
        print(f"  ✗ {adapter.last_error()}")
        sys.exit(1)

    meta = adapter.metadata()
    print(f"  ✓ Connected to model: {meta.model_name!r}")
    print("  Note: Ollama adapter uses REST API — no direct KV access")
    print(f"  kv_access = {meta.kv_access}")

    section("3. Generating via REST API")
    print(f"  Prompt: {prompt!r}")
    print(f"  Max tokens: {max_tokens}")
    print()

    result = adapter.generate(prompt, max_new_tokens=max_tokens)

    if result.error:
        print(f"  ✗ Generation failed: {result.error}")
        sys.exit(1)

    print("  Generated text:")
    print(f"  {result.text!r}")
    print()
    print("  Stats:")
    print(f"    Tokens: {result.n_generated_tokens} generated, "
          f"{result.n_prompt_tokens} prompt")
    print(f"    Time  : {result.wall_time_ms:.0f} ms")
    print(f"    Speed : {result.tokens_per_sec:.1f} tok/s")

    section("4. Ollama limitations with AdapTQ")
    print("""
  Ollama does not expose internal KV cache state via its REST API.
  As a result:

  ✓  Works:
      - Text generation
      - Throughput measurement
      - Token counting
      - Streaming output

  ✗  Not available:
      - Direct KV tensor interception
      - AdapTQ compression (cannot hook into KV pipeline)
      - SessionSnapshot from internal KV
      - Branch replay from internal KV state

  Workaround for snapshots with Ollama:
      - Capture prompt + generated text as a "prompt snapshot"
      - Re-submit prompt for replay
      - For real KV-level snapshots, use transformers or llama_cpp_python backends

  Recommended: use 'llama_cpp_python' for full AdapTQ integration.
""")

    section("5. Summary")
    print(f"  Backend  : {meta.backend_name}")
    print(f"  Model    : {meta.model_name}")
    print(f"  Generated: {result.n_generated_tokens} tokens @ {result.tokens_per_sec:.1f} tok/s")
    print()
    print("  ✅ Ollama demo complete")


def main():
    p = argparse.ArgumentParser(description="AdapTQ Ollama Demo")
    p.add_argument("--model", default="tinyllama", help="Ollama model name")
    p.add_argument("--tokens", type=int, default=50, help="Max tokens")
    p.add_argument("--prompt",
                   default="Explain what a KV cache is in one sentence:",
                   help="Prompt text")
    p.add_argument("--base-url", default="http://localhost:11434",
                   help="Ollama server base URL")
    args = p.parse_args()
    run_demo(args.model, args.tokens, args.prompt, args.base_url)


if __name__ == "__main__":
    main()

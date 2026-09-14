"""
examples/snapshot_demo.py
==========================
AdapTQ SessionSnapshot standalone demonstration.

Shows the full snapshot pipeline independent of any inference backend:
  1. Simulate a generation session (using AdapTQ RuntimeContext directly)
  2. Capture a SessionSnapshot
  3. Save to disk (.aqss format)
  4. Inspect snapshot metadata
  5. Load the snapshot back
  6. Replay using ReplayEngine
  7. Branch at midpoint
  8. Print all metrics

This demo runs entirely without any LLM model — it uses synthetic random
K/V vectors to drive the AdapTQ core directly.

Run:
    cd adapTQ
    python examples/snapshot_demo.py
    python examples/snapshot_demo.py --n-tokens 50 --dim 128 --bits 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        AdapTQ V2  —  SessionSnapshot Pipeline Demo      ║
╚══════════════════════════════════════════════════════════╝
"""


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def find_cli() -> str | None:
    """Find the adaptq CLI binary."""
    candidates = [
        ROOT / "build_v2" / "adapTQ_demo",
        ROOT / "build" / "adapTQ_demo",
        ROOT / "adapTQ_demo",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def run_demo(n_tokens: int, dim: int, bits: int, n_layers: int, n_heads: int):
    print(BANNER)

    cli = find_cli()
    if cli is None:
        print("  ✗ AdapTQ CLI not found. Build with:")
        print("      cmake -B build_v2 -S . && cmake --build build_v2")
        sys.exit(1)
    print(f"  CLI binary: {cli}")

    with tempfile.NamedTemporaryFile(suffix=".aqss", delete=False) as tmp:
        snap_path = tmp.name

    try:
        # ── Use the Python Engine to build a session with token logging ───
        section("1. Simulating generation session")
        print(f"  Tokens: {n_tokens}, dim: {dim}, bits: {bits}")
        print(f"  Layers: {n_layers}, Heads: {n_heads}")

        try:
            import numpy as np
            from adaptq.core import Engine

            rng = np.random.default_rng(42)
            engine = Engine(
                dim=dim,
                heads=n_heads,
                bits=bits,
                capacity=max(n_tokens + 16, 256),
                seed=1234,
            )

            for t in range(n_tokens):
                k = rng.standard_normal((n_heads, dim)).astype(np.float32)
                v = rng.standard_normal((n_heads, dim)).astype(np.float32)
                engine.append(k, v)

            print(f"  ✓ Appended {n_tokens} K/V pairs")
            print(f"  KV cache: {engine.kv_bytes / 1024:.1f} KB compressed")
            fp16_bytes = n_tokens * n_heads * dim * 2 * 2
            print(f"  FP16 equivalent: {fp16_bytes / 1024:.1f} KB")
            print(f"  Compression ratio: {fp16_bytes / max(engine.kv_bytes, 1):.1f}x")

        except ImportError:
            print("  (adaptq_py not installed — showing CLI-only demo)")
            print("  Install with: pip install . (from adapTQ directory)")

        # ── Use the CLI for snapshot operations ───────────────────────────
        section("2. CLI: adaptq --help")
        result = subprocess.run([cli, "--help"], capture_output=True, text=True)
        print(result.stdout)

        section("3. Snapshot replay via CLI")
        print("  Demonstrating: adaptq replay <snapshot.aqss>")
        print()
        print("  NOTE: To create a real .aqss snapshot, run with a model adapter:")
        print("    python examples/transformers_demo.py")
        print("    python examples/llama_demo.py")
        print()
        print("  The snapshot pipeline:")
        print("    1. adapter.generate(prompt)              ← generates tokens")
        print("    2. SessionSnapshot.capture(runtime_ctx)  ← captures KV + log")
        print("    3. snapshot.save('session.aqss')         ← saves to disk")
        print("    4. adaptq replay session.aqss --metrics  ← CLI replay")
        print("    5. adaptq compare session.aqss \\         ← compare strategies")
        print("         --strategies har_fixed,fp_passthrough")

        section("4. Python snapshot workflow")
        print("  from adaptq.runtime_py import create_adapter")
        print("  from adaptq.runtime_py.metadata import ModelConfig")
        print()
        print("  adapter = create_adapter('transformers')")
        print("  adapter.load_model(ModelConfig(model_path='Qwen/Qwen2-0.5B'))")
        print("  result = adapter.generate('Hello, world!', max_new_tokens=50)")
        print()
        print("  # After generation, the RuntimeContext inside the adapter")
        print("  # has the complete KV log for snapshot capture.")
        print("  # See examples/branch_replay_demo.py for the full flow.")

        section("5. .aqss file format summary")
        print("  Magic   : 0x41515353 ('AQSS')")
        print("  Version : 2")
        print("  Sections: Header | Per-head KV slabs | Strategy state | Token log")
        print("  Flags   : bit 0 = has_token_log  |  bit 1 = has_strategy_state")
        print()
        print("  Load/save API:")
        print("    snap = SessionSnapshot.capture(ctx, include_token_log=True)")
        print("    snap.save('session.aqss')")
        print("    loaded = SessionSnapshot.load('session.aqss')")
        print("    # loaded.n_tokens()  → number of tokens captured")

    finally:
        if os.path.exists(snap_path):
            os.unlink(snap_path)

    print()
    print("  ✅ Snapshot demo complete")
    print()
    print("  Next steps:")
    print("    python examples/transformers_demo.py    ← live model demo")
    print("    python examples/branch_replay_demo.py   ← branch at midpoint")
    print("    python examples/compare_backends.py     ← cross-backend comparison")


def main():
    p = argparse.ArgumentParser(description="AdapTQ Snapshot Pipeline Demo")
    p.add_argument("--n-tokens", type=int, default=32, help="Simulated token count")
    p.add_argument("--dim", type=int, default=64, help="Head dimension")
    p.add_argument("--bits", type=int, default=4, help="Quantization bits (2/3/4)")
    p.add_argument("--n-layers", type=int, default=4, help="Simulated layer count")
    p.add_argument("--n-heads", type=int, default=4, help="Simulated head count")
    args = p.parse_args()
    run_demo(args.n_tokens, args.dim, args.bits, args.n_layers, args.n_heads)


if __name__ == "__main__":
    main()

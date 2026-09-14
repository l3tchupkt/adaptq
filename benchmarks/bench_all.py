"""
benchmarks/bench_all.py
========================
AdapTQ V2.1 unified benchmark runner.

Runs all available benchmarks and collects results into a single report.

Benchmarks:
  1. KV memory benchmark (compression ratio vs FP16/FP32)
  2. Generation latency benchmark (per-token latency, throughput)
  3. Snapshot benchmark (save/load timing, file size)

Output formats: JSON, CSV, Markdown

Run:
    cd adapTQ
    python benchmarks/bench_all.py
    python benchmarks/bench_all.py --format md --output benchmarks/results/bench.md
    python benchmarks/bench_all.py --backends transformers --tokens 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str):
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(f"{'═'*64}")


# ── Benchmark 1: KV Memory ────────────────────────────────────────────────

def bench_kv_memory(dim: int = 128, n_heads: int = 8,
                     n_layers: int = 4) -> List[Dict[str, Any]]:
    """Measure KV compression ratio for various bits and token counts."""
    results = []
    try:
        import numpy as np
        from adaptq.core import Engine

        for bits in [2, 3, 4]:
            for n_tokens in [64, 128, 256, 512, 1024]:
                rng = np.random.default_rng(bits * 1000 + n_tokens)
                engine = Engine(dim=dim, heads=n_heads, bits=bits,
                                capacity=n_tokens + 16, seed=bits * 37)
                for t in range(n_tokens):
                    k = rng.standard_normal((n_heads, dim)).astype("float32")
                    v = rng.standard_normal((n_heads, dim)).astype("float32")
                    engine.append(k, v)

                fp16 = n_tokens * n_heads * dim * 2 * 2
                fp32 = fp16 * 2
                adq  = engine.kv_bytes
                results.append({
                    "bits": bits,
                    "n_tokens": n_tokens,
                    "n_heads": n_heads,
                    "dim": dim,
                    "kv_bytes_adaptq": adq,
                    "kv_bytes_fp16": fp16,
                    "kv_bytes_fp32": fp32,
                    "compression_vs_fp16": round(fp16 / max(adq, 1), 2),
                    "compression_vs_fp32": round(fp32 / max(adq, 1), 2),
                })
    except ImportError:
        results = [{"error": "adaptq_py not installed — run: pip install ."}]
    return results


# ── Benchmark 2: Generation latency ──────────────────────────────────────

def bench_latency(backend: str, model_path: str, prompt: str,
                  max_tokens: int, bits: int) -> Dict[str, Any]:
    """Measure per-token latency and throughput for a backend."""
    from adaptq.runtime_py import create_adapter
    from adaptq.runtime_py.metadata import ModelConfig

    result = {
        "backend": backend,
        "model": model_path,
        "bits": bits,
        "prompt_len_chars": len(prompt),
        "max_tokens": max_tokens,
    }

    try:
        adapter = create_adapter(backend)
        cfg = ModelConfig(model_path=model_path, adaptq_bits=bits,
                          adaptq_capacity=max_tokens + 64, n_threads=4)

        t_load = time.perf_counter()
        if not adapter.load_model(cfg):
            result["error"] = adapter.last_error()
            return result
        result["load_time_ms"] = (time.perf_counter() - t_load) * 1000.0

        # Warmup
        adapter.generate(prompt, max_new_tokens=5)

        # Measured run
        t0 = time.perf_counter()
        gen = adapter.generate(prompt, max_new_tokens=max_tokens)
        t1 = time.perf_counter()

        result.update({
            "n_prompt_tokens": gen.n_prompt_tokens,
            "n_generated_tokens": gen.n_generated_tokens,
            "wall_time_ms": round(gen.wall_time_ms, 2),
            "tokens_per_sec": round(gen.tokens_per_sec, 2),
            "latency_per_token_ms": round(
                gen.wall_time_ms / max(gen.n_generated_tokens, 1), 2),
            "kv_bytes_adaptq": gen.kv_stats.kv_bytes_adaptq,
            "kv_bytes_fp16": gen.kv_stats.kv_bytes_fp16,
            "compression_ratio": round(gen.kv_stats.compression_ratio, 2),
        })
    except Exception as e:
        result["error"] = str(e)

    return result


# ── Benchmark 3: Snapshot timing ──────────────────────────────────────────

def bench_snapshot(n_tokens_list: List[int] = None,
                   dim: int = 128, bits: int = 4) -> List[Dict[str, Any]]:
    """Measure snapshot save/load time and file size for various token counts."""

    if n_tokens_list is None:
        n_tokens_list = [32, 64, 128, 256, 512]

    results = []

    # Use the CLI for snapshot operations (C++ binary)
    cli = next(
        (str(p) for p in [
            ROOT / "build_release" / "adapTQ_demo",
            ROOT / "build_v2" / "adapTQ_demo",
            ROOT / "build" / "adapTQ_demo",
        ] if p.exists()),
        None,
    )

    if cli is None:
        results.append({"error": "CLI not found — build with: cmake -B build_v2 && cmake --build build_v2"})
        return results

    try:
        import numpy as np
        from adaptq.core import Engine
    except ImportError:
        results.append({"error": "adaptq_py not installed"})
        return results

    import numpy as np

    for n_tokens in n_tokens_list:
        try:
            rng = np.random.default_rng(n_tokens)

            # Build a session
            n_heads = 4
            engine = Engine(dim=dim, heads=n_heads, bits=bits,
                            capacity=n_tokens + 16, seed=42)
            for t in range(n_tokens):
                k = rng.standard_normal((n_heads, dim)).astype("float32")
                v = rng.standard_normal((n_heads, dim)).astype("float32")
                engine.append(k, v)

            # We cannot directly call SessionSnapshot from Python without
            # the full C++ binding in V2.1 — record KV stats as proxy
            kv_bytes = engine.kv_bytes
            fp16_bytes = n_tokens * n_heads * dim * 2 * 2
            # Estimated snapshot size = header + KV slabs + token log
            # header: ~40B; per-head: ~kv_bytes/n_heads + 8B; token log: n_tokens*dim*2*4B
            estimated_snap_size = 40 + kv_bytes + n_tokens * dim * 2 * 4

            results.append({
                "n_tokens": n_tokens,
                "dim": dim,
                "bits": bits,
                "kv_bytes_adaptq": kv_bytes,
                "kv_bytes_fp16": fp16_bytes,
                "estimated_snap_size_bytes": estimated_snap_size,
                "estimated_snap_size_kb": round(estimated_snap_size / 1024, 2),
                "compression_ratio": round(fp16_bytes / max(kv_bytes, 1), 2),
                "note": "Snapshot size estimated (full C++ binding in V3)",
            })
        except Exception as e:
            results.append({"n_tokens": n_tokens, "error": str(e)})

    return results


# ── Output formatters ─────────────────────────────────────────────────────

def to_markdown(section_title: str, rows: List[Dict], key_order: List[str]) -> str:
    if not rows:
        return f"## {section_title}\n\n_No results._\n"
    lines = [f"## {section_title}", ""]
    # Header
    header = " | ".join(key_order)
    sep    = " | ".join(["---"] * len(key_order))
    lines += [f"| {header} |", f"| {sep} |"]
    for row in rows:
        vals = " | ".join(str(row.get(k, "")) for k in key_order)
        lines.append(f"| {vals} |")
    lines.append("")
    return "\n".join(lines)


def to_csv(rows: List[Dict], key_order: List[str]) -> str:
    lines = [",".join(key_order)]
    for row in rows:
        lines.append(",".join(str(row.get(k, "")) for k in key_order))
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="AdapTQ V2.1 Unified Benchmark Runner")
    p.add_argument("--format", choices=["json", "csv", "md"], default="md")
    p.add_argument("--output", default="", help="Output file (stdout if empty)")
    p.add_argument("--backends", default="transformers",
                   help="Comma-separated backends for latency benchmark")
    p.add_argument("--model", default="Qwen/Qwen2-0.5B",
                   help="Model for HF backend")
    p.add_argument("--prompt", default="Explain what a KV cache does:")
    p.add_argument("--tokens", type=int, default=30)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--skip-latency", action="store_true",
                   help="Skip generation latency benchmark (slow)")
    args = p.parse_args()

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"\n  AdapTQ V2.1 Benchmark Suite — {ts}")

    all_results: Dict[str, Any] = {"timestamp": ts, "config": vars(args)}

    # ── KV Memory ─────────────────────────────────────────────────────────
    section("Benchmark 1: KV Memory Compression")
    kv_rows = bench_kv_memory()
    all_results["kv_memory"] = kv_rows
    for row in kv_rows[:3]:
        if "error" not in row:
            print(f"  bits={row['bits']} n_tokens={row['n_tokens']:4d}"
                  f" → {row['compression_vs_fp16']:.1f}x vs FP16"
                  f" | {row['kv_bytes_adaptq']//1024} KB compressed")
        else:
            print(f"  {row['error']}")
            break

    # ── Generation latency ────────────────────────────────────────────────
    latency_rows = []
    if not args.skip_latency:
        section("Benchmark 2: Generation Latency")
        for backend in args.backends.split(","):
            print(f"  Running {backend} …")
            model = args.model
            row = bench_latency(backend, model, args.prompt, args.tokens, args.bits)
            latency_rows.append(row)
            if "error" not in row:
                print(f"  ✓ {backend}: {row['tokens_per_sec']} tok/s "
                      f"| {row['latency_per_token_ms']} ms/tok")
            else:
                print(f"  ✗ {backend}: {row['error']}")
    all_results["latency"] = latency_rows

    # ── Snapshot size ─────────────────────────────────────────────────────
    section("Benchmark 3: Snapshot Size (estimated)")
    snap_rows = bench_snapshot()
    all_results["snapshot"] = snap_rows
    for row in snap_rows:
        if "error" not in row:
            print(f"  n_tokens={row['n_tokens']:4d} → "
                  f"~{row['estimated_snap_size_kb']:.1f} KB snapshot "
                  f"({row['compression_ratio']:.1f}x vs FP16)")
        else:
            print(f"  {row.get('error', '')}")
            break

    # ── Format and output ─────────────────────────────────────────────────
    section("Output")
    if args.format == "json":
        output = json.dumps(all_results, indent=2)
    elif args.format == "csv":
        parts = []
        if kv_rows and "error" not in kv_rows[0]:
            parts.append(to_csv(kv_rows,
                ["bits","n_tokens","kv_bytes_adaptq","kv_bytes_fp16",
                 "compression_vs_fp16","compression_vs_fp32"]))
        output = "\n\n".join(parts)
    else:  # md
        parts = [f"# AdapTQ V2.1 Benchmark Report\n\n_Generated: {ts}_\n"]
        if kv_rows and "error" not in kv_rows[0]:
            parts.append(to_markdown("KV Memory Compression", kv_rows,
                ["bits","n_tokens","kv_bytes_adaptq","kv_bytes_fp16",
                 "compression_vs_fp16","compression_vs_fp32"]))
        if latency_rows and "error" not in (latency_rows[0] if latency_rows else {"error":1}):
            parts.append(to_markdown("Generation Latency", latency_rows,
                ["backend","n_generated_tokens","wall_time_ms","tokens_per_sec",
                 "latency_per_token_ms","compression_ratio"]))
        if snap_rows and "error" not in snap_rows[0]:
            parts.append(to_markdown("Snapshot Size (estimated)", snap_rows,
                ["n_tokens","bits","kv_bytes_adaptq","estimated_snap_size_kb",
                 "compression_ratio"]))
        output = "\n".join(parts)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"  Results written to: {out_path}")
    else:
        print(output)

    print("\n  ✅ Benchmark suite complete")


if __name__ == "__main__":
    main()

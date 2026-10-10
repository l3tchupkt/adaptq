"""
benchmarks/benchmark_adaptq_la.py
===================================
Isolated benchmarks for AdapTQ-LA production components.

Benchmark categories (MUST be kept separate per research protocol):
  A. Codec/quantization overhead (pack + unpack per token)
  B. Isolated C++ attention step (single-head loop)
  C. Full production runtime (multi-head, multi-sequence)

This script does NOT run model.generate() or claim end-to-end speedups.
Any performance claims based on this script are restricted to these
isolated kernel measurements.

Usage:
    python -m benchmarks.benchmark_adaptq_la
    python -m benchmarks.benchmark_adaptq_la --dim 128 --heads 14 --seq 256
"""
import argparse
import time
import sys
from pathlib import Path
import numpy as np

# Make sure we can import from the production package
sys.path.insert(0, str(Path(__file__).parent.parent))
from adaptq.core import Engine


def _warmup(engine, k, v, q, n=3):
    """Warmup runs to avoid cold-cache effects."""
    for _ in range(n):
        engine.reset()
        engine.append(k, v)
        engine.compute(q)


def benchmark_codec(dim=64, bits_list=None, seq_len=128, n_reps=1000):
    """
    Category A: Codec overhead per token (quantize + pack).
    Measured by timing a single Engine.append() call.
    """
    if bits_list is None:
        bits_list = [(4, 4), (4, 2), (3, 2), (16, 2), (16, 16)]

    print("\n=== A. Codec / Quantization Overhead (single append per token) ===")
    print(f"dim={dim}, seq fill={seq_len}, reps={n_reps}")
    print(f"{'Config':<20} {'Median (µs)':>12} {'P95 (µs)':>10} {'P99 (µs)':>10}")
    print("-" * 55)

    rng = np.random.default_rng(42)
    k = rng.standard_normal((1, dim)).astype(np.float32)  # single-head
    v = rng.standard_normal((1, dim)).astype(np.float32)

    for k_bits, v_bits in bits_list:
        engine = Engine(dim=dim, heads=1, k_bits=k_bits, v_bits=v_bits,
                        capacity=seq_len + 10, seed=42)
        # Fill to seq_len then measure one more append
        for _ in range(seq_len):
            engine.append(k, v)

        times = []
        for _ in range(n_reps):
            engine.reset()
            # Pre-fill outside timing
            for _ in range(seq_len):
                engine.append(k, v)
            t0 = time.perf_counter()
            engine.append(k, v)
            times.append((time.perf_counter() - t0) * 1e6)

        times.sort()
        label = f"K{k_bits}-V{v_bits}"
        print(f"{label:<20} {np.median(times):>12.2f} {np.percentile(times, 95):>10.2f} {np.percentile(times, 99):>10.2f}")

    return times  # last config only (for smoke test)


def benchmark_attention_step(dim=64, heads=14, seq_len=128, n_reps=200):
    """
    Category B: Isolated C++ attention compute step (single compute() call).
    This is the kernel benchmark reported in the paper.
    Dense reference uses k_bits=16, v_bits=16.
    AdapTQ-LA uses k_bits=4, v_bits=2.
    """
    print(f"\n=== B. Isolated Attention Step Benchmark ===")
    print(f"dim={dim}, heads={heads}, context={seq_len}, reps={n_reps}")
    print(f"{'Config':<25} {'Median (ms)':>12} {'tok/s':>10} {'P95 (ms)':>10}")
    print("-" * 60)

    rng = np.random.default_rng(7)
    ks = [rng.standard_normal((heads, dim)).astype(np.float32) for _ in range(seq_len)]
    vs = [rng.standard_normal((heads, dim)).astype(np.float32) for _ in range(seq_len)]
    q  = rng.standard_normal((heads, dim)).astype(np.float32)

    configs = [
        ("Dense (K16-V16)", 16, 16),
        ("AdapTQ-LA (K4-V2)", 4, 2),
        ("AdapTQ-LA (K3-V2)", 3, 2),
        ("Layer-0 (K16-V2)", 16, 2),
    ]

    results = {}
    for name, k_bits, v_bits in configs:
        engine = Engine(dim=dim, heads=heads, k_bits=k_bits, v_bits=v_bits,
                        capacity=seq_len + 10, seed=42)
        # Fill cache
        for k, v in zip(ks, vs):
            engine.append(k, v)

        # Warmup
        for _ in range(5):
            engine.compute(q)

        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            engine.compute(q)
            times.append((time.perf_counter() - t0) * 1e3)

        times.sort()
        median_ms = np.median(times)
        p95_ms = np.percentile(times, 95)
        tps = 1000.0 / median_ms

        print(f"{name:<25} {median_ms:>12.3f} {tps:>10.1f} {p95_ms:>10.3f}")
        results[name] = {"median_ms": median_ms, "tps": tps, "p95_ms": p95_ms}

    # Compute ratio vs dense
    if "Dense (K16-V16)" in results and "AdapTQ-LA (K4-V2)" in results:
        dense_ms = results["Dense (K16-V16)"]["median_ms"]
        la_ms    = results["AdapTQ-LA (K4-V2)"]["median_ms"]
        ratio    = dense_ms / la_ms
        print(f"\nNote: K4-V2 vs Dense ratio = {ratio:.2f}x (isolated kernel only)")
        print("This is NOT an end-to-end generation speedup claim.")

    return results


def benchmark_memory(dim=64, heads=14, seq_len=512):
    """
    Category C: Memory accounting per configuration.
    """
    print(f"\n=== C. KV Memory Accounting ===")
    print(f"dim={dim}, heads={heads}, context={seq_len}")
    print(f"{'Config':<25} {'KV Bytes':>12} {'MB':>8} {'vs Dense':>12}")
    print("-" * 60)

    rng = np.random.default_rng(42)
    ks = [rng.standard_normal((heads, dim)).astype(np.float32) for _ in range(seq_len)]
    vs = [rng.standard_normal((heads, dim)).astype(np.float32) for _ in range(seq_len)]

    configs = [
        ("Dense (K16-V16)", 16, 16),
        ("Uniform 4-bit",    4,  4),
        ("Uniform 3-bit",    3,  3),
        ("Uniform 2-bit",    2,  2),
        ("AdapTQ-LA K4-V2",  4,  2),
        ("AdapTQ-LA K3-V2",  3,  2),
        ("Layer-0 (K16-V2)", 16, 2),
    ]

    dense_bytes = None
    for name, k_bits, v_bits in configs:
        engine = Engine(dim=dim, heads=heads, k_bits=k_bits, v_bits=v_bits,
                        capacity=seq_len + 10, seed=42)
        for k, v in zip(ks, vs):
            engine.append(k, v)

        nb = engine.kv_bytes
        mb = nb / (1024 ** 2)
        if dense_bytes is None:
            dense_bytes = nb
            ratio_str = "1.00x (baseline)"
        else:
            ratio = dense_bytes / nb if nb > 0 else float('inf')
            ratio_str = f"{ratio:.2f}x"

        print(f"{name:<25} {nb:>12,} {mb:>8.3f} {ratio_str:>12}")


def main():
    parser = argparse.ArgumentParser(description="AdapTQ-LA isolated benchmarks")
    parser.add_argument("--dim",   type=int, default=64)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--seq",   type=int, default=128)
    parser.add_argument("--reps",  type=int, default=200)
    args = parser.parse_args()

    print("AdapTQ-LA Production Benchmark Suite")
    print("=" * 60)
    print("IMPORTANT: These are isolated kernel benchmarks.")
    print("No end-to-end generation speedup is claimed.")
    print("=" * 60)

    benchmark_codec(dim=args.dim, seq_len=args.seq, n_reps=min(args.reps * 5, 2000))
    benchmark_attention_step(dim=args.dim, heads=args.heads, seq_len=args.seq, n_reps=args.reps)
    benchmark_memory(dim=args.dim, heads=args.heads, seq_len=args.seq)


if __name__ == "__main__":
    main()

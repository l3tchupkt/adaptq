#!/usr/bin/env python3
"""
AdapTQ benchmark — measures quantization throughput, MSE vs bits, and memory.
Pure Python; no external dependencies.
"""

import math
import random
import time

# --------------------------------------------------------------------------
# Reference implementation (copied from demo.py for standalone use)
# --------------------------------------------------------------------------
CB2 = [-1.5104, -0.4528, 0.4528, 1.5104]
CB3 = [-2.1529, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1529]
CB4 = [-2.7326, -2.0690, -1.5860, -1.1880, -0.8491, -0.5480, -0.2633, 0.0000,
        0.2633,  0.5480,  0.8491,  1.1880,  1.5860,  2.0690,  2.7326,  3.5714]
CODEBOOKS = {2: CB2, 3: CB3, 4: CB4}

def fwht(x):
    n = len(x); h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                a, b = x[i+j], x[i+j+h]
                x[i+j], x[i+j+h] = a+b, a-b
        h <<= 1
    s = 1.0 / math.sqrt(n)
    for i in range(n): x[i] *= s
    return x

def next_pow2(n):
    p = 1
    while p < n: p <<= 1
    return p

class Quantizer:
    def __init__(self, d, seed=42):
        self.d = d; self.p = next_pow2(d)
        rng = random.Random(seed)
        self.D = [1 if rng.random() > 0.5 else -1 for _ in range(self.p)]

    def quantize(self, x, bits):
        p = self.p; cb = CODEBOOKS[bits]
        buf = list(x) + [0.0] * (p - len(x))
        norm = math.sqrt(sum(v*v for v in buf) + 1e-12)
        buf = [v/norm * self.D[i] for i,v in enumerate(buf)]
        fwht(buf)
        sp = math.sqrt(p)
        indices = [min(range(len(cb)), key=lambda i,v=buf[j]*sp: abs(v-cb[i]))
                   for j in range(p)]
        return indices, norm

    def dequantize(self, indices, norm, bits):
        cb = CODEBOOKS[bits]; p = self.p
        sp = 1.0/math.sqrt(p)
        buf = [cb[idx] * sp for idx in indices]   # recover y
        fwht(buf)                                  # D·x/norm
        buf = [buf[i] * self.D[i] for i in range(p)]  # x/norm
        return [v * norm for v in buf[:self.d]]

def rand_unit(d, rng):
    v = [rng.gauss(0,1) for _ in range(d)]
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v]

def mse(a, b):
    return sum((x-y)**2 for x,y in zip(a,b)) / len(a)

# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------
def bench_mse_vs_bits(d=128, n=200):
    print(f"\n{'─'*55}")
    print(f"  MSE vs bits  (d={d}, n={n} vectors)")
    print(f"{'─'*55}")
    print(f"  {'bits':>4}  {'MSE':>12}  {'theory_bound':>14}  {'ratio':>6}")
    rng = random.Random(1337)
    q = Quantizer(d, seed=42)
    for bits in [2, 3, 4]:
        total = 0.0
        for _ in range(n):
            x = rand_unit(d, rng)
            idx, norm = q.quantize(x, bits)
            xr = q.dequantize(idx, norm, bits)
            total += mse(x, xr)
        avg_mse = total / n
        # Theoretical bound: C_HAR * sqrt(3*pi)/2 * 4^{-b}
        theory = 1.08 * math.sqrt(3*math.pi)/2 * 4**(-bits)
        ratio  = avg_mse / theory
        print(f"  {bits:>4}  {avg_mse:>12.4e}  {theory:>14.4e}  {ratio:>6.3f}")

def bench_throughput(d=128, n=1000, bits=4):
    print(f"\n{'─'*55}")
    print(f"  Throughput benchmark  (d={d}, n={n}, bits={bits})")
    print(f"{'─'*55}")
    rng = random.Random(999)
    q = Quantizer(d, seed=42)
    vecs = [rand_unit(d, rng) for _ in range(n)]

    t0 = time.perf_counter()
    results = [q.quantize(x, bits) for x in vecs]
    t1 = time.perf_counter()
    q_time = (t1-t0)*1e6/n

    t0 = time.perf_counter()
    for idx, norm in results:
        q.dequantize(idx, norm, bits)
    t1 = time.perf_counter()
    dq_time = (t1-t0)*1e6/n

    packed_b = (d * bits + 7) // 8
    fp32_b   = d * 4
    print(f"  Avg quantize:   {q_time:8.1f} us/vec")
    print(f"  Avg dequantize: {dq_time:8.1f} us/vec")
    print(f"  Packed bytes:   {packed_b} B  (vs FP32: {fp32_b} B,  {fp32_b/packed_b:.1f}x)")

def bench_attention(d=128, seq_len=128, bits=4):
    print(f"\n{'─'*55}")
    print(f"  Attention simulation  (d={d}, seq={seq_len}, bits={bits})")
    print(f"{'─'*55}")
    rng = random.Random(7)
    q   = Quantizer(d, seed=42)
    scale = 1.0 / math.sqrt(d)

    # Build KV cache
    t0 = time.perf_counter()
    cache = []
    for _ in range(seq_len):
        k = rand_unit(d, rng); v = rand_unit(d, rng)
        cache.append((q.quantize(k, bits), q.quantize(v, bits)))
    t1 = time.perf_counter()
    build_ms = (t1-t0)*1e3

    # Run 50 queries
    n_q = 50; total_ms = 0.0
    for _ in range(n_q):
        qvec = rand_unit(d, rng)
        t0 = time.perf_counter()
        logits = []
        for (ki, kn), _ in cache:
            kr = q.dequantize(ki, kn, bits)
            logits.append(sum(qvec[j]*kr[j] for j in range(d)) * scale)
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        s    = sum(exps); ws = [e/s for e in exps]
        out  = [0.0]*d
        for w, ((vi, vn), _) in zip(ws, cache):
            vr = q.dequantize(vi, vn, bits)
            for j in range(d): out[j] += w*vr[j]
        t1 = time.perf_counter()
        total_ms += (t1-t0)*1e3

    fp16_kv = seq_len * d * 2 * 2
    q_kv    = seq_len * (d * bits + 7) // 8 * 2
    print(f"  Cache build:    {build_ms:.2f} ms")
    print(f"  Avg attn query: {total_ms/n_q:.3f} ms  ({seq_len} tokens)")
    print(f"  FP16 KV mem:    {fp16_kv/1024:.1f} KB")
    print(f"  Q{bits} KV mem:     {q_kv/1024:.1f} KB")
    print(f"  Compression:    {fp16_kv/q_kv:.1f}x")

def bench_ring_buffer(d=128, bits=4, capacity=512, total=2048):
    print(f"\n{'─'*55}")
    print(f"  Ring buffer FIFO  (cap={capacity}, insert={total})")
    print(f"{'─'*55}")
    rng = random.Random(55)
    q   = Quantizer(d, seed=42)
    buf = []  # circular buffer (list-based)
    head = 0; size = 0

    t0 = time.perf_counter()
    for i in range(total):
        k = rand_unit(d, rng); v = rand_unit(d, rng)
        qk = q.quantize(k, bits); qv = q.quantize(v, bits)
        if size < capacity:
            buf.append((qk, qv, i))
            size += 1
        else:
            buf[head] = (qk, qv, i)
            head = (head + 1) % capacity
    t1 = time.perf_counter()

    mem = size * (d * bits + 7) // 8 * 2
    print(f"  Inserted:       {total} tokens")
    print(f"  Buffer size:    {size}/{capacity}")
    print(f"  Fill time:      {(t1-t0)*1e3:.2f} ms")
    print(f"  KV memory:      {mem/1024:.1f} KB")
    print(f"  Evicted:        {max(0, total - capacity)} tokens")

if __name__ == '__main__':
    print("=" * 55)
    print("  AdapTQ Python Benchmark Suite")
    print("=" * 55)

    bench_mse_vs_bits(d=128, n=500)
    bench_throughput(d=64,  n=1000, bits=4)
    bench_throughput(d=128, n=1000, bits=4)
    bench_throughput(d=128, n=1000, bits=3)
    bench_throughput(d=128, n=1000, bits=2)
    bench_attention(d=128, seq_len=64,  bits=4)
    bench_attention(d=128, seq_len=256, bits=4)
    bench_ring_buffer(d=128, bits=4, capacity=512, total=2048)

    print("\n" + "=" * 55)
    print("  Benchmark complete.")
    print("=" * 55)

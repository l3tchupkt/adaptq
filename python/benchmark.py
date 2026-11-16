
#!/usr/bin/env python3
"""AdapTQ standalone Python benchmark suite."""

import math
import random
import time

CODEBOOKS = {
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [-2.1529, -1.3439, -0.7560, -0.2451,
        0.2451, 0.7560, 1.3439, 2.1529],
    4: [-2.7326, -2.0690, -1.5860, -1.1880,
        -0.8491, -0.5480, -0.2633, 0.0,
        0.2633, 0.5480, 0.8491, 1.1880,
        1.5860, 2.0690, 2.7326, 3.5714],
}


def fwht(x):
    x = list(x)
    n = len(x)

    if n == 0 or n & (n - 1):
        raise ValueError("Length must be a non-zero power of 2")

    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                a, b = x[i + j], x[i + j + h]
                x[i + j], x[i + j + h] = a + b, a - b
        h <<= 1

    scale = 1 / math.sqrt(n)
    return [v * scale for v in x]


def next_pow2(n):
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("Dimension must be a positive integer")

    return 1 << (n - 1).bit_length()


class Quantizer:
    def __init__(self, d, seed=42):
        self.d = d
        self.p = next_pow2(d)
        rng = random.Random(seed)
        self.D = [1 if rng.random() > 0.5 else -1 for _ in range(self.p)]

    def quantize(self, x, bits):
        if bits not in CODEBOOKS:
            raise ValueError("Supported bit widths are 2, 3, and 4")
        if len(x) != self.d:
            raise ValueError(f"Expected {self.d} values, got {len(x)}")

        x = [float(v) for v in x]
        if not all(math.isfinite(v) for v in x):
            raise ValueError("Input values must be finite")

        norm = math.sqrt(sum(v * v for v in x))
        if norm == 0:
            raise ValueError("Cannot quantize a zero vector")

        buf = x + [0.0] * (self.p - self.d)
        buf = [v * self.D[i] / norm for i, v in enumerate(buf)]
        buf = fwht(buf)

        cb = CODEBOOKS[bits]
        scale = math.sqrt(self.p)
        indices = [
            min(range(len(cb)), key=lambda i: abs(buf[j] * scale - cb[i]))
            for j in range(self.p)
        ]

        return indices, norm

    def dequantize(self, indices, norm, bits):
        if bits not in CODEBOOKS:
            raise ValueError("Supported bit widths are 2, 3, and 4")
        if len(indices) != self.p:
            raise ValueError(f"Expected {self.p} indices, got {len(indices)}")
        if not isinstance(norm, (int, float)) or not math.isfinite(norm) or norm < 0:
            raise ValueError("Norm must be finite and non-negative")

        cb = CODEBOOKS[bits]
        scale = 1 / math.sqrt(self.p)

        if any(not isinstance(i, int) or isinstance(i, bool) or
               not 0 <= i < len(cb) for i in indices):
            raise ValueError("Invalid codebook index")

        buf = [cb[i] * scale for i in indices]
        buf = fwht(buf)
        buf = [v * self.D[i] * norm for i, v in enumerate(buf)]

        return buf[:self.d]


def rand_unit(d, rng):
    v = [rng.gauss(0, 1) for _ in range(d)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def mse(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("Inputs must have equal, non-zero lengths")
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def packed_size(d, bits):
    return (d * bits + 7) // 8


def bench_mse_vs_bits(d=128, n=200):
    if n < 1:
        raise ValueError("Sample count must be positive")

    print(f"\n{'─' * 55}\n  MSE vs bits (d={d}, n={n})\n{'─' * 55}")
    print(f"  {'bits':>4}  {'MSE':>12}  {'theory_bound':>14}  {'ratio':>6}")

    rng = random.Random(1337)
    q = Quantizer(d)

    for bits in (2, 3, 4):
        errors = []
        for _ in range(n):
            x = rand_unit(d, rng)
            indices, norm = q.quantize(x, bits)
            errors.append(mse(x, q.dequantize(indices, norm, bits)))

        avg = sum(errors) / n
        theory = 1.08 * math.sqrt(3 * math.pi) / 2 * 4 ** (-bits)
        print(f"  {bits:>4}  {avg:>12.4e}  {theory:>14.4e}  {avg/theory:>6.3f}")


def bench_throughput(d=128, n=1000, bits=4):
    if n < 1:
        raise ValueError("Vector count must be positive")

    print(f"\n{'─' * 55}\n  Throughput (d={d}, n={n}, bits={bits})\n{'─' * 55}")

    rng = random.Random(999)
    q = Quantizer(d)
    vectors = [rand_unit(d, rng) for _ in range(n)]

    start = time.perf_counter()
    results = [q.quantize(x, bits) for x in vectors]
    q_time = (time.perf_counter() - start) * 1e6 / n

    start = time.perf_counter()
    for indices, norm in results:
        q.dequantize(indices, norm, bits)
    dq_time = (time.perf_counter() - start) * 1e6 / n

    packed = packed_size(d, bits)
    fp32 = d * 4

    print(f"  Avg quantize:   {q_time:8.1f} us/vec")
    print(f"  Avg dequantize: {dq_time:8.1f} us/vec")
    print(f"  Packed bytes:   {packed} B (FP32: {fp32} B, {fp32/packed:.1f}x)")


def bench_attention(d=128, seq_len=128, bits=4):
    if seq_len < 1:
        raise ValueError("Sequence length must be positive")

    print(f"\n{'─' * 55}\n  Attention (d={d}, seq={seq_len}, bits={bits})\n{'─' * 55}")

    rng = random.Random(7)
    q = Quantizer(d)
    scale = 1 / math.sqrt(d)

    start = time.perf_counter()
    cache = []
    for _ in range(seq_len):
        k, v = rand_unit(d, rng), rand_unit(d, rng)
        cache.append((q.quantize(k, bits), q.quantize(v, bits)))
    build_ms = (time.perf_counter() - start) * 1e3

    n_queries = 50
    total_ms = 0.0

    for _ in range(n_queries):
        query = rand_unit(d, rng)
        start = time.perf_counter()

        logits = [
            sum(a * b for a, b in zip(
                query, q.dequantize(ki, kn, bits)
            )) * scale
            for (ki, kn), _ in cache
        ]

        max_logit = max(logits)
        weights = [math.exp(v - max_logit) for v in logits]
        total = sum(weights)
        weights = [v / total for v in weights]

        output = [0.0] * d
        for weight, (_, (vi, vn)) in zip(weights, cache):
            value = q.dequantize(vi, vn, bits)
            for j in range(d):
                output[j] += weight * value[j]

        total_ms += (time.perf_counter() - start) * 1e3

    fp16_kv = seq_len * d * 4
    q_kv = seq_len * packed_size(d, bits) * 2

    print(f"  Cache build:    {build_ms:.2f} ms")
    print(f"  Avg attn query: {total_ms/n_queries:.3f} ms ({seq_len} tokens)")
    print(f"  FP16 KV mem:    {fp16_kv/1024:.1f} KB")
    print(f"  Q{bits} KV mem:     {q_kv/1024:.1f} KB")
    print(f"  Compression:    {fp16_kv/q_kv:.1f}x")


def bench_ring_buffer(d=128, bits=4, capacity=512, total=2048):
    if capacity < 1 or total < 0:
        raise ValueError("Capacity must be positive and total must be non-negative")

    print(f"\n{'─' * 55}\n  Ring buffer (cap={capacity}, insert={total})\n{'─' * 55}")

    rng = random.Random(55)
    q = Quantizer(d)
    buf = []
    head = size = 0

    start = time.perf_counter()

    for i in range(total):
        k, v = rand_unit(d, rng), rand_unit(d, rng)
        item = (q.quantize(k, bits), q.quantize(v, bits), i)

        if size < capacity:
            buf.append(item)
            size += 1
        else:
            buf[head] = item
            head = (head + 1) % capacity

    elapsed = (time.perf_counter() - start) * 1e3
    memory = size * packed_size(d, bits) * 2

    print(f"  Inserted:       {total} tokens")
    print(f"  Buffer size:    {size}/{capacity}")
    print(f"  Fill time:      {elapsed:.2f} ms")
    print(f"  KV memory:      {memory/1024:.1f} KB")
    print(f"  Evicted:        {max(0, total-capacity)} tokens")


def main():
    print("=" * 55)
    print("  AdapTQ Python Benchmark Suite")
    print("=" * 55)

    bench_mse_vs_bits(d=128, n=500)

    for d, bits in ((64, 4), (128, 4), (128, 3), (128, 2)):
        bench_throughput(d=d, n=1000, bits=bits)

    for seq_len in (64, 256):
        bench_attention(d=128, seq_len=seq_len, bits=4)

    bench_ring_buffer(d=128, bits=4, capacity=512, total=2048)

    print("\n" + "=" * 55)
    print("  Benchmark complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()

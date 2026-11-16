
#!/usr/bin/env python3
"""AdapTQ Python demo and reference implementation."""

import math
import os
import random
import subprocess
import sys


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def fwht(values):
    n = len(values)

    if n == 0 or n & (n - 1):
        raise ValueError("Input length must be a non-zero power of 2")

    result = list(values)

    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                a = result[i + j]
                b = result[i + j + h]
                result[i + j] = a + b
                result[i + j + h] = a - b
        h <<= 1

    scale = 1.0 / math.sqrt(n)
    return [value * scale for value in result]


def next_pow2(n):
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("Input must be a positive integer")

    p = 1
    while p < n:
        p <<= 1
    return p


CB2 = [-1.5104, -0.4528, 0.4528, 1.5104]
CB3 = [-2.1529, -1.3439, -0.7560, -0.2451,
       0.2451, 0.7560, 1.3439, 2.1529]
CB4 = [-2.7326, -2.0690, -1.5860, -1.1880,
       -0.8491, -0.5480, -0.2633, 0.0000,
       0.2633, 0.5480, 0.8491, 1.1880,
       1.5860, 2.0690, 2.7326, 3.5714]

CODEBOOKS = {2: CB2, 3: CB3, 4: CB4}


def quantize_scalar(value, codebook):
    if not codebook:
        raise ValueError("Codebook cannot be empty")

    return min(
        range(len(codebook)),
        key=lambda i: abs(value - codebook[i])
    )


def dequantize_scalar(index, codebook):
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError("Codebook index must be an integer")

    if not 0 <= index < len(codebook):
        raise ValueError("Codebook index out of range")

    return codebook[index]


class PyQuantizer:
    def __init__(self, d, seed=42):
        self.d = d
        self.p = next_pow2(d)

        rng = random.Random(seed)
        self.D = [
            1 if rng.random() > 0.5 else -1
            for _ in range(self.p)
        ]

    def quantize(self, x, bits):
        if bits not in CODEBOOKS:
            raise ValueError("Supported bit widths are 2, 3, and 4")

        if len(x) != self.d:
            raise ValueError(f"Expected {self.d} values, got {len(x)}")

        values = [float(v) for v in x]

        if not all(math.isfinite(v) for v in values):
            raise ValueError("Input values must be finite")

        p = self.p
        buf = values + [0.0] * (p - self.d)
        norm = math.sqrt(sum(v * v for v in buf))

        if norm == 0:
            raise ValueError("Cannot quantize a zero vector")

        buf = [buf[i] * self.D[i] / norm for i in range(p)]
        buf = fwht(buf)

        codebook = CODEBOOKS[bits]
        scale = math.sqrt(p)

        indices = [
            quantize_scalar(buf[i] * scale, codebook)
            for i in range(p)
        ]

        return indices, norm

    def dequantize(self, indices, norm, bits):
        if bits not in CODEBOOKS:
            raise ValueError("Supported bit widths are 2, 3, and 4")

        if len(indices) != self.p:
            raise ValueError(f"Expected {self.p} indices, got {len(indices)}")

        if not isinstance(norm, (int, float)) or not math.isfinite(norm) or norm < 0:
            raise ValueError("Norm must be a finite, non-negative number")

        codebook = CODEBOOKS[bits]
        scale = 1.0 / math.sqrt(self.p)

        buf = [
            dequantize_scalar(index, codebook) * scale
            for index in indices
        ]

        buf = fwht(buf)
        buf = [buf[i] * self.D[i] for i in range(self.p)]

        return [value * norm for value in buf[:self.d]]


def rand_unit_vec(d, rng):
    if not isinstance(d, int) or isinstance(d, bool) or d < 1:
        raise ValueError("Vector dimension must be a positive integer")

    values = [rng.gauss(0, 1) for _ in range(d)]
    norm = math.sqrt(sum(value * value for value in values))

    if norm == 0:
        raise ValueError("Generated vector has zero magnitude")

    return [value / norm for value in values]


def mse(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("Inputs must have equal, non-zero lengths")

    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def run_python_demo():
    print("=" * 50)
    print("  AdapTQ Python Reference Demo")
    print("=" * 50)

    d = 128
    rng = random.Random(2024)

    print(f"\n--- Quantization MSE (d={d}) ---")

    quantizer = PyQuantizer(d, seed=999)
    x = rand_unit_vec(d, rng)

    for bits in (2, 3, 4):
        indices, norm = quantizer.quantize(x, bits)
        reconstructed = quantizer.dequantize(indices, norm, bits)

        error = mse(x, reconstructed)
        packed_bytes = (d * bits + 7) // 8
        ratio = (d * 4) / packed_bytes

        print(
            f"  {bits}-bit: MSE={error:.4e}  "
            f"packed={packed_bytes}B  ratio={ratio:.1f}x"
        )

    print("\n--- Attention Simulation ---")

    seq_len = 64
    attn_quantizer = PyQuantizer(d, seed=42)

    keys = []
    values = []

    for _ in range(seq_len):
        k = rand_unit_vec(d, rng)
        v = rand_unit_vec(d, rng)

        ki, kn = attn_quantizer.quantize(k, 4)
        vi, vn = attn_quantizer.quantize(v, 4)

        keys.append((ki, kn))
        values.append((vi, vn))

    query = rand_unit_vec(d, rng)
    scale = 1.0 / math.sqrt(d)

    logits = []

    for indices, norm in keys:
        reconstructed_key = attn_quantizer.dequantize(indices, norm, 4)
        logit = sum(
            query[j] * reconstructed_key[j]
            for j in range(d)
        ) * scale

        logits.append(logit)

    max_logit = max(logits)
    exp_logits = [math.exp(value - max_logit) for value in logits]
    total = sum(exp_logits)
    weights = [value / total for value in exp_logits]

    output = [0.0] * d

    for (indices, norm), weight in zip(values, weights):
        reconstructed_value = attn_quantizer.dequantize(
            indices, norm, 4
        )

        for j in range(d):
            output[j] += weight * reconstructed_value[j]

    output_norm = math.sqrt(sum(value * value for value in output))
    entropy = -sum(
        weight * math.log(weight + 1e-12)
        for weight in weights
    )

    print(f"  Seq len: {seq_len}  |attn_out|={output_norm:.4f}")
    print(f"  Top attention weight: {max(weights):.4f}")
    print(f"  Attention entropy: {entropy:.3f} nats")

    print("\n--- Memory Comparison ---")

    fp16_bytes = seq_len * d * 2 * 2
    q4_bytes = seq_len * ((d * 4 + 7) // 8) * 2

    print(f"  FP16 KV: {fp16_bytes / 1024:.1f} KB")
    print(f"  4-bit quantized KV: {q4_bytes / 1024:.1f} KB")
    print(f"  Compression: {fp16_bytes / q4_bytes:.1f}x")


def run_cpp_binary():
    build_dir = os.path.join(os.path.dirname(__file__), "..", "build")

    for candidate in ("adapTQ_demo", "adapTQ_demo.exe"):
        path = os.path.join(build_dir, candidate)

        if not os.path.isfile(path):
            continue

        print(f"\n{'=' * 50}")
        print(f"  Running compiled C++ binary: {candidate}")
        print(f"{'=' * 50}")

        try:
            result = subprocess.run([path], check=False)
            return result.returncode
        except OSError as exc:
            print(f"[error] Could not run C++ binary: {exc}", file=sys.stderr)
            return 1

    print("\n[info] C++ binary not found in build/. Run cmake + make first.")
    return 0


if __name__ == "__main__":
    run_python_demo()
    sys.exit(run_cpp_binary())

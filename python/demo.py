#!/usr/bin/env python3
"""
AdapTQ Python demo — calls the compiled C++ binary and parses its output.
Also provides a pure-Python reference implementation for validation.
"""

import subprocess
import os
import math
import random

# --------------------------------------------------------------------------
# Pure-Python reference: FWHT
# --------------------------------------------------------------------------
def fwht(x):
    n = len(x)
    assert (n & (n - 1)) == 0, "length must be power of 2"
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                a, b = x[i + j], x[i + j + h]
                x[i + j], x[i + j + h] = a + b, a - b
        h <<= 1
    scale = 1.0 / math.sqrt(n)
    for i in range(n): x[i] *= scale
    return x

def next_pow2(n):
    p = 1
    while p < n: p <<= 1
    return p

# --------------------------------------------------------------------------
# Pure-Python reference: codebooks
# --------------------------------------------------------------------------
CB2 = [-1.5104, -0.4528, 0.4528, 1.5104]
CB3 = [-2.1529, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1529]
CB4 = [-2.7326, -2.0690, -1.5860, -1.1880, -0.8491, -0.5480, -0.2633, 0.0000,
        0.2633,  0.5480,  0.8491,  1.1880,  1.5860,  2.0690,  2.7326,  3.5714]

CODEBOOKS = {2: CB2, 3: CB3, 4: CB4}

def quantize_scalar(v, cb):
    return min(range(len(cb)), key=lambda i: abs(v - cb[i]))

def dequantize_scalar(idx, cb):
    return cb[idx]

# --------------------------------------------------------------------------
# Pure-Python reference: HAR + MSE quantizer
# --------------------------------------------------------------------------
class PyQuantizer:
    def __init__(self, d, seed=42):
        self.d = d
        self.p = next_pow2(d)
        rng = random.Random(seed)
        self.D = [1 if rng.random() > 0.5 else -1 for _ in range(self.p)]

    def quantize(self, x, bits):
        p = self.p
        buf = list(x) + [0.0] * (p - len(x))
        norm = math.sqrt(sum(v*v for v in buf) + 1e-12)
        buf = [v / norm for v in buf]
        # apply D
        buf = [buf[i] * self.D[i] for i in range(p)]
        # fwht
        fwht(buf)
        cb  = CODEBOOKS[bits]
        sp  = math.sqrt(p)
        indices = [quantize_scalar(buf[i] * sp, cb) for i in range(p)]
        return indices, norm

    def dequantize(self, indices, norm, bits):
        cb = CODEBOOKS[bits]
        p  = self.p
        sp = 1.0 / math.sqrt(p)
        # recover rotated coords y (fwht_normed(D·x/norm))
        buf = [dequantize_scalar(idx, cb) * sp for idx in indices]
        # fwht_normed(y) = D·x/norm  (self-inverse property)
        fwht(buf)
        # apply D to cancel: D·(D·x/norm) = x/norm
        buf = [buf[i] * self.D[i] for i in range(p)]
        return [v * norm for v in buf[:self.d]]

def rand_unit_vec(d, rng):
    v = [rng.gauss(0, 1) for _ in range(d)]
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v]

def mse(a, b):
    return sum((x-y)**2 for x,y in zip(a,b)) / len(a)

# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
def run_python_demo():
    print("=" * 50)
    print("  AdapTQ Python Reference Demo")
    print("=" * 50)

    d   = 128
    rng = random.Random(2024)

    print(f"\n--- Quantization MSE (d={d}) ---")
    q = PyQuantizer(d, seed=999)
    x = rand_unit_vec(d, rng)

    for bits in [2, 3, 4]:
        indices, norm = q.quantize(x, bits)
        x_rec = q.dequantize(indices, norm, bits)
        err   = mse(x, x_rec)
        packed_bytes = (d * bits + 7) // 8
        ratio = (d * 4) / packed_bytes
        print(f"  {bits}-bit: MSE={err:.4e}  packed={packed_bytes}B  ratio={ratio:.1f}x")

    print("\n--- Attention Simulation ---")
    seq_len = 64
    attn_q  = PyQuantizer(d, seed=42)
    ks, vs  = [], []
    for i in range(seq_len):
        k = rand_unit_vec(d, rng)
        v = rand_unit_vec(d, rng)
        ki, kn = attn_q.quantize(k, 4)
        vi, vn = attn_q.quantize(v, 4)
        ks.append((ki, kn))
        vs.append((vi, vn))

    query = rand_unit_vec(d, rng)
    scale = 1.0 / math.sqrt(d)

    logits = []
    for ki, kn in ks:
        k_rec = attn_q.dequantize(ki, kn, 4)
        logit = sum(query[j] * k_rec[j] for j in range(d)) * scale
        logits.append(logit)

    max_l = max(logits)
    exp_l = [math.exp(l - max_l) for l in logits]
    s     = sum(exp_l)
    weights = [e/s for e in exp_l]

    out = [0.0] * d
    for (vi, vn), w in zip(vs, weights):
        v_rec = attn_q.dequantize(vi, vn, 4)
        for j in range(d): out[j] += w * v_rec[j]

    out_norm = math.sqrt(sum(x*x for x in out))
    print(f"  Seq len: {seq_len}  |attn_out|={out_norm:.4f}")
    print(f"  Top attention weight: {max(weights):.4f}")
    print(f"  Attention entropy: {-sum(w*math.log(w+1e-12) for w in weights):.3f} nats")

    print("\n--- Memory Comparison ---")
    fp16_bytes = seq_len * d * 2 * 2  # K+V, 2 bytes each
    q4_bytes   = seq_len * (d * 4 + 7) // 8 * 2
    print(f"  FP16 KV: {fp16_bytes/1024:.1f} KB")
    print(f"  4-bit quantized KV: {q4_bytes/1024:.1f} KB")
    print(f"  Compression: {fp16_bytes/q4_bytes:.1f}x")

# --------------------------------------------------------------------------
# Run C++ binary if available
# --------------------------------------------------------------------------
def run_cpp_binary():
    build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
    for candidate in ['adapTQ_demo', 'adapTQ_demo.exe']:
        path = os.path.join(build_dir, candidate)
        if os.path.isfile(path):
            print(f"\n{'='*50}")
            print(f"  Running compiled C++ binary: {candidate}")
            print(f"{'='*50}")
            result = subprocess.run([path], capture_output=False, text=True)
            return result.returncode
    print("\n[info] C++ binary not found in build/. Run cmake + make first.")
    return 0

if __name__ == '__main__':
    run_python_demo()
    run_cpp_binary()

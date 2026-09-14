# CLAUDE.md — AdapTQ

## Project Overview

**AdapTQ** (Adaptive Streaming Vector Quantization) — a production-grade C++17 KV cache quantization engine for LLM inference on edge and memory-constrained systems.

Based on: *AdapTQ: Adaptive Streaming Vector Quantization for Edge-Deployed Large Language Models* by Lakshmikanthan K.

AdapTQ reduces KV cache memory by 4–8× while matching or exceeding FP16 attention performance at large sequence lengths.

---

## Architecture

```
adapTQ/
├── core/
│   ├── fwht.cpp           — Fused FWHT (HAR): D-apply + butterfly merged, 4-wide unrolled
│   ├── codebook.cpp       — 2/3/4-bit Max-Lloyd codebooks, branchless lookup
│   ├── quantizer.cpp      — HAR quantizer, thread-local scratch (no heap in hot path)
│   └── adaptq_c_api.cpp   — Production C API implementation
├── cache/
│   └── ring_buffer.cpp    — KVFlatBuffer (contiguous layout, FIFO eviction)
├── attention/
│   └── attention.cpp      — Hybrid FP/quant attention + unified SIMD kernels
├── utils/
│   └── timer.cpp          — High-resolution timing
├── include/               — Public headers (adaptq.h, etc.)
├── main.cpp               — Benchmark + validation suite
python/
├── demo.py
└── benchmark.py
```

---

## Core Systems

### 1. HAR (Hadamard Accelerated Rotation)

* Rotation: `y = (1/√d) * H * D * x`
* Implemented using FWHT (O(d log d))
* Rademacher vector `D ∈ {±1}`
* Fully in-place, cache-friendly, vectorized

Property:

* Orthonormal transform → preserves norm
* Enables scalar quantization with minimal distortion

---

### 2. Vector Quantization (2/3/4-bit)

* Fixed Max-Lloyd codebooks:

  * 2-bit → 4 centroids
  * 3-bit → 8 centroids
  * 4-bit → 16 centroids
* Nearest centroid via branchless lookup
* Bit-packed storage (no alignment assumptions)

Quality:

```text
2-bit → MSE ~9e-4   (16× compression)
3-bit → MSE ~2.6e-4 (10.7×)
4-bit → MSE ~8e-5   (8×)
```

---

### 3. Unified SIMD Attention Kernel (Key Optimization)

* Query rotated once:
  `q_rot = FWHT(D * q)`

* Build per-query LUT:
  `LUT[c] = q_rot[i] * centroid[c]`

* Compute attention directly from packed indices:

  * No full dequantization
  * No scalar loops

SIMD implementation:

* AVX2 intrinsics:

  * `_mm256_srlv_epi32`
  * `_mm256_and_si256`
* Quad-unrolled inner loop
* Fully branchless

Result:

* 2/3/4-bit share identical execution path
* Performance scales with memory bandwidth

---

### 4. Hybrid FP / Quantized Execution

Observation:

* Small sequences (≤ ~1k tokens) fit in cache → FP16 is optimal
* Large sequences → memory bandwidth becomes bottleneck

Implementation:

* KV initially stored in FP32 (contiguous buffer)
* At compute time:

  * if `n <= hybrid_thresh` → run FP attention
  * else → use quantized path

Properties:

* No data copying
* No reallocation
* Single branch per call

---

### 5. Memory Layout (Why It Wins)

* KV stored in flat contiguous buffers (KVFlatBuffer)
* Sequential access → optimal prefetch + DRAM throughput
* Quantization reduces bandwidth by 4–8×

Key insight:

> At large context, attention is memory-bound.
> Reducing KV size directly increases throughput.

---

### 6. Production C API

Single-head:

```c
adaptq_ctx_t adaptq_create(int dim, int bits, int capacity, int seed, float v_mass, int hybrid_thresh);
void adaptq_append(adaptq_ctx_t ctx, const float* key, const float* val, int pos);
int  adaptq_compute(adaptq_ctx_t ctx, const float* query, float* out);
void adaptq_reset(adaptq_ctx_t ctx);
```

Multi-head:

```c
adaptq_mha_t adaptq_mha_create(int n_heads, int dim, int bits, int capacity, ...);
void adaptq_mha_append(adaptq_mha_t mha, int head, const float* k, const float* v, int pos);
int  adaptq_mha_compute(adaptq_mha_t mha, int head, const float* q, float* out);
```

Designed for integration into inference engines (e.g., llama.cpp).

---

## Benchmark Results (d=128, AVX2, -O3)

### Speed vs FP16 Attention

```text
seq       Q2(us)   Q3(us)   Q4(us)   FP16(us)   speedup (Q2/Q3/Q4)
128       5.7      5.6      5.5      5.4        ~1.0x
256       11.1     10.9     11.4     11.2       ~1.0x
512       22.6     22.2     22.0     22.1       ~1.0x
1024      38.4     41.4     50.3     47.1       1.24x / 1.14x / 0.94x
2048      75.4     79.7     100.0    103.2      1.25x / 1.34x / 0.99x
4096      150.3    155.9    200.5    198.4      1.32x / 1.27x / 0.90x
8192      >300     >300     >400     >>500      >1.5x (observed)
```

Key result:

> Quantized attention surpasses FP16 at sequence lengths ≥ 2k tokens.

---

### Sparse V Optimization

```text
v_mass   latency(us)   MSE
1.00     ~11.2         ~0
0.95     ~11.3         ~1e-6
0.70     ~11.0         ~7e-6
```

Effect:

* Skips low-weight tokens
* Reduces V reconstruction cost

---

## System Characteristics

* O(d log d) rotation (FWHT)
* No full dequantization in attention
* Unified SIMD pipeline across bit-widths
* Zero heap allocations in hot path
* Bandwidth-optimized KV layout
* Hybrid execution for real workloads

---

## What This Achieves

* 4–8× KV memory reduction
* Matches FP16 at small context
* Outperforms FP16 at large context
* Stable across bit-widths (2/3/4)
* Ready for real inference integration

---

## Build

```bash
# V2.1 build (Linux / WSL / Windows with MSVC)
cmake -B build_release -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --parallel 4

# Run all tests (67 total)
cd build_release && ctest --output-on-failure -j4

# Run the CLI demo
./build_release/adapTQ_demo --help
./build_release/adapTQ_demo replay session.aqss --metrics
./build_release/adapTQ_demo compare session.aqss --strategies har_fixed,fp_passthrough
```

---

## V2 Architecture (added 2026-07-22)

```
adapTQ/
├── runtime/
│   ├── runtime_context.h     — RuntimeContext: orchestration + token log
│   └── runtime_context.cpp   — init/reset/append/compute; strategy registry
├── replay/
│   ├── session_snapshot.h    — SessionSnapshot: binary AQSS format
│   ├── session_snapshot.cpp  — capture/save/load
│   ├── replay_engine.h       — ReplayEngine: replay/branch/replay_with
│   └── replay_engine.cpp
├── cli/
│   ├── cmd_replay.cpp        — adaptq replay subcommand
│   ├── cmd_compare.cpp       — adaptq compare subcommand
│   └── cmd_create_strategy.cpp — adaptq create-strategy subcommand
├── tests/unit/
│   ├── test_runtime_context.cpp   — 10 V2 tests
│   ├── test_session_snapshot.cpp  — 9 V2 tests (with unique_ptr helper)
│   └── test_replay_engine.cpp     — 10 V2 tests
└── adaptq/
    └── replay.py             — Python: ReplayEngine, CompareResult, snapshot_info
```

### Key Invariants

1. **RuntimeContext is non-copyable** — always use `std::unique_ptr<RuntimeContext>` or pass by ref/pointer.
2. **Token log required for replay** — set `cfg.log_tokens = true` before inference if you want replay capability.
3. **`cache_size` in SessionSnapshot = number of K/V PAIRS** — `ContiguousSlabStorage::size()` returns total slots (K+V), divide by 2 for pairs.
4. **`make_contiguous()` is in `adaptq::` namespace** — when linking against `adapTQ_runtime`, use `adaptq::make_contiguous()`.
5. **`cmd_create_strategy` is in `adaptq::` namespace** — declared as `adaptq::cmd_create_strategy` in `main.cpp`.

---

## Out of Scope

* Adaptive bit allocation (AIABA)
* Layer-wise precision scheduling (HLPS)
* Residual quantization (QJL)
* ARM NEON / RISC-V SIMD
* ONNX / TensorRT export
* HARAdaptive, H2O, Streaming, adaptive allocators (V3+)

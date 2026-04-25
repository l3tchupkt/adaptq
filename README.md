# AdapTQ: Adaptive Streaming Vector Quantization

**AdapTQ** is a C++17 KV cache quantization engine for LLM inference on edge and memory-constrained systems. It reduces KV cache memory footprint by **4–8×** while matching or exceeding FP16 attention performance at large sequence lengths. 

Designed independently of any specific ML framework, AdapTQ provides a stable C API and C++ vtable backend interface, alongside pluggable adapters for zero-copy integration into systems like `llama.cpp` and Python (`pybind11`).

## 🚀 Key Features

- **4-8x Memory Reduction**: 2/3/4-bit Max-Lloyd vector quantization codebooks.
- **Unified SIMD Pipeline**: 2, 3, and 4-bit decoding share a single, quad-unrolled branchless inner loop using AVX2 intrinsics. No scalar fallback loops in the hot path.
- **Fast Hadamard Rotation (HAR)**: $O(d \log d)$ fully in-place bounded rotation via Fast Walsh-Hadamard Transform effectively normalizes distributions and eliminates outliers.
- **Zero Heap Allocations**: Pure stack/thread-local memory buffers in the hot path.
- **Pre-Compute LUTs**: Dot products are executed directly against packed indices in SIMD registers—avoiding full dequantization inside the attention kernel.
- **Hybrid FP/Quant Execution**: Uses native FP32 for short sequences (cache-friendly) and dynamically seamlessly drops down to quantized evaluation for long, memory-bound sequences.

---

## 🏗 Architecture

```text
adapTQ/
├── include/
│   ├── adaptq.h                  ← Stable C API
│   ├── adaptq_backend.h          ← IAdaptQBackend C++ vtable + C vtable struct
│   └── adaptq_mha_backend.h      ← AdaptQMHABackend wrap
├── core/
│   ├── fwht.cpp                  ← Fused FWHT (HAR): D-apply + butterfly merged
│   ├── codebook.cpp              ← 2/3/4-bit Max-Lloyd codebooks, branchless lookup
│   ├── quantizer.cpp             ← HAR quantizer + ±3σ soft-clip variance norm
│   └── adaptq_c_api.cpp          ← C API Implementation
├── adapters/                     ← Framework Integration Layer
│   ├── adapter_standalone.c      ← Plain-C minimal implementation example
│   ├── adapter_llamacpp.cpp      ← `llama.cpp` adapter (zero GGML headers needed)
│   └── adapter_python.cpp        ← `pybind11` native Python adapter
├── cache/
│   └── ring_buffer.cpp           ← Contiguous flat buffer for sequential access
├── attention/
│   └── attention.cpp             — Hybrid FP/quant attention + SIMD kernels
└── utils/
```

## 🛠 Building

Requirements: `g++` (C++17, AVX2 support) or `clang++`, GNU Make.

```bash
# Clone the repository
git clone https://github.com/l3tchupkt/adaptq.git
cd adaptq

# Build the core library (libadaptq.so) and C++ demo
make

# Build all adapters (C Standalone & Llama.cpp shared lib)
make adapters

# Run the C standalone verification test
make test

# Install to /usr/local/lib
sudo make install
```

### Python Native Application

Use the heavily simplified native python API to manage sequence context directly in multi-head arrays implicitly.

```python
import numpy as np
from adaptq import Engine

engine = Engine(dim=128, heads=4, bits=4, capacity=2048)

k = np.random.randn(4, 128).astype(np.float32)
v = np.random.randn(4, 128).astype(np.float32)
q = np.random.randn(4, 128).astype(np.float32)

# Appends across all heads seamlessly
engine.append(k, v)
# Computes cross-attention dynamically 
output = engine.compute(q)
```

### PyTorch Integration (nn.Module)

For deep learning context flows, pass standard continuous PyTorch `(batch, heads, dim)` generation tensors through AdapTQ as a standard drop-in `torch.nn.Module`:

```python
import torch
from adaptq import AdaptQAttention

layer = AdaptQAttention(dim=128, heads=4, bits=4)

k = torch.randn(1, 4, 128)
v = torch.randn(1, 4, 128)
q = torch.randn(1, 4, 128)

out = layer(q, k, v)
```

## 📊 Real-world Benchmarks

Tested on standard AVX2 desktop hardware, using 4 heads, dimension 128, sequence reaching **4096 tokens**, and batch multi-query verification.

### Speed vs FP32 Baseline

*Quantized attention surpasses FP32/FP16 significantly for multi-thousand token contexts because large scale attention becomes heavily bandwidth-bound.*

**Final Stable Summary (4-bit, Sequence ≥ 256):**
- **Throughput**: ~1,100+ tokens/sec
- **Speedup**: ~10.18x stable average over FP32 NumPy equivalent
- **Memory**: 2.10 MB vs FP16 equivalent 8.39 MB (4.0x smaller)

### Quantization Fidelity

- **Cosine Similarity (vs FP32)**: `~0.947` stable average (1.000 = expected perfection)
- **Mean Squared Error (MSE)**: `~1.8e-04`

AdapTQ utilizes a specific $\pm 3\sigma$ variance bounding methodology on the HAR transformations. Unpredictable outlier inflation is smoothed mathematically to prevent maximum codebook distortion without breaking dot-product isometry.

## 🔌 Integration

AdapTQ features three layers of APIs: The standard un-opinionated C API, the vtable C++ API, and the Adapter API.

### C API (Standard Runtime)
Include `adaptq.h` and link against `libadaptq.so`.

```c
adaptq_mha_t mha = adaptq_mha_create(N_HEADS, HEAD_DIM, BITS, CAPACITY, 0, 0.95f, 512);

// Decode token: Add current K, V vectors
adaptq_mha_append(mha, head_idx, k_ptr, v_ptr, pos);

// Compute context: Query the attention
adaptq_mha_compute(mha, head_idx, q_ptr, out_ptr);

// Batched MHA evaluate
adaptq_mha_compute_batch(mha, head_idx, q_batch, num_queries, out_batch);
```

### llama.cpp Adapter

Using `AdapTQ` as the native KV Cache replacement during computation phase over GGML.
*(Requires using `llm_build_kqv` hooks. See `/integration/llama_cpp_patch.md` for full unified patch details.)*

```cpp
#include "adapters/adapter_llamacpp.h"

// Instantiate
LlamaCppAdaptQAdapter adapter(n_heads, head_dim, bits, capacity, seed, v_mass, hybrid_thr);

// Feed during graph resolution (raw float arrays)
adapter.feed_kv(head, key_array, val_array, token_pos);

// Output lookup
adapter.attention(head, query_array, out_array);
```

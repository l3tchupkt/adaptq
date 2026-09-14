# AdapTQ 0.2.3: Architectural Forensics & Abstraction Boundaries

## Current Architecture (0.2.2 Baseline)

### Dependency Map & Ownership Model
- **C API (`adaptq_c_api.cpp`)**: Exposes `adaptq_ctx_t` which wraps `AdapTQCtx`. This context directly owns an `AttentionHead` instance.
- **`AttentionHead`**: Owns its internal algorithmic components directly:
  - `Quantizer quant`: Responsible for FWHT and quantization/dequantization.
  - `KVFlatBuffer kv_buf`: Responsible for contiguous memory allocation of `k_data`, `v_data`, scales, and managing the ring buffer `head` and `size`.
- **Inner Loop (`attention.cpp`)**: 
  - The `AttentionHead::compute()` method performs the FWHT on the query.
  - It handles dynamic fallback to AVX2 via compiler flags `__builtin_cpu_supports` *within the method itself*.
  - Depending on the detected capabilities and `bits`, it explicitly calls either the template `compute_avx2<BITS>` or falls back to a scalar scanning loop (`kdot_scalar` and `vaccum_scalar`).

### Data Flow
1. **Append**: `adaptq_append()` -> `AttentionHead::append_kv()`. Key/val are quantized into local thread buffers via `Quantizer` and then `kv_buf.insert()` is called to copy them into the contiguous ring buffer. The FP32 variants are retained in `raw_kv` for the hybrid path.
2. **Compute**: `adaptq_compute()` -> `AttentionHead::compute()`. Queries are rotated via `fwht_forward()`. The AVX2/Scalar kernels compute un-normalized logits across the contiguous keys, apply softmax, accumulate values, and perform `fwht_inverse()`.

### Known Constraints & Hot Paths
- The hot path is heavily optimized with prefetching, 8-wide unrolling, and AVX2 `permutevar8x32`. 
- Intruding on the inner loop (per-token execution) with a vtable or virtual function call would destroy branch predictability and instruction cache locality. 

## Proposed Abstraction Boundaries (0.2.3)

To decouple the architecture without changing the numerical path or incurring vtable overhead in the inner loop, we will introduce three new interfaces:

### 1. `IStorageBackend`
Abstracts memory allocation, cache append, and capacity checking. 
- **Legacy Adapter**: `ContiguousStorageBackend` (which will wrap/replace the existing `KVFlatBuffer`).
- **Flow**: Returns the necessary pointers (`k_data`, `v_data`, `slots`) for the kernel to operate on. 

### 2. `IKernelBackend`
Abstracts the hardware-specific execution of the attention scan.
- **Legacy Adapters**: `AVX2KernelBackend` and `ScalarKernelBackend`.
- **Flow**: The kernel will expose a single coarse-grained virtual method (e.g. `compute_attention_scan(...)`) which takes the query, codebook, slot arrays, and storage pointers. This ensures exactly **one virtual call per head-query**, avoiding per-token overhead.

### 3. `IPolicy`
Abstracts retention, precision, and importance decisions. 
- **Legacy Adapter**: `DefaultPolicy` (will return the globally configured bits and no eviction).

### Integration into `AttentionHead`
`AttentionHead` will replace `KVFlatBuffer` with `std::unique_ptr<IStorageBackend>`. 
Instead of hardcoded `if (use_avx2)` branches, it will hold `std::unique_ptr<IKernelBackend>` (injected at creation by `KernelFactory`) and dispatch the coarse-grained scan.
The C API will retain ABI compatibility by defaulting to these legacy adapters.

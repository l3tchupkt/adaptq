# AdapTQ → llama.cpp Integration Guide

## Overview

This document shows exactly **where and how** to wire AdapTQ into llama.cpp.
Every change is self-contained; the existing FP16 path is kept as a fallback.

---

## Prerequisites

```bash
# Build AdapTQ shared library
cd /path/to/AdaptQ/adapTQ
make libadaptq.so

# Copy to llama.cpp root
cp libadaptq.so       /path/to/llama.cpp/
cp include/adaptq.h   /path/to/llama.cpp/
```

Add to llama.cpp's `CMakeLists.txt`:
```cmake
target_include_directories(llama PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(llama PRIVATE ${CMAKE_SOURCE_DIR}/libadaptq.so)
```

---

## Phase 1 — KV Cache Patch

### File: `llama.cpp` — struct `llama_kv_cache`

Locate the existing struct (around line 1300 in llama.cpp main branch):

```diff
 struct llama_kv_cache {
     bool has_shift = false;
     ...
     struct ggml_tensor * k = NULL;
     struct ggml_tensor * v = NULL;
     ...
+    // AdapTQ quantized KV — one MHA context per transformer layer
+    // NULL when AdapTQ is disabled.
+    void*  adaptq_mha = NULL;   // adaptq_mha_t per layer
+    int    adaptq_bits = 4;
+    bool   adaptq_enabled = false;
+    int    adaptq_hybrid_thresh = 1024;  // use FP16 below this seq len
 };
```

---

### File: `llama.cpp` — `llama_new_context_with_model`

Find where `kv_self` is initialised and add after the existing ggml tensor allocations:

```diff
+    // --- AdapTQ: initialise quantized KV caches ---
+    const bool use_adaptq = ggml_cpu_has_avx2()  // AVX2 required for max perf
+                         && (cparams.n_ctx >= 1024);  // only worthwhile at long ctx
+    if (use_adaptq) {
+        const int n_heads    = hparams.n_head;
+        const int head_dim   = hparams.n_embd / n_heads;
+        const int capacity   = cparams.n_ctx;
+        const int bits       = 4;          // 4-bit KV: 8× memory vs FP32
+        const float v_mass   = 0.95f;      // sparse-V: cover 95% softmax mass
+        const int hybrid_thr = ctx->kv_self.adaptq_hybrid_thresh;
+
+        // One MHA handle per transformer layer
+        ctx->kv_self.adaptq_layers.resize(hparams.n_layer);
+        for (int layer = 0; layer < (int)hparams.n_layer; ++layer) {
+            uint64_t seed = (uint64_t)layer * 0x9e3779b97f4a7c15ULL;
+            ctx->kv_self.adaptq_layers[layer] = adaptq_mha_create(
+                n_heads, head_dim, bits, capacity,
+                seed, v_mass, hybrid_thr);
+        }
+        ctx->kv_self.adaptq_enabled = true;
+        fprintf(stderr, "[AdapTQ] Enabled: %d layers × %d heads × %d-bit, "
+                "capacity=%d, hybrid_thresh=%d\n",
+                hparams.n_layer, n_heads, bits, capacity, hybrid_thr);
+    }
```

Also add cleanup in `llama_free`:

```diff
+    for (auto* mha : ctx->kv_self.adaptq_layers)
+        if (mha) adaptq_mha_destroy(mha);
+    ctx->kv_self.adaptq_layers.clear();
```

---

## Phase 2 — KV Store Intercept

### File: `llama.cpp` — inside `llm_build_kv_store` (or equivalent)

After computing `Kcur`/`Vcur` tensors, **before** writing them into the ggml K/V cache:

```diff
+    // --- AdapTQ: quantize and cache K/V ---
+    if (kv_self.adaptq_enabled) {
+        // Extract K and V as float arrays from ggml tensors
+        // (in llama.cpp these are already computed in cur_k/cur_v)
+        for (int h = 0; h < n_heads; ++h) {
+            float* k_head = (float*)cur_k->data + h * head_dim;
+            float* v_head = (float*)cur_v->data + h * head_dim;
+            adaptq_mha_append(kv_self.adaptq_layers[il],
+                              h, k_head, v_head, kv_head);
+        }
+        // Skip writing to ggml KV tensors (saves the FP16 copy)
+        goto skip_ggml_kv_write;
+    }
```

---

## Phase 3 — Attention Compute Intercept

### File: `llama.cpp` — inside `llm_build_kqv` (or the attention loop)

Replace the GGML softmax+weighted-V operation with AdapTQ's kernel:

```diff
+    if (kv_self.adaptq_enabled) {
+        const int n_qs = n_tokens; // batch size (num_queries)
+        // Process all heads in parallel
+        for (int h = 0; h < n_heads; ++h) {
+            float* q_heads  = (float*)cur_q->data + h * head_dim; // assumes continuous block
+            float* out_heads = (float*)cur_attn->data + h * head_dim; // assumes continuous
+            
+            if (n_qs > 1) {
+                // Batch (Multi-Query) Fast Attention: uses HW SIMD OpenMP
+                adaptq_mha_compute_batch(kv_self.adaptq_layers[il],
+                                         h, q_heads, n_qs, out_heads);
+            } else {
+                // Single token decode optimization
+                adaptq_mha_compute(kv_self.adaptq_layers[il],
+                                   h, q_heads, out_heads);
+            }
+        }
+        // head outputs are now in cur_attn; continue to projection layer
+        goto skip_ggml_attn;
+    }
```

---

## Phase 4 — Context Reset

### File: `llama.cpp` — `llama_kv_cache_clear`

```diff
 void llama_kv_cache_clear(struct llama_context * ctx) {
     ...
     kv_self.head = 0;
+    // Reset AdapTQ ring buffers
+    for (auto* mha : ctx->kv_self.adaptq_layers)
+        if (mha) adaptq_mha_reset(mha);
 }
```

---

## Phase 5 — Runtime Flag

Add to `llama_context_params`:

```diff
 struct llama_context_params {
     ...
+    bool use_adaptq          = false;   // enable AdapTQ KV cache
+    int  adaptq_bits         = 4;       // 2, 3, or 4
+    int  adaptq_hybrid_thresh = 1024;   // FP16 below this seq len
 };
```

---

## Memory Comparison

| Config | KV cache at 32k ctx, 32 layers, 32 heads, dim=128 |
|--------|---------------------------------------------------|
| FP16   | 32k × 32 × 32 × 128 × 2 × 2B = **1.0 GB** |
| 4-bit  | 32k × 32 × 32 × 128 × 2 × 0.5B = **250 MB** |
| 2-bit  | 32k × 32 × 32 × 128 × 2 × 0.25B = **125 MB** |

**4-bit AdapTQ uses 4× less KV memory than FP16.**

---

## Performance Projection (from the latest AdapTQ Hardware Benchmark)

| Seq length | FP16 attn | AdapTQ 4-bit | Speedup vs FP16 |
|------------|-----------|--------------|---------|
| 512  | 21.9 µs/head | 22.9 µs/head | **~0.96×** |
| 1024 | 48.6 µs/head | 43.3 µs/head | **1.12×** |
| 4096 | 210.8 µs/head| 162.8 µs/head| **1.29×** |
| 8192 | 569.3 µs/head| 407.5 µs/head| **1.40×** |
| 8192 (Q2) | 528.8 µs/head| 306.9 µs/head| **1.72×** |

At context lengths ≥ 1024 tokens, AdapTQ attention strictly **outperforms FP16** on x86 with AVX2.

---

## Logging / Debug

```c
fprintf(stderr, "[AdapTQ] KV usage: %.1f MB (FP16 would be %.1f MB)\n",
        adaptq_mha_total_kv_bytes(kv_layer) / 1e6,
        n_heads * capacity * head_dim * 2 * 2 / 1e6);
```

---

## Quick Test After Integration

```bash
./main -m tinyllama-1.1b-chat.gguf \
       --kv-quant adaptq --adaptq-bits 4 --ctx-size 8192 \
       -p "Tell me about large language models" -n 200

# Compare tokens/sec vs baseline (remove --kv-quant)
```

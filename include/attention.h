#pragma once
#include "quantizer.h"
#include "ring_buffer.h"
#include <vector>
#include <memory>
#include "adaptq/storage_backend.h"
#include "adaptq/kernel_backend.h"
#include "adaptq/policy.h"

// Attention head using abstract storage + kernel backends
struct AttentionHead {
  Quantizer quant;
  std::unique_ptr<IStorageBackend> storage;
  std::unique_ptr<IKernelBackend> kernel;
  std::unique_ptr<IPolicy> policy;
  
  int dim;
  int padded;
  int bits;
  // 0 = full; >0 = fraction of softmax mass to cover (e.g. 0.95)
  float v_mass_thresh;
  // Hybrid: if kv_buf.size < hybrid_thresh, use FP32 path (faster at small seq)
  int hybrid_thresh = 512;   // default 512, 0 = always quantized
  std::vector<float> raw_kv; // interleaved k+v, capped at hybrid_thresh tokens

  void init(int d, int b, int capacity, uint64_t seed, float v_mass = 0.f,
            int hyb = 512);
  void append_kv(const float *key, const float *val, int token_pos);

  // Full compute: rotate query, build LUT, K-scan, sparse V.
  // out must be float[dim]. Returns num active slots.
  int compute(const float *q, float *out) const;

  // Batch compute for multiple queries against the same KV cache
  int compute_batch(const float *queries, int num_queries, float *outs) const;

  size_t kv_bytes() const { return storage ? storage->kv_bytes() : 0; }
  size_t k_scan_bytes() const { return storage ? storage->k_scan_bytes() : 0; }
};

float dot_product(const float *a, const float *b, int n);
void softmax(float *x, int n);

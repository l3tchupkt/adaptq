/* adapter_llamacpp.cpp
 *
 * Pluggable adapter: llama.cpp  →  AdapTQ core
 *
 * This file has NO dependency on ggml.h / llama.cpp headers outside of the
 * types we receive as plain float pointers at call sites.
 *
 * Usage inside llama.cpp (after graph compute, with real data):
 *
 *   #include "adaptq/adapters/adapter_llamacpp.h"
 *
 *   // Create once per model context:
 *   LlamaCppAdaptQAdapter adapter(n_heads, head_dim, bits, capacity, seed,
 * v_mass, hybrid_thr);
 *
 *   // Inside the per-token decode loop (AFTER ggml_graph_compute):
 *   adapter.feed_kv(head, (float*)k_tensor->data + h*head_dim,
 * (float*)v_tensor->data + h*head_dim, pos); adapter.attention(head,
 * (float*)q_tensor->data + h*head_dim, (float*)out_tensor->data + h*head_dim);
 *
 * The adapter translates llama.cpp's raw float arrays into the pure
 * AdapTQ C API. No GGML types cross the boundary — only float*.
 */

#include "adapter_llamacpp.h"
#include <cstdio>
#include <cstring>


LlamaCppAdaptQAdapter::LlamaCppAdaptQAdapter(int n_heads, int head_dim,
                                             int bits, int capacity,
                                             uint64_t seed, float v_mass,
                                             int hybrid_thresh)
    : _backend(n_heads, head_dim, bits, capacity, seed, v_mass, hybrid_thresh) {
}

void LlamaCppAdaptQAdapter::feed_kv(int head, const float *key,
                                    const float *val, int token_pos) {
  _backend.append_kv(head, key, val, token_pos);
}

int LlamaCppAdaptQAdapter::attention(int head, const float *query, float *out) {
  return _backend.compute(head, query, out);
}

int LlamaCppAdaptQAdapter::attention_batch(int head, const float *queries,
                                           int n_queries, float *outs) {
  return _backend.compute_batch(head, queries, n_queries, outs);
}

void LlamaCppAdaptQAdapter::reset() { _backend.reset(); }

size_t LlamaCppAdaptQAdapter::kv_bytes() const { return _backend.kv_bytes(); }

void LlamaCppAdaptQAdapter::log_memory(int capacity) const {
  double used_mb = (double)_backend.kv_bytes() / 1e6;
  double fp16_mb =
      (double)_backend.n_heads() * capacity * _backend.head_dim() * 2 * 2 / 1e6;
  fprintf(stderr,
          "[AdapTQ/llama.cpp] KV: %.2f MB  (FP16 would be %.2f MB, "
          "%.1fx smaller)\n",
          used_mb, fp16_mb, fp16_mb / (used_mb + 1e-9));
}

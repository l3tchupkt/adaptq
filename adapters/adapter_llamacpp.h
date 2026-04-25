#pragma once
#include "../include/adaptq_mha_backend.h"
#include <cstddef>

/* -------------------------------------------------------------------------
 * LlamaCppAdaptQAdapter
 *
 * Thin adapter between llama.cpp's float tensor data and the AdapTQ core.
 * Include this file in llama.cpp builds. Zero GGML headers required here.
 * ------------------------------------------------------------------------- */
class LlamaCppAdaptQAdapter {
public:
  LlamaCppAdaptQAdapter(int n_heads, int head_dim, int bits, int capacity,
                        uint64_t seed = 0, float v_mass = 0.95f,
                        int hybrid_thresh = 512);

  /** Call once per incoming token, once per head, after graph compute. */
  void feed_kv(int head, const float *key, const float *val, int token_pos);

  /** Single-query attention forward (decode step). */
  int attention(int head, const float *query, float *out);

  /** Multi-query batch attention (prefill step). */
  int attention_batch(int head, const float *queries, int n_queries,
                      float *outs);

  /** Clear all cached KV (e.g. on context reset). */
  void reset();

  /** Compressed bytes in the KV cache. */
  size_t kv_bytes() const;

  /** Logs "KV: X MB vs FP16: Y MB" to stderr. */
  void log_memory(int capacity) const;

private:
  AdaptQMHABackend _backend;
};

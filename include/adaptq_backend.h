#pragma once

#include <stddef.h>
#include <stdint.h>

/* -------------------------------------------------------------------------
 * IAdaptQBackend — pure abstract backend interface
 *
 * Frameworks (llama.cpp, PyTorch, standalone) implement this interface.
 * Zero framework includes inside core AdapTQ.
 * All methods operate on raw float* buffers — no tensor types here.
 * ------------------------------------------------------------------------- */
struct IAdaptQBackend {
  virtual ~IAdaptQBackend() = default;

  /**
   * Append one token's (key, val) pair to the cache.
   * @param head_idx  which attention head [0, n_heads)
   * @param key       float[head_dim]
   * @param val       float[head_dim]
   * @param token_pos absolute token position (used for positional encoding)
   */
  virtual void append_kv(int head_idx, const float *key, const float *val,
                         int token_pos) = 0;

  /**
   * Compute attention output for a single query.
   * @param head_idx  which attention head
   * @param query     float[head_dim]
   * @param out       float[head_dim]  — result written here
   * @return          number of cached tokens used
   */
  virtual int compute(int head_idx, const float *query, float *out) = 0;

  /**
   * Compute attention output for a batch of queries against same KV cache.
   * Uses OpenMP internally — thread-safe when called from a single thread.
   * @param head_idx    which attention head
   * @param queries     float[num_queries * head_dim]
   * @param num_queries number of simultaneous queries
   * @param outs        float[num_queries * head_dim]
   * @return            number of cached tokens used
   */
  virtual int compute_batch(int head_idx, const float *queries, int num_queries,
                            float *outs) = 0;

  /** Reset the KV cache (keep configuration). */
  virtual void reset() = 0;

  /** Bytes consumed by compressed KV data. */
  virtual size_t kv_bytes() const = 0;

  /** Number of attention heads. */
  virtual int n_heads() const = 0;

  /** Head dimension. */
  virtual int head_dim() const = 0;
};

/* -------------------------------------------------------------------------
 * C-compatible function pointer vtable for pure-C callers.
 * Compatible with adaptq_* C API; assembled automatically by the factory.
 * ------------------------------------------------------------------------- */
typedef struct AdaptQBackendVTable {
  void (*append_kv)(void *ctx, int head, const float *k, const float *v,
                    int pos);
  int (*compute)(void *ctx, int head, const float *q, float *out);
  int (*compute_batch)(void *ctx, int head, const float *qs, int n,
                       float *outs);
  void (*reset)(void *ctx);
  size_t (*kv_bytes)(void *ctx);
  int (*n_heads)(void *ctx);
  int (*head_dim)(void *ctx);
  void (*destroy)(void *ctx);
  void *impl; /* opaque pointer to the concrete backend */
} AdaptQBackendVTable;

#ifdef __cplusplus
/**
 * Wrap any IAdaptQBackend into a C-compatible vtable.
 * The returned vtable does NOT take ownership of `backend`.
 */
AdaptQBackendVTable adaptq_make_vtable(IAdaptQBackend *backend);
#endif

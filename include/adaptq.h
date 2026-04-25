#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* -----------------------------------------------------------------------
 * AdapTQ — C API for LLM integration
 * Drop-in quantized KV cache for multi-head attention.
 * Thread-safe: each handle is independent.
 * ----------------------------------------------------------------------- */

typedef void *adaptq_ctx_t; /* opaque per-head context */
typedef void *adaptq_mha_t; /* opaque multi-head aggregate */

/* ---- Single head API ---- */

/**
 * Create a single-head KV cache context.
 * @param dim        head dimension (must be power-of-2 or padded internally)
 * @param bits       quantization bits: 2, 3, or 4
 * @param capacity   max tokens to cache (ring buffer size)
 * @param seed       Rademacher seed (unique per layer+head)
 * @param v_mass     sparse V threshold [0=full, 0.95=cover 95% mass]
 * @param hybrid_thresh  switch to FP16 path when n_tokens < this value
 *                       (0 = always use quantized)
 */
adaptq_ctx_t adaptq_create(int dim, int bits, int capacity, uint64_t seed,
                           float v_mass, int hybrid_thresh);

/** Free a single-head context. */
void adaptq_destroy(adaptq_ctx_t ctx);

/** Append one (key, val) pair. Float arrays of length dim. */
void adaptq_append(adaptq_ctx_t ctx, const float *key, const float *val,
                   int token_pos);

/**
 * Compute attention output.
 * @param ctx    context handle
 * @param query  float[dim]
 * @param out    float[dim]  — result written here
 * @return  number of tokens used (0 = empty)
 */
int adaptq_compute(adaptq_ctx_t ctx, const float *query, float *out);

/**
 * Compute batch attention (multi-query). Reuses K/V reads across queries for
 * massive bandwidth savings.
 * @param ctx         context handle
 * @param queries     float[num_queries * dim]
 * @param num_queries number of queries to compute simultaneously
 * @param outs        float[num_queries * dim] — results written here
 * @return  number of tokens used
 */
int adaptq_compute_batch(adaptq_ctx_t ctx, const float *queries,
                         int num_queries, float *outs);

/** Reset KV cache (keep config). */
void adaptq_reset(adaptq_ctx_t ctx);

/** Bytes consumed by compressed KV storage. */
size_t adaptq_kv_bytes(adaptq_ctx_t ctx);

/* ---- Multi-head API ---- */

/**
 * Create a multi-head context.
 * All heads share the same dim, bits, capacity, v_mass, and hybrid_thresh.
 * Seeds are derived per-head as: seed ^ (layer*1000 + head).
 */
adaptq_mha_t adaptq_mha_create(int n_heads, int dim, int bits, int capacity,
                               uint64_t base_seed, float v_mass,
                               int hybrid_thresh);

void adaptq_mha_destroy(adaptq_mha_t mha);

/** Append KV for one (layer_head) index.  head_idx < n_heads. */
void adaptq_mha_append(adaptq_mha_t mha, int head_idx, const float *key,
                       const float *val, int token_pos);

/** Compute attention for one head.  Returns active tokens. */
int adaptq_mha_compute(adaptq_mha_t mha, int head_idx, const float *query,
                       float *out);

/** Compute batch attention for one head. Returns active tokens. */
int adaptq_mha_compute_batch(adaptq_mha_t mha, int head_idx,
                             const float *queries, int num_queries,
                             float *outs);

/** Reset all heads. */
void adaptq_mha_reset(adaptq_mha_t mha);

/** Total KV bytes across all heads. */
size_t adaptq_mha_total_kv_bytes(adaptq_mha_t mha);

/* ---- Feature flags ---- */

#define ADAPTQ_FEAT_AVX2 (1u << 0)
#define ADAPTQ_FEAT_HYBRID (1u << 1)
#define ADAPTQ_FEAT_SPARSE_V (1u << 2)

/** Returns bitmask of features compiled in. */
unsigned int adaptq_features(void);

/** Version string, e.g. "3.1.0-avx2". */
const char *adaptq_version(void);

#ifdef __cplusplus
}
#endif

#include "../include/adaptq.h"
#include "../include/attention.h"
#include "../include/codebook.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

/* -----------------------------------------------------------------------
 * Thread-local error state
 * ----------------------------------------------------------------------- */

static thread_local char tl_error_buf[256] = "";

static void set_error(adaptq_error_t /*code*/, const char *msg) {
  snprintf(tl_error_buf, sizeof(tl_error_buf), "%s", msg);
}

const char *adaptq_last_error(void) { return tl_error_buf; }

static bool require_handle(const void *handle, const char *operation) {
  if (handle)
    return true;

  char message[256];
  snprintf(message, sizeof(message), "%s: null handle", operation);
  set_error(ADAPTQ_ERR_INVALID_ARG, message);
  return false;
}

static bool require_pointer(const void *pointer, const char *operation,
                            const char *parameter) {
  if (pointer)
    return true;

  char message[256];
  snprintf(message, sizeof(message), "%s: null %s", operation, parameter);
  set_error(ADAPTQ_ERR_INVALID_ARG, message);
  return false;
}

/* -----------------------------------------------------------------------
 * Internal state structs
 * ----------------------------------------------------------------------- */

struct AdapTQCtx {
  AttentionHead head;
  int hybrid_thresh; // use FP path when n_tokens < this (owned by AttentionHead::raw_kv)
  int dim;
};

struct AdapTQMHA {
  std::vector<AdapTQCtx *> heads;
  int n_heads;
};

/* -----------------------------------------------------------------------
 * Single-head API
 * ----------------------------------------------------------------------- */

adaptq_ctx_t adaptq_create(int dim, int bits, int capacity, uint64_t seed,
                           float v_mass, int hybrid_thresh) {
  if (dim <= 0 || bits < 2 || bits > 4 || capacity < 0 || hybrid_thresh < 0) {
    set_error(ADAPTQ_ERR_INVALID_ARG, "adaptq_create: invalid dimensions or capacity");
    return nullptr;
  }
  auto *ctx = new AdapTQCtx();
  ctx->dim = dim;
  ctx->hybrid_thresh = hybrid_thresh;
  ctx->head.init(dim, bits, capacity, seed, v_mass);
  return ctx;
}

void adaptq_destroy(adaptq_ctx_t h) {
  // delete(nullptr) is defined, so destruction remains a harmless cleanup
  // operation even when construction failed or ownership is optional.
  delete static_cast<AdapTQCtx *>(h);
}

void adaptq_append(adaptq_ctx_t h, const float *key, const float *val,
                   int token_pos) {
  if (!require_handle(h, "adaptq_append") ||
      !require_pointer(key, "adaptq_append", "key") ||
      !require_pointer(val, "adaptq_append", "val")) {
    return;
  }

  auto *ctx = static_cast<AdapTQCtx *>(h);
  // AttentionHead::append_kv maintains raw_kv[] for the hybrid path internally.
  // No duplicate FP copy here.
  ctx->head.append_kv(key, val, token_pos);
}

int adaptq_compute(adaptq_ctx_t h, const float *query, float *out) {
  if (!require_handle(h, "adaptq_compute") ||
      !require_pointer(query, "adaptq_compute", "query") ||
      !require_pointer(out, "adaptq_compute", "out")) {
    return -1;
  }

  auto *ctx = static_cast<AdapTQCtx *>(h);
  // Hybrid dispatch is handled entirely within AttentionHead::compute(),
  // which owns raw_kv[] as the single authoritative FP32 copy.
  return ctx->head.compute(query, out);
}

int adaptq_compute_batch(adaptq_ctx_t h, const float *queries, int num_queries,
                         float *outs) {
  if (!require_handle(h, "adaptq_compute_batch"))
    return -1;
  if (num_queries < 0) {
    set_error(ADAPTQ_ERR_INVALID_ARG,
              "adaptq_compute_batch: negative num_queries");
    return -1;
  }
  if (num_queries > 0 &&
      (!require_pointer(queries, "adaptq_compute_batch", "queries") ||
       !require_pointer(outs, "adaptq_compute_batch", "outs"))) {
    return -1;
  }

  auto *ctx = static_cast<AdapTQCtx *>(h);
  return ctx->head.compute_batch(queries, num_queries, outs);
}

void adaptq_reset(adaptq_ctx_t h) {
  if (!require_handle(h, "adaptq_reset"))
    return;

  auto *ctx = static_cast<AdapTQCtx *>(h);
  ctx->head.kv_buf.size = 0;
  ctx->head.kv_buf.head = 0;
  ctx->head.raw_kv.clear();  /* clear hybrid FP32 mirror — stale after reset */
  tl_error_buf[0] = '\0';
}

size_t adaptq_kv_bytes(adaptq_ctx_t h) {
  if (!require_handle(h, "adaptq_kv_bytes"))
    return 0;
  return static_cast<AdapTQCtx *>(h)->head.kv_bytes();
}

/* -----------------------------------------------------------------------
 * Multi-head API
 * ----------------------------------------------------------------------- */

adaptq_mha_t adaptq_mha_create(int n_heads, int dim, int bits, int capacity,
                               uint64_t base_seed, float v_mass,
                               int hybrid_thresh) {
  if (n_heads <= 0 || dim <= 0 || bits < 2 || bits > 4 || capacity < 0 || hybrid_thresh < 0) {
    set_error(ADAPTQ_ERR_INVALID_ARG, "adaptq_mha_create: invalid parameters");
    return nullptr;
  }
  auto *mha = new AdapTQMHA();
  mha->n_heads = n_heads;
  mha->heads.resize(n_heads);
  for (int i = 0; i < n_heads; ++i) {
    uint64_t seed = base_seed ^ ((uint64_t)i * 0xDEADBEEFCAFEULL);
    mha->heads[i] = static_cast<AdapTQCtx *>(
        adaptq_create(dim, bits, capacity, seed, v_mass, hybrid_thresh));
  }
  return mha;
}

void adaptq_mha_destroy(adaptq_mha_t h) {
  if (!h)
    return;

  auto *mha = static_cast<AdapTQMHA *>(h);
  for (auto *ctx : mha->heads)
    delete ctx;
  delete mha;
}

void adaptq_mha_append(adaptq_mha_t h, int head_idx, const float *key,
                       const float *val, int token_pos) {
  if (!require_handle(h, "adaptq_mha_append"))
    return;

  auto *mha = static_cast<AdapTQMHA *>(h);
  if (head_idx < 0 || head_idx >= mha->n_heads) {
    set_error(ADAPTQ_ERR_OUT_OF_BOUNDS,
              "adaptq_mha_append: head_idx out of range");
    return;
  }
  if (!require_pointer(key, "adaptq_mha_append", "key") ||
      !require_pointer(val, "adaptq_mha_append", "val")) {
    return;
  }

  tl_error_buf[0] = '\0';
  adaptq_append(mha->heads[head_idx], key, val, token_pos);
}

int adaptq_mha_compute(adaptq_mha_t h, int head_idx, const float *query,
                       float *out) {
  if (!require_handle(h, "adaptq_mha_compute"))
    return -1;

  auto *mha = static_cast<AdapTQMHA *>(h);
  if (head_idx < 0 || head_idx >= mha->n_heads) {
    set_error(ADAPTQ_ERR_OUT_OF_BOUNDS,
              "adaptq_mha_compute: head_idx out of range");
    return -1;
  }
  if (!require_pointer(query, "adaptq_mha_compute", "query") ||
      !require_pointer(out, "adaptq_mha_compute", "out")) {
    return -1;
  }

  tl_error_buf[0] = '\0';
  return adaptq_compute(mha->heads[head_idx], query, out);
}

int adaptq_mha_compute_batch(adaptq_mha_t h, int head_idx, const float *queries,
                             int num_queries, float *outs) {
  if (!require_handle(h, "adaptq_mha_compute_batch"))
    return -1;

  auto *mha = static_cast<AdapTQMHA *>(h);
  if (head_idx < 0 || head_idx >= mha->n_heads) {
    set_error(ADAPTQ_ERR_OUT_OF_BOUNDS,
              "adaptq_mha_compute_batch: head_idx out of range");
    return -1;
  }
  if (num_queries < 0) {
    set_error(ADAPTQ_ERR_INVALID_ARG,
              "adaptq_mha_compute_batch: negative num_queries");
    return -1;
  }
  if (num_queries > 0 &&
      (!require_pointer(queries, "adaptq_mha_compute_batch", "queries") ||
       !require_pointer(outs, "adaptq_mha_compute_batch", "outs"))) {
    return -1;
  }

  tl_error_buf[0] = '\0';
  return adaptq_compute_batch(mha->heads[head_idx], queries, num_queries, outs);
}

void adaptq_mha_reset(adaptq_mha_t h) {
  if (!require_handle(h, "adaptq_mha_reset"))
    return;

  auto *mha = static_cast<AdapTQMHA *>(h);
  for (auto *ctx : mha->heads)
    adaptq_reset(ctx);
}

size_t adaptq_mha_total_kv_bytes(adaptq_mha_t h) {
  if (!require_handle(h, "adaptq_mha_total_kv_bytes"))
    return 0;

  auto *mha = static_cast<AdapTQMHA *>(h);
  size_t total = 0;
  for (auto *ctx : mha->heads)
    total += ctx->head.kv_bytes();
  return total;
}

/* -----------------------------------------------------------------------
 * Feature flags + version
 * ----------------------------------------------------------------------- */

unsigned int adaptq_features(void) {
  unsigned int f = ADAPTQ_FEAT_HYBRID | ADAPTQ_FEAT_SPARSE_V;
#ifdef __AVX2__
  f |= ADAPTQ_FEAT_AVX2;
#endif
  return f;
}

const char *adaptq_version(void) {
#ifdef __AVX2__
  return "3.2.0-avx2";
#else
  return "3.2.0-scalar";
#endif
}

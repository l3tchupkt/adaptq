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
 * Internal state structs
 * ----------------------------------------------------------------------- */

struct AdapTQCtx {
  AttentionHead head;
  int hybrid_thresh; // use FP16 when n_tokens < this
  int dim;

  // FP16 hybrid storage (raw float32 — true FP16 on edge needs cast layer)
  std::vector<std::vector<float>> fp_k, fp_v;
  int fp_count = 0;

  float fp_attn(const float *q, float *out) const {
    int n = fp_count;
    if (n == 0) {
      memset(out, 0, dim * sizeof(float));
      return 0.f;
    }
    thread_local std::vector<float> logits;
    logits.resize(n);
    float scale = 1.f / sqrtf((float)dim);
    float mx = -1e30f;
    for (int i = 0; i < n; ++i) {
      float d = 0.f;
      const float *k = fp_k[i].data();
      for (int j = 0; j < dim; ++j)
        d += q[j] * k[j];
      logits[i] = d * scale;
      if (logits[i] > mx)
        mx = logits[i];
    }
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
      logits[i] = expf(logits[i] - mx);
      sum += logits[i];
    }
    float inv = 1.f / sum;
    memset(out, 0, dim * sizeof(float));
    for (int i = 0; i < n; ++i) {
      float w = logits[i] * inv;
      const float *v = fp_v[i].data();
      for (int j = 0; j < dim; ++j)
        out[j] += w * v[j];
    }
    return (float)n;
  }
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
  auto *ctx = new AdapTQCtx();
  ctx->dim = dim;
  ctx->hybrid_thresh = hybrid_thresh;
  ctx->head.init(dim, bits, capacity, seed, v_mass);
  if (hybrid_thresh > 0) {
    ctx->fp_k.reserve(hybrid_thresh);
    ctx->fp_v.reserve(hybrid_thresh);
  }
  return ctx;
}

void adaptq_destroy(adaptq_ctx_t h) { delete static_cast<AdapTQCtx *>(h); }

void adaptq_append(adaptq_ctx_t h, const float *key, const float *val,
                   int token_pos) {
  auto *ctx = static_cast<AdapTQCtx *>(h);
  ctx->head.append_kv(key, val, token_pos);

  // Also maintain FP cache if hybrid mode is active
  if (ctx->hybrid_thresh > 0 && ctx->fp_count < ctx->hybrid_thresh) {
    if ((int)ctx->fp_k.size() <= ctx->fp_count) {
      ctx->fp_k.push_back(std::vector<float>(key, key + ctx->dim));
      ctx->fp_v.push_back(std::vector<float>(val, val + ctx->dim));
    } else {
      ctx->fp_k[ctx->fp_count].assign(key, key + ctx->dim);
      ctx->fp_v[ctx->fp_count].assign(val, val + ctx->dim);
    }
    ctx->fp_count++;
  }
}

int adaptq_compute(adaptq_ctx_t h, const float *query, float *out) {
  auto *ctx = static_cast<AdapTQCtx *>(h);
  int n = ctx->head.kv_buf.size;

  // Hybrid: use FP path for short contexts
  if (ctx->hybrid_thresh > 0 && n < ctx->hybrid_thresh) {
    ctx->fp_attn(query, out);
    return n;
  }
  return ctx->head.compute(query, out);
}

int adaptq_compute_batch(adaptq_ctx_t h, const float *queries, int num_queries,
                         float *outs) {
  auto *ctx = static_cast<AdapTQCtx *>(h);
  return ctx->head.compute_batch(queries, num_queries, outs);
}

void adaptq_reset(adaptq_ctx_t h) {
  auto *ctx = static_cast<AdapTQCtx *>(h);
  ctx->head.kv_buf.size = 0;
  ctx->head.kv_buf.head = 0;
  ctx->fp_count = 0;
}

size_t adaptq_kv_bytes(adaptq_ctx_t h) {
  return static_cast<AdapTQCtx *>(h)->head.kv_bytes();
}

/* -----------------------------------------------------------------------
 * Multi-head API
 * ----------------------------------------------------------------------- */

adaptq_mha_t adaptq_mha_create(int n_heads, int dim, int bits, int capacity,
                               uint64_t base_seed, float v_mass,
                               int hybrid_thresh) {
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
  auto *mha = static_cast<AdapTQMHA *>(h);
  for (auto *ctx : mha->heads)
    delete ctx;
  delete mha;
}

void adaptq_mha_append(adaptq_mha_t h, int head_idx, const float *key,
                       const float *val, int token_pos) {
  auto *mha = static_cast<AdapTQMHA *>(h);
  adaptq_append(mha->heads[head_idx], key, val, token_pos);
}

int adaptq_mha_compute(adaptq_mha_t h, int head_idx, const float *query,
                       float *out) {
  auto *mha = static_cast<AdapTQMHA *>(h);
  return adaptq_compute(mha->heads[head_idx], query, out);
}

int adaptq_mha_compute_batch(adaptq_mha_t h, int head_idx, const float *queries,
                             int num_queries, float *outs) {
  auto *mha = static_cast<AdapTQMHA *>(h);
  return adaptq_compute_batch(mha->heads[head_idx], queries, num_queries, outs);
}

void adaptq_mha_reset(adaptq_mha_t h) {
  auto *mha = static_cast<AdapTQMHA *>(h);
  for (auto *ctx : mha->heads)
    adaptq_reset(ctx);
}

size_t adaptq_mha_total_kv_bytes(adaptq_mha_t h) {
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

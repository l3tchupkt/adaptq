#include "runtime_context.h"
#include "../include/adaptq/config.h"
#include "../include/adaptq/metrics.h"
#include "../include/adaptq/quality.h"
#include "../storage/contiguous_slab.h"
#include "../include/codebook.h"
#include "../include/fwht.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

/* -------------------------------------------------------------------------
 * runtime/runtime_context.cpp
 *
 * Strategy / storage factories are included directly here as static
 * compilation units to avoid ODR issues (same pattern as test_conformance).
 * The RuntimeContext TU is the sole owner of these definitions.
 * ----------------------------------------------------------------------- */

/* Pull in strategy implementations (compiled into this TU only). */
#include "../strategies/har_fixed.cpp"
#include "../strategies/fp_passthrough.cpp"
/* Pull in kernel backend implementations. */
namespace adaptq {

/* =========================================================================
 * Strategy / storage factories
 * ========================================================================= */

static IKVStrategy *make_har_fixed()       { return new HARFixedStrategy();      }
static IKVStrategy *make_fp_passthrough()  { return new FPPassthroughStrategy(); }
IStorageBackend *make_contiguous()  { return new ContiguousSlabStorage(); }

/* =========================================================================
 * Registry — maps string names to factories (for CLI use)
 * ========================================================================= */
struct StrategyRegistryEntry {
    const char      *name;
    StrategyFactory  factory;
};

static const StrategyRegistryEntry kStrategyRegistry[] = {
    { "har_fixed",      make_har_fixed      },
    { "fp_passthrough", make_fp_passthrough },
};
static constexpr int kRegistrySize =
    static_cast<int>(sizeof(kStrategyRegistry) / sizeof(kStrategyRegistry[0]));

StrategyFactory strategy_factory_by_name(const char *name) {
    for (int i = 0; i < kRegistrySize; ++i)
        if (strcmp(kStrategyRegistry[i].name, name) == 0)
            return kStrategyRegistry[i].factory;
    return nullptr;
}

const char **strategy_names(int *out_count) {
    static const char *names[kRegistrySize];
    for (int i = 0; i < kRegistrySize; ++i)
        names[i] = kStrategyRegistry[i].name;
    *out_count = kRegistrySize;
    return names;
}

/* =========================================================================
 * Helpers
 * ========================================================================= */

static int next_pow2_rt(int n) {
    if (n <= 1) return 1;
    if (n > (1 << 30)) return 1 << 30;

    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* =========================================================================
 * RuntimeContext::init
 * ========================================================================= */

void RuntimeContext::init(const RuntimeContextConfig &cfg) {
    init(cfg, make_har_fixed, make_contiguous);
}

void RuntimeContext::init(const RuntimeContextConfig &cfg,
                          StrategyFactory             strategy_fn,
                          StorageFactory              storage_fn) {
    if (cfg.n_layers <= 0 || cfg.n_heads <= 0 || cfg.dim <= 0 ||
        cfg.bits < 2 || cfg.bits > 4 || cfg.capacity <= 0 ||
        cfg.dim > (1 << 30) ||
        cfg.n_layers > std::numeric_limits<int>::max() / cfg.n_heads ||
        !strategy_fn || !storage_fn) {
        throw std::invalid_argument("RuntimeContext::init: invalid configuration");
    }
    cfg_      = cfg;
    token_pos_ = 0;
    token_log_.clear();
    token_log_data_.clear();

    int n = cfg.n_layers * cfg.n_heads;
    strategies_.clear();  strategies_.reserve(n);
    storages_.clear();    storages_.reserve(n);
    cache_sizes_.assign(n, 0);
    recent_quality_.assign(n, -1.f);
    recent_latency_.assign(n, 0.f);

    /* slot_bytes = max packed bytes for one vector.
     * For HAR 4-bit: (padded * 4 + 7) / 8 = padded / 2 bytes.
     * For FP32:      dim * 4 bytes.
     * Use the larger bound to keep storage backend generic.  */
    int padded       = next_pow2_rt(cfg_.dim);
    int slot_bytes_q = (padded * cfg.bits + 7) / 8;
    int slot_bytes_f = cfg_.dim * (int)sizeof(float);
    int slot_bytes   = std::max(slot_bytes_q, slot_bytes_f);

    HeadConfig hcfg;
    hcfg.dim      = cfg_.dim;
    hcfg.bits     = cfg.bits;
    hcfg.capacity = cfg.capacity;
    hcfg.v_mass   = cfg.v_mass;

    for (int l = 0; l < cfg.n_layers; ++l) {
        for (int h = 0; h < cfg.n_heads; ++h) {
            hcfg.seed = (uint64_t)l * 65537ULL + (uint64_t)h;

            auto *strat = strategy_fn();
            strat->init(hcfg);
            strategies_.emplace_back(strat);

            auto *stor = storage_fn();
            /* Storage holds K and V slots — capacity × 2. */
            stor->init(cfg.capacity * 2, slot_bytes);
            storages_.emplace_back(stor);
        }
    }

    /* Select best available kernel backend (process-global). */
    kernel_ = select_kernel_backend();

    /* Quality oracle — default ensemble. */
    oracle_ = make_default_quality_oracle();

    /* Scratch buffers — sized for padded dimension. */
    q_rot_.assign(padded, 0.f);
    v_acc_.assign(padded, 0.f);
    k_results_.reserve(cfg.capacity);
    v_results_.reserve(cfg.capacity);
    logits_.reserve(cfg.capacity);
}

void RuntimeContext::set_policy(IPolicy *policy) {
    policy_ = policy;
    if (policy_) {
        RuntimeConfig rc;
        rc.n_layers             = cfg_.n_layers;
        rc.n_heads              = cfg_.n_heads;
        rc.dim                  = cfg_.dim;
        rc.bits                 = cfg_.bits;
        rc.capacity             = cfg_.capacity;
        rc.v_mass               = cfg_.v_mass;
        rc.memory_budget_mb     = cfg_.memory_budget_mb;
        rc.quality_floor        = cfg_.quality_floor;
        rc.latency_hard_limit_us = cfg_.latency_hard_limit_us;
        policy_->init(cfg_.n_layers, cfg_.n_heads, rc);
    }
}

void RuntimeContext::reset() {
    token_pos_ = 0;
    token_log_.clear();
    token_log_data_.clear();
    int n = cfg_.n_layers * cfg_.n_heads;
    for (int i = 0; i < n; ++i) {
        cache_sizes_[i]    = 0;
        recent_quality_[i] = -1.f;
        recent_latency_[i] = 0.f;
        storages_[i]->reset();
        strategies_[i]->reset();
    }
}

/* =========================================================================
 * ExecutionContext factory
 * ========================================================================= */

ExecutionContext RuntimeContext::make_ctx(int layer, int head) const {
    int idx = head_idx(layer, head);
    float mem_used = (float)storages_[idx]->bytes_used() / (1024.f * 1024.f);

    ExecutionContext ctx{};
    ctx.layer            = layer;
    ctx.head             = head;
    ctx.token_pos        = token_pos_;
    ctx.cache_size       = cache_sizes_[idx];
    ctx.cache_capacity   = cfg_.capacity;
    ctx.memory_used_mb   = mem_used;
    ctx.memory_budget_mb = cfg_.memory_budget_mb;
    ctx.approaching_budget = (mem_used > 0.9f * cfg_.memory_budget_mb);
    ctx.recent_quality   = recent_quality_[idx];
    ctx.recent_latency_us = recent_latency_[idx];
    ctx.storage          = storages_[idx].get();
    ctx.kernel           = kernel_;
    ctx.quality_oracle   = oracle_.get();
    ctx.quality_floor    = cfg_.quality_floor;
    ctx.latency_hard_limit_us = cfg_.latency_hard_limit_us;
    return ctx;
}

/* =========================================================================
 * IKVStrategy / IStorageBackend accessors
 * ========================================================================= */

IKVStrategy *RuntimeContext::get_strategy(int layer, int head) const {
    return strategies_[head_idx(layer, head)].get();
}

IStorageBackend *RuntimeContext::get_storage(int layer, int head) const {
    return storages_[head_idx(layer, head)].get();
}

/* =========================================================================
 * append — hot path
 * ========================================================================= */

void RuntimeContext::append(int          layer,
                            int          head,
                            const float *k_vec,
                            const float *v_vec) {
    assert(layer >= 0 && layer < cfg_.n_layers);
    assert(head  >= 0 && head  < cfg_.n_heads);

    int idx = head_idx(layer, head);
    ExecutionContext ctx = make_ctx(layer, head);

    /* Optionally select strategy from policy. */
    IKVStrategy *strat = strategies_[idx].get();
    if (policy_) {
        IKVStrategy *ps = policy_->select_strategy(layer, head, ctx);
        if (ps) strat = ps;
    }

    ICompression *comp = strat->compression();
    IEviction    *evic = strat->eviction();

    /* Compress K. */
    comp->compress(k_vec, cfg_.dim, true, ctx);

    /* Compress V. The storage backend maintains independent K/V regions,
     * so K and V remain paired even when the FIFO rings wrap. */
    int n = cache_sizes_[idx];
    comp->compress(v_vec, cfg_.dim, false, ctx);

    /* Eviction decision (after both K and V are stored). The storage ring
     * performs the physical overwrite; this callback remains available to
     * policies for observability. */
    ctx.cache_size = std::min(n + 1, cfg_.capacity);
    EvictionDecision d = evic->on_append(ctx);
    if (d.evict && d.evict_slot >= 0) {
        storages_[idx]->free_slot((StorageSlot)d.evict_slot);
    }

    cache_sizes_[idx] = std::min(n + 1, cfg_.capacity);

    /* Log K/V FP32 vectors if snapshot logging is enabled. */
    if (cfg_.log_tokens) {
        /* Append K then V floats to the flat data buffer. */
        size_t base = token_log_data_.size();
        token_log_data_.resize(base + 2 * cfg_.dim);
        memcpy(token_log_data_.data() + base,              k_vec, cfg_.dim * sizeof(float));
        memcpy(token_log_data_.data() + base + cfg_.dim,   v_vec, cfg_.dim * sizeof(float));

        TokenLogEntry entry;
        entry.layer  = layer;
        entry.head   = head;
        entry.dim    = cfg_.dim;
        entry.k_fp32 = nullptr; /* filled by snapshot capture from token_log_data_ */
        entry.v_fp32 = nullptr;
        token_log_.push_back(entry);
    }

    /* Advance token position after the last head of each layer. */
    if (layer == cfg_.n_layers - 1 && head == cfg_.n_heads - 1)
        ++token_pos_;
}

/* =========================================================================
 * compute — hot path
 * ========================================================================= */

ComputeMetrics RuntimeContext::compute(int          layer,
                                       int          head,
                                       const float *q_vec,
                                       float       *out) {
    assert(layer >= 0 && layer < cfg_.n_layers);
    assert(head  >= 0 && head  < cfg_.n_heads);

    auto t0 = std::chrono::high_resolution_clock::now();

    int idx = head_idx(layer, head);
    int n   = cache_sizes_[idx];

    ExecutionContext ctx = make_ctx(layer, head);

    IKVStrategy  *strat = strategies_[idx].get();
    IStorageBackend *st = storages_[idx].get();

    ComputeMetrics m{};
    m.layer         = layer;
    m.head          = head;
    m.n_tokens_used = n;
    m.avg_bits_per_dim = -1.f;
    m.quality          = -1.f;

    if (n == 0) {
        memset(out, 0, cfg_.dim * sizeof(float));
        m.latency_us = 0.f;
        return m;
    }

    int padded = next_pow2_rt(cfg_.dim);

    /* ---- 1. Rotate query (FWHT forward) -------------------------------- */
    q_rot_.assign(padded, 0.f);
    memcpy(q_rot_.data(), q_vec, cfg_.dim * sizeof(float));

    /* L2-normalise */
    float qnorm = 0.f;
    for (int i = 0; i < cfg_.dim; ++i) qnorm += q_vec[i] * q_vec[i];
    qnorm = sqrtf(qnorm + 1e-12f);
    float inv_qn = 1.f / qnorm;
    for (int i = 0; i < padded; ++i) q_rot_[i] *= inv_qn;

    /* Apply kernel FWHT forward — needs Rademacher D from the strategy's
     * underlying Quantizer. We call the kernel backend directly if the
     * strategy uses HARFixedStrategy internals. For V1 we call the scalar
     * fwht_forward directly since all strategies use the same Rademacher D.
     *
     * Design note: a proper V2 extension would expose "query rotation" as
     * an IKernelBackend method. For now, IKernelBackend::fwht_forward
     * takes a D vector; we get D from the strategy cast if possible,
     * otherwise fall back to an identity D (all +1). */
    /* Attempt dynamic cast to get the Rademacher D — strategy-specific. */


    /* Read one K slot to determine the format_tag. */
    CompressResult sample = st->read(0);
    const uint8_t ftag   = sample.format_tag;
    const bool is_fp32   = (ftag == 0xFF);

    if (is_fp32) {
        /* ---- FP32 path: plain dot products ----------------------------- */
        logits_.resize(n);
        const float attn_scale = 1.f / sqrtf((float)cfg_.dim);
        float mx = -1e30f;

        for (int i = 0; i < n; ++i) {
            CompressResult kr = st->read((StorageSlot)i);
            const float *k  = reinterpret_cast<const float *>(kr.data);
            float dot = 0.f;
            for (int d = 0; d < cfg_.dim; ++d) dot += q_vec[d] * k[d];
            logits_[i] = dot * attn_scale;
            if (logits_[i] > mx) mx = logits_[i];
        }

        /* Softmax. */
        float sv = 0.f;
        for (int i = 0; i < n; ++i) { logits_[i] = expf(logits_[i] - mx); sv += logits_[i]; }
        float inv_s = 1.f / sv;
        for (int i = 0; i < n; ++i) logits_[i] *= inv_s;

        m.logit_max = *std::max_element(logits_.begin(), logits_.begin() + n);
        m.logit_min = *std::min_element(logits_.begin(), logits_.begin() + n);

        /* V accumulation. */
        v_acc_.assign(cfg_.dim, 0.f);
        for (int i = 0; i < n; ++i) {
            CompressResult vr = st->read((StorageSlot)(ctx.cache_capacity + i));
            const float *v = reinterpret_cast<const float *>(vr.data);
            float w = logits_[i];
            for (int d = 0; d < cfg_.dim; ++d) v_acc_[d] += w * v[d];
        }
        memcpy(out, v_acc_.data(), cfg_.dim * sizeof(float));

        /* Quality: FP32 is always perfect. */
        recent_quality_[idx] = 1.f;

    } else {
        /* ---- HAR quantized path: delegate to HARFixedStrategy ---------- */
        /* compute_full() does the full HAR rotation + kdot + softmax + vaccum. */
        HARFixedStrategy *har = dynamic_cast<HARFixedStrategy *>(strat);
        if (har) {
            har->compute_full(q_vec, out, ctx);
            IQuality *q = har->quality();
            if (q) recent_quality_[idx] = q->estimated_quality();
        } else {
            /* Unknown quantized strategy: safe fallback → zero output. */
            memset(out, 0, cfg_.dim * sizeof(float));
        }
        m.logit_max = 0.f;
        m.logit_min = 0.f;
    }

    /* ---- 5. AttentionFeedback → eviction ------------------------------- */
    {
        static thread_local std::vector<int> attn_slots;
        attn_slots.resize(n);
        for (int i = 0; i < n; ++i) attn_slots[i] = i;
        float lat = 0.f;
        if (n > 0) {
            auto t1 = std::chrono::high_resolution_clock::now();
            lat = (float)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        }
        AttentionFeedback fb;
        fb.weights    = logits_.data();
        fb.slots      = attn_slots.data();
        fb.n          = n;
        fb.latency_us = lat;
        strat->eviction()->on_attention(fb, ctx);
    }

    /* ---- 6. Metrics ---------------------------------------------------- */
    auto t1 = std::chrono::high_resolution_clock::now();
    float lat_us = (float)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    recent_latency_[idx] = lat_us;
    m.latency_us         = lat_us;
    m.quality            = recent_quality_[idx];
    m.n_tokens_evicted   = 0;  /* eviction count maintained by eviction policy */

    /* IQuality report. */
    IQuality *iq = strat->quality();
    if (iq) m.avg_bits_per_dim = iq->avg_bits_per_dim();

    /* ---- 7. Policy on_metrics ----------------------------------------- */
    if (policy_) {
        policy_->on_metrics(layer, head, m);
    }

    return m;
}

} /* namespace adaptq */




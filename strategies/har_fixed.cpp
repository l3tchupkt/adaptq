#include <adaptq/strategy.h>
#include <adaptq/context.h>
#include <adaptq/storage.h>
#include <adaptq/config.h>
#include <adaptq/quality.h>
#include <codebook.h>
#include <fwht.h>
#include <quantizer.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <ostream>
#include <istream>
#include <vector>


/* -------------------------------------------------------------------------
 * strategies/har_fixed.cpp — HARFixedStrategy : IKVStrategy
 *
 * The current AdapTQ algorithm, lifted verbatim into the plugin interface.
 * HAR = Hadamard + Adaptive Rademacher (FWHT + random diagonal rotation).
 *
 * Algorithm:
 *   Append: L2-normalise → apply D (Rademacher) → FWHT → ±3σ soft-clip
 *           → Max-Lloyd VQ at `bits` (2/3/4) → pack indices → store scale
 *   Compute: Rotate query identically → dot-products against packed K →
 *            softmax → sparse-V accumulation → inverse FWHT → rescale
 *
 * This strategy produces identical results to the old AttentionHead::compute()
 * for the quantized path. Hybrid (FP32 warm-up) is delegated to the runtime.
 *
 * Capabilities: ICompression, IEviction (FIFO), IQuality, IReplayHooks
 * ----------------------------------------------------------------------- */

namespace adaptq {

/* ---- format_tag for packed HAR data ------------------------------------ */
static constexpr uint8_t HARTAG_2BIT = 0x02;
static constexpr uint8_t HARTAG_3BIT = 0x03;
static constexpr uint8_t HARTAG_4BIT = 0x04;

static inline uint8_t bits_to_tag(int bits) {
    switch (bits) {
    case 2:  return HARTAG_2BIT;
    case 3:  return HARTAG_3BIT;
    default: return HARTAG_4BIT;
    }
}

/* ---- Thread-local scratch (sizes match quantizer.cpp constants) -------- */
static constexpr int TL_BUF = 1024;

struct ThreadLocalWorkspace {
  std::vector<float> logits;
  std::vector<int> slots;
  std::vector<int> ord;
  std::vector<float> q_rot;
  std::vector<float> v_acc;

  void ensure_capacity(int n, int padded) {
    if (logits.size() < (size_t)n) {
      logits.resize(n);
      slots.resize(n);
      ord.resize(n);
    }
    if (q_rot.size() < (size_t)padded) {
      q_rot.resize(padded);
      v_acc.resize(padded);
    }
  }
};

static thread_local ThreadLocalWorkspace tl_ws;

/* ========================================================================
 * HARCompression — ICompression
 * ====================================================================== */
class HARCompression final : public ICompression {
public:
    Quantizer *quant  = nullptr;
    int        dim    = 0;
    int        padded = 0;
    int        bits   = 4;

    CompressResult compress(const float           *x,
                            int                  /*dim*/,
                            bool                 /*is_key*/,
                            const ExecutionContext &ctx) override {
        assert(quant && padded <= TL_BUF);
        /* quantize_into: normalise → D-apply → FWHT → clip → VQ → pack */
        static thread_local uint8_t tmp[TL_BUF];  /* at most 1024 bytes */
        float scale = quant->quantize_into(x, bits, tmp);
        int   pb    = (padded * bits + 7) / 8;
        StorageSlot slot = ctx.storage->write(tmp, pb, scale, bits_to_tag(bits));
        return ctx.storage->read(slot);
    }

    void decompress(const CompressResult &r,
                    int                   d,
                    float                *out) const override {
        assert(quant);
        quant->dequantize_raw(r.data, r.scale, padded, bits, out);
    }
};

/* ========================================================================
 * HARFIFOEviction — IEviction (ring-buffer FIFO, importance-unaware)
 * ====================================================================== */
class HARFIFOEviction final : public IEviction {
public:
    EvictionDecision on_append(const ExecutionContext &ctx) override {
        if (ctx.cache_size >= ctx.cache_capacity)
            return {true, ctx.cache_size % ctx.cache_capacity};
        return {false, -1};
    }
    void on_attention(const AttentionFeedback & /*fb*/,
                      const ExecutionContext   & /*ctx*/) override {}
};

/* ========================================================================
 * HARQuality — IQuality
 * Reports average bits and a simple reconstruction-based quality estimate.
 * ====================================================================== */
class HARQuality final : public IQuality {
public:
    int   bits_  = 4;
    float quality_ = -1.f; /* -1 = no compute() yet */

    explicit HARQuality(int b) : bits_(b) {}
    float avg_bits_per_dim()  const override { return (float)bits_; }
    float estimated_quality() const override { return quality_; }
    void  update(float q)                    { quality_ = q; }
};

/* ========================================================================
 * HARReplayHooks — IReplayHooks
 * HARFixed has no per-token internal state beyond what the storage holds.
 * ====================================================================== */
class HARReplayHooks final : public IReplayHooks {
public:
    void serialize_state(std::ostream &out) const override {
        uint32_t v = 1;
        out.write(reinterpret_cast<const char *>(&v), sizeof(v));
    }
    void deserialize_state(std::istream &in) override {
        uint32_t v = 0;
        in.read(reinterpret_cast<char *>(&v), sizeof(v));
    }
};

/* ========================================================================
 * HARFixedStrategy — IKVStrategy
 * ====================================================================== */

/* HeadConfig is defined in include/adaptq/config.h */

class HARFixedStrategy final : public IKVStrategy {
public:
    void init(const HeadConfig &cfg) override {
        cfg_    = cfg;
        quant_  = std::make_unique<Quantizer>();
        quant_->init(cfg.dim, cfg.seed);
        compress_.quant  = quant_.get();
        compress_.dim    = cfg.dim;
        compress_.padded = quant_->padded;
        compress_.bits   = cfg.bits;
        quality_         = std::make_unique<HARQuality>(cfg.bits);
        v_mass_          = cfg.v_mass;
    }

    void reset() override {
        if (quality_) quality_->update(-1.f);
    }

    const char *name() const override { return "har_fixed"; }

    ICompression *compression() override { return &compress_; }
    IEviction    *eviction()    override { return &evict_;    }
    IQuality     *quality()     override { return quality_.get(); }
    IReplayHooks *replay_hooks()override { return &replay_;   }

    /* ---- Custom compute helpers called by the runtime ------------------ */

    /**
     * Compute attention for a single query using the full HAR path.
     * Called by the runtime after rotating the query — strategy takes
     * over if has_custom_kdot() == false.
     *
     * NOTE: In V1 the runtime still calls AttentionHead::compute() for the
     * actual hot path. HARFixedStrategy exists to validate the interface
     * and provide the reference implementation for conformance tests.
     * The hot-path wiring from RuntimeContext → HARFixedStrategy is the
     * next step after RuntimeContext is written.
     */
    int compute_full(const float *q,
                     float       *out,
                     const ExecutionContext &ctx) {
        const int n      = ctx.cache_size;
        const int padded = compress_.padded;
        const int dim    = compress_.dim;
        const int bits   = compress_.bits;
        if (!n) { memset(out, 0, dim * sizeof(float)); return 0; }

        tl_ws.ensure_capacity(n, padded);
        float *logits = tl_ws.logits.data();
        int *slots    = tl_ws.slots.data();
        int *ord      = tl_ws.ord.data();

        assert(padded <= TL_BUF);

        /* 1. Rotate query (L2-normalise → D-apply → FWHT → scale) */
        float *qr = tl_ws.q_rot.data();
        memcpy(qr, q, dim * sizeof(float));
        for (int i = dim; i < padded; ++i) qr[i] = 0.f;
        float qn = 0.f;
        for (int i = 0; i < dim; ++i) qn += q[i] * q[i];
        qn = sqrtf(qn + 1e-12f);
        float inv_qn = (qn > 1e-12f) ? (1.f / qn) : 0.f;
        for (int i = 0; i < padded; ++i) qr[i] *= inv_qn;
        fwht_forward(qr, quant_->D.data(), padded);
        float sp = sqrtf((float)padded);
        for (int i = 0; i < padded; ++i) qr[i] *= sp * qn;

        const float *cb     = get_codebook(bits);
        const float  attn_s = 1.f / (sqrtf((float)dim) * (float)padded);
        const float  isp    = 1.f / sqrtf((float)padded);

        /* 2. Build slot list from storage backend */
        for (int i = 0; i < n; ++i) slots[i] = i;

        /* 3. K-dot + softmax (scalar reference path) */
        float mx = -1e30f;
        for (int i = 0; i < n; ++i) {
            CompressResult kr = ctx.storage->read((StorageSlot)slots[i]);
            float dot = kdot_ref(qr, kr.data, cb, padded, bits);
            logits[i] = dot * attn_s * kr.scale;
            if (logits[i] > mx) mx = logits[i];
        }
        float sv = 0.f;
        for (int i = 0; i < n; ++i) {
            logits[i] = expf(logits[i] - mx);
            sv += logits[i];
        }
        float inv_s = (sv > 1e-12f && std::isfinite(sv)) ? (1.f / sv) : (n > 0 ? 1.f / (float)n : 0.f);
        for (int i = 0; i < n; ++i) logits[i] *= inv_s;

        /* 4. V accumulation with optional sparse-V */
        float *acc = tl_ws.v_acc.data();
        memset(acc, 0, padded * sizeof(float));

        if (v_mass_ <= 0.f) {
            for (int i = 0; i < n; ++i) {
                CompressResult vr = ctx.storage->read((StorageSlot)slots[i] + (StorageSlot)ctx.cache_capacity);
                vaccum_ref(acc, vr.data, vr.scale * logits[i] * isp, cb, padded, bits);
            }
        } else {
            /* Sparse-V: accumulate only tokens contributing to v_mass_ of total */
            for (int i = 0; i < n; ++i) ord[i] = i;
            std::sort(ord, ord + n,
                      [&](int a, int b){ return logits[a] > logits[b]; });
            float mass = 0.f;
            for (int i = 0; i < n && mass < v_mass_; ++i) {
                int idx = ord[i];
                CompressResult vr = ctx.storage->read((StorageSlot)slots[idx] + (StorageSlot)ctx.cache_capacity);
                vaccum_ref(acc, vr.data, vr.scale * logits[idx] * isp, cb, padded, bits);
                mass += logits[idx];
            }
        }

        /* 5. Inverse FWHT + copy to output */
        fwht_inverse(acc, quant_->D.data(), padded);
        memcpy(out, acc, dim * sizeof(float));

        /* 6. Update quality estimate (entropy proxy) */
        if (quality_) {
            float entropy = 0.f;
            for (int i = 0; i < n; ++i) {
                float w = logits[i];
                if (w > 1e-12f) entropy -= w * logf(w);
            }
            /* Normalise: max entropy = log(n). High entropy → safe quantisation. */
            float max_entropy = (n > 1) ? logf((float)n) : 1.f;
            quality_->update(std::min(1.f, entropy / (max_entropy + 1e-12f)));
        }

        return n;
    }

private:
    HeadConfig                  cfg_;
    std::unique_ptr<Quantizer>  quant_;
    HARCompression              compress_;
    HARFIFOEviction             evict_;
    std::unique_ptr<HARQuality> quality_;
    HARReplayHooks              replay_;
    float                       v_mass_ = 0.f;

    /* Scalar K-dot: identical to kdot_scalar in attention.cpp */
    static float kdot_ref(const float    *qr,
                           const uint8_t  *packed,
                           const float    *cb,
                           int             padded,
                           int             bits) {
        float s = 0.f;
        int   i = 0;
        if (bits == 4) {
            for (int b = 0; b < padded/2; ++b, i += 2) {
                uint8_t byte = packed[b];
                s += qr[i] * cb[byte >> 4] + qr[i+1] * cb[byte & 0xF];
            }
        } else if (bits == 2) {
            for (int b = 0; b < padded/4; ++b, i += 4) {
                uint8_t byte = packed[b];
                s += qr[i]   * cb[(byte>>6)&3] + qr[i+1] * cb[(byte>>4)&3]
                   + qr[i+2] * cb[(byte>>2)&3] + qr[i+3] * cb[ byte    &3];
            }
        } else { /* 3-bit */
            for (int g = 0; g < padded/8; ++g, i += 8) {
                const uint8_t *p = packed + g*3;
                s += qr[i]   * cb[(p[0]>>5)&7]
                   + qr[i+1] * cb[(p[0]>>2)&7]
                   + qr[i+2] * cb[((p[0]&3)<<1)|(p[1]>>7)]
                   + qr[i+3] * cb[(p[1]>>4)&7]
                   + qr[i+4] * cb[(p[1]>>1)&7]
                   + qr[i+5] * cb[((p[1]&1)<<2)|(p[2]>>6)]
                   + qr[i+6] * cb[(p[2]>>3)&7]
                   + qr[i+7] * cb[ p[2]    &7];
            }
        }
        return s;
    }

    static void vaccum_ref(float         *acc,
                            const uint8_t *packed,
                            float          w_scale,
                            const float   *cb,
                            int            padded,
                            int            bits) {
        int i = 0;
        if (bits == 4) {
            for (int b = 0; b < padded/2; ++b, i += 2) {
                uint8_t byte = packed[b];
                acc[i]   += w_scale * cb[byte >> 4];
                acc[i+1] += w_scale * cb[byte & 0xF];
            }
        } else if (bits == 2) {
            for (int b = 0; b < padded/4; ++b, i += 4) {
                uint8_t byte = packed[b];
                acc[i]   += w_scale * cb[(byte>>6)&3];
                acc[i+1] += w_scale * cb[(byte>>4)&3];
                acc[i+2] += w_scale * cb[(byte>>2)&3];
                acc[i+3] += w_scale * cb[ byte    &3];
            }
        } else {
            for (int g = 0; g < padded/8; ++g, i += 8) {
                const uint8_t *p = packed + g*3;
                acc[i]   += w_scale * cb[(p[0]>>5)&7];
                acc[i+1] += w_scale * cb[(p[0]>>2)&7];
                acc[i+2] += w_scale * cb[((p[0]&3)<<1)|(p[1]>>7)];
                acc[i+3] += w_scale * cb[(p[1]>>4)&7];
                acc[i+4] += w_scale * cb[(p[1]>>1)&7];
                acc[i+5] += w_scale * cb[((p[1]&1)<<2)|(p[2]>>6)];
                acc[i+6] += w_scale * cb[(p[2]>>3)&7];
                acc[i+7] += w_scale * cb[ p[2]    &7];
            }
        }
    }
};

} /* namespace adaptq */

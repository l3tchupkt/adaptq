#include <adaptq/strategy.h>
#include <adaptq/context.h>
#include <adaptq/storage.h>
#include <adaptq/config.h>
#include <cstring>
#include <memory>
#include <ostream>
#include <istream>


/* -------------------------------------------------------------------------
 * strategies/fp_passthrough.cpp — FPPassthroughStrategy : IKVStrategy
 *
 * Stores K and V as raw float32 without any compression.
 * Compression ratio: 1.0× (no savings; baseline for quality comparison).
 *
 * Purpose:
 *   1. Quality baseline — compare all other strategies against FP passthrough
 *      in `adaptq compare` to measure quality delta.
 *   2. Warm-up path — use as default strategy for the first N tokens before
 *      switching to a compressed strategy via CostDrivenPolicy.
 *   3. Critical layer fallback — CostDrivenPolicy assigns FPPassthrough
 *      to heads where compressed quality falls below quality_floor.
 *   4. Research sanity check — any benchmark result should match the
 *      FP32 attention output to within FP32 round-off when using this strategy.
 *
 * Eviction: FIFO (ring-buffer). No importance tracking needed.
 * Quality:  estimated_quality() always returns 1.0 (exact FP32).
 * Replay:   full state serialization via serialize_state()/deserialize_state().
 * ----------------------------------------------------------------------- */

namespace adaptq {

/* format_tag for FP32 storage. kdot_batch and vaccum_batch in the kernel
 * backends check this tag and use the FP32 path instead of the codebook.  */
static constexpr uint8_t FPTAG_F32 = 0xFF;

/* ---- Capability: FPCompression --------------------------------------- */
class FPCompression final : public ICompression {
public:
    int dim    = 0;
    int padded = 0;

    CompressResult compress(const float           *x,
                            int                    dim,
                            bool                  /*is_key*/,
                            const ExecutionContext &ctx) override {
        /* Write raw float32 bytes directly into the storage backend. */
        int bytes = dim * sizeof(float);
        StorageSlot slot = ctx.storage->write(
            reinterpret_cast<const uint8_t *>(x),
            bytes,
            1.0f,        /* scale = 1: no quantization */
            FPTAG_F32
        );
        const CompressResult r = ctx.storage->read(slot);
        return r;
    }

    void decompress(const CompressResult &r,
                    int                   dim,
                    float                *out) const override {
        memcpy(out, r.data, dim * sizeof(float));
    }

    /* No custom kernels — runtime dispatches to IKernelBackend. */
};

/* ---- Capability: FIFOEviction ----------------------------------------- */
class FIFOEviction final : public IEviction {
public:
    EvictionDecision on_append(const ExecutionContext &ctx) override {
        /* Evict the oldest slot when at capacity.
         * ContiguousSlabStorage handles ring-buffer eviction internally
         * on the next write(), so no explicit slot is needed here.       */
        if (ctx.cache_size >= ctx.cache_capacity)
            return {true, ctx.cache_size % ctx.cache_capacity};
        return {false, -1};
    }

    void on_attention(const AttentionFeedback & /*fb*/,
                      const ExecutionContext   & /*ctx*/) override {
        /* FIFO: no importance tracking. Attention weights are ignored.   */
    }
};

/* ---- Capability: FPQuality -------------------------------------------- */
class FPQuality final : public IQuality {
public:
    int dim_ = 0;
    explicit FPQuality(int dim) : dim_(dim) {}
    float avg_bits_per_dim()  const override { return 32.0f; }
    float estimated_quality() const override { return 1.0f;  }
};

/* ---- Capability: FPReplayHooks ---------------------------------------- */
class FPReplayHooks final : public IReplayHooks {
public:
    /* FPPassthrough has no internal state beyond what the storage holds. */
    void serialize_state(std::ostream &out) const override {
        /* Write a version marker. */
        uint32_t version = 1;
        out.write(reinterpret_cast<const char *>(&version), sizeof(version));
    }
    void deserialize_state(std::istream &in) override {
        uint32_t version = 0;
        in.read(reinterpret_cast<char *>(&version), sizeof(version));
        /* Nothing else to restore for FIFO/FP32. */
    }
};

/* ---- FPPassthroughStrategy : IKVStrategy ------------------------------ */
class FPPassthroughStrategy final : public IKVStrategy {
public:
    void init(const HeadConfig &config) override;
    void reset()                        override;
    const char *name()            const override { return "fp_passthrough"; }

    ICompression *compression() override { return &compress_; }
    IEviction    *eviction()    override { return &evict_;    }
    IQuality     *quality()     override { return quality_.get(); }
    IReplayHooks *replay_hooks()override { return &replay_;   }

private:
    FPCompression                   compress_;
    FIFOEviction                    evict_;
    std::unique_ptr<FPQuality>      quality_;
    FPReplayHooks                   replay_;
};

void FPPassthroughStrategy::init(const HeadConfig &config) {
    /* HeadConfig defined in config.h — dim and capacity at minimum. */
    quality_ = std::make_unique<FPQuality>(config.dim);
    (void)config;
}

void FPPassthroughStrategy::reset() {
    /* No internal state to reset beyond what the storage backend handles. */
}

} /* namespace adaptq */

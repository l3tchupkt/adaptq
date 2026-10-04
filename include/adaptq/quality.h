#pragma once
#include <memory>
#include <vector>
#include "context.h"

namespace adaptq {

class IQualityOracle {
public:
    virtual ~IQualityOracle() = default;

    /**
     * Estimate attention output quality after a compute() call.
     *
     * @param attn_weights  float[n] post-softmax attention weights
     * @param logits        float[n] pre-softmax logits
     * @param n             number of cached tokens used in this compute
     * @param ctx           ExecutionContext for the head that just ran
     * @return              quality estimate in [0, 1].
     *                      0 = very low quality (high quantization error).
     *                      1 = indistinguishable from FP32.
     *                      Returns -1.f if estimation is not possible
     *                      (e.g., no samples available yet).
     */
    virtual float estimate(const float           *attn_weights,
                           const float           *logits,
                           int                    n,
                           const ExecutionContext &ctx) = 0;

    /** Human-readable name for diagnostics and profiler output. */
    virtual const char *name() const = 0;
};

/* ---- EnsembleOracle — weighted combination of constituent oracles ------ */
/**
 * Combines multiple IQualityOracle instances with weights.
 * The default oracle configured by the runtime is:
 *   ReconstructionOracle (weight 0.5)
 *   EntropyOracle        (weight 0.3)
 *   LogitSpreadOracle    (weight 0.2)
 *
 * Researchers can substitute a different ensemble or a standalone oracle.
 */
class EnsembleOracle : public IQualityOracle {
public:
    struct Entry {
        std::unique_ptr<IQualityOracle> oracle;
        float weight;
    };

    void add(std::unique_ptr<IQualityOracle> oracle, float weight) {
        entries_.push_back({std::move(oracle), weight});
    }

    float estimate(const float           *attn_weights,
                   const float           *logits,
                   int                    n,
                   const ExecutionContext &ctx) override {
        if (entries_.empty()) return -1.f;
        float total_weight = 0.f, total_quality = 0.f;
        for (auto &e : entries_) {
            float q = e.oracle->estimate(attn_weights, logits, n, ctx);
            if (q < 0.f) continue;  /* oracle not ready; skip */
            total_quality += e.weight * q;
            total_weight  += e.weight;
        }
        return (total_weight > 0.f) ? (total_quality / total_weight) : -1.f;
    }

    const char *name() const override { return "ensemble"; }

private:
    std::vector<Entry> entries_;
};

/**
 * Build the default ensemble oracle used by the runtime.
 * Weights: Reconstruction(0.5), Entropy(0.3), LogitSpread(0.2).
 * Declared here; defined in core/quality_oracle.cpp.
 */
std::unique_ptr<IQualityOracle> make_default_quality_oracle();

} /* namespace adaptq */

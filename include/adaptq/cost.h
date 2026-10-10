#pragma once
#include <cfloat>

/* -------------------------------------------------------------------------
 * adaptq/cost.h — ICostFunction, CostWeights, and built-in presets
 *
 * C++17-compatible. Designated-initializer syntax (.field = value) was
 * removed; use the CostWeights() constructor helpers instead.
 * ----------------------------------------------------------------------- */

namespace adaptq {

/* ---- CostWeights — the coefficient vector ----------------------------- */

struct CostWeights {
    float memory_mb_per_unit    = 1.0f;
    float latency_us_per_unit   = 0.5f;
    float quality_loss_per_unit = 10.0f;
    float bandwidth_gbps        = 0.0f;
    float energy_joules         = 0.0f;
    float memory_hard_limit_mb  = FLT_MAX;
    float quality_floor         = 0.0f;
    float latency_hard_limit_us = FLT_MAX;
};

/* ---- CostFunctionState — runtime state passed to weights() ------------ */

struct CostFunctionState {
    float memory_used_mb    = 0.f;
    float memory_budget_mb  = FLT_MAX;
    float cache_fill_ratio  = 0.f;
    bool  approaching_budget = false;
    float recent_quality    = 1.f;
    float quality_floor     = 0.f;
    int   token_pos         = 0;
    float recent_latency_us = 0.f;
};

/* ---- OperatingPoint --------------------------------------------------- */

struct OperatingPoint {
    float memory_mb     = 0.f;
    float latency_us    = 0.f;
    float quality       = 1.f;
    float bandwidth_gbps= 0.f;
    float energy_joules = 0.f;
};

/* ---- ICostFunction ---------------------------------------------------- */

class ICostFunction {
public:
    virtual ~ICostFunction() = default;

    virtual CostWeights weights(const CostFunctionState &state) const = 0;

    virtual float cost(const OperatingPoint    &op,
                       const CostFunctionState &state) const {
        CostWeights w = weights(state);
        float c = w.memory_mb_per_unit    *  op.memory_mb
                + w.latency_us_per_unit   *  op.latency_us
                + w.quality_loss_per_unit * (1.f - op.quality)
                + w.bandwidth_gbps        *  op.bandwidth_gbps
                + w.energy_joules         *  op.energy_joules;
        if (op.memory_mb < 0.f || op.memory_mb > state.memory_budget_mb)
            c += 1e9f;
        if (op.quality < state.quality_floor)
            c += 1e9f;
        return c;
    }

    virtual const char *name() const = 0;
};

/* ---- ConstantCostFunction --------------------------------------------- */

class ConstantCostFunction : public ICostFunction {
public:
    explicit ConstantCostFunction(CostWeights w) : w_(w) {}
    CostWeights weights(const CostFunctionState &) const override { return w_; }
    const char *name() const override { return "constant"; }
private:
    CostWeights w_;
};

/* ---- AdaptiveCostFunction --------------------------------------------- */

class AdaptiveCostFunction : public ICostFunction {
public:
    explicit AdaptiveCostFunction(CostWeights base) : base_(base) {}

    CostWeights weights(const CostFunctionState &s) const override {
        CostWeights w = base_;
        w.memory_mb_per_unit *= (1.f + 3.f * s.cache_fill_ratio);
        if (s.approaching_budget) w.memory_mb_per_unit *= 10.f;
        w.quality_loss_per_unit *= (1.f - 0.2f * s.cache_fill_ratio);
        return w;
    }
    const char *name() const override { return "adaptive"; }
private:
    CostWeights base_;
};

/* ---- CostPreset — C++17-compatible factory helpers -------------------- */
namespace CostPreset {

inline ICostFunction *EdgeDevice() {
    CostWeights w;
    w.memory_mb_per_unit    = 10.0f;
    w.latency_us_per_unit   = 2.0f;
    w.quality_loss_per_unit = 5.0f;
    w.memory_hard_limit_mb  = 256.f;
    return new ConstantCostFunction(w);
}

inline ICostFunction *LaptopInference() {
    CostWeights w;
    w.memory_mb_per_unit    = 3.0f;
    w.latency_us_per_unit   = 2.0f;
    w.quality_loss_per_unit = 8.0f;
    return new AdaptiveCostFunction(w);
}

inline ICostFunction *ServerThroughput() {
    CostWeights w;
    w.memory_mb_per_unit    = 1.0f;
    w.latency_us_per_unit   = 5.0f;
    w.quality_loss_per_unit = 8.0f;
    return new ConstantCostFunction(w);
}

inline ICostFunction *ResearchAccuracy() {
    CostWeights w;
    w.memory_mb_per_unit    = 1.0f;
    w.latency_us_per_unit   = 1.0f;
    w.quality_loss_per_unit = 20.0f;
    return new ConstantCostFunction(w);
}

inline ICostFunction *EnergyEfficient() {
    CostWeights w;
    w.memory_mb_per_unit    = 2.0f;
    w.latency_us_per_unit   = 1.0f;
    w.quality_loss_per_unit = 8.0f;
    w.energy_joules         = 5.0f;
    return new ConstantCostFunction(w);
}

} /* namespace CostPreset */

} /* namespace adaptq */

#include "adaptq/cost.h"

#include <algorithm>
#include <cmath>
#include <vector>

/* -------------------------------------------------------------------------
 * core/cost_optimizer.cpp — Pareto optimizer using ICostFunction
 *
 * Implements the calibration + selection phase of the Constraint-Driven
 * Runtime. In V1 this runs offline (on a recorded session) or on the first
 * 256 tokens of a new session.
 *
 * Phase 1 (Calibration): probe a grid of (bits, eviction_aggressiveness)
 *   and measure (memory, latency, quality) for each.
 *
 * Phase 2 (Selection): find the grid point minimizing ICostFunction::cost()
 *   subject to hard constraint penalties (automatically included in cost).
 *
 * Phase 3 (Control loop): every 128 tokens, re-evaluate cost at the current
 *   operating point. If cost increased > 20%, re-run Phase 2.
 *   Stability guard: do not re-select more than once per 64 tokens.
 *
 * In V1 the control loop fires but only logs the recommendation; actual
 * strategy switching requires CostDrivenPolicy (V2). The optimizer is
 * written as a standalone utility usable from `adaptq optimize`.
 * ----------------------------------------------------------------------- */

namespace adaptq {

/* ---- OperatingPoint observation record -------------------------------- */
struct Observation {
    int   bits;
    float eviction_aggressiveness;  /* 0=none, 1=maximum */
    float memory_mb;
    float latency_us;
    float quality;
    float cost;                     /* computed from ICostFunction */
};

/* ---- CostOptimizer ---------------------------------------------------- */
class CostOptimizer {
public:
    explicit CostOptimizer(ICostFunction *cost_fn) : cost_fn_(cost_fn) {}

    /**
     * Run Phase 1: grid search over (bits × eviction_aggressiveness).
     * In V1 the observations are supplied externally from a calibration run.
     * In V2 this will call the runtime to measure each point.
     */
    void add_observation(int bits, float eviction_agg,
                         float memory_mb, float latency_us, float quality,
                         const CostFunctionState &state) {
        Observation obs;
        obs.bits                 = bits;
        obs.eviction_aggressiveness = eviction_agg;
        obs.memory_mb            = memory_mb;
        obs.latency_us           = latency_us;
        obs.quality              = quality;
        OperatingPoint op{memory_mb, latency_us, quality, 0.f, 0.f};
        obs.cost = cost_fn_->cost(op, state);
        observations_.push_back(obs);
    }

    /**
     * Phase 2: select the minimum-cost observation.
     * Returns the index into observations_ of the selected operating point.
     * Returns -1 if no valid observation (all have infinite cost from hard
     * constraint violations).
     */
    int select() const {
        int    best_idx  = -1;
        float  best_cost = 1e38f;
        for (int i = 0; i < (int)observations_.size(); ++i) {
            if (observations_[i].cost < best_cost) {
                best_cost = observations_[i].cost;
                best_idx  = i;
            }
        }
        return best_idx;
    }

    const Observation *selected_observation() const {
        int idx = select();
        return (idx >= 0) ? &observations_[idx] : nullptr;
    }

    /**
     * Phase 3: re-evaluate current cost and check if re-selection is needed.
     * Returns true if re-selection is recommended.
     *
     * @param current_cost  cost of the currently active operating point
     * @param threshold     fractional cost increase triggering re-selection
     */
    bool should_reselect(float current_cost, float threshold = 0.20f) const {
        const Observation *best = selected_observation();
        if (!best) return false;
        float relative_increase = (current_cost - best->cost) / (best->cost + 1e-12f);
        return relative_increase > threshold;
    }

    void clear_observations() { observations_.clear(); }
    const std::vector<Observation> &observations() const { return observations_; }

private:
    ICostFunction              *cost_fn_;
    std::vector<Observation>    observations_;
};

/**
 * Build a default calibration grid.
 * Returns a vector of (bits, eviction_agg) pairs to probe.
 */
std::vector<std::pair<int, float>> default_calibration_grid() {
    std::vector<std::pair<int, float>> grid;
    for (int bits : {2, 3, 4}) {
        for (float agg : {0.0f, 0.5f, 1.0f}) {
            grid.push_back({bits, agg});
        }
    }
    return grid;
}

} /* namespace adaptq */

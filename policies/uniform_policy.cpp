#include "../../include/adaptq/policy.h"
#include "../../include/adaptq/strategy.h"
#include "../../include/adaptq/storage.h"
#include "../../include/adaptq/context.h"

/* -------------------------------------------------------------------------
 * policies/uniform_policy.cpp — UniformPolicy : IPolicy
 *
 * Selects the same IKVStrategy and IStorageBackend for every (layer, head)
 * across the entire session.
 *
 * This is the V1 default policy. It replicates the behavior of the old
 * hard-coded AttentionHead — one strategy, one storage type, no switching.
 *
 * Usage:
 *   Runtime rt;
 *   rt.set_policy(std::make_unique<UniformPolicy>(
 *       my_strategy_instance,
 *       my_storage_instance
 *   ));
 *
 * CostDrivenPolicy (V2) builds on top of this by selecting from a pool of
 * strategies and calling on_metrics() to decide when to switch.
 * ----------------------------------------------------------------------- */

namespace adaptq {

class UniformPolicy final : public IPolicy {
public:
    /**
     * @param strategy  Strategy to use for all heads. Non-owning.
     *                  The runtime owns the strategy lifetime.
     * @param storage   Storage backend to use for all heads. Non-owning.
     */
    UniformPolicy(IKVStrategy *strategy, IStorageBackend *storage)
        : strategy_(strategy), storage_(storage) {}

    void init(int          /*n_layers*/,
              int          /*n_heads*/,
              const RuntimeConfig & /*config*/) override {
        /* Uniform: no per-layer or per-head state to initialize. */
    }

    IKVStrategy *select_strategy(int                    /*layer*/,
                                  int                   /*head*/,
                                  const ExecutionContext & /*ctx*/) override {
        return strategy_;
    }

    IStorageBackend *select_storage(int                    /*layer*/,
                                    int                   /*head*/,
                                    const ExecutionContext & /*ctx*/) override {
        return storage_;
    }

    void on_metrics(int                  /*layer*/,
                    int                  /*head*/,
                    const ComputeMetrics & /*m*/) override {
        /* Uniform: no adaptation; metrics are ignored. */
    }

    const char *name() const override { return "uniform"; }

private:
    IKVStrategy     *strategy_;
    IStorageBackend *storage_;
};

} /* namespace adaptq */

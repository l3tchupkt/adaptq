#pragma once
#include "context.h"
#include "config.h"
#include "metrics.h"
#include "strategy.h"
#include "storage.h"

/* -------------------------------------------------------------------------
 * adaptq/policy.h — IPolicy interface
 *
 * IPolicy sits one level above IKVStrategy. It decides:
 *   - which strategy to use for each (layer, head) at each token
 *   - which storage backend to pair with it
 *   - when to switch strategy mid-session
 *
 * The runtime calls IPolicy before each append/compute operation.
 * The runtime owns all strategy and storage instances; IPolicy returns
 * non-owning pointers to the selected objects.
 *
 * Built-in policies:
 *   UniformPolicy       — same strategy and storage for all heads (default)
 *   LayerAdaptivePolicy — per-layer config, different strategy per layer
 *   CostDrivenPolicy    — minimizes ICostFunction; enables mid-session switch
 * ----------------------------------------------------------------------- */

namespace adaptq {

class IPolicy {
public:
    virtual ~IPolicy() = default;

    /**
     * Called once at runtime initialization.
     * Policy pre-allocates any per-layer or per-head internal state.
     */
    virtual void init(int n_layers, int n_heads,
                      const RuntimeConfig &config) = 0;

    /**
     * Select which strategy to use for this (layer, head) at this token.
     * May return a different strategy than the previous call — mid-session
     * switching is explicitly allowed and is the key feature of
     * CostDrivenPolicy.
     *
     * @param layer  transformer layer index
     * @param head   attention head index
     * @param ctx    current ExecutionContext (memory usage, quality, etc.)
     * @return       non-owning pointer to the selected IKVStrategy
     */
    virtual IKVStrategy *select_strategy(int                   layer,
                                         int                   head,
                                         const ExecutionContext &ctx) = 0;

    /**
     * Select which storage backend to use for this (layer, head).
     * Called once per head at context creation; storage backends are
     * not switched mid-session (only strategies are).
     *
     * @return non-owning pointer to the selected IStorageBackend
     */
    virtual IStorageBackend *select_storage(int                   layer,
                                            int                   head,
                                            const ExecutionContext &ctx) = 0;

    /**
     * Called after each compute() with observed metrics.
     * CostDrivenPolicy uses this to update cost estimates and potentially
     * queue a strategy switch for the next token.
     */
    virtual void on_metrics(int                  layer,
                            int                  head,
                            const ComputeMetrics &m) = 0;

    /** Human-readable name for diagnostics. */
    virtual const char *name() const = 0;
};

} /* namespace adaptq */

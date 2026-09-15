#pragma once
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>
#include "../include/adaptq/config.h"
#include "../include/adaptq/context.h"
#include "../include/adaptq/metrics.h"
#include "../include/adaptq/strategy.h"
#include "../include/adaptq/storage.h"
#include "../include/adaptq/kernel.h"
#include "../include/adaptq/policy.h"
#include "../include/adaptq/quality.h"

/* -------------------------------------------------------------------------
 * runtime/runtime_context.h — RuntimeContext
 *
 * RuntimeContext is the top-level orchestration object for one inference
 * session. It owns every plugin instance and wires together:
 *
 *   IPolicy
 *     → IKVStrategy  (n_layers × n_heads instances, one per head)
 *     → IStorageBackend (n_layers × n_heads instances)
 *   IKernelBackend   (one process-global selection, shared across all heads)
 *   IQualityOracle   (one per context, shared)
 *
 * Ownership:
 *   RuntimeContext owns strategy and storage instances (unique_ptr vectors).
 *   IPolicy is borrowed — caller manages the policy lifetime.
 *   IKernelBackend is borrowed from select_kernel_backend() (process-global).
 *   IQualityOracle is owned if created by RuntimeContext::init().
 *
 * Thread safety:
 *   RuntimeContext is NOT thread-safe. Use one instance per thread.
 *   Within a single RuntimeContext, all append/compute calls are sequential.
 * ----------------------------------------------------------------------- */

namespace adaptq {

/* ---- Per-token entry stored in the token log (for snapshot replay) ----- */
struct TokenLogEntry {
    int    layer;
    int    head;
    int    dim;
    /* k_vec and v_vec are owned by the dynamic storage in token_log_data_ */
    const float *k_fp32;   /* non-owning view into token_log_data_         */
    const float *v_fp32;
};

/* ---- RuntimeContext configuration -------------------------------------- */
struct RuntimeContextConfig {
    int   n_layers   = 1;
    int   n_heads    = 1;
    int   dim        = 128;
    int   bits       = 4;
    int   capacity   = 4096;
    float v_mass     = 0.f;
    float memory_budget_mb      = 2048.f;
    float quality_floor         = 0.f;
    float latency_hard_limit_us = 0.f;
    bool  log_tokens = false;   /* enable token log for snapshot replay    */
};

/* ---- Strategy factory function type ------------------------------------ */
using StrategyFactory = IKVStrategy *(*)();
using StorageFactory  = IStorageBackend *(*)();

/* ---- RuntimeContext ---------------------------------------------------- */
class RuntimeContext {
public:
    RuntimeContext() = default;
    ~RuntimeContext() = default;

    RuntimeContext(const RuntimeContext &)            = delete;
    RuntimeContext &operator=(const RuntimeContext &) = delete;

    /* ---- Initialization -------------------------------------------- */

    /**
     * Initialize the context with default factories:
     *   strategy = HARFixedStrategy
     *   storage  = ContiguousSlabStorage
     *   kernel   = select_kernel_backend() (best available)
     *   oracle   = make_default_quality_oracle()
     *
     * Allocates n_layers × n_heads strategy and storage instances.
     */
    void init(const RuntimeContextConfig &cfg);

    /**
     * Initialize with explicit strategy/storage factories.
     * Called by ReplayEngine when replaying with alternate strategies.
     */
    void init(const RuntimeContextConfig &cfg,
              StrategyFactory             strategy_fn,
              StorageFactory              storage_fn);

    /**
     * Override the policy (called before any append/compute).
     * policy lifetime must outlive this RuntimeContext.
     */
    void set_policy(IPolicy *policy);

    /* ---- Session lifecycle ----------------------------------------- */

    /**
     * Reset all per-session state (storage, strategy, quality).
     * Keeps configuration, factory selections, and policy.
     */
    void reset();

    /* ---- Hot path ----------------------------------------------------- */

    /**
     * Append one K/V pair for (layer, head) at the current token position.
     *
     * @param layer   transformer layer [0, n_layers)
     * @param head    attention head   [0, n_heads)
     * @param k_vec   float[dim] key vector
     * @param v_vec   float[dim] value vector
     */
    void append(int          layer,
                int          head,
                const float *k_vec,
                const float *v_vec);

    /**
     * Compute attention output for (layer, head) given a query.
     *
     * Uses the kernel backend for FWHT + kdot_batch + vaccum_batch,
     * falling back to strategy's custom kernels if has_custom_kdot().
     *
     * @param layer   transformer layer [0, n_heads)
     * @param head    attention head    [0, n_heads)
     * @param q_vec   float[dim] query vector
     * @param out     float[dim] output (written by this call)
     * @return        ComputeMetrics for this call
     */
    ComputeMetrics compute(int          layer,
                           int          head,
                           const float *q_vec,
                           float       *out);

    /* ---- Accessors ----------------------------------------------------- */

    int n_layers()   const { return cfg_.n_layers; }
    int n_heads()    const { return cfg_.n_heads;  }
    int dim()        const { return cfg_.dim;      }
    int token_pos()  const { return token_pos_;    }

    IKVStrategy     *get_strategy(int layer, int head) const;
    IStorageBackend *get_storage (int layer, int head) const;
    IKernelBackend  *get_kernel  ()                    const { return kernel_; }
    IQualityOracle  *get_oracle  ()                    const { return oracle_.get(); }

    /* ---- Token log (snapshot support) --------------------------------- */

    /** Returns the recorded token log entries (empty if log_tokens=false). */
    const std::vector<TokenLogEntry> &token_log() const { return token_log_; }

    /** Raw float storage backing the token log (keeps ownership). */
    const std::vector<float> &token_log_data() const { return token_log_data_; }

private:
    /* Build ExecutionContext for a (layer, head, token_pos). */
    ExecutionContext make_ctx(int layer, int head) const;

    /* Index into flat strategy/storage vectors. */
    int head_idx(int layer, int head) const {
        if (layer < 0 || layer >= cfg_.n_layers || head < 0 || head >= cfg_.n_heads)
            throw std::out_of_range("RuntimeContext: layer or head index out of range");
        return layer * cfg_.n_heads + head;
    }

    /* Per-head state. */
    std::vector<std::unique_ptr<IKVStrategy>>    strategies_;
    std::vector<std::unique_ptr<IStorageBackend>> storages_;
    std::vector<int>                             cache_sizes_;    /* per head */
    std::vector<float>                           recent_quality_; /* per head */
    std::vector<float>                           recent_latency_; /* per head */

    /* Shared infrastructure (non-owning for kernel, owned for oracle). */
    IKernelBackend                   *kernel_  = nullptr;
    std::unique_ptr<IQualityOracle>   oracle_;
    IPolicy                          *policy_  = nullptr;

    /* Config and session state. */
    RuntimeContextConfig cfg_;
    int                  token_pos_ = 0;

    /* Token log for snapshot replay. */
    std::vector<TokenLogEntry> token_log_;
    std::vector<float>         token_log_data_;  /* owns the float data */

    /* Scratch buffers for compute() — allocated once, reused. */
    std::vector<float>         q_rot_;           /* rotated query [padded] */
    std::vector<float>         v_acc_;           /* V accumulator [padded]  */
    std::vector<CompressResult> k_results_;      /* K vectors for kdot_batch */
    std::vector<CompressResult> v_results_;      /* V vectors for vaccum_batch*/
    std::vector<float>          logits_;         /* pre-softmax logits      */
};

} /* namespace adaptq */

/* ---- Registry and factory accessors (defined in runtime_context.cpp) --- */
namespace adaptq {

/**
 * Returns the StrategyFactory for the named strategy, or nullptr if unknown.
 * Available names in V2: "har_fixed", "fp_passthrough".
 */
StrategyFactory strategy_factory_by_name(const char *name);

/**
 * Returns a pointer to the array of known strategy names, and sets *out_count.
 */
const char **strategy_names(int *out_count);

/**
 * Returns a new ContiguousSlabStorage instance (caller takes ownership).
 * Used as the default StorageFactory in RuntimeContext::init().
 */
IStorageBackend *make_contiguous();

} /* namespace adaptq (registry) */


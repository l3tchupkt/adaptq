#pragma once
#include <cstddef>
#include <cstdint>

/* -------------------------------------------------------------------------
 * adaptq/context.h — ExecutionContext and supporting structs
 *
 * ExecutionContext is the single object passed to every IKVStrategy method.
 * It replaces individual parameter lists and makes the strategy API stable:
 * new runtime state fields are added here without changing method signatures.
 *
 * Non-owning pointers inside ExecutionContext are valid only for the duration
 * of the call. Strategies must not store them.
 * ----------------------------------------------------------------------- */

namespace adaptq {

/* Forward declarations — implementations in their own headers. */
struct IStorageBackend;
struct IKernelBackend;
struct IQualityOracle;

/* ---- Per-token metadata passed into compress() and on_append() --------- */
struct TokenContext {
    int layer;
    int head;
    int token_pos;       /* absolute position in the full sequence */
    int cache_size;      /* number of tokens currently in cache    */
    int cache_capacity;  /* maximum tokens the cache can hold      */
    float memory_used_mb;
    float memory_budget_mb;
    bool  approaching_budget; /* memory_used > 90% of budget       */
};

/* ---- Post-compute feedback passed into on_attention() ------------------ */
struct AttentionFeedback {
    const float *weights;  /* softmax weights for all cached tokens [n]    */
    const int   *slots;    /* slot indices corresponding to each weight [n] */
    int          n;        /* number of cached tokens                      */
    float        latency_us;
};

/* ---- Full context for strategy method calls ---------------------------- */
struct ExecutionContext {
    /* Identity */
    int layer;
    int head;
    int token_pos;

    /* Cache state */
    int   cache_size;
    int   cache_capacity;
    float memory_used_mb;
    float memory_budget_mb;
    bool  approaching_budget;  /* memory_used > 90% of budget */

    /* Runtime observations (rolling estimates from the last compute call) */
    float recent_quality;       /* last IQualityOracle estimate [0, 1]  */
    float recent_latency_us;    /* last compute() latency in µs         */

    /* Infrastructure access — non-owning, valid for the duration of the call */
    IStorageBackend *storage;
    IKernelBackend  *kernel;
    IQualityOracle  *quality_oracle;  /* strategy may query; may be nullptr */

    /* Active constraints */
    float quality_floor;           /* abort compression if quality < this */
    float latency_hard_limit_us;   /* 0 = no limit                        */
};

} /* namespace adaptq */

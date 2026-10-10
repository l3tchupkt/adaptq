#pragma once
#include <cstddef>
#include <iosfwd>
#include "config.h"
#include "storage.h"
#include "context.h"

/* -------------------------------------------------------------------------
 * adaptq/strategy.h — IKVStrategy and its capability sub-interfaces
 *
 * IKVStrategy is a capability provider, not a God Object.
 * It is a single entry point (one registration, one object per head),
 * but each capability is a focused, independently evolvable sub-interface.
 *
 * Required capabilities:  compression(), eviction()
 * Optional capabilities:  quality(), replay_hooks()
 *
 * The runtime queries capabilities at call time via virtual dispatch.
 * Optional capabilities return nullptr by default; existing strategies
 * are unaffected when new optional capabilities are added to this file.
 *
 * All method arguments use ExecutionContext instead of individual parameters.
 * New runtime state fields are added to ExecutionContext, not to these
 * method signatures.
 * ----------------------------------------------------------------------- */

namespace adaptq {


/* ========================================================================
 * Capability: ICompression
 * Responsible for compress, decompress, and optional custom kernels.
 * ====================================================================== */
struct ICompression {
    virtual ~ICompression() = default;

    /**
     * Compress one K or V vector into a CompressResult.
     * Strategy chooses format, precision, and algorithm for this token.
     * The CompressResult is written into ctx.storage and the slot stored
     * by the runtime; the strategy must call ctx.storage->write().
     *
     * @param x      float[dim] input vector
     * @param dim    head dimension
     * @param is_key true = Key, false = Value
     * @param ctx    ExecutionContext (contains storage backend, cache state, etc.)
     * @return       CompressResult with slot filled in
     */
    virtual CompressResult compress(const float          *x,
                                    int                   dim,
                                    bool                  is_key,
                                    const ExecutionContext &ctx) = 0;

    /**
     * Decompress a CompressResult back into float[dim].
     * Used for the FP hybrid path and quality oracle sampling.
     */
    virtual void decompress(const CompressResult  &r,
                            int                    dim,
                            float                 *out) const = 0;

    /* --- Optional custom kernel overrides ---
     * If false (default), the runtime dispatches K-dot and V-accumulate
     * to the active IKernelBackend. Override to provide a strategy-specific
     * kernel (e.g., RVQ with a multi-stage LUT). */

    virtual bool  has_custom_kdot()  const { return false; }

    /**
     * Compute dot product between a rotated query and a compressed K.
     * Only called if has_custom_kdot() == true.
     *
     * @param q_rot   float[padded] — query after HAR rotation
     * @param k       compressed K CompressResult
     * @param padded  power-of-2 padded dimension
     */
    virtual float kdot(const float          *q_rot,
                       const CompressResult  &k,
                       int                   padded) const { return 0.f; }

    virtual bool has_custom_vaccum() const { return false; }

    /**
     * Accumulate weighted V contribution into acc.
     * Only called if has_custom_vaccum() == true.
     *
     * @param acc     float[padded] accumulator (in/out)
     * @param v       compressed V CompressResult
     * @param weight  softmax attention weight for this token
     * @param padded  power-of-2 padded dimension
     */
    virtual void vaccum(float                *acc,
                        const CompressResult  &v,
                        float                 weight,
                        int                   padded) const {}
};


/* ========================================================================
 * Capability: IEviction
 * Decides which tokens to evict and tracks importance.
 * ====================================================================== */

struct EvictionDecision {
    bool evict;
    int  evict_slot; /* meaningful only if evict == true; -1 otherwise */
};

struct IEviction {
    virtual ~IEviction() = default;

    /**
     * Called after each compress(). Returns which slot (if any) to evict.
     * The runtime frees the evicted slot in the storage backend.
     *
     * @param ctx   current ExecutionContext (cache_size, memory_used, etc.)
     * @return      EvictionDecision
     */
    virtual EvictionDecision on_append(const ExecutionContext &ctx) = 0;

    /**
     * Called after each compute(). Provides attention weights so the policy
     * can update importance estimates for future eviction decisions.
     *
     * @param fb    AttentionFeedback containing weights and slot indices
     * @param ctx   ExecutionContext for the head that just ran compute()
     */
    virtual void on_attention(const AttentionFeedback &fb,
                              const ExecutionContext   &ctx) = 0;
};


/* ========================================================================
 * Capability: IQuality (optional)
 * Reports compression quality estimates for the controller and profiler.
 * ====================================================================== */
struct IQuality {
    virtual ~IQuality() = default;

    /** Average bits per dimension across all currently cached tokens.
     *  For fixed-precision strategies, this is the configured bits value.
     *  For adaptive strategies, this is a weighted average.          */
    virtual float avg_bits_per_dim()  const = 0;

    /** Quality estimate for the last compute() call, in [0, 1].
     *  Uses internal signals: reconstruction error, entropy, etc.
     *  Returns -1.f if no compute() has been called yet.            */
    virtual float estimated_quality() const = 0;
};


/* ========================================================================
 * Capability: IReplayHooks (optional)
 * Serializes internal state for session snapshots.
 * ====================================================================== */
struct IReplayHooks {
    virtual ~IReplayHooks() = default;

    /**
     * Serialize strategy internal state (importance counters, EMA buffers,
     * heavy-hitter tables, etc.) into out.
     * Called by the replay engine when saving a SessionSnapshot.
     */
    virtual void serialize_state(std::ostream &out) const = 0;

    /**
     * Restore strategy internal state from in.
     * Called by the replay engine before replaying a branch.
     */
    virtual void deserialize_state(std::istream &in) = 0;
};


/* ========================================================================
 * IKVStrategy — Single entry point, capability provider
 * One instance per head per layer.
 * ====================================================================== */
class IKVStrategy {
public:
    virtual ~IKVStrategy() = default;

    /* ---- Lifecycle ---- */

    /** Called once at context creation. Strategy allocates internal state. */
    virtual void init(const HeadConfig &config) = 0;

    /** Called on context reset. Clears state, keeps config and capacity. */
    virtual void reset() = 0;

    /** Unique name used in benchmarks, comparison CLI, and session logs. */
    virtual const char *name() const = 0;

    /* ---- Required capabilities ---- */

    /** Returns the compression capability. Must not return nullptr. */
    virtual ICompression *compression() = 0;

    /** Returns the eviction capability. Must not return nullptr. */
    virtual IEviction    *eviction()    = 0;

    /* ---- Optional capabilities ----
     * Return nullptr if not implemented. The runtime checks before calling.
     * Adding new optional capabilities here does not break existing strategies.
     */

    /** Returns quality reporting capability, or nullptr. */
    virtual IQuality     *quality()      { return nullptr; }

    /** Returns replay serialization hooks, or nullptr. */
    virtual IReplayHooks *replay_hooks() { return nullptr; }

    /* Reserve slots for future capabilities without ABI breakage. */
};

} /* namespace adaptq */

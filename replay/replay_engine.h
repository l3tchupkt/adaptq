#pragma once
#include <string>
#include <vector>
#include "../include/adaptq/metrics.h"
#include "../include/adaptq/strategy.h"
#include "../runtime/runtime_context.h"
#include "session_snapshot.h"

/* -------------------------------------------------------------------------
 * replay/replay_engine.h — ReplayEngine
 *
 * Drives a RuntimeContext from a SessionSnapshot, with support for:
 *   - Full deterministic replay from token 0
 *   - Branching: warm-up to token N, then return a ready context
 *   - Cross-strategy replay: replay with a different IKVStrategy
 *
 * ReplayEngine does NOT own the RuntimeContext it drives. Callers create
 * and own the context; the engine only reads configuration from it.
 * ----------------------------------------------------------------------- */

namespace adaptq {

/* ---- ReplayReport — result of a full or cross-strategy replay ---------- */
struct ReplayReport {
    int    n_tokens_replayed    = 0;
    double wall_time_ms         = 0.0;
    bool   collect_metrics      = false;

    /* Per-token metrics (populated only if collect_metrics = true). */
    std::vector<ComputeMetrics> metrics;

    /* Per-strategy name tag. */
    std::string strategy_name;
};

/* ---- ReplayEngine ------------------------------------------------------- */
class ReplayEngine {
public:
    /**
     * Create a ReplayEngine bound to ctx.
     * ctx must outlive the ReplayEngine.
     *
     * @param collect_metrics  If true, all replay / branch calls collect
     *                         per-token ComputeMetrics into ReplayReport::metrics.
     */
    explicit ReplayEngine(bool collect_metrics = false);

    /* ---- Full replay --------------------------------------------------- */

    /**
     * Re-feed all tokens from snap into a fresh RuntimeContext, using the
     * strategy that was active when the snapshot was taken.
     *
     * Requires snap.has_token_log() == true.
     *
     * The returned ReplayReport contains timing and, if collect_metrics,
     * per-token ComputeMetrics.
     */
    ReplayReport replay(const SessionSnapshot  &snap,
                        RuntimeContext         &ctx) const;

    /* ---- Branch replay ------------------------------------------------- */

    /**
     * Warm-up replay to from_token, then restore strategy internal state
     * and return. The RuntimeContext is ready for continued inference from
     * token `from_token`.
     *
     * Warm-up replay re-feeds the first `from_token` (layer=0, head=0)
     * entries from snap.token_log() silently (no output comparison).
     * Strategy state is restored via IReplayHooks::deserialize_state() after
     * warm-up if strategy_state is present in the snapshot.
     *
     * @param snap         Snapshot to branch from.
     * @param ctx          RuntimeContext to restore into (must be reset first).
     * @param from_token   Token index to branch at [0, snap.n_tokens()).
     */
    void branch(const SessionSnapshot &snap,
                RuntimeContext        &ctx,
                int                    from_token) const;

    /* ---- Cross-strategy replay ----------------------------------------- */

    /**
     * Replay snap.token_log() using an alternative strategy factory.
     * The alternative strategy name appears in the returned ReplayReport.
     *
     * @param snap          Snapshot containing the token log.
     * @param strategy_name Name of the alternate strategy (must be in registry).
     * @param cfg           Config for the replay RuntimeContext.
     * @return              ReplayReport for the alternate strategy run.
     */
    ReplayReport replay_with(const SessionSnapshot     &snap,
                             const std::string         &strategy_name,
                             const RuntimeContextConfig &cfg) const;

private:
    bool collect_metrics_;

    /* Feed one token's K/V pair into ctx for all layers and heads.
     * The token log is organized as one entry per (layer, head) per token.
     * We accumulate entries and feed them when a full token round is complete. */
    void feed_token(RuntimeContext                         &ctx,
                    const std::vector<SnapshotTokenEntry>  &log,
                    int                                     token_idx,
                    int                                     n_layers,
                    int                                     n_heads,
                    ReplayReport                           *report) const;
};

} /* namespace adaptq */

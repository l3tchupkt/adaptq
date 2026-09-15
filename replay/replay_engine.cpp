#include "replay_engine.h"
#include "../runtime/runtime_context.h"
#include "../include/adaptq/strategy.h"
#include <cassert>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <vector>

/* -------------------------------------------------------------------------
 * replay/replay_engine.cpp — ReplayEngine implementation
 * ----------------------------------------------------------------------- */

namespace adaptq {

/* These symbols are defined in runtime_context.cpp (same link unit). */
extern StrategyFactory strategy_factory_by_name(const char *name);
extern IStorageBackend *make_contiguous();

static void validate_token_log(const SessionSnapshot &snap) {
    if (!snap.has_token_log())
        return;

    const int n_tokens = snap.n_tokens();
    const int n_layers = snap.n_layers();
    const int n_heads = snap.n_heads();
    const int dim = snap.dim();

    if (n_tokens < 0 || n_layers <= 0 || n_heads <= 0 || dim <= 0) {
        throw std::runtime_error(
            "ReplayEngine: invalid snapshot dimensions for token log");
    }

    const size_t expected_entries =
        static_cast<size_t>(n_tokens) * static_cast<size_t>(n_layers) *
        static_cast<size_t>(n_heads);
    const auto &log = snap.token_log();

    if (log.size() != expected_entries) {
        throw std::runtime_error(
            "ReplayEngine: incomplete token log (expected " +
            std::to_string(expected_entries) + " entries, found " +
            std::to_string(log.size()) + ")");
    }

    for (size_t i = 0; i < log.size(); ++i) {
        const SnapshotTokenEntry &entry = log[i];
        if (entry.layer < 0 || entry.layer >= n_layers ||
            entry.head < 0 || entry.head >= n_heads || entry.dim != dim ||
            entry.k_fp32.size() != static_cast<size_t>(dim) ||
            entry.v_fp32.size() != static_cast<size_t>(dim)) {
            throw std::runtime_error(
                "ReplayEngine: invalid token log entry at index " +
                std::to_string(i));
        }
    }
}

/* =========================================================================
 * Constructor
 * ========================================================================= */

ReplayEngine::ReplayEngine(bool collect_metrics)
    : collect_metrics_(collect_metrics) {}

/* =========================================================================
 * feed_token — private helper
 *
 * Token log layout:
 *   One SnapshotTokenEntry per (layer, head) combination per token.
 *   Entries are in the order they were appended: for token T, entries
 *   occupy indices [T * n_layers * n_heads  …  (T+1) * n_layers * n_heads).
 *   Within that range, order is (l=0,h=0), (l=0,h=1), …, (l=L-1, h=H-1).
 * ========================================================================= */

void ReplayEngine::feed_token(RuntimeContext                         &ctx,
                               const std::vector<SnapshotTokenEntry>  &log,
                               int                                     token_idx,
                               int                                     n_layers,
                               int                                     n_heads,
                               ReplayReport                           *report) const {
    int stride    = n_layers * n_heads;
    int base_idx  = token_idx * stride;

    static thread_local std::vector<float> out_buf;
    out_buf.resize(ctx.dim());

    for (int l = 0; l < n_layers; ++l) {
        for (int h = 0; h < n_heads; ++h) {
            int entry_idx = base_idx + l * n_heads + h;
            if (entry_idx >= (int)log.size()) {
                throw std::runtime_error(
                    "ReplayEngine: token log ended before token " +
                    std::to_string(token_idx) + " was fully replayed");
            }

            const SnapshotTokenEntry &e = log[entry_idx];
            ctx.append(l, h, e.k_fp32.data(), e.v_fp32.data());

            /* compute() with zero query — used only for side-effects
             * (eviction feedback, quality update) during warm-up.
             * The output is discarded in warm-up / branch mode. */
            if (collect_metrics_ && report) {
                static thread_local std::vector<float> q_zero;
                q_zero.assign(ctx.dim(), 0.f);
                ComputeMetrics m = ctx.compute(l, h, q_zero.data(), out_buf.data());
                report->metrics.push_back(m);
            }
        }
    }
}

/* =========================================================================
 * replay()
 * ========================================================================= */

ReplayReport ReplayEngine::replay(const SessionSnapshot &snap,
                                   RuntimeContext        &ctx) const {
    if (!snap.has_token_log())
        throw std::runtime_error("ReplayEngine::replay: snapshot has no token log. "
                                 "Capture with include_token_log=true.");

    validate_token_log(snap);

    ReplayReport report;
    report.collect_metrics = collect_metrics_;
    report.strategy_name   = ctx.get_strategy(0, 0)
                               ? ctx.get_strategy(0, 0)->name()
                               : "unknown";

    /* Reset context before replay. */
    ctx.reset();

    const auto &log = snap.token_log();
    int n_tok    = snap.n_tokens();
    int n_layers = snap.n_layers();
    int n_heads  = snap.n_heads();

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < n_tok; ++t) {
        feed_token(ctx, log, t, n_layers, n_heads, &report);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    report.n_tokens_replayed = n_tok;
    report.wall_time_ms =
        (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
        / 1000.0;

    return report;
}

/* =========================================================================
 * branch()
 * ========================================================================= */

void ReplayEngine::branch(const SessionSnapshot &snap,
                           RuntimeContext        &ctx,
                           int                    from_token) const {
    if (!snap.has_token_log())
        throw std::runtime_error("ReplayEngine::branch: snapshot has no token log.");

    validate_token_log(snap);

    if (from_token < 0 || from_token > snap.n_tokens())
        throw std::runtime_error("ReplayEngine::branch: from_token out of range.");

    int n_layers = snap.n_layers();
    int n_heads  = snap.n_heads();
    const auto &log = snap.token_log();

    /* Reset and warm-up replay silently to from_token. */
    ctx.reset();

    /* Temporarily disable collect_metrics for warm-up. */
    for (int t = 0; t < from_token; ++t) {
        feed_token(ctx, log, t, n_layers, n_heads, nullptr);
    }

    /* Restore strategy state if available. */
    if (snap.has_strategy_state()) {
        const auto &heads = snap.heads();
        for (int l = 0; l < n_layers; ++l) {
            for (int h = 0; h < n_heads; ++h) {
                int hi = l * n_heads + h;
                if (hi >= (int)heads.size()) continue;

                const HeadSnapshot &hs = heads[hi];
                if (hs.strategy_state.empty()) continue;

                IKVStrategy *strat = ctx.get_strategy(l, h);
                if (!strat) continue;
                IReplayHooks *rh = strat->replay_hooks();
                if (!rh) continue;

                std::string blob(hs.strategy_state.begin(),
                                 hs.strategy_state.end());
                std::istringstream in(blob);
                rh->deserialize_state(in);
            }
        }
    }
}

/* =========================================================================
 * replay_with() — cross-strategy replay
 * ========================================================================= */

ReplayReport ReplayEngine::replay_with(const SessionSnapshot      &snap,
                                        const std::string          &strategy_name,
                                        const RuntimeContextConfig &cfg) const {
    if (!snap.has_token_log())
        throw std::runtime_error("ReplayEngine::replay_with: snapshot has no token log.");

    validate_token_log(snap);

    StrategyFactory sfn = strategy_factory_by_name(strategy_name.c_str());
    if (!sfn)
        throw std::runtime_error("ReplayEngine::replay_with: unknown strategy '"
                                 + strategy_name + "'");

    /* Build a fresh RuntimeContext with the alternate strategy. */
    RuntimeContext alt_ctx;
    alt_ctx.init(cfg, sfn, make_contiguous);


    ReplayReport report;
    report.collect_metrics = collect_metrics_;
    report.strategy_name   = strategy_name;

    const auto &log = snap.token_log();
    int n_tok    = snap.n_tokens();
    int n_layers = snap.n_layers();
    int n_heads  = snap.n_heads();

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < n_tok; ++t) {
        feed_token(alt_ctx, log, t, n_layers, n_heads, &report);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    report.n_tokens_replayed = n_tok;
    report.wall_time_ms =
        (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
        / 1000.0;

    return report;
}

} /* namespace adaptq */

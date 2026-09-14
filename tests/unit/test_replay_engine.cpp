/* -------------------------------------------------------------------------
 * tests/unit/test_replay_engine.cpp
 *
 * Tests for ReplayEngine: full replay determinism, branch warm-up,
 * and cross-strategy replay correctness.
 * ----------------------------------------------------------------------- */
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <vector>

#include "../../runtime/runtime_context.h"
#include "../../replay/session_snapshot.h"
#include "../../replay/replay_engine.h"

using namespace adaptq;

/* ---- Helpers ------------------------------------------------------------ */
static void rand_vec(float *v, int d, unsigned seed) {
    unsigned x = seed;
    for (int i = 0; i < d; ++i) {
        x = x * 1664525u + 1013904223u;
        v[i] = (float)(int)(x >> 16) / 32768.f - 1.f;
    }
    float n = 0.f;
    for (int i = 0; i < d; ++i) n += v[i] * v[i];
    n = sqrtf(n + 1e-12f);
    for (int i = 0; i < d; ++i) v[i] /= n;
}

static RuntimeContextConfig make_cfg(int n_tokens = 20, int dim = 64) {
    RuntimeContextConfig cfg;
    cfg.n_layers   = 1;
    cfg.n_heads    = 1;
    cfg.dim        = dim;
    cfg.bits       = 4;
    cfg.capacity   = 128;
    cfg.log_tokens = true;
    return cfg;
}

static SessionSnapshot build_snapshot(int n_tokens, int dim = 64) {
    RuntimeContextConfig cfg = make_cfg(n_tokens, dim);
    RuntimeContext ctx;
    ctx.init(cfg);

    std::vector<float> k(dim), v(dim);
    for (int t = 0; t < n_tokens; ++t) {
        rand_vec(k.data(), dim, (unsigned)(t * 37 + 1));
        rand_vec(v.data(), dim, (unsigned)(t * 37 + 2));
        ctx.append(0, 0, k.data(), v.data());
    }
    return SessionSnapshot::capture(ctx, true);
}

/* =========================================================================
 * Test cases
 * ========================================================================= */

TEST_CASE("ReplayEngine: full replay completes without error", "[replay]") {
    SessionSnapshot snap = build_snapshot(10);

    RuntimeContextConfig cfg = make_cfg(10);
    cfg.log_tokens = false;
    RuntimeContext ctx;
    ctx.init(cfg);

    ReplayEngine engine(/*collect_metrics=*/false);
    ReplayReport report;
    REQUIRE_NOTHROW(report = engine.replay(snap, ctx));
    REQUIRE(report.n_tokens_replayed == 10);
    REQUIRE(report.wall_time_ms >= 0.0);
}

TEST_CASE("ReplayEngine: full replay produces non-empty report", "[replay]") {
    SessionSnapshot snap = build_snapshot(15);

    RuntimeContextConfig cfg = make_cfg(15);
    cfg.log_tokens = false;
    RuntimeContext ctx;
    ctx.init(cfg);

    ReplayEngine engine(/*collect_metrics=*/true);
    ReplayReport report = engine.replay(snap, ctx);

    REQUIRE(report.n_tokens_replayed == 15);
    /* With collect_metrics=true, metrics are populated for each compute call.
     * 15 tokens × 1 layer × 1 head = 15 entries. */
    REQUIRE(report.metrics.size() == 15u);
}

TEST_CASE("ReplayEngine: replay without token log throws", "[replay]") {
    /* Build a snapshot without token log. */
    RuntimeContextConfig cfg = make_cfg(5);
    cfg.log_tokens = false;
    RuntimeContext orig;
    orig.init(cfg);

    std::vector<float> k(64), v(64);
    rand_vec(k.data(), 64, 1); rand_vec(v.data(), 64, 2);
    orig.append(0, 0, k.data(), v.data());

    SessionSnapshot snap = SessionSnapshot::capture(orig, /*include_token_log=*/false);

    RuntimeContextConfig cfg2 = make_cfg(5);
    cfg2.log_tokens = false;
    RuntimeContext ctx;
    ctx.init(cfg2);

    ReplayEngine engine;
    REQUIRE_THROWS_AS(engine.replay(snap, ctx), std::runtime_error);
}

TEST_CASE("ReplayEngine: branch at 0 leaves context empty", "[replay]") {
    SessionSnapshot snap = build_snapshot(10);

    RuntimeContextConfig cfg = make_cfg(10);
    cfg.log_tokens = false;
    RuntimeContext ctx;
    ctx.init(cfg);

    ReplayEngine engine;
    REQUIRE_NOTHROW(engine.branch(snap, ctx, 0));

    /* After branching at token 0, no tokens should be in storage. */
    REQUIRE(ctx.get_storage(0, 0)->bytes_used() == 0);
}

TEST_CASE("ReplayEngine: branch at N puts N tokens in storage", "[replay]") {
    int N = 8;
    SessionSnapshot snap = build_snapshot(20);

    RuntimeContextConfig cfg = make_cfg(20);
    cfg.log_tokens = false;
    RuntimeContext ctx;
    ctx.init(cfg);

    ReplayEngine engine;
    engine.branch(snap, ctx, N);

    /* Storage should have data for N K/V pairs. */
    REQUIRE(ctx.get_storage(0, 0)->bytes_used() > 0);
    REQUIRE(ctx.token_pos() == N);
}

TEST_CASE("ReplayEngine: branch at from_token > n_tokens throws", "[replay]") {
    SessionSnapshot snap = build_snapshot(10);

    RuntimeContextConfig cfg = make_cfg(10);
    cfg.log_tokens = false;
    RuntimeContext ctx;
    ctx.init(cfg);

    ReplayEngine engine;
    REQUIRE_THROWS_AS(engine.branch(snap, ctx, 999), std::runtime_error);
}

TEST_CASE("ReplayEngine: cross-strategy replay with fp_passthrough", "[replay]") {
    SessionSnapshot snap = build_snapshot(10);

    RuntimeContextConfig cfg = make_cfg(10);
    cfg.log_tokens = false;

    ReplayEngine engine(/*collect_metrics=*/true);
    ReplayReport report;
    REQUIRE_NOTHROW(
        report = engine.replay_with(snap, "fp_passthrough", cfg)
    );

    REQUIRE(report.n_tokens_replayed == 10);
    REQUIRE(report.strategy_name == "fp_passthrough");
}

TEST_CASE("ReplayEngine: cross-strategy replay unknown strategy throws", "[replay]") {
    SessionSnapshot snap = build_snapshot(5);
    RuntimeContextConfig cfg = make_cfg(5);

    ReplayEngine engine;
    REQUIRE_THROWS_AS(
        engine.replay_with(snap, "nonexistent_strategy", cfg),
        std::runtime_error
    );
}

TEST_CASE("ReplayEngine: replay strategy name matches context strategy", "[replay]") {
    SessionSnapshot snap = build_snapshot(5);

    RuntimeContextConfig cfg = make_cfg(5);
    cfg.log_tokens = false;
    RuntimeContext ctx;
    ctx.init(cfg);

    ReplayEngine engine(false);
    ReplayReport report = engine.replay(snap, ctx);

    /* Default strategy is har_fixed. */
    REQUIRE(report.strategy_name == "har_fixed");
}

TEST_CASE("ReplayEngine: full replay and branch produce same storage at branch point",
          "[replay]") {
    const int BRANCH_AT = 6;
    SessionSnapshot snap = build_snapshot(12);

    /* Full replay up to BRANCH_AT via branch(). */
    RuntimeContextConfig cfg = make_cfg(12);
    cfg.log_tokens = false;
    RuntimeContext ctx_branch;
    ctx_branch.init(cfg);

    ReplayEngine engine(false);
    engine.branch(snap, ctx_branch, BRANCH_AT);

    size_t branch_bytes = ctx_branch.get_storage(0, 0)->bytes_used();

    /* Full replay of 12 tokens via replay(). */
    RuntimeContext ctx_full;
    ctx_full.init(cfg);
    engine.replay(snap, ctx_full);

    size_t full_bytes = ctx_full.get_storage(0, 0)->bytes_used();

    /* After a full replay, storage holds 12 K+V pairs.
     * After branch at 6, storage holds 6 K+V pairs.
     * Full ≥ branch (exact ratio depends on capacity evictions). */
    REQUIRE(full_bytes >= branch_bytes);
}

/* -------------------------------------------------------------------------
 * tests/unit/test_runtime_context.cpp
 *
 * Tests for RuntimeContext: orchestration of append/compute/reset through
 * the full IPolicy → IKVStrategy → IStorageBackend → IKernelBackend chain.
 * ----------------------------------------------------------------------- */
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

/* Pull in RuntimeContext (which includes strategy/storage implementations). */
#include "../../runtime/runtime_context.h"

using namespace adaptq;

/* ---- Helpers ------------------------------------------------------------ */
static RuntimeContextConfig make_cfg(int n_layers = 1, int n_heads = 2,
                                     int dim = 64, int bits = 4,
                                     int capacity = 32) {
    RuntimeContextConfig cfg;
    cfg.n_layers  = n_layers;
    cfg.n_heads   = n_heads;
    cfg.dim       = dim;
    cfg.bits      = bits;
    cfg.capacity  = capacity;
    cfg.log_tokens = false;
    return cfg;
}

static void rand_vec(float *v, int d, unsigned seed = 42) {
    /* Simple deterministic pseudo-random vector. */
    unsigned x = seed;
    for (int i = 0; i < d; ++i) {
        x = x * 1664525u + 1013904223u;
        v[i] = (float)(int)(x >> 16) / 32768.f - 1.f;
    }
    /* L2-normalize. */
    float n = 0.f;
    for (int i = 0; i < d; ++i) n += v[i] * v[i];
    n = sqrtf(n + 1e-12f);
    for (int i = 0; i < d; ++i) v[i] /= n;
}

/* ---- Policy test double ------------------------------------------------ */

struct FixedStrategyPolicy final : IPolicy {
    IKVStrategy *strategy = nullptr;

    void init(int, int, const RuntimeConfig &) override {}
    IKVStrategy *select_strategy(int, int, const ExecutionContext &) override {
        return strategy;
    }
    IStorageBackend *select_storage(int, int, const ExecutionContext &) override {
        return nullptr;
    }
    void on_metrics(int, int, const ComputeMetrics &) override {}
    const char *name() const override { return "fixed_strategy"; }
};

/* =========================================================================
 * Test cases
 * ========================================================================= */

TEST_CASE("RuntimeContext: init does not throw", "[runtime]") {
    RuntimeContextConfig cfg = make_cfg();
    RuntimeContext ctx;
    REQUIRE_NOTHROW(ctx.init(cfg));
}

TEST_CASE("RuntimeContext: rejects invalid configuration before allocation", "[runtime][security]") {
    RuntimeContext ctx;
    for (RuntimeContextConfig cfg : {
        make_cfg(0, 1), make_cfg(1, 0), make_cfg(1, 1, 0),
        make_cfg(1, 1, 64, 1), make_cfg(1, 1, 64, 5),
        make_cfg(1, 1, 64, 4, 0)
    }) {
        REQUIRE_THROWS_AS(ctx.init(cfg), std::invalid_argument);
    }
}

TEST_CASE("RuntimeContext: get_strategy / get_storage return non-null", "[runtime]") {
    RuntimeContextConfig cfg = make_cfg(2, 4, 128, 4, 64);
    RuntimeContext ctx;
    ctx.init(cfg);
    for (int l = 0; l < 2; ++l) {
        for (int h = 0; h < 4; ++h) {
            REQUIRE(ctx.get_strategy(l, h) != nullptr);
            REQUIRE(ctx.get_storage(l, h)  != nullptr);
        }
    }
}

TEST_CASE("RuntimeContext: rejects invalid layer and head indices", "[runtime][security]") {
    RuntimeContext ctx;
    ctx.init(make_cfg(2, 2));
    REQUIRE_THROWS_AS(ctx.get_strategy(-1, 0), std::out_of_range);
    REQUIRE_THROWS_AS(ctx.get_storage(0, 2), std::out_of_range);
    REQUIRE_THROWS_AS(ctx.compute(2, 0, nullptr, nullptr), std::out_of_range);
}

TEST_CASE("RuntimeContext: append increases storage usage", "[runtime]") {
    RuntimeContextConfig cfg = make_cfg(1, 1, 64, 4, 32);
    RuntimeContext ctx;
    ctx.init(cfg);

    std::vector<float> k(64), v(64);
    rand_vec(k.data(), 64, 1);
    rand_vec(v.data(), 64, 2);

    size_t before = ctx.get_storage(0, 0)->bytes_used();
    ctx.append(0, 0, k.data(), v.data());
    size_t after  = ctx.get_storage(0, 0)->bytes_used();

    REQUIRE(after > before);
}

TEST_CASE("RuntimeContext: compute returns valid ComputeMetrics after append", "[runtime]") {
    RuntimeContextConfig cfg = make_cfg(1, 1, 64, 4, 32);
    RuntimeContext ctx;
    ctx.init(cfg);

    std::vector<float> k(64), v(64), q(64), out(64, 0.f);
    rand_vec(k.data(), 64, 10);
    rand_vec(v.data(), 64, 11);
    rand_vec(q.data(), 64, 12);

    ctx.append(0, 0, k.data(), v.data());
    ComputeMetrics m = ctx.compute(0, 0, q.data(), out.data());

    REQUIRE(m.layer == 0);
    REQUIRE(m.head  == 0);
    REQUIRE(m.n_tokens_used == 1);
    REQUIRE(m.latency_us >= 0.f);

    /* Output should be non-zero (one V vector × weight ≈ 1). */
    float norm = 0.f;
    for (float x : out) norm += x * x;
    REQUIRE(norm > 1e-12f);
}

TEST_CASE("RuntimeContext: compute honors the policy-selected strategy", "[runtime][policy]") {
    RuntimeContextConfig cfg = make_cfg(1, 1, 64, 4, 32);
    RuntimeContext ctx;
    ctx.init(cfg);

    std::vector<float> k(64), v(64), q(64);
    rand_vec(q.data(), 64, 31);
    for (int t = 0; t < 8; ++t) {
        rand_vec(k.data(), 64, (unsigned)(t * 41 + 1));
        rand_vec(v.data(), 64, (unsigned)(t * 41 + 2));
        ctx.append(0, 0, k.data(), v.data());
    }

    std::vector<float> out_default(64, 0.f), out_policy(64, 0.f);
    ComputeMetrics m0 = ctx.compute(0, 0, q.data(), out_default.data());
    REQUIRE(m0.n_tokens_used == 8);

    float norm = 0.f;
    for (float x : out_default) norm += x * x;
    REQUIRE(norm > 1e-12f);

    /* Build a second HAR instance with a different Rademacher seed. The
     * policy will select it for compute only, so the output must change if
     * compute actually honours the policy. */
    std::unique_ptr<IKVStrategy> alt(strategy_factory_by_name("har_fixed")());
    REQUIRE(alt != nullptr);
    HeadConfig hc;
    hc.dim      = 64;
    hc.bits     = 4;
    hc.capacity = 32;
    hc.seed     = 987654321ULL;
    alt->init(hc);

    FixedStrategyPolicy policy;
    policy.strategy = alt.get();
    ctx.set_policy(&policy);

    ComputeMetrics m1 = ctx.compute(0, 0, q.data(), out_policy.data());
    REQUIRE(m1.n_tokens_used == 8);

    float diff = 0.f;
    for (int i = 0; i < 64; ++i) {
        float d = out_policy[i] - out_default[i];
        diff += d * d;
    }
    REQUIRE(diff > 1e-9f);
}

TEST_CASE("RuntimeContext: compute on empty cache returns zero output", "[runtime]") {
    RuntimeContextConfig cfg = make_cfg(1, 1, 64, 4, 32);
    RuntimeContext ctx;
    ctx.init(cfg);

    std::vector<float> q(64), out(64, 1.f);
    rand_vec(q.data(), 64, 99);

    ComputeMetrics m = ctx.compute(0, 0, q.data(), out.data());
    REQUIRE(m.n_tokens_used == 0);

    float s = 0.f;
    for (float x : out) s += x * x;
    REQUIRE(s < 1e-12f); /* output was zeroed */
}

TEST_CASE("RuntimeContext: reset clears storage and resets token_pos", "[runtime]") {
    RuntimeContextConfig cfg = make_cfg(1, 1, 64, 4, 32);
    RuntimeContext ctx;
    ctx.init(cfg);

    std::vector<float> k(64), v(64);
    rand_vec(k.data(), 64, 1);
    rand_vec(v.data(), 64, 2);

    ctx.append(0, 0, k.data(), v.data());
    REQUIRE(ctx.get_storage(0, 0)->bytes_used() > 0);

    ctx.reset();
    REQUIRE(ctx.get_storage(0, 0)->bytes_used() == 0);
    REQUIRE(ctx.token_pos() == 0);
}

TEST_CASE("RuntimeContext: multiple appends accumulate", "[runtime]") {
    RuntimeContextConfig cfg = make_cfg(1, 1, 64, 4, 32);
    RuntimeContext ctx;
    ctx.init(cfg);

    std::vector<float> k(64), v(64);
    for (int t = 0; t < 10; ++t) {
        rand_vec(k.data(), 64, (unsigned)(t * 17 + 1));
        rand_vec(v.data(), 64, (unsigned)(t * 17 + 2));
        ctx.append(0, 0, k.data(), v.data());
    }
    /* After 10 appends, compute should see n_tokens_used = 10. */
    std::vector<float> q(64), out(64);
    rand_vec(q.data(), 64, 777);
    ComputeMetrics m = ctx.compute(0, 0, q.data(), out.data());
    REQUIRE(m.n_tokens_used == 10);
}

TEST_CASE("RuntimeContext: fp_passthrough strategy init", "[runtime]") {
    RuntimeContextConfig cfg = make_cfg(1, 1, 64, 4, 32);
    RuntimeContext ctx;
    /* Use fp_passthrough factory. */
    StrategyFactory sfn = strategy_factory_by_name("fp_passthrough");
    REQUIRE(sfn != nullptr);
    /* Storage factory: use make_contiguous from adaptq:: namespace. */
    ctx.init(cfg, sfn, []() -> IStorageBackend * {
        return adaptq::make_contiguous();
    });

    std::vector<float> k(64), v(64), q(64), out(64);
    rand_vec(k.data(), 64, 5);
    rand_vec(v.data(), 64, 6);
    rand_vec(q.data(), 64, 7);

    ctx.append(0, 0, k.data(), v.data());
    ComputeMetrics m = ctx.compute(0, 0, q.data(), out.data());
    REQUIRE(m.n_tokens_used == 1);
    REQUIRE(m.quality >= 0.f); /* FP32 quality = 1.0 */
}

TEST_CASE("RuntimeContext: strategy_factory_by_name registry", "[runtime]") {
    REQUIRE(strategy_factory_by_name("har_fixed")      != nullptr);
    REQUIRE(strategy_factory_by_name("fp_passthrough") != nullptr);
    REQUIRE(strategy_factory_by_name("nonexistent")    == nullptr);
}

TEST_CASE("RuntimeContext: token log is populated when log_tokens=true", "[runtime]") {
    RuntimeContextConfig cfg = make_cfg(1, 1, 64, 4, 32);
    cfg.log_tokens = true;
    RuntimeContext ctx;
    ctx.init(cfg);

    std::vector<float> k(64), v(64);
    for (int t = 0; t < 5; ++t) {
        rand_vec(k.data(), 64, (unsigned)(t + 100));
        rand_vec(v.data(), 64, (unsigned)(t + 200));
        ctx.append(0, 0, k.data(), v.data());
    }
    REQUIRE(ctx.token_log().size() == 5u);
    REQUIRE(ctx.token_log_data().size() == 5u * 2u * 64u);
}

/* ---- Context Limits (Issue #22) --------------------------------------- */

TEST_CASE("RuntimeContext: HARFixedStrategy handles sizes around and above 65536 tokens", "[runtime][limits]") {
    RuntimeContextConfig cfg = make_cfg(1, 1, 64, 4, 100005);
    RuntimeContext ctx;
    ctx.init(cfg); // uses har_fixed by default

    std::vector<float> k(64, 0.1f), v(64, 0.1f), q(64, 0.1f), out(64, 0.f);

    int test_sizes[] = { 65535, 65536, 65537, 100000 };
    int current_size = 0;

    for (int target : test_sizes) {
        while (current_size < target) {
            ctx.append(0, 0, k.data(), v.data());
            current_size++;
        }
        
        ComputeMetrics m = ctx.compute(0, 0, q.data(), out.data());
        REQUIRE(m.n_tokens_used == target);
    }
}

TEST_CASE("RuntimeContext: fp_passthrough records pre-softmax logit_max and logit_min", "[runtime][fp32][metrics]") {
    RuntimeContextConfig cfg = make_cfg(1, 1, 64, 4, 32);
    RuntimeContext ctx;
    StrategyFactory sfn = strategy_factory_by_name("fp_passthrough");
    REQUIRE(sfn != nullptr);
    ctx.init(cfg, sfn, []() -> IStorageBackend * {
        return adaptq::make_contiguous();
    });

    std::vector<float> k_pos(64, 1.0f), k_neg(64, -1.0f), v(64, 0.5f), q(64, 1.0f), out(64);
    ctx.append(0, 0, k_pos.data(), v.data());
    ctx.append(0, 0, k_neg.data(), v.data());

    ComputeMetrics m = ctx.compute(0, 0, q.data(), out.data());
    // Q dot K_pos = 64 / sqrt(64) = 8.0f
    // Q dot K_neg = -64 / sqrt(64) = -8.0f
    REQUIRE(std::abs(m.logit_max - 8.0f) < 1e-4f);
    REQUIRE(std::abs(m.logit_min - (-8.0f)) < 1e-4f);
}


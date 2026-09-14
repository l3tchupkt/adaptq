#pragma once
/* -------------------------------------------------------------------------
 * tests/conformance/strategy_conformance.h
 *
 * Reusable conformance harness for IKVStrategy implementations.
 * Include this header and instantiate run_conformance<YourStrategy>() in
 * a Catch2 TEST_CASE to run the full IKVStrategy contract test suite.
 *
 * Usage:
 *   #include "tests/conformance/strategy_conformance.h"
 *   #include "strategies/my_strategy.h"    // class definition
 *
 *   TEST_CASE("MyStrategy conformance", "[conformance]") {
 *       run_conformance<adaptq::MyStrategy>("MyStrategy");
 *   }
 * ----------------------------------------------------------------------- */

#include <catch2/catch_test_macros.hpp>
#include "../../include/adaptq/strategy.h"
#include "../../include/adaptq/context.h"
#include "../../include/adaptq/storage.h"
#include "../../include/adaptq/config.h"
#include "../../storage/contiguous_slab.h"   /* inline header */
#include <cstring>
#include <sstream>
#include <string>

namespace adaptq {

/* ---- Test helpers ------------------------------------------------------- */

inline HeadConfig make_cfg(int dim = 128, int bits = 4, int capacity = 256) {
    HeadConfig cfg;
    cfg.dim      = dim;
    cfg.bits     = bits;
    cfg.capacity = capacity;
    cfg.seed     = 0xDEADBEEFULL;
    cfg.v_mass   = 0.f;
    return cfg;
}

inline ExecutionContext make_ctx(ContiguousSlabStorage *st,
                                 int dim = 128, int cap = 256) {
    ExecutionContext ctx{};
    ctx.layer             = 0;
    ctx.head              = 0;
    ctx.token_pos         = 0;
    ctx.cache_size        = 0;
    ctx.cache_capacity    = cap;
    ctx.memory_budget_mb  = 512.f;
    ctx.storage           = st;
    ctx.quality_floor     = 0.f;
    return ctx;
}

/* ---- Generic conformance runner ---------------------------------------- */

template <typename StratT>
void run_conformance(const char *strategy_name) {

    SECTION(std::string(strategy_name) + ": lifecycle") {
        ContiguousSlabStorage st;
        st.init(256, 128 * (int)sizeof(float));  /* dim * sizeof(float) for FP32 passthrough */
        StratT s;
        s.init(make_cfg());
        REQUIRE_NOTHROW(s.reset());
    }

    SECTION(std::string(strategy_name) + ": required capabilities non-null") {
        ContiguousSlabStorage st;
        st.init(256, 128 * (int)sizeof(float));
        StratT s;
        s.init(make_cfg());
        REQUIRE(s.compression() != nullptr);
        REQUIRE(s.eviction()    != nullptr);
    }

    SECTION(std::string(strategy_name) + ": name() non-null and non-empty") {
        StratT s;
        s.init(make_cfg());
        REQUIRE(s.name() != nullptr);
        REQUIRE(std::string(s.name()).size() > 0);
    }

    SECTION(std::string(strategy_name) + ": compress() writes to storage") {
        ContiguousSlabStorage st;
        st.init(256, 128 * (int)sizeof(float));  /* large enough for FP32 */
        StratT s;
        s.init(make_cfg());
        ExecutionContext ctx = make_ctx(&st);

        float key[128];
        for (int i = 0; i < 128; ++i) key[i] = (float)i / 128.f;

        CompressResult r = s.compression()->compress(key, 128, true, ctx);
        REQUIRE(r.data  != nullptr);
        REQUIRE(r.slot  != ADAPTQ_INVALID_SLOT);
        REQUIRE(st.bytes_used() > 0);
    }

    SECTION(std::string(strategy_name) + ": decompress() produces non-zero output") {
        ContiguousSlabStorage st;
        st.init(256, 128 * (int)sizeof(float));
        StratT s;
        s.init(make_cfg());
        ExecutionContext ctx = make_ctx(&st);

        float key[128], out[128] = {};
        for (int i = 0; i < 128; ++i) key[i] = (float)(i + 1) / 128.f;

        CompressResult r = s.compression()->compress(key, 128, true, ctx);
        s.compression()->decompress(r, 128, out);

        double norm = 0.0;
        for (int i = 0; i < 128; ++i) norm += (double)out[i] * out[i];
        REQUIRE(norm > 1e-12);
    }

    SECTION(std::string(strategy_name) + ": on_append() at capacity evicts") {
        ContiguousSlabStorage st;
        st.init(4, 128 * (int)sizeof(float));
        StratT s;
        s.init(make_cfg(128, 4, 4));
        ExecutionContext ctx = make_ctx(&st, 128, 4);

        for (int i = 0; i < 4; ++i) {
            ctx.cache_size = i;
            EvictionDecision d = s.eviction()->on_append(ctx);
            REQUIRE(d.evict == false);
        }
        ctx.cache_size = 4;
        EvictionDecision d = s.eviction()->on_append(ctx);
        REQUIRE(d.evict == true);
    }

    SECTION(std::string(strategy_name) + ": on_attention() does not crash") {
        ContiguousSlabStorage st;
        st.init(256, 128 * (int)sizeof(float));
        StratT s;
        s.init(make_cfg());
        ExecutionContext ctx = make_ctx(&st);

        float weights[4] = {0.4f, 0.3f, 0.2f, 0.1f};
        int   att_slots[4] = {0, 1, 2, 3};
        AttentionFeedback fb{weights, att_slots, 4, 100.f};
        REQUIRE_NOTHROW(s.eviction()->on_attention(fb, ctx));
    }

    SECTION(std::string(strategy_name) + ": optional IQuality sub-contract") {
        StratT s;
        s.init(make_cfg());
        IQuality *q = s.quality();
        if (q) {
            float bits = q->avg_bits_per_dim();
            REQUIRE(bits >= 2.f);
            REQUIRE(bits <= 32.f);
            float qual = q->estimated_quality();
            REQUIRE((qual == -1.f || (qual >= 0.f && qual <= 1.f)));
        }
    }

    SECTION(std::string(strategy_name) + ": optional IReplayHooks sub-contract") {
        StratT s;
        s.init(make_cfg());
        IReplayHooks *rh = s.replay_hooks();
        if (rh) {
            std::ostringstream out;
            REQUIRE_NOTHROW(rh->serialize_state(out));
            std::istringstream in(out.str());
            REQUIRE_NOTHROW(rh->deserialize_state(in));
        }
    }
}

} /* namespace adaptq */

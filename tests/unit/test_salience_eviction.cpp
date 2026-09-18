#include <catch2/catch_test_macros.hpp>
#include "../../cache/salience_eviction.h"
#include <vector>
#include <unordered_map>
#include <cmath>

using namespace adaptq::cache;

TEST_CASE("SalienceEvictionPolicy: initialization validation", "[cache][salience]") {
    SECTION("Valid configuration") {
        SalienceEvictionConfig cfg;
        cfg.max_cache_capacity = 128;
        cfg.sink_token_count = 4;
        cfg.recent_window_size = 32;
        cfg.decay_factor = 0.9;
        REQUIRE_NOTHROW(SalienceEvictionPolicy(cfg));
    }

    SECTION("Invalid capacity vs sink+window") {
        SalienceEvictionConfig cfg;
        cfg.max_cache_capacity = 30;
        cfg.sink_token_count = 10;
        cfg.recent_window_size = 20; // 10 + 20 >= 30
        REQUIRE_THROWS_AS(SalienceEvictionPolicy(cfg), std::invalid_argument);
    }

    SECTION("Invalid decay factor") {
        SalienceEvictionConfig cfg;
        cfg.decay_factor = 0.0;
        REQUIRE_THROWS_AS(SalienceEvictionPolicy(cfg), std::invalid_argument);

        cfg.decay_factor = 1.1;
        REQUIRE_THROWS_AS(SalienceEvictionPolicy(cfg), std::invalid_argument);
    }
}

TEST_CASE("SalienceEvictionPolicy: insertion and token tagging", "[cache][salience]") {
    SalienceEvictionConfig cfg;
    cfg.max_cache_capacity = 20;
    cfg.sink_token_count = 3;
    cfg.recent_window_size = 4;
    cfg.decay_factor = 0.95;
    SalienceEvictionPolicy policy(cfg);

    for (size_t i = 0; i < 10; ++i) {
        policy.record_insertion(1000 + i, i);
    }

    REQUIRE(policy.size() == 10);
    REQUIRE(policy.contains(1000));
    REQUIRE(policy.contains(1009));
    REQUIRE_FALSE(policy.contains(9999));

    // Check sink tags
    REQUIRE(policy.get_metadata(1000).is_sink == true);
    REQUIRE(policy.get_metadata(1001).is_sink == true);
    REQUIRE(policy.get_metadata(1002).is_sink == true);
    REQUIRE(policy.get_metadata(1003).is_sink == false);

    // Check recent tags (last 4: 1006, 1007, 1008, 1009)
    REQUIRE(policy.get_metadata(1006).is_recent == true);
    REQUIRE(policy.get_metadata(1007).is_recent == true);
    REQUIRE(policy.get_metadata(1008).is_recent == true);
    REQUIRE(policy.get_metadata(1009).is_recent == true);
    REQUIRE(policy.get_metadata(1005).is_recent == false);
}

TEST_CASE("SalienceEvictionPolicy: salience score updates and decay", "[cache][salience]") {
    SalienceEvictionConfig cfg;
    cfg.max_cache_capacity = 20;
    cfg.sink_token_count = 2;
    cfg.recent_window_size = 3;
    cfg.decay_factor = 0.5; // 50% decay
    SalienceEvictionPolicy policy(cfg);

    policy.record_insertion(10, 0); // Sink
    policy.record_insertion(11, 1); // Sink
    policy.record_insertion(12, 2); // Middle candidate
    policy.record_insertion(13, 3); // Middle candidate
    policy.record_insertion(14, 4); // Recent
    policy.record_insertion(15, 5); // Recent

    // Boost salience of token 12 significantly
    std::unordered_map<uint64_t, double> weights;
    weights[12] = 10.0;
    weights[13] = 0.1;
    policy.update_salience_scores(weights);

    const auto& meta12 = policy.get_metadata(12);
    const auto& meta13 = policy.get_metadata(13);

    // Token 12 prior was 1.0 -> decayed to 0.5 + 10.0 = 10.5
    REQUIRE(meta12.cumulative_salience > 10.0);
    // Token 13 prior was 1.0 -> decayed to 0.5 + 0.1 = 0.6
    REQUIRE(meta13.cumulative_salience < 1.0);
    REQUIRE(meta12.access_count == 2);
}

TEST_CASE("SalienceEvictionPolicy: eviction candidate selection protects sinks and recents", "[cache][salience]") {
    SalienceEvictionConfig cfg;
    cfg.max_cache_capacity = 10;
    cfg.sink_token_count = 2; // Tokens 0, 1 protected
    cfg.recent_window_size = 3; // Tokens 7, 8, 9 protected
    cfg.decay_factor = 0.9;
    SalienceEvictionPolicy policy(cfg);

    for (size_t i = 0; i < 10; ++i) {
        policy.record_insertion(i, i);
    }

    // Assign heavy-hitter status to token 3 and 5
    std::unordered_map<uint64_t, double> weights;
    weights[3] = 50.0;
    weights[5] = 40.0;
    weights[2] = 0.01;
    weights[4] = 0.05;
    weights[6] = 0.02;
    policy.update_salience_scores(weights);

    // Eligible candidates for eviction are {2, 4, 6} (since 0, 1 are sink, 7, 8, 9 are recent, 3 and 5 are high salience)
    auto candidates = policy.select_eviction_candidates(2);
    REQUIRE(candidates.size() == 2);

    // Token 2 and 6 have lower salience than token 4
    REQUIRE((candidates[0] == 2 || candidates[0] == 6));
    REQUIRE((candidates[1] == 2 || candidates[1] == 6 || candidates[1] == 4));

    // Sink tokens (0, 1) and recent tokens (7, 8, 9) must never be in candidates
    for (uint64_t c : candidates) {
        REQUIRE(c != 0);
        REQUIRE(c != 1);
        REQUIRE(c != 7);
        REQUIRE(c != 8);
        REQUIRE(c != 9);
        REQUIRE(c != 3); // Heavy hitter
        REQUIRE(c != 5); // Heavy hitter
    }
}

TEST_CASE("SalienceEvictionPolicy: capacity enforcement and stats", "[cache][salience]") {
    SalienceEvictionConfig cfg;
    cfg.max_cache_capacity = 6;
    cfg.sink_token_count = 1;
    cfg.recent_window_size = 2;
    cfg.decay_factor = 0.9;
    SalienceEvictionPolicy policy(cfg);

    for (size_t i = 0; i < 8; ++i) {
        policy.record_insertion(100 + i, i);
    }

    REQUIRE(policy.size() == 8);

    size_t evicted_count = policy.enforce_capacity();
    REQUIRE(evicted_count == 2);
    REQUIRE(policy.size() == 6);

    const auto& stats = policy.stats();
    REQUIRE(stats.total_tokens_inserted == 8);
    REQUIRE(stats.total_evictions == 2);
    REQUIRE(stats.eviction_pass_count == 1);

    policy.clear();
    REQUIRE(policy.size() == 0);
    REQUIRE(policy.stats().total_tokens_inserted == 0);
}

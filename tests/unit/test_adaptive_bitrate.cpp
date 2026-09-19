#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "../../strategies/adaptive_bitrate.h"

using namespace adaptq;

TEST_CASE("AdaptiveBitratePolicy: configuration validation", "[strategy][adaptive_bitrate]") {
    AdaptiveBitrateConfig cfg;
    cfg.sink_tokens = 4;
    cfg.recent_window = 32;
    cfg.mid_window = 128;
    cfg.high_bits = 4;
    cfg.mid_bits = 3;
    cfg.low_bits = 2;

    REQUIRE_NOTHROW(AdaptiveBitratePolicy(cfg));

    // Invalid negative sink tokens
    cfg.sink_tokens = -1;
    REQUIRE_THROWS_AS(AdaptiveBitratePolicy(cfg), std::invalid_argument);
    cfg.sink_tokens = 4;

    // Invalid windows: mid <= recent
    cfg.mid_window = 32;
    REQUIRE_THROWS_AS(AdaptiveBitratePolicy(cfg), std::invalid_argument);
    cfg.mid_window = 128;

    // Invalid bits: high < mid
    cfg.high_bits = 2;
    cfg.mid_bits = 3;
    REQUIRE_THROWS_AS(AdaptiveBitratePolicy(cfg), std::invalid_argument);
}

TEST_CASE("AdaptiveBitratePolicy: window and sink tier assignment", "[strategy][adaptive_bitrate]") {
    AdaptiveBitrateConfig cfg;
    cfg.sink_tokens = 4;
    cfg.recent_window = 16;
    cfg.mid_window = 64;
    cfg.high_bits = 4;
    cfg.mid_bits = 3;
    cfg.low_bits = 2;
    cfg.entropy_threshold = 10.0f; // High threshold so entropy doesn't trigger

    AdaptiveBitratePolicy policy(cfg);
    const int SEQ_LEN = 100;

    // Attention sink tokens: 0, 1, 2, 3 -> always 4 bits
    for (int t = 0; t < 4; ++t) {
        REQUIRE(policy.determine_bits(t, SEQ_LEN) == 4);
    }

    // Cold context: token 10 (SEQ_LEN - 10 = 90 > mid_window 64) -> 2 bits
    REQUIRE(policy.determine_bits(10, SEQ_LEN) == 2);
    REQUIRE(policy.determine_bits(25, SEQ_LEN) == 2);

    // Mid context: token 50 (SEQ_LEN - 50 = 50 <= mid_window 64 and > recent_window 16) -> 3 bits
    REQUIRE(policy.determine_bits(50, SEQ_LEN) == 3);
    REQUIRE(policy.determine_bits(70, SEQ_LEN) == 3);

    // Recent window: token 90 (SEQ_LEN - 90 = 10 <= recent_window 16) -> 4 bits
    REQUIRE(policy.determine_bits(90, SEQ_LEN) == 4);
    REQUIRE(policy.determine_bits(99, SEQ_LEN) == 4);

    REQUIRE(policy.total_tokens_seen() == 10);
    REQUIRE(policy.average_bits_per_token() > 2.0);
    REQUIRE(policy.average_bits_per_token() <= 4.0);
}

TEST_CASE("AdaptiveBitratePolicy: entropy-driven precision promotion", "[strategy][adaptive_bitrate]") {
    AdaptiveBitrateConfig cfg;
    cfg.sink_tokens = 2;
    cfg.recent_window = 10;
    cfg.mid_window = 50;
    cfg.high_bits = 4;
    cfg.mid_bits = 3;
    cfg.low_bits = 2;
    cfg.entropy_threshold = 1.0f;

    AdaptiveBitratePolicy policy(cfg);
    const int SEQ_LEN = 100;
    const int DIM = 8;

    // Low-variance cold token vector -> remains 2 bits
    std::vector<float> flat_vec(DIM, 0.5f);
    int bits_normal = policy.determine_bits(20, SEQ_LEN, flat_vec.data(), DIM);
    REQUIRE(bits_normal == 2);

    // High-variance cold token vector (alternating large spikes) -> promoted from 2 to 3 bits!
    std::vector<float> spike_vec = {5.0f, -5.0f, 5.0f, -5.0f, 5.0f, -5.0f, 5.0f, -5.0f};
    int bits_promoted = policy.determine_bits(20, SEQ_LEN, spike_vec.data(), DIM);
    REQUIRE(bits_promoted == 3);
}

TEST_CASE("AdaptiveBitrateStrategy: lifecycle and position tracking", "[strategy][adaptive_bitrate]") {
    AdaptiveBitrateStrategy strat;
    REQUIRE(std::string(strat.name()) == "adaptive_bitrate");

    StrategyConfig sc;
    sc.dim = 64;
    sc.capacity = 512;
    strat.init(sc);

    REQUIRE(strat.current_token_position() == 0);
    strat.on_append(100);
    REQUIRE(strat.current_token_position() == 100);

    // Query token allocation
    int bits = strat.determine_bit_allocation(0, nullptr, 0);
    REQUIRE(bits == 4); // Sink token

    REQUIRE(strat.average_bitrate() > 0.0);

    strat.reset();
    REQUIRE(strat.current_token_position() == 0);
    REQUIRE(strat.average_bitrate() == 0.0);
}

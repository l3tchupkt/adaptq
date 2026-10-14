#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <numeric>
#include <vector>

#include "../../attention/sparse_attention.h"

using namespace adaptq;

TEST_CASE("SparseAttentionKernel: config validation", "[attention][sparse]") {
    SparseAttentionConfig cfg;
    cfg.window_size = 128;
    cfg.sink_tokens = 4;
    cfg.enable_top_k = false;

    REQUIRE_NOTHROW(SparseAttentionKernel(cfg));

    // Invalid window size
    cfg.window_size = 0;
    REQUIRE_THROWS_AS(SparseAttentionKernel(cfg), std::invalid_argument);
    cfg.window_size = 128;

    // Invalid negative sinks
    cfg.sink_tokens = -1;
    REQUIRE_THROWS_AS(SparseAttentionKernel(cfg), std::invalid_argument);
    cfg.sink_tokens = 4;

    // Invalid Top-K
    cfg.enable_top_k = true;
    cfg.top_k = 0;
    REQUIRE_THROWS_AS(SparseAttentionKernel(cfg), std::invalid_argument);
    cfg.enable_top_k = false;

    // Invalid threshold
    cfg.prune_threshold = 1.5f;
    REQUIRE_THROWS_AS(SparseAttentionKernel(cfg), std::invalid_argument);
}

TEST_CASE("SparseAttentionKernel: sliding window and sink index selection", "[attention][sparse]") {
    SparseAttentionConfig cfg;
    cfg.window_size = 10;
    cfg.sink_tokens = 2;
    SparseAttentionKernel kernel(cfg);

    std::vector<int> active;

    // Case 1: Sequence smaller than window
    kernel.select_active_indices(5, 4, active);
    REQUIRE(active.size() == 5);
    for (int i = 0; i < 5; ++i) REQUIRE(active[i] == i);

    // Case 2: Sequence larger than window (e.g. query_pos = 50, n_tokens = 100)
    // Sinks: 0, 1
    // Window: [50 - 10 + 1, 50] = [41, 50] (10 tokens)
    // Total active: 2 + 10 = 12 tokens
    kernel.select_active_indices(100, 50, active);
    REQUIRE(active.size() == 12);
    REQUIRE(active[0] == 0);
    REQUIRE(active[1] == 1);
    REQUIRE(active[2] == 41);
    REQUIRE(active.back() == 50);
}

TEST_CASE("SparseAttentionKernel: top-k logit filtering", "[attention][sparse]") {
    SparseAttentionConfig cfg;
    cfg.window_size = 20;
    cfg.sink_tokens = 2;
    cfg.enable_top_k = true;
    cfg.top_k = 4; // 2 sinks + top 2 from window candidates
    SparseAttentionKernel kernel(cfg);

    std::vector<int> active;
    kernel.select_active_indices(100, 50, active); // Window has tokens 32..50

    // Fabricate logits where tokens 45 and 49 are very high
    std::vector<float> logits(100, 0.0f);
    logits[45] = 10.0f;
    logits[49] = 15.0f;
    logits[35] = 1.0f;

    kernel.apply_top_k(logits, active);

    REQUIRE(active.size() == 4);
    REQUIRE(active[0] == 0);  // Sink 0 preserved
    REQUIRE(active[1] == 1);  // Sink 1 preserved
    REQUIRE(active[2] == 45); // Top 2 candidate
    REQUIRE(active[3] == 49); // Top 1 candidate
}

TEST_CASE("SparseAttentionKernel: sparse softmax and V accumulation", "[attention][sparse]") {
    SparseAttentionConfig cfg;
    cfg.prune_threshold = 0.05f; // Zero out weights below 5%
    SparseAttentionKernel kernel(cfg);

    std::vector<int> active = {0, 1, 10};
    std::vector<float> logits = {10.0f, 10.0f, -5.0f}; // token 10 will have negligible exp

    std::vector<float> probs;
    kernel.compute_sparse_softmax(logits, active, probs);

    REQUIRE(probs.size() == 3);
    REQUIRE_THAT(probs[0], Catch::Matchers::WithinRel(0.5f, 1e-3f));
    REQUIRE_THAT(probs[1], Catch::Matchers::WithinRel(0.5f, 1e-3f));
    REQUIRE(probs[2] == 0.0f); // Pruned!

    // Value accumulation
    const int DIM = 4;
    std::vector<float> v_matrix(11 * DIM, 0.0f);
    // V for token 0: [1, 1, 1, 1]
    // V for token 1: [3, 3, 3, 3]
    for (int d = 0; d < DIM; ++d) {
        v_matrix[0 * DIM + d] = 1.0f;
        v_matrix[1 * DIM + d] = 3.0f;
        v_matrix[10 * DIM + d] = 100.0f; // should not contribute due to pruning
    }

    std::vector<float> out_acc(DIM, 0.0f);
    kernel.accumulate_sparse_v(probs, active, v_matrix.data(), DIM, out_acc.data());

    // Expected: 0.5 * [1,1,1,1] + 0.5 * [3,3,3,3] = [2,2,2,2]
    for (int d = 0; d < DIM; ++d) {
        REQUIRE_THAT(out_acc[d], Catch::Matchers::WithinRel(2.0f, 1e-3f));
    }
}

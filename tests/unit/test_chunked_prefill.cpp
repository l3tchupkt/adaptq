#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "../../attention/chunked_prefill.h"

using namespace adaptq;

TEST_CASE("ChunkedPrefillEngine: configuration bounds", "[attention][chunked_prefill]") {
    REQUIRE_THROWS_AS(ChunkedPrefillEngine(0, 128), std::invalid_argument);
    REQUIRE_THROWS_AS(ChunkedPrefillEngine(-8, 128), std::invalid_argument);
    REQUIRE_THROWS_AS(ChunkedPrefillEngine(64, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(ChunkedPrefillEngine(64, -10), std::invalid_argument);

    ChunkedPrefillEngine engine(64, 32);
    REQUIRE(engine.dim() == 64);
    REQUIRE(engine.chunk_size() == 32);
    REQUIRE(engine.total_tokens_ingested() == 0);
}

TEST_CASE("ChunkedPrefillEngine: exact mathematical equivalence to dense attention", "[attention][chunked_prefill]") {
    const int DIM = 16;
    const int N_TOKENS = 50;
    const int CHUNK_SIZE = 10;

    std::vector<float> query(DIM, 0.5f);
    std::vector<float> k_full(N_TOKENS * DIM);
    std::vector<float> v_full(N_TOKENS * DIM);

    for (int i = 0; i < N_TOKENS; ++i) {
        for (int d = 0; d < DIM; ++d) {
            k_full[i * DIM + d] = std::sin(static_cast<float>(i * DIM + d) * 0.1f);
            v_full[i * DIM + d] = std::cos(static_cast<float>(i * DIM + d) * 0.1f);
        }
    }

    // 1. Dense reference computation in single pass
    float scale = 1.0f / std::sqrt(static_cast<float>(DIM));
    std::vector<float> logits(N_TOKENS, 0.0f);
    float max_logit = -1e30f;

    for (int i = 0; i < N_TOKENS; ++i) {
        float dot = 0.0f;
        for (int d = 0; d < DIM; ++d) {
            dot += query[d] * k_full[i * DIM + d];
        }
        dot *= scale;
        logits[i] = dot;
        if (dot > max_logit) max_logit = dot;
    }

    float sum_exp = 0.0f;
    std::vector<float> probs(N_TOKENS, 0.0f);
    for (int i = 0; i < N_TOKENS; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (int i = 0; i < N_TOKENS; ++i) {
        probs[i] /= sum_exp;
    }

    std::vector<float> expected_out(DIM, 0.0f);
    for (int i = 0; i < N_TOKENS; ++i) {
        for (int d = 0; d < DIM; ++d) {
            expected_out[d] += probs[i] * v_full[i * DIM + d];
        }
    }

    // 2. Chunked prefill execution
    ChunkedPrefillEngine engine(DIM, CHUNK_SIZE);

    int processed = 0;
    while (processed < N_TOKENS) {
        int this_chunk = std::min(CHUNK_SIZE, N_TOKENS - processed);
        const float *k_ptr = k_full.data() + processed * DIM;
        const float *v_ptr = v_full.data() + processed * DIM;

        engine.process_chunk(k_ptr, v_ptr, this_chunk, query.data());
        processed += this_chunk;
    }

    REQUIRE(engine.total_tokens_ingested() == N_TOKENS);

    std::vector<float> chunked_out(DIM, 0.0f);
    engine.finalize(chunked_out.data());

    // 3. Verify exact numerical match
    for (int d = 0; d < DIM; ++d) {
        REQUIRE_THAT(chunked_out[d], Catch::Matchers::WithinRel(expected_out[d], 1e-4f));
    }

    // 4. Test reset
    engine.reset();
    REQUIRE(engine.total_tokens_ingested() == 0);
    std::vector<float> reset_out(DIM, 1.0f);
    engine.finalize(reset_out.data());
    for (int d = 0; d < DIM; ++d) {
        REQUIRE(reset_out[d] == 0.0f);
    }
}

TEST_CASE("ChunkedPrefillEngine: null argument handling", "[attention][chunked_prefill]") {
    ChunkedPrefillEngine engine(16, 8);
    std::vector<float> buf(16, 1.0f);

    REQUIRE_THROWS_AS(engine.process_chunk(nullptr, buf.data(), 1, buf.data()), std::invalid_argument);
    REQUIRE_THROWS_AS(engine.process_chunk(buf.data(), nullptr, 1, buf.data()), std::invalid_argument);
    REQUIRE_THROWS_AS(engine.process_chunk(buf.data(), buf.data(), 1, nullptr), std::invalid_argument);
    REQUIRE_THROWS_AS(engine.finalize(nullptr), std::invalid_argument);
}

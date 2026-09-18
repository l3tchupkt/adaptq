#include <catch2/catch_test_macros.hpp>
#include "../../attention/softmax_avx2.h"
#include "../../include/attention.h"
#include <cmath>
#include <vector>

TEST_CASE("AVX2 softmax matches scalar reference", "[attention][avx2]") {
    const std::vector<int> sizes = {1, 2, 3, 7, 8, 9, 15, 16, 31, 32, 63, 64};

    for (int n : sizes) {
        std::vector<float> expected(n);
        std::vector<float> actual(n);
        for (int i = 0; i < n; ++i) {
            expected[i] = std::sin((float)i * 0.37f) * 4.0f
                        - std::cos((float)i * 0.19f) * 2.0f
                        + (float)(i % 5) * 0.11f;
        }
        actual = expected;

        softmax(expected.data(), n);
        softmax_avx2(actual.data(), n);

        float expected_sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            expected_sum += expected[i];
            REQUIRE(std::isfinite(actual[i]));
            REQUIRE(std::abs(actual[i] - expected[i]) < 2e-6f);
        }
        REQUIRE(std::abs(expected_sum - 1.0f) < 2e-6f);
    }
}

TEST_CASE("AVX2 softmax handles extreme logits like scalar reference", "[attention][avx2][stability]") {
    const std::vector<float> reference = {
        -80.0f, -40.0f, -10.0f, -1.0f, 0.0f, 1.0f, 10.0f, 40.0f
    };
    std::vector<float> expected = reference;
    std::vector<float> actual = reference;

    softmax(expected.data(), (int)expected.size());
    softmax_avx2(actual.data(), (int)actual.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(std::isfinite(actual[i]));
        REQUIRE(std::abs(actual[i] - expected[i]) < 2e-6f);
    }
}

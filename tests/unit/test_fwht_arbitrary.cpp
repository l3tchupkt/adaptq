#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "../../core/fwht_arbitrary.h"

using namespace adaptq;

TEST_CASE("ArbitraryDimFWHTAdapter: dimension calculation and bounds", "[core][fwht]") {
    // Power of two
    ArbitraryDimFWHTAdapter adapter64(64);
    REQUIRE(adapter64.original_dim() == 64);
    REQUIRE(adapter64.padded_dim() == 64);
    REQUIRE(adapter64.is_power_of_two() == true);
    REQUIRE_THAT(adapter64.energy_scale(), Catch::Matchers::WithinRel(1.0f, 1e-5f));

    // Non power of two: 48 -> 64
    ArbitraryDimFWHTAdapter adapter48(48);
    REQUIRE(adapter48.original_dim() == 48);
    REQUIRE(adapter48.padded_dim() == 64);
    REQUIRE(adapter48.is_power_of_two() == false);
    REQUIRE(adapter48.energy_scale() > 1.0f);

    // Non power of two: 80 -> 128
    ArbitraryDimFWHTAdapter adapter80(80);
    REQUIRE(adapter80.original_dim() == 80);
    REQUIRE(adapter80.padded_dim() == 128);

    // Non power of two: 96 -> 128
    ArbitraryDimFWHTAdapter adapter96(96);
    REQUIRE(adapter96.original_dim() == 96);
    REQUIRE(adapter96.padded_dim() == 128);

    // Invalid dimension
    REQUIRE_THROWS_AS(ArbitraryDimFWHTAdapter(0), std::invalid_argument);
    REQUIRE_THROWS_AS(ArbitraryDimFWHTAdapter(-10), std::invalid_argument);
}

TEST_CASE("ArbitraryDimFWHTAdapter: roundtrip reconstruction for irregular dimensions", "[core][fwht]") {
    std::vector<int> test_dims = {32, 48, 64, 80, 96, 160};

    for (int d : test_dims) {
        ArbitraryDimFWHTAdapter adapter(d, 42ULL);
        int pad_d = adapter.padded_dim();

        std::vector<float> original(d);
        for (int i = 0; i < d; ++i) {
            original[i] = std::sin(static_cast<float>(i + 1) * 0.2f);
        }

        std::vector<float> transformed(pad_d, 0.0f);
        adapter.forward(original.data(), transformed.data());

        // Transformed vector should have non-zero energy
        float norm_rot = adapter.compute_l2_norm(transformed.data(), pad_d);
        float norm_orig = adapter.compute_l2_norm(original.data(), d);
        REQUIRE(norm_rot > 0.0f);
        REQUIRE(norm_orig > 0.0f);

        // Inverse transform back to original dimension
        std::vector<float> reconstructed(d, 0.0f);
        adapter.inverse(transformed.data(), reconstructed.data());

        // Verify reconstruction error
        float max_diff = 0.0f;
        for (int i = 0; i < d; ++i) {
            float diff = std::abs(reconstructed[i] - original[i]);
            if (diff > max_diff) max_diff = diff;
        }

        REQUIRE(max_diff < 1e-4f);
    }
}

TEST_CASE("ArbitraryDimFWHTAdapter: null pointer safety", "[core][fwht]") {
    ArbitraryDimFWHTAdapter adapter(64);
    std::vector<float> buf(64, 1.0f);

    REQUIRE_THROWS_AS(adapter.forward(nullptr, buf.data()), std::invalid_argument);
    REQUIRE_THROWS_AS(adapter.forward(buf.data(), nullptr), std::invalid_argument);
    REQUIRE_THROWS_AS(adapter.inverse(nullptr, buf.data()), std::invalid_argument);
    REQUIRE_THROWS_AS(adapter.inverse(buf.data(), nullptr), std::invalid_argument);
}

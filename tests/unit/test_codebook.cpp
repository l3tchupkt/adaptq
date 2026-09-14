/* Catch2 v3 — link against Catch2::Catch2WithMain */
#include <catch2/catch_test_macros.hpp>
#include "../../include/codebook.h"
#include "../../include/quantizer.h"
#include <cmath>
#include <cfloat>
#include <limits>
#include <vector>
#include <random>

/* -------------------------------------------------------------------------
 * tests/unit/test_codebook.cpp
 *
 * Issue #16: Comprehensive boundary condition and extreme outlier tests
 * for Max-Lloyd quantization codebooks and Quantizer pipeline.
 * ----------------------------------------------------------------------- */

TEST_CASE("Max-Lloyd Codebook Centroid Ordering and Sizes", "[codebook][boundary]") {
    // CB2 (2-bit, 4 centroids) must be strictly ascending
    for (int i = 0; i < 3; ++i) {
        REQUIRE(CB2[i] < CB2[i + 1]);
    }
    // CB3 (3-bit, 8 centroids) must be strictly ascending
    for (int i = 0; i < 7; ++i) {
        REQUIRE(CB3[i] < CB3[i + 1]);
    }
    // CB4 (4-bit, 16 centroids) must be strictly ascending
    for (int i = 0; i < 15; ++i) {
        REQUIRE(CB4[i] < CB4[i + 1]);
    }

    // codebook_size returns clamped valid sizes
    REQUIRE(codebook_size(2) == 4);
    REQUIRE(codebook_size(3) == 8);
    REQUIRE(codebook_size(4) == 16);
    REQUIRE(codebook_size(0) == 4);   // clamped lower
    REQUIRE(codebook_size(1) == 4);   // clamped lower
    REQUIRE(codebook_size(5) == 16);  // clamped upper

    // get_codebook always returns valid non-null pointer
    REQUIRE(get_codebook(2) == CB2);
    REQUIRE(get_codebook(3) == CB3);
    REQUIRE(get_codebook(4) == CB4);
    REQUIRE(get_codebook(0) != nullptr);
    REQUIRE(get_codebook(99) != nullptr);
}

TEST_CASE("Centroid Exact Matching and Reconstruction", "[codebook][boundary]") {
    // Verify that quantizing exactly on a centroid returns that exact centroid index
    for (int i = 0; i < 4; ++i) {
        REQUIRE(quantize_fast(CB2[i], 2) == i);
        REQUIRE(quantize_scalar(CB2[i], CB2, 4) == i);
        REQUIRE(dequantize_scalar(i, CB2) == CB2[i]);
    }
    for (int i = 0; i < 8; ++i) {
        REQUIRE(quantize_fast(CB3[i], 3) == i);
        REQUIRE(quantize_scalar(CB3[i], CB3, 8) == i);
        REQUIRE(dequantize_scalar(i, CB3) == CB3[i]);
    }
    for (int i = 0; i < 16; ++i) {
        REQUIRE(quantize_fast(CB4[i], 4) == i);
        REQUIRE(quantize_scalar(CB4[i], CB4, 16) == i);
        REQUIRE(dequantize_scalar(i, CB4) == CB4[i]);
    }

    // Defensive dequantize_scalar checks
    REQUIRE(dequantize_scalar(0, nullptr) == 0.0f);
    REQUIRE(dequantize_scalar(-1, CB4) == CB4[0]);
}

TEST_CASE("Extreme Outlier Values (Infinity and FLT_MAX)", "[codebook][boundary]") {
    const float pos_inf = std::numeric_limits<float>::infinity();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const float huge_pos = 1e30f;
    const float huge_neg = -1e30f;
    const float flt_max = FLT_MAX;
    const float flt_min_neg = -FLT_MAX;

    // +inf and huge positive values must map strictly to highest centroid
    REQUIRE(quantize_fast(pos_inf, 2) == 3);
    REQUIRE(quantize_fast(pos_inf, 3) == 7);
    REQUIRE(quantize_fast(pos_inf, 4) == 15);
    REQUIRE(quantize_fast(flt_max, 4) == 15);
    REQUIRE(quantize_fast(huge_pos, 4) == 15);

    // -inf and huge negative values must map strictly to index 0
    REQUIRE(quantize_fast(neg_inf, 2) == 0);
    REQUIRE(quantize_fast(neg_inf, 3) == 0);
    REQUIRE(quantize_fast(neg_inf, 4) == 0);
    REQUIRE(quantize_fast(flt_min_neg, 4) == 0);
    REQUIRE(quantize_fast(huge_neg, 4) == 0);

    // Scalar fallback handles extreme outliers safely
    REQUIRE(quantize_scalar(pos_inf, CB4, 16) == 15);
    REQUIRE(quantize_scalar(neg_inf, CB4, 16) == 0);
    REQUIRE(quantize_scalar(huge_pos, CB4, 16) == 15);
    REQUIRE(quantize_scalar(huge_neg, CB4, 16) == 0);
}

TEST_CASE("Subnormals and Zero Values", "[codebook][boundary]") {
    const float zero_pos = 0.0f;
    const float zero_neg = -0.0f;
    const float flt_min = FLT_MIN;
    const float subnormal = 1e-38f;

    // 0.0f is exact midpoint for symmetric codebooks
    REQUIRE(quantize_fast(zero_pos, 2) >= 1);
    REQUIRE(quantize_fast(zero_pos, 2) <= 2);
    REQUIRE(quantize_fast(zero_neg, 2) >= 1);
    REQUIRE(quantize_fast(zero_neg, 2) <= 2);

    // In CB4, 0.0000f is centroid 7
    REQUIRE(quantize_fast(zero_pos, 4) == 7);
    REQUIRE(quantize_fast(zero_neg, 4) == 7);
    REQUIRE(quantize_scalar(zero_pos, CB4, 16) == 7);

    // Subnormal numbers close to 0 map near center
    int idx_sub = quantize_fast(subnormal, 4);
    REQUIRE((idx_sub == 7 || idx_sub == 8));
    int idx_min = quantize_fast(flt_min, 4);
    REQUIRE((idx_min == 7 || idx_min == 8));
}

TEST_CASE("NaN Handling Robustness", "[codebook][boundary]") {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    const float snan = std::numeric_limits<float>::signaling_NaN();

    for (int bits : {2, 3, 4}) {
        int idx_q = quantize_fast(qnan, bits);
        int idx_s = quantize_fast(snan, bits);
        int max_idx = (1 << bits) - 1;

        // Must return valid in-bounds index
        REQUIRE(idx_q >= 0);
        REQUIRE(idx_q <= max_idx);
        REQUIRE(idx_s >= 0);
        REQUIRE(idx_s <= max_idx);

        // Scalar fallback must also not crash or index out of bounds
        const float *cb = get_codebook(bits);
        int sc_q = quantize_scalar(qnan, cb, 1 << bits);
        REQUIRE(sc_q >= 0);
        REQUIRE(sc_q <= max_idx);
    }
}

TEST_CASE("Midpoint Decision Threshold Boundaries", "[codebook][boundary]") {
    // For 2-bit: CB2 = {-1.5104, -0.4528, 0.4528, 1.5104}
    // Midpoints: -0.9816, 0.0, 0.9816
    const float eps = 1e-4f;
    const float t0 = (-1.5104f - 0.4528f) * 0.5f; // -0.9816
    const float t1 = 0.0f;
    const float t2 = (0.4528f + 1.5104f) * 0.5f;  //  0.9816

    REQUIRE(quantize_fast(t0 - eps, 2) == 0);
    REQUIRE(quantize_fast(t0 + eps, 2) == 1);
    REQUIRE(quantize_fast(t1 - eps, 2) == 1);
    REQUIRE(quantize_fast(t1 + eps, 2) == 2);
    REQUIRE(quantize_fast(t2 - eps, 2) == 2);
    REQUIRE(quantize_fast(t2 + eps, 2) == 3);
}

TEST_CASE("Fast vs Scalar Consistency Sweep Across Large Range", "[codebook][consistency]") {
    // Sweep [-10.0, 10.0] in fine steps
    for (int bits : {2, 3, 4}) {
        const float *cb = get_codebook(bits);
        int cb_sz = 1 << bits;
        for (float v = -10.0f; v <= 10.0f; v += 0.025f) {
            int idx_fast = quantize_fast(v, bits);
            int idx_scalar = quantize_scalar(v, cb, cb_sz);
            // On exact decision threshold ties, both adjacent indices are mathematically valid.
            // Check that difference is at most 1, and distance to centroids is essentially identical.
            int diff = std::abs(idx_fast - idx_scalar);
            if (diff > 0) {
                float dist_fast = fabsf(v - cb[idx_fast]);
                float dist_scalar = fabsf(v - cb[idx_scalar]);
                REQUIRE(fabsf(dist_fast - dist_scalar) < 1e-4f);
            } else {
                REQUIRE(idx_fast == idx_scalar);
            }
        }
    }
}

TEST_CASE("Quantizer Vector-Level Outlier Robustness", "[quantizer][outliers]") {
    Quantizer q;
    q.init(128, 0x12345678ULL);

    SECTION("Single massive outlier spike") {
        std::vector<float> x(128, 0.0f);
        x[0] = 1e7f; // Huge outlier spike

        for (int bits : {2, 3, 4}) {
            QuantizedVec qv = q.quantize(x.data(), bits);
            REQUIRE(std::isfinite(qv.scale));
            REQUIRE(qv.scale > 0.0f);

            std::vector<float> reconstructed(128, 0.0f);
            q.dequantize(qv, reconstructed.data());

            for (int i = 0; i < 128; ++i) {
                REQUIRE(std::isfinite(reconstructed[i]));
            }
        }
    }

    SECTION("All-zero input vector") {
        std::vector<float> x(128, 0.0f);

        for (int bits : {2, 3, 4}) {
            QuantizedVec qv = q.quantize(x.data(), bits);
            REQUIRE(std::isfinite(qv.scale));

            std::vector<float> reconstructed(128, 0.0f);
            q.dequantize(qv, reconstructed.data());

            for (int i = 0; i < 128; ++i) {
                REQUIRE(std::isfinite(reconstructed[i]));
            }
        }
    }

    SECTION("Alternating large sign vector") {
        std::vector<float> x(128);
        for (int i = 0; i < 128; ++i) {
            x[i] = (i % 2 == 0) ? 5000.0f : -5000.0f;
        }

        for (int bits : {2, 3, 4}) {
            QuantizedVec qv = q.quantize(x.data(), bits);
            REQUIRE(std::isfinite(qv.scale));

            std::vector<float> reconstructed(128, 0.0f);
            q.dequantize(qv, reconstructed.data());

            for (int i = 0; i < 128; ++i) {
                REQUIRE(std::isfinite(reconstructed[i]));
            }
        }
    }
}

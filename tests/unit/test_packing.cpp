/* Catch2 v3 — link against Catch2::Catch2WithMain */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../../include/quantizer.h"
#include "../../include/codebook.h"
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

/* -------------------------------------------------------------------------
 * tests/unit/test_packing.cpp
 *
 * Validates bit-packing round-trips for 2, 3, and 4-bit precision.
 * Also validates the full quantize → dequantize round-trip via Quantizer.
 * ----------------------------------------------------------------------- */

static std::vector<uint8_t> random_indices(int n, int bits, uint64_t seed) {
    int max_idx = (1 << bits) - 1;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(0, max_idx);
    std::vector<uint8_t> idx(n);
    for (auto &x : idx) x = (uint8_t)dist(rng);
    return idx;
}

static int max_err_u8(const uint8_t *a, const uint8_t *b, int n) {
    int mx = 0;
    for (int i = 0; i < n; ++i)
        mx = std::max(mx, (int)std::abs((int)a[i] - (int)b[i]));
    return mx;
}

TEST_CASE("4-bit pack/unpack round-trip", "[packing]") {
    for (int padded : {32, 64, 128, 256, 512, 1024}) {
        auto indices = random_indices(padded, 4, 0xABCDULL ^ (uint64_t)padded);
        int  pb      = padded / 2;
        std::vector<uint8_t> packed(pb, 0), unpacked(padded, 0);
        pack_indices(indices.data(), padded, 4, packed.data());
        unpack_indices(packed.data(), padded, 4, unpacked.data());
        CAPTURE(padded);
        REQUIRE(max_err_u8(indices.data(), unpacked.data(), padded) == 0);
    }
}

TEST_CASE("2-bit pack/unpack round-trip", "[packing]") {
    for (int padded : {32, 64, 128, 256, 512, 1024}) {
        auto indices = random_indices(padded, 2, 0xCAFEULL ^ (uint64_t)padded);
        int  pb      = padded / 4;
        std::vector<uint8_t> packed(pb, 0), unpacked(padded, 0);
        pack_indices(indices.data(), padded, 2, packed.data());
        unpack_indices(packed.data(), padded, 2, unpacked.data());
        CAPTURE(padded);
        REQUIRE(max_err_u8(indices.data(), unpacked.data(), padded) == 0);
    }
}

TEST_CASE("3-bit pack/unpack round-trip (padded must be mult of 8)", "[packing]") {
    for (int padded : {8, 16, 24, 32, 64, 128, 256}) {
        auto indices = random_indices(padded, 3, 0xBEEFULL ^ (uint64_t)padded);
        int  pb      = padded * 3 / 8;
        std::vector<uint8_t> packed(pb, 0), unpacked(padded, 0);
        pack_indices(indices.data(), padded, 3, packed.data());
        unpack_indices(packed.data(), padded, 3, unpacked.data());
        CAPTURE(padded);
        REQUIRE(max_err_u8(indices.data(), unpacked.data(), padded) == 0);
    }
}

TEST_CASE("Unpacked indices are in [0, 2^bits-1]", "[packing]") {
    for (int bits : {2, 3, 4}) {
        int padded = 128;
        auto indices = random_indices(padded, bits, 9999ULL);
        int  pb      = (padded * bits + 7) / 8;
        std::vector<uint8_t> packed(pb, 0), unpacked(padded, 0);
        pack_indices(indices.data(), padded, bits, packed.data());
        unpack_indices(packed.data(), padded, bits, unpacked.data());
        int max_val = (1 << bits) - 1;
        for (int i = 0; i < padded; ++i) {
            CAPTURE(bits, i, (int)unpacked[i]);
            REQUIRE((int)unpacked[i] <= max_val);
        }
    }
}

static std::vector<uint8_t> reference_quantized_indices(
    const Quantizer &q, const std::vector<float> &input, int bits,
    float &scale) {
    std::vector<float> buf(q.padded, 0.f);
    float norm = 0.f;
    for (int i = 0; i < q.dim; ++i) {
        buf[i] = input[i];
        norm += input[i] * input[i];
    }

    norm = std::sqrt(norm + 1e-12f);
    const float inv_norm = 1.f / norm;
    for (int i = 0; i < q.padded; ++i)
        buf[i] *= inv_norm;

    fwht_forward(buf.data(), q.D.data(), q.padded);

    const float sp = std::sqrt((float)q.padded);
    float sum2 = 0.f;
    for (int i = 0; i < q.padded; ++i) {
        const float value = buf[i] * sp;
        sum2 += value * value;
    }

    const float clip = 3.f * std::sqrt(sum2 / (float)q.padded + 1e-12f);
    const float inv_clip = 1.f / (clip + 1e-12f);

    std::vector<uint8_t> indices(q.padded);
    for (int i = 0; i < q.padded; ++i) {
        float value = buf[i] * sp;
        if (value > clip)
            value = clip;
        if (value < -clip)
            value = -clip;
        value *= inv_clip;
        indices[i] = (uint8_t)quantize_fast(value, bits);
    }

    scale = norm * clip;
    return indices;
}

TEST_CASE("Quantizer SIMD staging matches scalar reference", "[quantizer][avx2]") {
    for (int dim : {9, 17, 33, 65, 127, 128}) {
        Quantizer q;
        q.init(dim, 0xA55A1234ULL ^ (uint64_t)dim);

        std::vector<float> input(dim);
        for (int i = 0; i < dim; ++i)
            input[i] = std::sin(0.17f * (float)(i + 1)) * (1.f + 0.01f * i);

        for (int bits : {2, 3, 4}) {
            float reference_scale = 0.f;
            const auto reference =
                reference_quantized_indices(q, input, bits, reference_scale);

            const QuantizedVec actual = q.quantize(input.data(), bits);
            std::vector<uint8_t> actual_indices(q.padded);
            unpack_indices(actual.data.data(), q.padded, bits,
                           actual_indices.data());

            CAPTURE(dim, bits, reference_scale, actual.scale);
            REQUIRE(actual_indices == reference);
            REQUIRE(actual.scale == Catch::Approx(reference_scale).epsilon(1e-5f));
        }
    }
}

TEST_CASE("Quantizer quantize/dequantize MSE within expected bounds", "[quantizer]") {
    Quantizer q;
    q.init(128, 0xDEADC0DEULL);

    std::mt19937_64 rng(42);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> x(128);
    for (auto &v : x) v = dist(rng);

    for (int bits : {2, 3, 4}) {
        QuantizedVec qv = q.quantize(x.data(), bits);
        std::vector<float> out(128, 0.f);
        q.dequantize(qv, out.data());

        double mse_val = 0.0;
        for (int i = 0; i < 128; ++i) {
            double d = (double)x[i] - (double)out[i];
            mse_val += d * d;
        }
        mse_val /= 128;
        CAPTURE(bits, mse_val);
        /* NOTE: MSE is vs the raw input x. The Quantizer applies the full HAR
         * pipeline (L2-norm → Rademacher → FWHT → ±3σ clip → VQ), so the MSE
         * between x and the reconstruction includes multi-stage transform error.
         * Thresholds are set at 2× the observed values:
         *   bits=4: ~0.076 observed  bits=3: ~0.249 observed  bits=2: ~0.714 */
        if (bits == 4) REQUIRE(mse_val < 0.15);
        if (bits == 3) REQUIRE(mse_val < 0.50);
        if (bits == 2) REQUIRE(mse_val < 1.50);
    }
}

TEST_CASE("Quantizer init and pack_indices parameter validation", "[quantizer][safety]") {
    Quantizer q;
    REQUIRE_THROWS_AS(q.init(0, 42), std::invalid_argument);
    REQUIRE_THROWS_AS(q.init(-64, 42), std::invalid_argument);
    REQUIRE_THROWS_AS(q.init(1025, 42), std::invalid_argument);

    uint8_t in_buf[16] = {};
    uint8_t out_buf[16] = {};
    REQUIRE_THROWS_AS(pack_indices(in_buf, 16, 1, out_buf), std::invalid_argument);
    REQUIRE_THROWS_AS(pack_indices(in_buf, 16, 5, out_buf), std::invalid_argument);
    REQUIRE_THROWS_AS(unpack_indices(in_buf, 16, 1, out_buf), std::invalid_argument);
    REQUIRE_THROWS_AS(unpack_indices(in_buf, 16, 6, out_buf), std::invalid_argument);
}

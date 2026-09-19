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

static void reference_dequantize(const Quantizer &quantizer,
                                const std::vector<uint8_t> &packed, int padded,
                                int bits, float scale, std::vector<float> &out) {
    std::vector<uint8_t> indices(padded);
    unpack_indices(packed.data(), padded, bits, indices.data());

    std::vector<float> buf(padded);
    const float inv_sq = 1.f / std::sqrt((float)padded);
    const float *cb = get_codebook(bits);
    for (int i = 0; i < padded; ++i)
        buf[i] = cb[indices[i]] * inv_sq;

    fwht_inverse(buf.data(), quantizer.D.data(), padded);
    out.resize(quantizer.dim);
    for (int i = 0; i < quantizer.dim; ++i)
        out[i] = buf[i] * scale;
}

TEST_CASE("Dequantization SIMD path matches scalar reconstruction",
          "[quantizer][avx2][dequantize]") {
    for (int dim : {33, 64, 127, 128}) {
        Quantizer q;
        q.init(dim, 0xD311AULL ^ (uint64_t)dim);
        const int padded = q.padded;

        for (int bits : {2, 3, 4}) {
            std::vector<uint8_t> indices(padded);
            for (int i = 0; i < padded; ++i)
                indices[i] = (uint8_t)((i * 5 + bits + dim) & ((1 << bits) - 1));

            const int packed_bytes = (padded * bits + 7) / 8;
            std::vector<uint8_t> packed(packed_bytes);
            pack_indices(indices.data(), padded, bits, packed.data());

            std::vector<float> expected;
            reference_dequantize(q, packed, padded, bits, 1.25f, expected);

            QuantizedVec quantized;
            quantized.data = packed;
            quantized.scale = 1.25f;
            quantized.dim = padded;
            quantized.bits = bits;

            std::vector<float> actual(dim, 0.f);
            q.dequantize(quantized, actual.data());

            CAPTURE(dim, bits);
            REQUIRE(actual.size() == expected.size());
            for (int i = 0; i < dim; ++i)
                REQUIRE(actual[i] == Catch::Approx(expected[i]).epsilon(1e-6f));
        }
    }
}

TEST_CASE("Raw dequantization SIMD path matches scalar reconstruction",
          "[quantizer][avx2][dequantize]") {
    for (int dim : {33, 64, 127, 128}) {
        Quantizer q;
        q.init(dim, 0x5A17ULL ^ (uint64_t)dim);
        const int padded = q.padded;

        for (int bits : {2, 3, 4}) {
            std::vector<uint8_t> indices(padded);
            for (int i = 0; i < padded; ++i)
                indices[i] = (uint8_t)((i * 7 + bits) & ((1 << bits) - 1));

            const int packed_bytes = (padded * bits + 7) / 8;
            std::vector<uint8_t> packed(packed_bytes);
            pack_indices(indices.data(), padded, bits, packed.data());

            std::vector<float> expected;
            reference_dequantize(q, packed, padded, bits, 0.75f, expected);

            std::vector<float> actual(dim, 0.f);
            q.dequantize_raw(packed.data(), 0.75f, padded, bits, actual.data());

            CAPTURE(dim, bits);
            for (int i = 0; i < dim; ++i)
                REQUIRE(actual[i] == Catch::Approx(expected[i]).epsilon(1e-6f));
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

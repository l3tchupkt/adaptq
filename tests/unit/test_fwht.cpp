/* Catch2 v3 — link against Catch2::Catch2WithMain; no CATCH_CONFIG_MAIN needed */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../../include/fwht.h"
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

/* -------------------------------------------------------------------------
 * tests/unit/test_fwht.cpp
 *
 * Validates that fwht_forward followed by fwht_inverse is a true inverse:
 *   fwht_inverse( fwht_forward(x) ) == x
 *
 * Tolerance: MSE < 1e-10 (well within FP32 round-off budget).
 * Also validates next_pow2() and gen_rademacher() invariants.
 * ----------------------------------------------------------------------- */

static std::vector<float> random_vec(int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> v(n);
    for (auto &x : v) x = dist(rng);
    return v;
}

static double mse(const float *a, const float *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return s / n;
}

TEST_CASE("next_pow2 returns correct values", "[fwht]") {
    REQUIRE(next_pow2(1)   == 1);
    REQUIRE(next_pow2(2)   == 2);
    REQUIRE(next_pow2(3)   == 4);
    REQUIRE(next_pow2(64)  == 64);
    REQUIRE(next_pow2(65)  == 128);
    REQUIRE(next_pow2(127) == 128);
    REQUIRE(next_pow2(128) == 128);
    REQUIRE(next_pow2(129) == 256);
    REQUIRE(next_pow2(512) == 512);
    REQUIRE(next_pow2(513) == 1024);
}

TEST_CASE("gen_rademacher produces only +/-1", "[fwht]") {
    std::vector<int8_t> D(256);
    gen_rademacher(D.data(), 256, 0xABCD1234ULL);
    for (int i = 0; i < 256; ++i)
        REQUIRE((D[i] == 1 || D[i] == -1));
}

TEST_CASE("gen_rademacher is seed-deterministic", "[fwht]") {
    std::vector<int8_t> D1(128), D2(128);
    gen_rademacher(D1.data(), 128, 12345ULL);
    gen_rademacher(D2.data(), 128, 12345ULL);
    REQUIRE(memcmp(D1.data(), D2.data(), 128) == 0);
}

TEST_CASE("gen_rademacher differs across seeds", "[fwht]") {
    std::vector<int8_t> D1(128), D2(128);
    gen_rademacher(D1.data(), 128, 1ULL);
    gen_rademacher(D2.data(), 128, 2ULL);
    int diff = 0;
    for (int i = 0; i < 128; ++i) diff += (D1[i] != D2[i]);
    REQUIRE(diff > 10);
}

/* ---- Round-trip: fwht_inverse(fwht_forward(x)) == x ------------------- */
static void check_roundtrip(int dim, uint64_t vec_seed, uint64_t d_seed) {
    int padded = next_pow2(dim);
    std::vector<int8_t> D(padded);
    gen_rademacher(D.data(), padded, d_seed);

    auto orig = random_vec(padded, vec_seed);
    std::vector<float> x = orig;

    fwht_forward(x.data(), D.data(), padded);
    fwht_inverse(x.data(), D.data(), padded);

    double err = mse(x.data(), orig.data(), padded);
    CAPTURE(dim, padded, err);
    REQUIRE(err < 1e-10);
}

TEST_CASE("FWHT round-trip MSE < 1e-10 for common head dims", "[fwht]") {
    for (int dim : {32, 64, 96, 128, 160, 192, 256}) {
        check_roundtrip(dim, 42ULL,     0xDEAD);
        check_roundtrip(dim, 123456ULL, 0xBEEF);
    }
}

TEST_CASE("FWHT round-trip for non-power-of-2 dims (padding path)", "[fwht]") {
    for (int dim : {3, 5, 7, 10, 15, 33, 100, 200}) {
        check_roundtrip(dim, 99ULL, 0xCAFE);
    }
}

TEST_CASE("FWHT accepts D sized to d for non-power-of-2 dimensions", "[fwht][safety]") {
    for (int dim : {3, 5, 7, 10, 15, 33, 96, 100, 160, 192, 200}) {
        std::vector<int8_t> D(dim);
        gen_rademacher(D.data(), dim, 0xCAFE);

        auto x = random_vec(dim, 99ULL);
        fwht_forward(x.data(), D.data(), dim);
        for (float value : x)
            REQUIRE(std::isfinite(value));
    }
}

TEST_CASE("FWHT is linear: forward(a+b) == forward(a)+forward(b)", "[fwht]") {
    int padded = 64;
    std::vector<int8_t> D(padded);
    gen_rademacher(D.data(), padded, 7ULL);

    auto a = random_vec(padded, 1ULL);
    auto b = random_vec(padded, 2ULL);

    std::vector<float> apb(padded), fa = a, fb = b;
    for (int i = 0; i < padded; ++i) apb[i] = a[i] + b[i];

    fwht_forward(fa.data(),  D.data(), padded);
    fwht_forward(fb.data(),  D.data(), padded);
    fwht_forward(apb.data(), D.data(), padded);

    std::vector<float> fapb(padded);
    for (int i = 0; i < padded; ++i) fapb[i] = fa[i] + fb[i];
    REQUIRE(mse(apb.data(), fapb.data(), padded) < 1e-10);
}

TEST_CASE("next_pow2 boundary and overflow safety", "[fwht][safety]") {
    REQUIRE(next_pow2(0) == 1);
    REQUIRE(next_pow2(-10) == 1);
    REQUIRE(next_pow2(1 << 30) == (1 << 30));
    // Verify no hang or infinite loop on values exceeding 1 << 30
    REQUIRE(next_pow2((1 << 30) + 1) == (1 << 30));
    REQUIRE(next_pow2(2147483647) == (1 << 30));
}

TEST_CASE("FWHT safety on null, empty, or out-of-bounds input", "[fwht][safety]") {
    float buf[16] = {};
    int8_t D[16] = {};
    // Should safely return without crashing
    fwht_forward(nullptr, D, 16);
    fwht_forward(buf, nullptr, 16);
    fwht_forward(buf, D, 0);
    fwht_forward(buf, D, -5);

    fwht_inverse(nullptr, D, 16);
    fwht_inverse(buf, nullptr, 16);
    fwht_inverse(buf, D, 0);
    fwht_inverse(buf, D, -5);

    // Dimension exceeding pad buffer throws std::invalid_argument
    std::vector<float> big_x(1025, 1.0f);
    std::vector<int8_t> big_D(2048, 1);
    REQUIRE_THROWS_AS(fwht_forward(big_x.data(), big_D.data(), 1025), std::invalid_argument);
    REQUIRE_THROWS_AS(fwht_inverse(big_x.data(), big_D.data(), 1025), std::invalid_argument);
}

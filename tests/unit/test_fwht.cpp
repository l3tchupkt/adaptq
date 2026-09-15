/* Catch2 v3 — link against Catch2::Catch2WithMain; no CATCH_CONFIG_MAIN needed */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../../include/fwht.h"
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

/* -------------------------------------------------------------------------
 * tests/unit/test_fwht.cpp
 *
 * Validates that fwht_forward followed by fwht_inverse is a true inverse
 * when the transform dimension is already a power of two.
 *
 * Non-power-of-two dimensions use internal zero-padding. The public D vector
 * only covers the caller's input dimension; padded coordinates are treated
 * as having an implicit +1 sign. Those dimensions are covered separately
 * by a contract/safety test because the padded transform output is truncated
 * back to the caller-provided d elements and is therefore not invertible from
 * that truncated representation alone.
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
    std::vector<int8_t> D(dim);
    gen_rademacher(D.data(), dim, d_seed);

    auto orig = random_vec(dim, vec_seed);
    std::vector<float> x = orig;

    fwht_forward(x.data(), D.data(), dim);
    fwht_inverse(x.data(), D.data(), dim);

    double err = mse(x.data(), orig.data(), dim);
    CAPTURE(dim, err);
    REQUIRE(err < 1e-10);
}

TEST_CASE("FWHT round-trip MSE < 1e-10 for power-of-2 head dims", "[fwht]") {
    for (int dim : {32, 64, 128, 256, 512}) {
        check_roundtrip(dim, 42ULL,     0xDEAD);
        check_roundtrip(dim, 123456ULL, 0xBEEF);
    }
}

TEST_CASE("FWHT accepts D sized to d for non-power-of-2 dimensions", "[fwht]") {
    for (int dim : {3, 5, 7, 10, 15, 33, 96, 100, 160, 192, 200}) {
        std::vector<int8_t> D(dim, 0x7F);
        gen_rademacher(D.data(), dim, 0xCAFE);

        auto x = random_vec(dim, 99ULL);
        fwht_forward(x.data(), D.data(), dim);
        fwht_inverse(x.data(), D.data(), dim);

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

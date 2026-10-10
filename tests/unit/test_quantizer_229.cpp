/* Catch2 v3 — link against Catch2::Catch2WithMain */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../../include/codebook.h"
#include "../../include/quantizer.h"
#include "../../include/fwht.h"
#include <cmath>
#include <cstring>
#include <random>
#include <set>
#include <vector>
#include <algorithm>
#include <numeric>

/* =========================================================================
 * tests/unit/test_quantizer_229.cpp
 *
 * Issue #229: Correct Lloyd-Max codebook and quantisation scaling
 *
 * Tests:
 *   A. Level utilisation — verifies no pathological level collapse
 *   B. Reconstruction metrics — MSE, RMSE, relative L2, cosine similarity,
 *      max absolute error; uses deterministic seeds
 *   C. Dimension/padding overhead — reports and tests all required dims
 *   D. FWHT correctness — round-trip and properties
 *   E. Codebook symmetry — all three codebooks must be symmetric
 * ========================================================================= */

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<float> gaussian_vec(int n, uint64_t seed, float mean = 0.f, float stddev = 1.f) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(mean, stddev);
    std::vector<float> v(n);
    for (auto &x : v) x = dist(rng);
    return v;
}

struct ReconMetrics {
    double mse;
    double rmse;
    double rel_l2;    // ||x - x'||_2 / ||x||_2
    double cosine;    // dot(x, x') / (||x|| * ||x'||)
    double max_abs;   // max |x[i] - x'[i]|
};

static ReconMetrics compute_metrics(const float *orig, const float *recon, int n) {
    ReconMetrics m = {};
    double sum_sq_err = 0, sum_sq_orig = 0, dot = 0;
    double sum_sq_recon = 0;
    m.max_abs = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = (double)orig[i] - (double)recon[i];
        sum_sq_err  += diff * diff;
        sum_sq_orig += (double)orig[i] * (double)orig[i];
        sum_sq_recon+= (double)recon[i] * (double)recon[i];
        dot         += (double)orig[i] * (double)recon[i];
        if (std::abs(diff) > m.max_abs) m.max_abs = std::abs(diff);
    }
    m.mse   = sum_sq_err / n;
    m.rmse  = std::sqrt(m.mse);
    m.rel_l2 = (sum_sq_orig > 1e-20) ? std::sqrt(sum_sq_err / sum_sq_orig) : 0.0;
    double denom = std::sqrt(sum_sq_orig) * std::sqrt(sum_sq_recon);
    m.cosine = (denom > 1e-20) ? (dot / denom) : 0.0;
    return m;
}

// ---------------------------------------------------------------------------
// A. Level utilisation tests
// ---------------------------------------------------------------------------

TEST_CASE("Level utilisation — no pathological collapse (issue #229)", "[quantizer][level_util]") {
    // For a large Gaussian sample, almost all levels should be used.
    // We require at least (2^bits - 2) distinct levels for 4-bit,
    // and at least (2^bits - 1) distinct levels for 2/3-bit.
    // This detects the old bug where the [-1,1] normalisation caused
    // extreme values of the N(0,1)-range codebook to be unreachable.

    Quantizer q;
    q.init(128, 0xDEADBEEFULL);

    // We quantize 2000 Gaussian vectors (enough to exercise all levels)
    const int N_VECS = 2000;
    const int DIM = 128;

    for (int bits : {2, 3, 4}) {
        int cb_size = 1 << bits;
        // Minimum distinct levels we require (allow missing at most 1 extreme level)
        int min_levels = cb_size - 1;

        std::set<int> seen_indices;
        std::mt19937_64 rng(bits * 12345ULL);
        std::normal_distribution<float> dist(0.f, 1.f);

        std::vector<uint8_t> unpacked(128, 0);
        for (int vec = 0; vec < N_VECS; ++vec) {
            std::vector<float> x(DIM);
            for (auto &v : x) v = dist(rng);

            QuantizedVec qv = q.quantize(x.data(), bits);
            unpack_indices(qv.data.data(), qv.dim, bits, unpacked.data());
            for (int i = 0; i < qv.dim; ++i)
                seen_indices.insert((int)unpacked[i]);

            if ((int)seen_indices.size() >= cb_size) break;
        }

        CAPTURE(bits, cb_size, seen_indices.size());
        // Require at least (cb_size - 1) levels used: allow 1 extreme to be very rare
        REQUIRE((int)seen_indices.size() >= min_levels);
    }
}

TEST_CASE("Level utilisation — 4-bit uses all 16 levels on large sample", "[quantizer][level_util]") {
    // With the corrected N(0,1) domain codebook, all 16 levels MUST appear
    // in a large enough Gaussian sample. The extreme levels (0 and 15,
    // corresponding to ±2.4) occur with probability ≈ P(|Z|>2.1) ≈ 3.6%.
    // With 5000 vectors × 128 dimensions, we expect many hits at extremes.

    Quantizer q;
    q.init(128, 0xCAFEBABEULL);

    std::set<int> seen_indices;
    std::mt19937_64 rng(42ULL);
    std::normal_distribution<float> dist(0.f, 1.f);

    // 5000 vectors is enough to see all 16 levels
    std::vector<uint8_t> unpacked(128, 0);
    for (int vec = 0; vec < 5000 && seen_indices.size() < 16; ++vec) {
        std::vector<float> x(128);
        for (auto &v : x) v = dist(rng);
        QuantizedVec qv = q.quantize(x.data(), 4);
        unpack_indices(qv.data.data(), qv.dim, 4, unpacked.data());
        for (int i = 0; i < qv.dim; ++i)
            seen_indices.insert((int)unpacked[i]);
    }

    CAPTURE(seen_indices.size());
    // All 16 levels must appear
    REQUIRE((int)seen_indices.size() == 16);
}

// ---------------------------------------------------------------------------
// B. Reconstruction metrics — deterministic seeds
// ---------------------------------------------------------------------------

TEST_CASE("Reconstruction metrics: MSE/RMSE/cosine/rel-L2/max-abs (issue #229)", "[quantizer][reconstruction]") {
    // Tests reconstruction quality for each bit width.
    // Thresholds are based on theoretical Lloyd-Max MSE for N(0,1)
    // multiplied by a 5x margin to account for the FWHT transform pipeline.
    // If these tests fail, the codebook domain mismatch is not fully fixed.

    for (int head_dim : {64, 128}) {
        Quantizer q;
        q.init(head_dim, 0xABCD1234ULL ^ (uint64_t)head_dim);

        const int N_VECS = 200;
        std::mt19937_64 rng(777ULL);
        std::normal_distribution<float> dist(0.f, 1.f);

        for (int bits : {2, 3, 4}) {
            double total_mse = 0, total_cosine = 0, total_rel_l2 = 0;

            std::vector<float> x(head_dim), recon(head_dim);
            for (int vec = 0; vec < N_VECS; ++vec) {
                for (auto &v : x) v = dist(rng);
                QuantizedVec qv = q.quantize(x.data(), bits);
                q.dequantize(qv, recon.data());

                auto m = compute_metrics(x.data(), recon.data(), head_dim);
                total_mse    += m.mse;
                total_cosine += m.cosine;
                total_rel_l2 += m.rel_l2;
            }
            double avg_mse    = total_mse    / N_VECS;
            double avg_cosine = total_cosine / N_VECS;
            double avg_rel_l2 = total_rel_l2 / N_VECS;

            CAPTURE(head_dim, bits, avg_mse, avg_cosine, avg_rel_l2);

            // Thresholds: 5× theoretical Lloyd-Max MSE for N(0,1)
            // Theory: 4-bit~0.0120, 3-bit~0.0397, 2-bit~0.1175
            // (actual values are larger due to pipeline transform overhead)
            if (bits == 4) {
                REQUIRE(avg_mse    < 0.10);
                REQUIRE(avg_cosine > 0.90);
                REQUIRE(avg_rel_l2 < 0.40);
            } else if (bits == 3) {
                REQUIRE(avg_mse    < 0.20);
                REQUIRE(avg_cosine > 0.80);
                REQUIRE(avg_rel_l2 < 0.55);
            } else {  // 2-bit
                REQUIRE(avg_mse    < 0.50);
                REQUIRE(avg_cosine > 0.60);
                REQUIRE(avg_rel_l2 < 0.80);
            }
        }
    }
}

TEST_CASE("Reconstruction: single deterministic vector (issue #229)", "[quantizer][reconstruction]") {
    // Fixed-seed single-vector test for reproducibility.
    Quantizer q;
    q.init(128, 0x12345678ULL);

    auto x = gaussian_vec(128, 42ULL);

    for (int bits : {2, 3, 4}) {
        QuantizedVec qv = q.quantize(x.data(), bits);
        std::vector<float> recon(128, 0.f);
        q.dequantize(qv, recon.data());

        auto m = compute_metrics(x.data(), recon.data(), 128);

        CAPTURE(bits, m.mse, m.rmse, m.rel_l2, m.cosine, m.max_abs);

        // Basic sanity: reconstruction should be finite and non-trivial
        REQUIRE(std::isfinite(m.mse));
        REQUIRE(std::isfinite(m.cosine));
        REQUIRE(m.cosine > 0.0);  // output must be in the same direction

        // Reconstruction must be better than a zero vector (trivial baseline)
        // Zero vector cosine is 0; we need > 0 for any reasonable quantizer.
        REQUIRE(m.cosine > 0.5);  // must be clearly aligned
    }
}

// ---------------------------------------------------------------------------
// C. Dimension and padding overhead
// ---------------------------------------------------------------------------

struct PaddingInfo {
    int dim;
    int padded;
    int payload_bits;     // actual coded bits (padded × bits_per_value)
    int actual_bits;      // dim × bits_per_value (no padding)
    double overhead_pct;  // (padded - dim) / dim * 100
};

static PaddingInfo measure_padding(int dim, int bits) {
    PaddingInfo p;
    p.dim = dim;
    p.padded = 1;
    while (p.padded < dim) p.padded <<= 1;
    p.payload_bits = p.padded * bits;
    p.actual_bits  = dim * bits;
    p.overhead_pct = (p.padded > dim)
        ? (100.0 * (p.padded - dim) / dim)
        : 0.0;
    return p;
}

TEST_CASE("Dimension and padding overhead reporting (issue #229)", "[quantizer][padding]") {
    // This test documents padding overhead for each common head dimension.
    // It is an informational test — it WILL NOT fail unless padding is
    // catastrophically wrong (e.g., more than 2x the actual data).

    const int test_dims[] = {64, 96, 128, 192, 256, 384, 512};

    for (int dim : test_dims) {
        for (int bits : {2, 3, 4}) {
            auto p = measure_padding(dim, bits);
            CAPTURE(dim, bits, p.padded, p.overhead_pct);

            // Padding must never more than double the dimension
            REQUIRE(p.padded <= dim * 2);

            // Quantizer must accept this dimension
            Quantizer q;
            if (dim <= 512) {  // within TL buffer limit
                q.init(dim, 42ULL);
                auto x = gaussian_vec(dim, 99ULL);
                QuantizedVec qv = q.quantize(x.data(), bits);
                REQUIRE(qv.dim == p.padded);
                REQUIRE((int)qv.data.size() == (p.padded * bits + 7) / 8);
            }
        }
    }

    // Explicitly test the d=96 padding case mentioned in the spec
    {
        auto p = measure_padding(96, 4);
        CAPTURE("d=96 pads to", p.padded, "overhead%", p.overhead_pct);
        // d=96 should pad to 128 (next power of 2)
        REQUIRE(p.padded == 128);
        // Overhead: (128-96)/96 = 33.3%
        REQUIRE(p.overhead_pct > 30.0);
        REQUIRE(p.overhead_pct < 40.0);
    }
}

TEST_CASE("Quantizer round-trip for all required head dimensions", "[quantizer][dimensions]") {
    // Tests 64, 96, 128, 192, 256, 384, 512 — all dimensions from the spec.
    // For each, verifies pack/unpack round-trip and reconstruction is sane.

    const int test_dims[] = {64, 96, 128, 192, 256, 384, 512};

    for (int dim : test_dims) {
        Quantizer q;
        q.init(dim, (uint64_t)dim * 0xABCDEFULL);

        auto x = gaussian_vec(dim, (uint64_t)dim * 42ULL);

        for (int bits : {2, 3, 4}) {
            QuantizedVec qv = q.quantize(x.data(), bits);
            std::vector<float> recon(dim, 0.f);
            q.dequantize(qv, recon.data());

            auto m = compute_metrics(x.data(), recon.data(), dim);

            CAPTURE(dim, bits, m.mse, m.cosine);
            REQUIRE(std::isfinite(m.mse));
            REQUIRE(std::isfinite(m.cosine));
            REQUIRE(m.cosine > 0.0);  // not completely orthogonal
        }
    }
}

// ---------------------------------------------------------------------------
// D. FWHT correctness (supplementary to test_fwht.cpp)
// ---------------------------------------------------------------------------

TEST_CASE("FWHT round-trip MSE for dims 64,96,128,192,256,384,512", "[fwht][dimensions]") {
    // Explicitly tests round-trip at all spec dimensions.
    // The non-power-of-two dims (96, 192, 384) use the padding path.

    for (int dim : {64, 96, 128, 192, 256, 384, 512}) {
        int padded = 1;
        while (padded < dim) padded <<= 1;

        std::vector<int8_t> D(padded);
        gen_rademacher(D.data(), padded, (uint64_t)dim * 12345ULL);

        auto orig = gaussian_vec(padded, (uint64_t)dim * 999ULL);
        std::vector<float> x = orig;

        fwht_forward(x.data(), D.data(), padded);
        fwht_inverse(x.data(), D.data(), padded);

        double mse = 0.0;
        for (int i = 0; i < padded; ++i) {
            double d = (double)x[i] - (double)orig[i];
            mse += d * d;
        }
        mse /= padded;

        CAPTURE(dim, padded, mse);
        REQUIRE(mse < 1e-10);
    }
}

TEST_CASE("FWHT output is approximately N(0,1/p) for L2-normalised input", "[fwht][distribution]") {
    // After fwht_forward, values are (1/√p)·H·D·x.
    // If x is L2-normalised (||x||=1), then after scaling by √p,
    // the values should be approximately N(0,1) by CLT.
    // This test validates the distribution assumption made by the quantizer.

    int padded = 128;
    std::vector<int8_t> D(padded);
    gen_rademacher(D.data(), padded, 0xDEADBEEFULL);

    const int N = 10000;
    double sum = 0, sum2 = 0;
    int total = 0;
    std::mt19937_64 rng(42ULL);
    std::normal_distribution<float> dist(0.f, 1.f);

    for (int trial = 0; trial < N; ++trial) {
        std::vector<float> x(padded);
        double norm2 = 0;
        for (auto &v : x) { v = dist(rng); norm2 += (double)v * (double)v; }
        // L2-normalise
        float inv = (float)(1.0 / std::sqrt(norm2 + 1e-12));
        for (auto &v : x) v *= inv;

        fwht_forward(x.data(), D.data(), padded);
        float sp = sqrtf((float)padded);
        for (int i = 0; i < padded; ++i) {
            double val = (double)x[i] * sp;
            sum  += val;
            sum2 += val * val;
            ++total;
        }
    }

    double mean = sum / total;
    double var  = sum2 / total - mean * mean;

    CAPTURE(mean, var);
    // Mean should be near 0
    REQUIRE(std::abs(mean) < 0.05);
    // Variance should be near 1.0 (N(0,1))
    REQUIRE(var > 0.80);
    REQUIRE(var < 1.20);
}

// ---------------------------------------------------------------------------
// E. Codebook symmetry (also in test_codebook.cpp but repeated here for
//    completeness of the #229 test suite)
// ---------------------------------------------------------------------------

TEST_CASE("All codebooks are symmetric about zero (issue #229 invariant)", "[codebook][symmetry]") {
    const float eps = 1e-4f;

    for (int i = 0; i < 4; ++i)
        REQUIRE(std::abs(CB2[i] + CB2[3 - i]) < eps);

    for (int i = 0; i < 8; ++i)
        REQUIRE(std::abs(CB3[i] + CB3[7 - i]) < eps);

    for (int i = 0; i < 16; ++i)
        REQUIRE(std::abs(CB4[i] + CB4[15 - i]) < eps);
}

TEST_CASE("CB4 centroids are in N(0,1) domain (issue #229 regression)", "[codebook][domain]") {
    // The extreme centroids of CB4 must be within typical N(0,1) range.
    // The old CB4[15]=3.5714 was outside the expected range for a 4-bit
    // Lloyd-Max codebook for N(0,1). The new value should be ≤ 2.8.
    REQUIRE(CB4[15] < 2.8f);   // correct: 2.4008
    REQUIRE(CB4[0]  > -2.8f);  // correct: -2.4008
    // And it must be symmetric
    REQUIRE(std::abs(CB4[0] + CB4[15]) < 1e-4f);
}

TEST_CASE("Quantizer scale is strictly positive and finite", "[quantizer][scale]") {
    // After issue #229 fix, scale = ||x|| (L2 norm of input).
    // It must always be positive and finite for non-zero input.
    Quantizer q;
    q.init(128, 42ULL);

    auto x = gaussian_vec(128, 1ULL);
    for (int bits : {2, 3, 4}) {
        QuantizedVec qv = q.quantize(x.data(), bits);
        CAPTURE(bits, qv.scale);
        REQUIRE(std::isfinite(qv.scale));
        REQUIRE(qv.scale > 0.f);
        // The scale should equal ||x||
        double expected_norm = 0;
        for (auto v : x) expected_norm += (double)v * (double)v;
        expected_norm = std::sqrt(expected_norm);
        // Allow 0.1% tolerance for the epsilon in sqrtf
        REQUIRE(std::abs(qv.scale - (float)expected_norm) < (float)expected_norm * 0.001f + 1e-6f);
    }
}

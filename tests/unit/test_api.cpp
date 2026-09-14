/* Catch2 v3 — link against Catch2::Catch2WithMain */
#include <catch2/catch_test_macros.hpp>
#include "../../include/adaptq.h"
#include <cstring>
#include <cmath>
#include <string>

/* -------------------------------------------------------------------------
 * tests/unit/test_api.cpp
 * C ABI smoke tests — covers every public function in adaptq.h.
 * ----------------------------------------------------------------------- */

static void fill_vec(float *v, int n, float val) {
    for (int i = 0; i < n; ++i) v[i] = val;
}

/* ---- Single head -------------------------------------------------------- */

TEST_CASE("adaptq_create returns non-null handle", "[api]") {
    adaptq_ctx_t h = adaptq_create(128, 4, 1024, 42, 0.f, 0);
    REQUIRE(h != nullptr);
    adaptq_destroy(h);
}

TEST_CASE("adaptq_last_error is empty after successful create", "[api]") {
    adaptq_ctx_t h = adaptq_create(64, 4, 512, 1, 0.f, 0);
    REQUIRE(h != nullptr);
    REQUIRE(std::string(adaptq_last_error()).empty());
    adaptq_destroy(h);
}

TEST_CASE("adaptq_kv_bytes is 0 before any append", "[api]") {
    adaptq_ctx_t h = adaptq_create(128, 4, 1024, 1, 0.f, 0);
    REQUIRE(adaptq_kv_bytes(h) == 0);
    adaptq_destroy(h);
}

TEST_CASE("adaptq_append increases kv_bytes", "[api]") {
    adaptq_ctx_t h = adaptq_create(128, 4, 1024, 1, 0.f, 0);
    float k[128] = {}, v[128] = {};
    fill_vec(k, 128, 0.5f); fill_vec(v, 128, 0.3f);
    adaptq_append(h, k, v, 0);
    REQUIRE(adaptq_kv_bytes(h) > 0);
    adaptq_destroy(h);
}

TEST_CASE("adaptq_compute returns active token count", "[api]") {
    adaptq_ctx_t h = adaptq_create(128, 4, 1024, 1, 0.f, 0);
    float k[128], v[128], q[128], out[128];
    fill_vec(k, 128, 0.1f); fill_vec(v, 128, 0.2f); fill_vec(q, 128, 0.5f);
    adaptq_append(h, k, v, 0);
    adaptq_append(h, k, v, 1);
    int n = adaptq_compute(h, q, out);
    REQUIRE(n == 2);
    adaptq_destroy(h);
}

TEST_CASE("adaptq_compute output is non-zero after append", "[api]") {
    adaptq_ctx_t h = adaptq_create(128, 4, 1024, 1, 0.f, 0);
    float k[128], v[128], q[128], out[128];
    for (int i = 0; i < 128; ++i) {
        k[i] = (float)i / 128.f;
        v[i] = 1.f - k[i];
        q[i] = k[i];
    }
    adaptq_append(h, k, v, 0);
    adaptq_compute(h, q, out);
    double norm = 0.0;
    for (int i = 0; i < 128; ++i) norm += (double)out[i] * out[i];
    REQUIRE(norm > 1e-10);
    adaptq_destroy(h);
}

TEST_CASE("adaptq_compute on empty cache returns 0", "[api]") {
    adaptq_ctx_t h = adaptq_create(128, 4, 1024, 1, 0.f, 0);
    float q[128] = {}, out[128];
    int n = adaptq_compute(h, q, out);
    REQUIRE(n == 0);
    adaptq_destroy(h);
}

TEST_CASE("adaptq_reset zeroes cache", "[api]") {
    adaptq_ctx_t h = adaptq_create(128, 4, 1024, 1, 0.f, 0);
    float k[128], v[128];
    fill_vec(k, 128, 1.f); fill_vec(v, 128, 1.f);
    adaptq_append(h, k, v, 0);
    REQUIRE(adaptq_kv_bytes(h) > 0);
    adaptq_reset(h);
    REQUIRE(adaptq_kv_bytes(h) == 0);
    adaptq_destroy(h);
}

TEST_CASE("adaptq_create with invalid parameters returns null", "[api][security]") {
    REQUIRE(adaptq_create(0, 4, 1024, 42, 0.f, 0) == nullptr);
    REQUIRE(adaptq_create(-1, 4, 1024, 42, 0.f, 0) == nullptr);
    REQUIRE(adaptq_create(128, 0, 1024, 42, 0.f, 0) == nullptr);
    REQUIRE(adaptq_create(128, 4, -1, 42, 0.f, 0) == nullptr);
    REQUIRE(adaptq_create(128, 4, 1024, 42, 0.f, -100) == nullptr);
    REQUIRE(std::string(adaptq_last_error()).size() > 0);
}

/* ---- Multi-head -------------------------------------------------------- */

TEST_CASE("adaptq_mha_create / destroy", "[api][mha]") {
    adaptq_mha_t mha = adaptq_mha_create(8, 64, 4, 512, 0, 0.f, 0);
    REQUIRE(mha != nullptr);
    adaptq_mha_destroy(mha);
}

TEST_CASE("adaptq_mha_append + compute basic contract", "[api][mha]") {
    adaptq_mha_t mha = adaptq_mha_create(4, 128, 4, 256, 7, 0.f, 0);
    float k[128], v[128], q[128], out[128];
    fill_vec(k, 128, 0.3f); fill_vec(v, 128, 0.7f); fill_vec(q, 128, 0.5f);
    for (int hi = 0; hi < 4; ++hi) adaptq_mha_append(mha, hi, k, v, 0);
    for (int hi = 0; hi < 4; ++hi) {
        int n = adaptq_mha_compute(mha, hi, q, out);
        REQUIRE(n == 1);
    }
    adaptq_mha_destroy(mha);
}

TEST_CASE("adaptq_mha_append: out-of-range head_idx sets error", "[api][mha]") {
    adaptq_mha_t mha = adaptq_mha_create(4, 64, 4, 128, 0, 0.f, 0);
    float k[64], v[64];
    fill_vec(k, 64, 1.f); fill_vec(v, 64, 1.f);
    adaptq_mha_append(mha, 99, k, v, 0);
    REQUIRE(std::string(adaptq_last_error()).size() > 0);
    adaptq_mha_destroy(mha);
}

TEST_CASE("adaptq_mha_compute: out-of-range head_idx returns -1", "[api][mha]") {
    adaptq_mha_t mha = adaptq_mha_create(4, 64, 4, 128, 0, 0.f, 0);
    float q[64], out[64];
    int n = adaptq_mha_compute(mha, -1, q, out);
    REQUIRE(n == -1);
    REQUIRE(std::string(adaptq_last_error()).size() > 0);
    adaptq_mha_destroy(mha);
}

TEST_CASE("adaptq_mha_reset clears all heads", "[api][mha]") {
    adaptq_mha_t mha = adaptq_mha_create(4, 64, 4, 128, 0, 0.f, 0);
    float k[64], v[64];
    fill_vec(k, 64, 1.f); fill_vec(v, 64, 1.f);
    for (int hi = 0; hi < 4; ++hi) adaptq_mha_append(mha, hi, k, v, 0);
    REQUIRE(adaptq_mha_total_kv_bytes(mha) > 0);
    adaptq_mha_reset(mha);
    REQUIRE(adaptq_mha_total_kv_bytes(mha) == 0);
    adaptq_mha_destroy(mha);
}

TEST_CASE("adaptq_mha_create with invalid parameters returns null", "[api][mha][security]") {
    REQUIRE(adaptq_mha_create(0, 128, 4, 1024, 0, 0.f, 0) == nullptr);
    REQUIRE(adaptq_mha_create(-1, 128, 4, 1024, 0, 0.f, 0) == nullptr);
    REQUIRE(adaptq_mha_create(4, 0, 4, 1024, 0, 0.f, 0) == nullptr);
    REQUIRE(adaptq_mha_create(4, 128, -1, 1024, 0, 0.f, 0) == nullptr);
    REQUIRE(std::string(adaptq_last_error()).size() > 0);
}

TEST_CASE("adaptq_mha_append: exact upper bound head_idx sets error", "[api][mha][security]") {
    adaptq_mha_t mha = adaptq_mha_create(4, 64, 4, 128, 0, 0.f, 0);
    float k[64], v[64];
    adaptq_mha_append(mha, 4, k, v, 0); // 4 is out of bounds for size 4
    REQUIRE(std::string(adaptq_last_error()).size() > 0);
    adaptq_mha_destroy(mha);
}

/* ---- Feature flags + version ------------------------------------------ */

TEST_CASE("adaptq_version returns non-empty string", "[api]") {
    const char *ver = adaptq_version();
    REQUIRE(ver != nullptr);
    REQUIRE(std::string(ver).size() > 0);
}

TEST_CASE("adaptq_features returns valid bitmask", "[api]") {
    unsigned f = adaptq_features();
    REQUIRE((f & ADAPTQ_FEAT_HYBRID)   != 0);
    REQUIRE((f & ADAPTQ_FEAT_SPARSE_V) != 0);
}

/* ---- Context Limits (Issue #22) --------------------------------------- */

TEST_CASE("adaptq context handles sizes around and above 65536 tokens", "[api][limits]") {
    // We test 65535, 65536, 65537, and 100000.
    // We create a single cache with capacity 100005 to cover everything.
    adaptq_ctx_t h = adaptq_create(64, 4, 100005, 42, 0.f, 0);
    REQUIRE(h != nullptr);

    float k[64] = {}, v[64] = {}, q[64] = {}, out[64];
    fill_vec(k, 64, 0.1f);
    fill_vec(v, 64, 0.1f);
    fill_vec(q, 64, 0.1f);

    int test_sizes[] = { 65535, 65536, 65537, 100000 };
    int current_size = 0;

    for (int target : test_sizes) {
        // Append tokens until we reach the target size
        while (current_size < target) {
            adaptq_append(h, k, v, current_size);
            current_size++;
        }
        
        // Compute should not assert/crash and return exactly the target size
        int n = adaptq_compute(h, q, out);
        REQUIRE(n == target);
    }

    adaptq_destroy(h);
}

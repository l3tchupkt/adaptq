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
    REQUIRE(adaptq_create(128, 1, 1024, 42, 0.f, 0) == nullptr);
    REQUIRE(adaptq_create(128, 5, 1024, 42, 0.f, 0) == nullptr);
    REQUIRE(adaptq_create(128, 4, -1, 42, 0.f, 0) == nullptr);
    REQUIRE(adaptq_create(128, 4, 1024, 42, 0.f, -100) == nullptr);
    REQUIRE(std::string(adaptq_last_error()).size() > 0);
}

TEST_CASE("adaptq_create rejects unsupported quantization bit widths", "[api][security]") {
    for (int bits : {1, 5, 6, 8, 16}) {
        REQUIRE(adaptq_create(128, bits, 1024, 42, 0.f, 0) == nullptr);
        REQUIRE(std::string(adaptq_last_error()).size() > 0);
    }
}

TEST_CASE("adaptq_create accepts all supported quantization bit widths", "[api]") {
    for (int bits : {2, 3, 4}) {
        adaptq_ctx_t h = adaptq_create(128, bits, 1024, 42, 0.f, 0);
        REQUIRE(h != nullptr);
        adaptq_destroy(h);
    }
}

TEST_CASE("single-head API rejects null handles and buffers without crashing", "[api][security]") {
    float k[128] = {}, v[128] = {}, q[128] = {}, out[128] = {};

    REQUIRE(adaptq_compute(nullptr, q, out) == -1);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_compute: null handle") != std::string::npos);

    REQUIRE(adaptq_compute(nullptr, nullptr, nullptr) == -1);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_compute: null handle") != std::string::npos);

    adaptq_ctx_t h = adaptq_create(128, 4, 16, 1, 0.f, 0);
    REQUIRE(h != nullptr);

    adaptq_append(h, nullptr, v, 0);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_append: null key") != std::string::npos);

    adaptq_append(h, k, nullptr, 0);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_append: null val") != std::string::npos);

    REQUIRE(adaptq_compute(h, nullptr, out) == -1);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_compute: null query") != std::string::npos);

    REQUIRE(adaptq_compute(h, q, nullptr) == -1);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_compute: null out") != std::string::npos);

    REQUIRE(adaptq_compute_batch(h, nullptr, 1, out) == -1);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_compute_batch: null queries") != std::string::npos);

    REQUIRE(adaptq_compute_batch(h, q, 1, nullptr) == -1);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_compute_batch: null outs") != std::string::npos);

    REQUIRE(adaptq_compute_batch(h, nullptr, 0, nullptr) == 0);

    adaptq_reset(nullptr);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_reset: null handle") != std::string::npos);
    REQUIRE(adaptq_kv_bytes(nullptr) == 0);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_kv_bytes: null handle") != std::string::npos);

    adaptq_destroy(nullptr);
    adaptq_destroy(h);
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

TEST_CASE("adaptq_mha_create rejects unsupported quantization bit widths", "[api][mha][security]") {
    for (int bits : {1, 5, 6, 8, 16}) {
        REQUIRE(adaptq_mha_create(4, 128, bits, 1024, 0, 0.f, 0) == nullptr);
        REQUIRE(std::string(adaptq_last_error()).size() > 0);
    }
}

TEST_CASE("adaptq_mha_create accepts all supported quantization bit widths", "[api][mha]") {
    for (int bits : {2, 3, 4}) {
        adaptq_mha_t mha = adaptq_mha_create(4, 128, bits, 1024, 0, 0.f, 0);
        REQUIRE(mha != nullptr);
        adaptq_mha_destroy(mha);
    }
}

TEST_CASE("adaptq_mha_append: exact upper bound head_idx sets error", "[api][mha][security]") {
    adaptq_mha_t mha = adaptq_mha_create(4, 64, 4, 128, 0, 0.f, 0);
    float k[64], v[64];
    adaptq_mha_append(mha, 4, k, v, 0); // 4 is out of bounds for size 4
    REQUIRE(std::string(adaptq_last_error()).size() > 0);
    adaptq_mha_destroy(mha);
}

TEST_CASE("multi-head API rejects null handles and buffers without crashing", "[api][mha][security]") {
    float k[64] = {}, v[64] = {}, q[64] = {}, out[64] = {};

    REQUIRE(adaptq_mha_compute(nullptr, 0, q, out) == -1);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_mha_compute: null handle") != std::string::npos);

    adaptq_mha_append(nullptr, 0, k, v, 0);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_mha_append: null handle") != std::string::npos);

    REQUIRE(adaptq_mha_compute_batch(nullptr, 0, q, 1, out) == -1);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_mha_compute_batch: null handle") != std::string::npos);

    adaptq_mha_reset(nullptr);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_mha_reset: null handle") != std::string::npos);

    REQUIRE(adaptq_mha_total_kv_bytes(nullptr) == 0);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_mha_total_kv_bytes: null handle") != std::string::npos);

    adaptq_mha_destroy(nullptr);

    adaptq_mha_t mha = adaptq_mha_create(2, 64, 4, 4, 0, 0.f, 0);
    INFO(adaptq_last_error());
    REQUIRE(mha != nullptr);

    adaptq_mha_append(mha, 0, nullptr, v, 0);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_mha_append: null key") != std::string::npos);

    adaptq_mha_append(mha, 0, k, nullptr, 0);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_mha_append: null val") != std::string::npos);

    REQUIRE(adaptq_mha_compute(mha, 0, nullptr, out) == -1);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_mha_compute: null query") != std::string::npos);

    REQUIRE(adaptq_mha_compute(mha, 0, q, nullptr) == -1);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_mha_compute: null out") != std::string::npos);

    REQUIRE(adaptq_mha_compute_batch(mha, 0, nullptr, 1, out) == -1);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_mha_compute_batch: null queries") != std::string::npos);

    REQUIRE(adaptq_mha_compute_batch(mha, 0, q, 1, nullptr) == -1);
    REQUIRE(std::string(adaptq_last_error()).find("adaptq_mha_compute_batch: null outs") != std::string::npos);

    REQUIRE(adaptq_mha_compute_batch(mha, 0, nullptr, 0, nullptr) == 0);

    adaptq_mha_destroy(mha);
}

/* ---- Feature flags + version ------------------------------------------ */

TEST_CASE("adaptq_version returns non-empty string", "[api]") {
    const char *ver = adaptq_version();
    REQUIRE(ver != nullptr);
    REQUIRE(std::string(ver).size() > 0);
}


TEST_CASE("adaptq AVX2 feature matches reported version", "[api]") {
    const unsigned f = adaptq_features();
    const std::string version = adaptq_version();
#ifdef ADAPTQ_HAS_AVX2_BACKEND
    REQUIRE((f & ADAPTQ_FEAT_AVX2) != 0);
    REQUIRE(version == "3.2.0-avx2");
#else
    REQUIRE((f & ADAPTQ_FEAT_AVX2) == 0);
    REQUIRE(version == "3.2.0-scalar");
#endif
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

/* ---- Boundary Conditions (Issue #16) ---------------------------------- */

TEST_CASE("Max-Lloyd codebooks handle boundary conditions and outlier vectors without crashing", "[api][security][boundary]") {
    adaptq_ctx_t h = adaptq_create(64, 4, 128, 42, 0.f, 0);
    REQUIRE(h != nullptr);

    float q[64] = {}, out[64] = {};
    float k_zeros[64] = {}, v_zeros[64] = {};
    float k_nans[64], v_nans[64];
    float k_huge[64], v_huge[64];

    for (int i = 0; i < 64; ++i) {
        k_nans[i] = std::nanf("");
        v_nans[i] = std::nanf("");
        k_huge[i] = 1e38f;
        v_huge[i] = -1e38f;
    }

    adaptq_append(h, k_zeros, v_zeros, 0);
    adaptq_append(h, k_nans, v_nans, 1);
    adaptq_append(h, k_huge, v_huge, 2);

    int n = adaptq_compute(h, q, out);
    REQUIRE(n == 3);

    adaptq_destroy(h);
}

/* ---- Flat Buffer & Aligned Allocator Safety (Issue #52) --------------- */
#include "../../include/ring_buffer.h"

TEST_CASE("AlignedAllocator throws std::bad_alloc on huge allocation request", "[allocator][safety]") {
    AlignedAllocator<uint8_t, 64> alloc;
    REQUIRE_THROWS_AS(alloc.allocate(size_t(-1) / 2), std::bad_alloc);
}

TEST_CASE("KVFlatBuffer handles initialization, insert, and cleanup safely", "[cache][flatbuffer]") {
    KVFlatBuffer buf;
    buf.init(16, 64, 4);
    REQUIRE(buf.capacity == 16);
    REQUIRE(buf.size == 0);
    REQUIRE(buf.k_data != nullptr);
    REQUIRE(buf.v_data != nullptr);

    std::vector<uint8_t> dummy_k(buf.packed_bytes, 0x12);
    std::vector<uint8_t> dummy_v(buf.packed_bytes, 0x34);

    int idx = buf.insert(dummy_k.data(), 1.0f, dummy_v.data(), 2.0f, 0);
    REQUIRE(idx == 0);
    REQUIRE(buf.size == 1);
    REQUIRE(buf.k_ptr(0)[0] == 0x12);
    REQUIRE(buf.v_ptr(0)[0] == 0x34);
    REQUIRE(buf.kv_bytes() == (size_t)buf.packed_bytes * 2);

    buf.free_aligned();
    REQUIRE(buf.k_data == nullptr);
    REQUIRE(buf.v_data == nullptr);
}

/* ---- Softmax Numerical Stability Tests (Issue #58) -------------------- */
#include "../../include/attention.h"

TEST_CASE("softmax handles empty, null, single element, and zero-sum safely", "[attention][softmax][stability]") {
    // Null and empty
    softmax(nullptr, 10);
    float dummy = 5.0f;
    softmax(&dummy, 0);
    softmax(&dummy, -1);

    // Single element
    float single[1] = { 42.0f };
    softmax(single, 1);
    REQUIRE(single[0] == 1.0f);

    // Normal multi-element
    float arr[3] = { 1.0f, 2.0f, 3.0f };
    softmax(arr, 3);
    float sum = arr[0] + arr[1] + arr[2];
    REQUIRE(std::abs(sum - 1.0f) < 1e-5f);
    REQUIRE(arr[2] > arr[1]);
    REQUIRE(arr[1] > arr[0]);

    // Extreme negative logits resulting in zero-sum underflow
    float extreme[3] = { -1e30f, -1e30f, -1e30f };
    softmax(extreme, 3);
    // Should fall back to uniform distribution
    for (int i = 0; i < 3; ++i) {
        REQUIRE(std::abs(extreme[i] - (1.0f / 3.0f)) < 1e-5f);
    }
}

TEST_CASE("hybrid path invalidates raw_kv on ring buffer eviction to prevent stale tokens", "[attention][hybrid][eviction]") {
    // capacity = 4, hybrid_thresh = 16 (capacity < hybrid_thresh)
    adaptq_ctx_t h = adaptq_create(64, 4, 4, 42, 0.f, 16);
    REQUIRE(h != nullptr);

    float k[64] = {}, v_old[64] = {}, v_new[64] = {}, q[64] = {}, out[64] = {};
    fill_vec(k, 64, 1.0f);
    fill_vec(q, 64, 1.0f);
    fill_vec(v_old, 64, 1.0f);
    fill_vec(v_new, 64, 10.0f);

    // Fill capacity (4 tokens with v = 1.0)
    for (int i = 0; i < 4; ++i) {
        adaptq_append(h, k, v_old, i);
    }
    int n1 = adaptq_compute(h, q, out);
    REQUIRE(n1 == 4);

    // Overwrite all 4 tokens with v = 10.0 (triggers circular ring buffer eviction)
    for (int i = 4; i < 8; ++i) {
        adaptq_append(h, k, v_new, i);
    }
    int n2 = adaptq_compute(h, q, out);
    REQUIRE(n2 == 4);
    // Output should now reflect v_new (approx 10.0), NOT stale v_old (approx 1.0)
    REQUIRE(out[0] > 5.0f);

    adaptq_destroy(h);
}


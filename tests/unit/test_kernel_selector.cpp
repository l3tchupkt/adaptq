/* Catch2 v3 — link against Catch2::Catch2WithMain */
#include <catch2/catch_test_macros.hpp>
#include "../../include/adaptq/kernel.h"
#include "../../include/fwht.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* -------------------------------------------------------------------------
 * tests/unit/test_kernel_selector.cpp
 *
 * Issue #7: Automated AVX2 capability detection, XCR0 OS validation,
 * portable scalar fallback, and environment override testing.
 * ----------------------------------------------------------------------- */

namespace adaptq {
// Backend factory declarations
IKernelBackend *create_scalar_backend();
IKernelBackend *create_avx2_backend();
}

#if defined(_WIN32)
static void set_env_var(const char *name, const char *val) {
    _putenv_s(name, val);
}
static void unset_env_var(const char *name) {
    _putenv_s(name, "");
}
#else
static void set_env_var(const char *name, const char *val) {
    setenv(name, val, 1);
}
static void unset_env_var(const char *name) {
    unsetenv(name);
}
#endif

TEST_CASE("AVX2 Capability Detection and OS XCR0 State", "[kernel][avx2]") {
    bool has_avx2 = adaptq::cpu_supports_avx2();
    // cpu_supports_avx2 returns a clean boolean without illegal-instruction faults
    INFO("Host CPU + OS AVX2 support status: " << (has_avx2 ? "ENABLED" : "DISABLED/FALLBACK"));
    SUCCEED("cpu_supports_avx2 evaluated cleanly");
}

TEST_CASE("AVX2 Backend Factory Contract", "[kernel][avx2]") {
    adaptq::IKernelBackend *backend = adaptq::create_avx2_backend();

    if (backend) {
        REQUIRE(backend->is_available());
        REQUIRE(std::string(backend->name()) == "avx2");
    } else {
        SUCCEED("AVX2 backend is unavailable in this portable build");
    }
}

TEST_CASE("Scalar Backend Factory and Interface Conformance", "[kernel][scalar]") {
    adaptq::IKernelBackend *scalar_backend = adaptq::create_scalar_backend();
    REQUIRE(scalar_backend != nullptr);
    REQUIRE(scalar_backend->is_available() == true);
    REQUIRE(std::string(scalar_backend->name()) == "scalar");
}

TEST_CASE("AVX2 FWHT matches scalar backend", "[kernel][avx2][fwht]") {
    adaptq::IKernelBackend *scalar = adaptq::create_scalar_backend();
    adaptq::IKernelBackend *avx2 = adaptq::create_avx2_backend();
    REQUIRE(scalar != nullptr);
    if (!avx2) {
        SUCCEED("AVX2 backend is unavailable in this build");
        return;
    }

    const int sizes[] = {1, 2, 4, 8, 16, 32, 64};
    for (int n : sizes) {
        std::vector<float> input(n);
        for (int i = 0; i < n; ++i)
            input[i] = static_cast<float>((i * 17) % 23 - 11) / 7.f;
        std::vector<int8_t> d(n);
        gen_rademacher(d.data(), n, static_cast<uint64_t>(1234 + n));

        std::vector<float> scalar_forward = input;
        std::vector<float> avx_forward = input;
        scalar->fwht_forward(scalar_forward.data(), d.data(), n);
        avx2->fwht_forward(avx_forward.data(), d.data(), n);
        for (int i = 0; i < n; ++i)
            REQUIRE(std::abs(scalar_forward[i] - avx_forward[i]) <= 1e-5f);

        std::vector<float> scalar_inverse = input;
        std::vector<float> avx_inverse = input;
        scalar->fwht_inverse(scalar_inverse.data(), d.data(), n);
        avx2->fwht_inverse(avx_inverse.data(), d.data(), n);
        for (int i = 0; i < n; ++i)
            REQUIRE(std::abs(scalar_inverse[i] - avx_inverse[i]) <= 1e-5f);
    }
}

TEST_CASE("Runtime Environment Variable Fallback Override", "[kernel][fallback]") {
    // Ensure clean initial state
    unset_env_var("ADAPTQ_DISABLE_AVX2");
    unset_env_var("ADAPTQ_FORCE_SCALAR");
    unset_env_var("ADAPTQ_BACKEND");

    REQUIRE(adaptq::is_scalar_forced() == false);

    SECTION("Override via ADAPTQ_DISABLE_AVX2=1") {
        set_env_var("ADAPTQ_DISABLE_AVX2", "1");
        REQUIRE(adaptq::is_scalar_forced() == true);
        adaptq::IKernelBackend *backend = adaptq::select_kernel_backend();
        REQUIRE(backend != nullptr);
        REQUIRE(std::string(backend->name()) == "scalar");
        unset_env_var("ADAPTQ_DISABLE_AVX2");
    }

    SECTION("Override via ADAPTQ_FORCE_SCALAR=1") {
        set_env_var("ADAPTQ_FORCE_SCALAR", "1");
        REQUIRE(adaptq::is_scalar_forced() == true);
        adaptq::IKernelBackend *backend = adaptq::select_kernel_backend();
        REQUIRE(backend != nullptr);
        REQUIRE(std::string(backend->name()) == "scalar");
        unset_env_var("ADAPTQ_FORCE_SCALAR");
    }

    SECTION("Override via ADAPTQ_BACKEND=scalar") {
        set_env_var("ADAPTQ_BACKEND", "scalar");
        REQUIRE(adaptq::is_scalar_forced() == true);
        adaptq::IKernelBackend *backend = adaptq::select_kernel_backend();
        REQUIRE(backend != nullptr);
        REQUIRE(std::string(backend->name()) == "scalar");
        unset_env_var("ADAPTQ_BACKEND");
    }

    // Restore clean state
    unset_env_var("ADAPTQ_DISABLE_AVX2");
    unset_env_var("ADAPTQ_FORCE_SCALAR");
    unset_env_var("ADAPTQ_BACKEND");
}

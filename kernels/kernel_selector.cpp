#include "../include/adaptq/kernel.h"
#include <cstdlib>
#include <cstring>

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#define ADAPTQ_X86_OR_X64 1
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#include <x86intrin.h>
#endif
#else
#define ADAPTQ_X86_OR_X64 0
#endif

namespace adaptq {

IKernelBackend *create_scalar_backend();
IKernelBackend *create_avx2_backend();

bool is_scalar_forced() {
    const char *disable_avx2 = std::getenv("ADAPTQ_DISABLE_AVX2");
    if (disable_avx2 && (std::strcmp(disable_avx2, "1") == 0 ||
                         std::strcmp(disable_avx2, "true") == 0 ||
                         std::strcmp(disable_avx2, "TRUE") == 0)) {
        return true;
    }
    const char *force_scalar = std::getenv("ADAPTQ_FORCE_SCALAR");
    if (force_scalar && (std::strcmp(force_scalar, "1") == 0 ||
                         std::strcmp(force_scalar, "true") == 0 ||
                         std::strcmp(force_scalar, "TRUE") == 0)) {
        return true;
    }
    const char *backend = std::getenv("ADAPTQ_BACKEND");
    if (backend && (std::strcmp(backend, "scalar") == 0 ||
                    std::strcmp(backend, "SCALAR") == 0)) {
        return true;
    }
    return false;
}

bool cpu_supports_avx2() {
#if ADAPTQ_X86_OR_X64
#if defined(_MSC_VER)
    int regs[4] = {};
    __cpuid(regs, 0);
    if (regs[0] < 7)
        return false;

    __cpuidex(regs, 1, 0);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    if (!osxsave || !avx)
        return false;

    // Verify OS has enabled both XMM (bit 1) and YMM (bit 2) state saving via XCR0
    unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6ULL) != 0x6ULL)
        return false;

    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    // First, verify basic AVX2 CPUID support
    if (!__builtin_cpu_supports("avx2"))
        return false;

    // Query CPUID leaf 1 to ensure OSXSAVE is present before calling xgetbv
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        return false;

    const bool osxsave = (ecx & (1U << 27)) != 0;
    if (!osxsave)
        return false;

    // Verify OS manages XMM and YMM state saving via XCR0
    uint32_t xcr0_lo = 0, xcr0_hi = 0;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    uint64_t xcr0 = ((uint64_t)xcr0_hi << 32) | xcr0_lo;
    if ((xcr0 & 0x6ULL) != 0x6ULL)
        return false;

    return true;
#else
    return false;
#endif
#else
    // Non-x86 architectures (e.g. ARM/AArch64, RISC-V) gracefully fall back to scalar
    return false;
#endif
}

bool cpu_supports_fma() {
#if ADAPTQ_X86_OR_X64
#if defined(_MSC_VER)
    int regs[4] = {};
    __cpuidex(regs, 1, 0);

    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    const bool fma = (regs[2] & (1 << 12)) != 0;
    if (!osxsave || !avx || !fma)
        return false;

    const unsigned long long xcr0 = _xgetbv(0);
    return (xcr0 & 0x6ULL) == 0x6ULL;
#elif defined(__GNUC__) || defined(__clang__)
    if (!__builtin_cpu_supports("fma"))
        return false;

    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        return false;

    const bool osxsave = (ecx & (1U << 27)) != 0;
    const bool avx = (ecx & (1U << 28)) != 0;
    if (!osxsave || !avx)
        return false;

    uint32_t xcr0_lo = 0, xcr0_hi = 0;
    __asm__ volatile("xgetbv"
                     : "=a"(xcr0_lo), "=d"(xcr0_hi)
                     : "c"(0));
    const uint64_t xcr0 = (static_cast<uint64_t>(xcr0_hi) << 32) | xcr0_lo;
    return (xcr0 & 0x6ULL) == 0x6ULL;
#else
    return false;
#endif
#else
    return false;
#endif
}

IKernelBackend *select_kernel_backend() {
    if (!is_scalar_forced() && cpu_supports_avx2() && cpu_supports_fma()) {
        if (IKernelBackend *backend = create_avx2_backend())
            return backend;
    }
    return create_scalar_backend();
}

} /* namespace adaptq */

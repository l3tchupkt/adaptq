#pragma once
#include "storage.h"

/* -------------------------------------------------------------------------
 * adaptq/kernel.h — IKernelBackend interface
 *
 * IKernelBackend provides the SIMD compute primitives used by the attention
 * kernel. It is separate from IKVStrategy: multiple strategies can share
 * the same kernel backend, and the backend is selected at runtime based on
 * CPU capability (AVX2, AVX-512, NEON, scalar fallback).
 *
 * All methods operate on rotated float buffers (after FWHT + D-apply).
 * Quantized data is accessed via CompressResult::data and format_tag.
 *
 * Built-in backends (in order of preference):
 *   AVX512KernelBackend  — 16-wide permutevar (Skylake-X+)
 *   AVX2KernelBackend    — current kdot4_quad implementation
 *   NEONKernelBackend    — ARM NEON (V4)
 *   ScalarKernelBackend  — portable reference, always available
 * ----------------------------------------------------------------------- */

namespace adaptq {

class IKernelBackend {
public:
    virtual ~IKernelBackend() = default;

    /* ---- FWHT ---------------------------------------------------------- */

    /**
     * Apply Rademacher diagonal D and forward Walsh-Hadamard transform.
     * Operates in-place on x[padded]. padded must be a power of 2.
     */
    virtual void fwht_forward(float         *x,
                              const int8_t  *D,
                              int            padded) = 0;

    /**
     * Inverse transform: D * (1/sqrt(p)) * H * y, in-place on x[padded].
     */
    virtual void fwht_inverse(float         *x,
                              const int8_t  *D,
                              int            padded) = 0;

    /* ---- K-dot batch --------------------------------------------------- */

    /**
     * Compute dot products between a rotated query and N compressed K vectors.
     *
     * @param q_rot        float[padded] — rotated query
     * @param k_results    array of N CompressResult (compressed K vectors)
     * @param N            number of tokens
     * @param padded       power-of-2 padded dimension
     * @param bits         quantization bits (2, 3, or 4) — may differ per
     *                     result in future adaptive builds; currently uniform
     * @param logits_out   float[N] — output dot products
     */
    virtual void kdot_batch(const float          *q_rot,
                            const CompressResult *k_results,
                            int                   N,
                            int                   padded,
                            int                   bits,
                            float                *logits_out) = 0;

    /* ---- V-accumulate batch -------------------------------------------- */

    /**
     * Accumulate weighted V contributions into acc.
     *
     * @param acc         float[padded] accumulator (in/out, zero-initialized by caller)
     * @param v_results   array of N CompressResult (compressed V vectors)
     * @param weights     float[N] softmax attention weights
     * @param N           number of tokens
     * @param padded      power-of-2 padded dimension
     * @param bits        quantization bits
     */
    virtual void vaccum_batch(float                *acc,
                              const CompressResult *v_results,
                              const float          *weights,
                              int                   N,
                              int                   padded,
                              int                   bits) = 0;

    /* ---- Runtime detection --------------------------------------------- */

    /**
     * Returns true if this backend can run on the current CPU/hardware.
     * The kernel registry calls is_available() to select the best backend.
     */
    virtual bool        is_available() const = 0;

    /** Human-readable name for diagnostics and benchmark output. */
    virtual const char *name()         const = 0;
};

/**
 * Returns the best available IKernelBackend for this process.
 * Checks AVX-512, AVX2, NEON, scalar in priority order.
 * The returned pointer is process-global; do not delete.
 */
IKernelBackend *select_kernel_backend();

/**
 * Returns true if the host CPU supports AVX2 and OS has enabled AVX state management.
 */
bool cpu_supports_avx2();

/**
 * Returns true if scalar fallback was requested via environment variables
 * (e.g. ADAPTQ_DISABLE_AVX2=1, ADAPTQ_FORCE_SCALAR=1, ADAPTQ_BACKEND=scalar).
 */
bool is_scalar_forced();

} /* namespace adaptq */


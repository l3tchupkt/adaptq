#include "../../include/adaptq/kernel.h"
#include "../../include/codebook.h"
#include "../../include/fwht.h"
#include <cassert>
#include <cmath>
#include <cstring>

/* -------------------------------------------------------------------------
 * kernels/scalar/kdot_scalar.cpp — Portable reference IKernelBackend
 *
 * This backend runs on any CPU. It is the correctness reference:
 * all other backends (AVX2, AVX-512, NEON) must produce results within
 * FP32 rounding tolerance of this implementation on the same inputs.
 *
 * No SIMD intrinsics. No compiler-specific attributes. Should compile
 * cleanly under GCC, Clang, and MSVC on x86, ARM, RISC-V, and WASM.
 * ----------------------------------------------------------------------- */

namespace adaptq {

class ScalarKernelBackend final : public IKernelBackend {
public:
    /* ---- FWHT ---------------------------------------------------------- */

    void fwht_forward(float *x, const int8_t *D, int padded) override {
        ::fwht_forward(x, D, padded);
    }

    void fwht_inverse(float *x, const int8_t *D, int padded) override {
        ::fwht_inverse(x, D, padded);
    }

    /* ---- K-dot batch --------------------------------------------------- */

    void kdot_batch(const float          *q_rot,
                    const CompressResult *k_results,
                    int                   N,
                    int                   padded,
                    int                   bits,
                    float                *logits_out) override {
        const float *cb  = get_codebook(bits);
        const float  inv = 1.f / (float)padded;

        for (int i = 0; i < N; ++i) {
            const CompressResult &kr = k_results[i];
            float dot = kdot_single(q_rot, kr.data, cb, padded, bits);
            logits_out[i] = dot * kr.scale * inv;
        }
    }

    /* ---- V-accumulate batch -------------------------------------------- */

    void vaccum_batch(float                *acc,
                      const CompressResult *v_results,
                      const float          *weights,
                      int                   N,
                      int                   padded,
                      int                   bits) override {
        const float *cb = get_codebook(bits);
        memset(acc, 0, padded * sizeof(float));

        for (int i = 0; i < N; ++i) {
            if (weights[i] == 0.f) continue;
            vaccum_single(acc, v_results[i].data, v_results[i].scale,
                          weights[i], cb, padded, bits);
        }
    }

    /* ---- Capability detection ------------------------------------------ */

    bool        is_available() const override { return true; }  /* always */
    const char *name()         const override { return "scalar"; }

private:
    /* Compute dot product between q_rot and one packed K vector.
     * Unpacks indices on the fly; no temporary buffer needed.             */
    static float kdot_single(const float    *q_rot,
                              const uint8_t  *packed,
                              const float    *cb,
                              int             padded,
                              int             bits) {
        float dot = 0.f;
        int   i   = 0;

        if (bits == 4) {
            int n = padded >> 1;
            for (int b = 0; b < n; ++b) {
                uint8_t byte = packed[b];
                dot += q_rot[i++] * cb[byte >> 4];
                dot += q_rot[i++] * cb[byte & 0xF];
            }
        } else if (bits == 2) {
            int n = padded >> 2;
            for (int b = 0; b < n; ++b) {
                uint8_t byte = packed[b];
                dot += q_rot[i++] * cb[(byte >> 6) & 3];
                dot += q_rot[i++] * cb[(byte >> 4) & 3];
                dot += q_rot[i++] * cb[(byte >> 2) & 3];
                dot += q_rot[i++] * cb[ byte        & 3];
            }
        } else {
            assert(bits == 3);
            int groups = padded / 8;
            for (int g = 0; g < groups; ++g) {
                const uint8_t *p = packed + g * 3;
                dot += q_rot[i++] * cb[(p[0] >> 5) & 7];
                dot += q_rot[i++] * cb[(p[0] >> 2) & 7];
                dot += q_rot[i++] * cb[((p[0] & 3) << 1) | (p[1] >> 7)];
                dot += q_rot[i++] * cb[(p[1] >> 4) & 7];
                dot += q_rot[i++] * cb[(p[1] >> 1) & 7];
                dot += q_rot[i++] * cb[((p[1] & 1) << 2) | (p[2] >> 6)];
                dot += q_rot[i++] * cb[(p[2] >> 3) & 7];
                dot += q_rot[i++] * cb[ p[2]        & 7];
            }
        }
        return dot;
    }

    /* Accumulate one weighted V contribution into acc. */
    static void vaccum_single(float         *acc,
                               const uint8_t *packed,
                               float          scale,
                               float          weight,
                               const float   *cb,
                               int            padded,
                               int            bits) {
        const float w = weight * scale;
        int         i = 0;

        if (bits == 4) {
            int n = padded >> 1;
            for (int b = 0; b < n; ++b) {
                uint8_t byte = packed[b];
                acc[i++] += w * cb[byte >> 4];
                acc[i++] += w * cb[byte & 0xF];
            }
        } else if (bits == 2) {
            int n = padded >> 2;
            for (int b = 0; b < n; ++b) {
                uint8_t byte = packed[b];
                acc[i++] += w * cb[(byte >> 6) & 3];
                acc[i++] += w * cb[(byte >> 4) & 3];
                acc[i++] += w * cb[(byte >> 2) & 3];
                acc[i++] += w * cb[ byte        & 3];
            }
        } else {
            assert(bits == 3);
            int groups = padded / 8;
            for (int g = 0; g < groups; ++g) {
                const uint8_t *p = packed + g * 3;
                acc[i++] += w * cb[(p[0] >> 5) & 7];
                acc[i++] += w * cb[(p[0] >> 2) & 7];
                acc[i++] += w * cb[((p[0] & 3) << 1) | (p[1] >> 7)];
                acc[i++] += w * cb[(p[1] >> 4) & 7];
                acc[i++] += w * cb[(p[1] >> 1) & 7];
                acc[i++] += w * cb[((p[1] & 1) << 2) | (p[2] >> 6)];
                acc[i++] += w * cb[(p[2] >> 3) & 7];
                acc[i++] += w * cb[ p[2]        & 7];
            }
        }
    }
};

IKernelBackend *create_scalar_backend() {
    static ScalarKernelBackend instance;
    return &instance;
}
} /* namespace adaptq */

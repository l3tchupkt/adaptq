#include "../../include/adaptq/kernel.h"
#include "../../include/codebook.h"
#include "../../include/fwht.h"
#include <cassert>
#include <cstring>
#include <cmath>

#ifdef __AVX2__
#include <immintrin.h>

/* -------------------------------------------------------------------------
 * kernels/avx2/kdot_avx2.cpp — AVX2 IKernelBackend
 *
 * Extracted from attention/attention.cpp. The actual SIMD code is identical
 * to the original; this file just wraps it in the IKernelBackend interface.
 *
 * Key kernels:
 *   lup8()        — permutevar 16-entry LUT lookup (4 cycles vs 40 for gather)
 *   kdot1<BITS>   — single-token K-dot using lup8
 *   kdot4_quad<BITS> — 4-token K-dot sharing q_rot loads (peak bandwidth)
 *   vaccum1<BITS> — single-token V accumulation
 *   vaccum4<BITS> — 4-token V accumulation fused
 * ----------------------------------------------------------------------- */

/* ---- AVX2 primitive helpers (identical to attention.cpp) -------------- */

static inline __m256 lup8(const __m256i idx, const __m256 cl, const __m256 ch) {
    const __m256i m7  = _mm256_set1_epi32(7);
    __m256i       gt7 = _mm256_cmpgt_epi32(idx, m7);
    __m256i       i7  = _mm256_and_si256(idx, m7);
    return _mm256_blendv_ps(_mm256_permutevar8x32_ps(cl, i7),
                            _mm256_permutevar8x32_ps(ch, i7),
                            _mm256_castsi256_ps(gt7));
}

static inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(_mm_hadd_ps(lo, lo));
}

/* ---- Decode<BITS>: unpack 8 indices from packed bytes into int32×8 ---- */
template<int BITS> struct Decode;

template<> struct Decode<4> {
    static constexpr int step = 4;
    static inline __m256i f(const uint8_t *p) {
        uint32_t val; memcpy(&val, p, 4);
        __m128i h = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)((val >> 4) & 0x0F0F0F0F)));
        __m128i l = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)( val       & 0x0F0F0F0F)));
        return _mm256_set_m128i(_mm_unpackhi_epi32(h, l), _mm_unpacklo_epi32(h, l));
    }
};
template<> struct Decode<3> {
    static constexpr int step = 3;
    static inline __m256i f(const uint8_t *p) {
        uint32_t val = ((uint32_t)p[0]<<16)|((uint32_t)p[1]<<8)|(uint32_t)p[2];
        __m256i v256   = _mm256_set1_epi32(val);
        __m256i shifts = _mm256_setr_epi32(21,18,15,12,9,6,3,0);
        return _mm256_and_si256(_mm256_srlv_epi32(v256, shifts), _mm256_set1_epi32(7));
    }
};
template<> struct Decode<2> {
    static constexpr int step = 2;
    static inline __m256i f(const uint8_t *p) {
        uint16_t val; memcpy(&val, p, 2);
        __m256i v256   = _mm256_set1_epi32(val);
        __m256i shifts = _mm256_setr_epi32(6,4,2,0,14,12,10,8);
        return _mm256_and_si256(_mm256_srlv_epi32(v256, shifts), _mm256_set1_epi32(3));
    }
};

/* ---- Single-token K-dot ----------------------------------------------- */
template<int BITS>
static float kdot1_avx2(const float    *__restrict qr,
                         const uint8_t  *__restrict pk,
                         const __m256   cl, const __m256 ch,
                         int padded) {
    __m256 s0 = _mm256_setzero_ps();
    int b = 0;
    for (int j = 0; j < padded; j += 8) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(qr + j),
                              lup8(Decode<BITS>::f(pk + b), cl, ch), s0);
        b += Decode<BITS>::step;
    }
    return hsum8(s0);
}

/* ---- 4-token K-dot (shares q_rot loads) -------------------------------- */
template<int BITS>
static void kdot4_quad_avx2(const float    *__restrict qr,
                              const uint8_t  *k0, const uint8_t *k1,
                              const uint8_t  *k2, const uint8_t *k3,
                              const __m256 cl, const __m256 ch,
                              int padded, float out[4]) {
    __m256 s0={}, s1={}, s2={}, s3={};
    int b = 0;
    for (int j = 0; j < padded; j += 16) {
        __m256 qL = _mm256_loadu_ps(qr + j);
        s0 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k0+b), cl, ch), qL, s0);
        s1 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k1+b), cl, ch), qL, s1);
        s2 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k2+b), cl, ch), qL, s2);
        s3 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k3+b), cl, ch), qL, s3);
        b += Decode<BITS>::step;
        __m256 qH = _mm256_loadu_ps(qr + j + 8);
        s0 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k0+b), cl, ch), qH, s0);
        s1 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k1+b), cl, ch), qH, s1);
        s2 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k2+b), cl, ch), qH, s2);
        s3 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k3+b), cl, ch), qH, s3);
        b += Decode<BITS>::step;
    }
    out[0] = hsum8(s0); out[1] = hsum8(s1);
    out[2] = hsum8(s2); out[3] = hsum8(s3);
}

/* ---- Single-token V accumulation -------------------------------------- */
template<int BITS>
static void vaccum1_avx2(float          *__restrict acc,
                           const uint8_t  *__restrict vp,
                           float ew, const __m256 cl, const __m256 ch,
                           int padded) {
    __m256 ev = _mm256_set1_ps(ew);
    int b = 0;
    for (int j = 0; j < padded; j += 8) {
        __m256 ra = _mm256_loadu_ps(acc + j);
        _mm256_storeu_ps(acc + j,
            _mm256_fmadd_ps(lup8(Decode<BITS>::f(vp+b), cl, ch), ev, ra));
        b += Decode<BITS>::step;
    }
}

/* ---- 4-token V accumulation ------------------------------------------- */
template<int BITS>
static void vaccum4_avx2(float *__restrict acc,
                           const uint8_t *v0, const uint8_t *v1,
                           const uint8_t *v2, const uint8_t *v3,
                           float e0, float e1, float e2, float e3,
                           const __m256 cl, const __m256 ch, int padded) {
    __m256 q0=_mm256_set1_ps(e0), q1=_mm256_set1_ps(e1);
    __m256 q2=_mm256_set1_ps(e2), q3=_mm256_set1_ps(e3);
    int b = 0;
    for (int j = 0; j < padded; j += 16) {
        __m256 ra = _mm256_loadu_ps(acc + j);
        ra = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v0+b), cl, ch), q0, ra);
        ra = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v1+b), cl, ch), q1, ra);
        ra = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v2+b), cl, ch), q2, ra);
        ra = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v3+b), cl, ch), q3, ra);
        _mm256_storeu_ps(acc + j, ra);
        b += Decode<BITS>::step;
        __m256 rb = _mm256_loadu_ps(acc + j + 8);
        rb = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v0+b), cl, ch), q0, rb);
        rb = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v1+b), cl, ch), q1, rb);
        rb = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v2+b), cl, ch), q2, rb);
        rb = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v3+b), cl, ch), q3, rb);
        _mm256_storeu_ps(acc + j + 8, rb);
        b += Decode<BITS>::step;
    }
}

/* ---- AVX2 FWHT helpers (8-wide butterfly + normalize) ------------------ */
static inline void fwht_butterfly_avx2(float *lo, float *hi, int len) {
    int j = 0;
    for (; j + 7 < len; j += 8) {
        __m256 a = _mm256_loadu_ps(lo + j);
        __m256 b = _mm256_loadu_ps(hi + j);
        _mm256_storeu_ps(lo + j, _mm256_add_ps(a, b));
        _mm256_storeu_ps(hi + j, _mm256_sub_ps(a, b));
    }
    for (; j < len; ++j) {
        float a = lo[j], b = hi[j];
        lo[j] = a + b;
        hi[j] = a - b;
    }
}

static inline void fwht_scale_avx2(float *x, int n, float scale, const int8_t *D, int d) {
    __m256 vs = _mm256_set1_ps(scale);
    int i = 0;
    // Fast path when D covers the whole vector
    if (D && d >= n) {
        for (; i + 7 < n; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            // D is +-1 per lane; build float signs on the fly
            __m256 signs = _mm256_set_ps((float)D[i+7], (float)D[i+6], (float)D[i+5], (float)D[i+4],
                                         (float)D[i+3], (float)D[i+2], (float)D[i+1], (float)D[i]);
            _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_mul_ps(v, vs), signs));
        }
    } else {
        for (; i + 7 < n; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            _mm256_storeu_ps(x + i, _mm256_mul_ps(v, vs));
        }
    }
    for (; i < n; ++i) x[i] *= scale;
}

namespace adaptq {

/* =========================================================================
 * AVX2KernelBackend : IKernelBackend
 * ======================================================================= */
class AVX2KernelBackend final : public IKernelBackend {
public:
    void fwht_forward(float *x, const int8_t *D, int p) override {
        if (!x || !D || p <= 0) return;
        if (p < 8) { ::fwht_forward(x, D, p); return; }
        // Fused first pass (D + len=1) scalar, rest 8-wide
        for (int i = 0; i < p; i += 2) {
            float a = x[i] * (float)D[i];
            float b = x[i+1] * (float)D[i+1];
            x[i] = a + b;
            x[i+1] = a - b;
        }
        for (int len = 2; len < p; len <<= 1) {
            for (int i = 0; i < p; i += len << 1) {
                fwht_butterfly_avx2(x + i, x + i + len, len);
            }
        }
        float inv = 1.0f / sqrtf((float)p);
        __m256 vinv = _mm256_set1_ps(inv);
        int i = 0;
        for (; i + 7 < p; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            _mm256_storeu_ps(x + i, _mm256_mul_ps(v, vinv));
        }
        for (; i < p; ++i) x[i] *= inv;
    }
    void fwht_inverse(float *x, const int8_t *D, int p) override {
        if (!x || !D || p <= 0) return;
        if (p < 8) { ::fwht_inverse(x, D, p); return; }
        for (int len = 1; len < p; len <<= 1) {
            for (int i = 0; i < p; i += len << 1) {
                fwht_butterfly_avx2(x + i, x + i + len, len);
            }
        }
        float scale = 1.0f / sqrtf((float)p);
        fwht_scale_avx2(x, p, scale, D, p);
        // Apply D signs for the tail when D length < p (padded case handled above)
        // fwht_scale_avx2 already folded D when d>=n; for padded the extra
        // lanes use +1 which matches the scalar reference.
    }

    /* K-dot: 4-token batches via kdot4_quad, scalar tail via kdot1 */
    void kdot_batch(const float          *q_rot,
                    const CompressResult *kr,
                    int N, int padded, int bits,
                    float *logits_out) override {
        const float *cb = get_codebook(bits);
        __m256 cl = _mm256_loadu_ps(cb), ch = _mm256_loadu_ps(cb + 8);
        int i = 0;
        for (; i + 3 < N; i += 4) {
            float d[4];
#define KDOT4(B) kdot4_quad_avx2<B>(q_rot, \
    kr[i].data, kr[i+1].data, kr[i+2].data, kr[i+3].data, cl, ch, padded, d)
            if      (bits==4) KDOT4(4);
            else if (bits==3) KDOT4(3);
            else              KDOT4(2);
#undef KDOT4
            for (int k = 0; k < 4; ++k)
                logits_out[i+k] = d[k] * kr[i+k].scale / (float)padded;
        }
        for (; i < N; ++i) {
            float d;
            if      (bits==4) d = kdot1_avx2<4>(q_rot, kr[i].data, cl, ch, padded);
            else if (bits==3) d = kdot1_avx2<3>(q_rot, kr[i].data, cl, ch, padded);
            else              d = kdot1_avx2<2>(q_rot, kr[i].data, cl, ch, padded);
            logits_out[i] = d * kr[i].scale / (float)padded;
        }
    }

    /* V-accumulate: 4-token batches, scalar tail */
    void vaccum_batch(float                *acc,
                      const CompressResult *vr,
                      const float          *weights,
                      int N, int padded, int bits) override {
        const float *cb = get_codebook(bits);
        __m256 cl = _mm256_loadu_ps(cb), ch = _mm256_loadu_ps(cb + 8);
        memset(acc, 0, (size_t)padded * sizeof(float));
        int i = 0;
        for (; i + 3 < N; i += 4) {
            float e0 = weights[i]   * vr[i].scale;
            float e1 = weights[i+1] * vr[i+1].scale;
            float e2 = weights[i+2] * vr[i+2].scale;
            float e3 = weights[i+3] * vr[i+3].scale;
#define VACC4(B) vaccum4_avx2<B>(acc, \
    vr[i].data, vr[i+1].data, vr[i+2].data, vr[i+3].data, \
    e0, e1, e2, e3, cl, ch, padded)
            if      (bits==4) VACC4(4);
            else if (bits==3) VACC4(3);
            else              VACC4(2);
#undef VACC4
        }
        for (; i < N; ++i) {
            float ew = weights[i] * vr[i].scale;
            if      (bits==4) vaccum1_avx2<4>(acc, vr[i].data, ew, cl, ch, padded);
            else if (bits==3) vaccum1_avx2<3>(acc, vr[i].data, ew, cl, ch, padded);
            else              vaccum1_avx2<2>(acc, vr[i].data, ew, cl, ch, padded);
        }
    }

    bool        is_available() const override { return true; }
    const char *name()         const override { return "avx2"; }
};

IKernelBackend *create_avx2_backend() {
    static AVX2KernelBackend instance;
    return &instance;
}
} /* namespace adaptq */

#else  /* !__AVX2__ */

namespace adaptq {

/* Keep the factory symbol available in portable builds. The selector will
 * fall back to the scalar backend when this TU has no AVX2 implementation. */
IKernelBackend *create_avx2_backend() {
    return nullptr;
}

} /* namespace adaptq */

#endif /* __AVX2__ */

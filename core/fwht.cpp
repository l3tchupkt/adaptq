#include "../include/fwht.h"
#include <cassert>
#include <cmath>
#include <cstring>

// Thread-local pad buffer so fwht_forward/inverse never heap-allocate.
// Must be >= max next_pow2(head_dim) used by any context in this process.
// Increase ADAPTQ_FWHT_PAD_BUF if you need head_dim > 512.
static constexpr int ADAPTQ_FWHT_PAD_BUF = 1024;
static thread_local float tl_pad_buf[ADAPTQ_FWHT_PAD_BUF];

int next_pow2(int n) {
    if (n <= 1) return 1;
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

void gen_rademacher(int8_t* D, int d, uint64_t seed) {
    uint64_t s = seed ? seed : 0xdeadbeefcafe1234ULL;
    for (int i = 0; i < d; ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        D[i] = (s & 1) ? 1 : -1;
    }
}

// Fused forward FWHT:
//   Pass 0: apply Rademacher D and first butterfly (len=1) in one sweep.
//   Passes 1..log2(n)-1: 4-wide unrolled butterfly.
//   Final pass: normalization by 1/sqrt(n), unrolled 4-wide.
static void fwht_fused(float* x, const int8_t* D, int d, int n) {
    // D is defined only for the original d-dimensional input. When the
    // input is padded to n > d, the implicit signs for padded coordinates
    // are +1 because those coordinates are initially zero.
    for (int i = 0; i < n; i += 2) {
        float a = x[i]   * (i < d ? (float)D[i] : 1.0f);
        float b = x[i+1] * (i + 1 < d ? (float)D[i+1] : 1.0f);
        x[i]   = a + b;
        x[i+1] = a - b;
    }
    // Remaining butterfly passes, 4-wide inner loop
    for (int len = 2; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            float* lo = x + i;
            float* hi = x + i + len;
            int j = 0;
            for (; j + 3 < len; j += 4) {
                float a0=lo[j],   a1=lo[j+1], a2=lo[j+2], a3=lo[j+3];
                float b0=hi[j],   b1=hi[j+1], b2=hi[j+2], b3=hi[j+3];
                lo[j]  =a0+b0; lo[j+1]=a1+b1; lo[j+2]=a2+b2; lo[j+3]=a3+b3;
                hi[j]  =a0-b0; hi[j+1]=a1-b1; hi[j+2]=a2-b2; hi[j+3]=a3-b3;
            }
            for (; j < len; ++j) {
                float a=lo[j], b=hi[j]; lo[j]=a+b; hi[j]=a-b;
            }
        }
    }
    // Normalize 1/sqrt(n), unrolled 4-wide
    float inv = 1.0f / sqrtf((float)n);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        x[i]*=inv; x[i+1]*=inv; x[i+2]*=inv; x[i+3]*=inv;
    }
    for (; i < n; ++i) x[i] *= inv;
}

// Plain butterfly (no D, no normalization) used by fwht_inverse.
static void fwht_raw_unrolled(float* x, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            float* lo = x + i;
            float* hi = x + i + len;
            int j = 0;
            for (; j + 3 < len; j += 4) {
                float a0=lo[j],   a1=lo[j+1], a2=lo[j+2], a3=lo[j+3];
                float b0=hi[j],   b1=hi[j+1], b2=hi[j+2], b3=hi[j+3];
                lo[j]  =a0+b0; lo[j+1]=a1+b1; lo[j+2]=a2+b2; lo[j+3]=a3+b3;
                hi[j]  =a0-b0; hi[j+1]=a1-b1; hi[j+2]=a2-b2; hi[j+3]=a3-b3;
            }
            for (; j < len; ++j) {
                float a=lo[j], b=hi[j]; lo[j]=a+b; hi[j]=a-b;
            }
        }
    }
}

void fwht_forward(float* x, const int8_t* D, int d) {
    int p = next_pow2(d);
    float* work = x;
    if (p != d) {
        assert(p <= ADAPTQ_FWHT_PAD_BUF &&
               "FWHT padded dimension exceeds tl_pad_buf. "
               "Increase ADAPTQ_FWHT_PAD_BUF in fwht.cpp.");
        memcpy(tl_pad_buf, x, d * sizeof(float));
        memset(tl_pad_buf + d, 0, (p - d) * sizeof(float));
        work = tl_pad_buf;
    }
    fwht_fused(work, D, d, p);
    if (p != d) memcpy(x, work, d * sizeof(float));
}

void fwht_inverse(float* x, const int8_t* D, int d) {
    // Inverse of (1/sqrt(p))*H*D*x is D*(1/sqrt(p))*H*y.
    // For padded coordinates beyond d, D is implicitly +1.
    int p = next_pow2(d);
    float* work = x;
    if (p != d) {
        assert(p <= ADAPTQ_FWHT_PAD_BUF &&
               "FWHT padded dimension exceeds tl_pad_buf. "
               "Increase ADAPTQ_FWHT_PAD_BUF in fwht.cpp.");
        memcpy(tl_pad_buf, x, d * sizeof(float));
        memset(tl_pad_buf + d, 0, (p - d) * sizeof(float));
        work = tl_pad_buf;
    }
    fwht_raw_unrolled(work, p);
    float scale = 1.0f / sqrtf((float)p);
    int i = 0;
    for (; i + 3 < p; i += 4) {
        work[i]  =work[i]  *scale*(i < d ? (float)D[i] : 1.0f);
        work[i+1]=work[i+1]*scale*(i+1 < d ? (float)D[i+1] : 1.0f);
        work[i+2]=work[i+2]*scale*(i+2 < d ? (float)D[i+2] : 1.0f);
        work[i+3]=work[i+3]*scale*(i+3 < d ? (float)D[i+3] : 1.0f);
    }
    for (; i < p; ++i)
        work[i] = work[i] * scale * (i < d ? (float)D[i] : 1.0f);
    if (p != d) memcpy(x, work, d * sizeof(float));
}

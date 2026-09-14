#include "../include/attention.h"
#include "../include/codebook.h"
#include <algorithm>
#include <cmath>
#include <cstring>

/* Portable cache prefetch — __builtin_prefetch is GCC/Clang only */
#if defined(_MSC_VER)
#  include <xmmintrin.h>
#  define ADAPTQ_PREFETCH(ptr) _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T1)
#else
#  define ADAPTQ_PREFETCH(ptr) __builtin_prefetch((ptr), 0, 1)
#endif

#ifdef __AVX2__
#include <immintrin.h>

// Permutevar 4-bit lookup: 8 indices in ~8 cycles vs ~40 for gather
static inline __m256 lup8(const __m256i idx, const __m256 cl, const __m256 ch) {
  const __m256i m7 = _mm256_set1_epi32(7);
  __m256i gt7 = _mm256_cmpgt_epi32(idx, m7);
  __m256i i7 = _mm256_and_si256(idx, m7);
  return _mm256_blendv_ps(_mm256_permutevar8x32_ps(cl, i7),
                          _mm256_permutevar8x32_ps(ch, i7),
                          _mm256_castsi256_ps(gt7));
}

// ---------------------------------------------------------------------------
// Templatized Decoding logic: extracts 8 indices into an int32×8 vector
// for permutevar8x32.
// ---------------------------------------------------------------------------
template <int BITS> struct Decode;

template <> struct Decode<4> {
  static constexpr int step = 4;
  static inline __m256i f(const uint8_t *p) {
    uint32_t val;
    memcpy(&val, p, 4);
    __m128i h =
        _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)((val >> 4) & 0x0F0F0F0F)));
    __m128i l = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)(val & 0x0F0F0F0F)));
    return _mm256_set_m128i(_mm_unpackhi_epi32(h, l), _mm_unpacklo_epi32(h, l));
  }
};

template <> struct Decode<3> {
  static constexpr int step = 3;
  static inline __m256i f(const uint8_t *p) {
    uint32_t val =
        ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | (uint32_t)p[2];
    __m256i v256 = _mm256_set1_epi32(val);
    __m256i shifts = _mm256_setr_epi32(21, 18, 15, 12, 9, 6, 3, 0);
    return _mm256_and_si256(_mm256_srlv_epi32(v256, shifts),
                            _mm256_set1_epi32(7));
  }
};

template <> struct Decode<2> {
  static constexpr int step = 2;
  static inline __m256i f(const uint8_t *p) {
    uint16_t val;
    memcpy(&val, p, 2);
    __m256i v256 = _mm256_set1_epi32(val);
    __m256i shifts = _mm256_setr_epi32(6, 4, 2, 0, 14, 12, 10, 8);
    return _mm256_and_si256(_mm256_srlv_epi32(v256, shifts),
                            _mm256_set1_epi32(3));
  }
};

// Horizontal sum of __m256 → scalar (uses XMM path, no extra YMM pressure)
static inline float hsum8(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(_mm_hadd_ps(lo, lo));
}

// ---------------------------------------------------------------------------
// SINGLE-TOKEN K-dot: used for tail processing
// ---------------------------------------------------------------------------
template <int BITS>
static float kdot1(const float *__restrict qr, const uint8_t *__restrict pk,
                   const __m256 cl, const __m256 ch, int padded) {
  __m256 s0 = _mm256_setzero_ps();
  int b = 0;
  for (int j = 0; j < padded; j += 8) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(qr + j),
                         lup8(Decode<BITS>::f(pk + b), cl, ch), s0);
    b += Decode<BITS>::step;
  }
  return hsum8(s0);
}

// ---------------------------------------------------------------------------
// 4-TOKEN K-dot: all 4 tokens processed in one loop, sharing q_rot loads.
// ---------------------------------------------------------------------------
template <int BITS>
static void kdot4_quad(const float *__restrict qr, const uint8_t *__restrict k0,
                       const uint8_t *__restrict k1,
                       const uint8_t *__restrict k2,
                       const uint8_t *__restrict k3, const __m256 cl,
                       const __m256 ch, int padded, float out[4]) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
  int b = 0;
  for (int j = 0; j < padded; j += 16) {
    __m256 qL = _mm256_loadu_ps(qr + j);
    s0 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k0 + b), cl, ch), qL, s0);
    s1 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k1 + b), cl, ch), qL, s1);
    s2 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k2 + b), cl, ch), qL, s2);
    s3 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k3 + b), cl, ch), qL, s3);
    b += Decode<BITS>::step;
    __m256 qH = _mm256_loadu_ps(qr + j + 8);
    s0 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k0 + b), cl, ch), qH, s0);
    s1 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k1 + b), cl, ch), qH, s1);
    s2 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k2 + b), cl, ch), qH, s2);
    s3 = _mm256_fmadd_ps(lup8(Decode<BITS>::f(k3 + b), cl, ch), qH, s3);
    b += Decode<BITS>::step;
  }
  out[0] = hsum8(s0);
  out[1] = hsum8(s1);
  out[2] = hsum8(s2);
  out[3] = hsum8(s3);
}

// ---------------------------------------------------------------------------
// V-accum 4-token batch: 1 load + 4 FMAs + 1 store per 16-pos block.
// ---------------------------------------------------------------------------
template <int BITS>
static void vaccum4(float *__restrict acc, const uint8_t *v0, const uint8_t *v1,
                    const uint8_t *v2, const uint8_t *v3, float e0, float e1,
                    float e2, float e3, const __m256 cl, const __m256 ch,
                    int padded) {
  __m256 q0 = _mm256_set1_ps(e0), q1 = _mm256_set1_ps(e1);
  __m256 q2 = _mm256_set1_ps(e2), q3 = _mm256_set1_ps(e3);
  int b = 0;
  for (int j = 0; j < padded; j += 16) {
    __m256 ra = _mm256_loadu_ps(acc + j);
    ra = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v0 + b), cl, ch), q0, ra);
    ra = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v1 + b), cl, ch), q1, ra);
    ra = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v2 + b), cl, ch), q2, ra);
    ra = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v3 + b), cl, ch), q3, ra);
    _mm256_storeu_ps(acc + j, ra);
    b += Decode<BITS>::step;

    __m256 rb = _mm256_loadu_ps(acc + j + 8);
    rb = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v0 + b), cl, ch), q0, rb);
    rb = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v1 + b), cl, ch), q1, rb);
    rb = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v2 + b), cl, ch), q2, rb);
    rb = _mm256_fmadd_ps(lup8(Decode<BITS>::f(v3 + b), cl, ch), q3, rb);
    _mm256_storeu_ps(acc + j + 8, rb);
    b += Decode<BITS>::step;
  }
}

template <int BITS>
static void vaccum1(float *__restrict acc, const uint8_t *__restrict vp,
                    float ew, const __m256 cl, const __m256 ch, int padded) {
  __m256 ev = _mm256_set1_ps(ew);
  int b = 0;
  for (int j = 0; j < padded; j += 8) {
    __m256 ra = _mm256_loadu_ps(acc + j);
    _mm256_storeu_ps(
        acc + j,
        _mm256_fmadd_ps(lup8(Decode<BITS>::f(vp + b), cl, ch), ev, ra));
    b += Decode<BITS>::step;
  }
}

#endif // __AVX2__

#include <vector>

struct AttentionWorkspace {
  std::vector<float> logits;
  std::vector<int> slots;
  std::vector<int> ord;
  std::vector<float> q_rot;
  std::vector<float> v_accum;

  void ensure_capacity(int n, int padded) {
    if (logits.size() < (size_t)n) {
      logits.resize(n);
      slots.resize(n);
      ord.resize(n);
    }
    if (q_rot.size() < (size_t)padded) {
      q_rot.resize(padded);
      v_accum.resize(padded);
    }
  }
};

static thread_local AttentionWorkspace tl_ws;

float dot_product(const float *a, const float *b, int n) {
  float s = 0.f;
  for (int i = 0; i < n; ++i)
    s += a[i] * b[i];
  return s;
}
void softmax(float *x, int n) {
  float mx = x[0];
  for (int i = 1; i < n; ++i)
    if (x[i] > mx)
      mx = x[i];
  float s = 0.f;
  for (int i = 0; i < n; ++i) {
    x[i] = expf(x[i] - mx);
    s += x[i];
  }
  float inv = 1.f / s;
  for (int i = 0; i < n; ++i)
    x[i] *= inv;
}

[[maybe_unused]]
static float kdot_scalar(const float *q, const uint8_t *p, const float *cb,
                         int nb, int bits) {
  float s = 0.f;
  if (bits == 2) {
    for (int i = 0; i < nb; ++i) {
      uint8_t v = p[i];
      int j = i << 2;
      s += q[j] * cb[(v >> 6) & 3] + q[j + 1] * cb[(v >> 4) & 3] +
           q[j + 2] * cb[(v >> 2) & 3] + q[j + 3] * cb[v & 3];
    }
  } else if (bits == 3) {
    int g = nb / 3;
    for (int i = 0; i < g; ++i) {
      const uint8_t *r = p + i * 3;
      int j = i * 8;
      s += q[j] * cb[(r[0] >> 5) & 7] + q[j + 1] * cb[(r[0] >> 2) & 7] +
           q[j + 2] * cb[((r[0] & 3) << 1) | (r[1] >> 7)] +
           q[j + 3] * cb[(r[1] >> 4) & 7] + q[j + 4] * cb[(r[1] >> 1) & 7] +
           q[j + 5] * cb[((r[1] & 1) << 2) | (r[2] >> 6)] +
           q[j + 6] * cb[(r[2] >> 3) & 7] + q[j + 7] * cb[r[2] & 7];
    }
  } else {
    for (int i = 0; i + 3 < nb; i += 4) {
      uint8_t b0 = p[i], b1 = p[i + 1], b2 = p[i + 2], b3 = p[i + 3];
      int j = i << 1;
      s += q[j] * cb[b0 >> 4] + q[j + 1] * cb[b0 & 15] +
           q[j + 2] * cb[b1 >> 4] + q[j + 3] * cb[b1 & 15] +
           q[j + 4] * cb[b2 >> 4] + q[j + 5] * cb[b2 & 15] +
           q[j + 6] * cb[b3 >> 4] + q[j + 7] * cb[b3 & 15];
    }
  }
  return s;
}

static void vaccum_scalar(float *acc, const uint8_t *vp, const float *ecb,
                          int nb, int bits) {
  if (bits == 2) {
    for (int b = 0; b < nb; ++b) {
      uint8_t v = vp[b];
      int j = b << 2;
      acc[j] += ecb[(v >> 6) & 3];
      acc[j + 1] += ecb[(v >> 4) & 3];
      acc[j + 2] += ecb[(v >> 2) & 3];
      acc[j + 3] += ecb[v & 3];
    }
  } else if (bits == 3) {
    int g = nb / 3;
    for (int i = 0; i < g; ++i) {
      const uint8_t *r = vp + i * 3;
      int j = i * 8;
      acc[j] += ecb[(r[0] >> 5) & 7];
      acc[j + 1] += ecb[(r[0] >> 2) & 7];
      acc[j + 2] += ecb[((r[0] & 3) << 1) | (r[1] >> 7)];
      acc[j + 3] += ecb[(r[1] >> 4) & 7];
      acc[j + 4] += ecb[(r[1] >> 1) & 7];
      acc[j + 5] += ecb[((r[1] & 1) << 2) | (r[2] >> 6)];
      acc[j + 6] += ecb[(r[2] >> 3) & 7];
      acc[j + 7] += ecb[r[2] & 7];
    }
  } else {
    for (int b = 0; b + 3 < nb; b += 4) {
      uint8_t b0 = vp[b], b1 = vp[b + 1], b2 = vp[b + 2], b3 = vp[b + 3];
      int j = b << 1;
      acc[j] += ecb[b0 >> 4];
      acc[j + 1] += ecb[b0 & 15];
      acc[j + 2] += ecb[b1 >> 4];
      acc[j + 3] += ecb[b1 & 15];
      acc[j + 4] += ecb[b2 >> 4];
      acc[j + 5] += ecb[b2 & 15];
      acc[j + 6] += ecb[b3 >> 4];
      acc[j + 7] += ecb[b3 & 15];
    }
  }
}

// ---------------------------------------------------------------------------
void AttentionHead::init(int d, int b, int cap, uint64_t seed, float v_mass,
                         int hyb) {
  dim = d;
  bits = b;
  v_mass_thresh = v_mass;
  hybrid_thresh = hyb;
  quant.init(d, seed);
  padded = quant.padded;
  kv_buf.init(cap, padded, b);
  if (hyb > 0)
    raw_kv.reserve((size_t)hyb * 2 * d);
}
void AttentionHead::append_kv(const float *key, const float *val, int pos) {
  static thread_local uint8_t tmp_k[8192], tmp_v[8192];
  float ks = quant.quantize_into(key, bits, tmp_k);
  float vs = quant.quantize_into(val, bits, tmp_v);
  kv_buf.insert(tmp_k, ks, tmp_v, vs, pos);
  // Mirror raw floats for hybrid FP path (only up to threshold)
  if (hybrid_thresh > 0 && (int)raw_kv.size() < hybrid_thresh * 2 * dim) {
    raw_kv.insert(raw_kv.end(), key, key + dim);
    raw_kv.insert(raw_kv.end(), val, val + dim);
  }
}

#ifdef __AVX2__
template <int BITS>
static void compute_avx2(const float *qr, float *acc, const float *cb,
                         const uint8_t *kb, const uint8_t *vb,
                         const float *kscale, const float *vscale, float attn_s,
                         float isp, int *slots, int n, int pb, int padded,
                         float v_mass_thresh, float *logits) {
  const __m256 cl = _mm256_loadu_ps(cb), ch = _mm256_loadu_ps(cb + 8);
  float mx = -1e30f;
  int i = 0;
  for (; i + 3 < n; i += 4) {
    int s0 = slots[i], s1 = slots[i + 1], s2 = slots[i + 2], s3 = slots[i + 3];
    if (i + 7 < n) {
      ADAPTQ_PREFETCH(kb + (size_t)slots[i + 4] * pb);
      ADAPTQ_PREFETCH(kb + (size_t)slots[i + 5] * pb);
      ADAPTQ_PREFETCH(kb + (size_t)slots[i + 6] * pb);
      ADAPTQ_PREFETCH(kb + (size_t)slots[i + 7] * pb);
    }
    ADAPTQ_PREFETCH(vb + (size_t)s0 * pb);
    ADAPTQ_PREFETCH(vb + (size_t)s1 * pb);
    ADAPTQ_PREFETCH(vb + (size_t)s2 * pb);
    ADAPTQ_PREFETCH(vb + (size_t)s3 * pb);

    float d[4];
    kdot4_quad<BITS>(qr, kb + (size_t)s0 * pb, kb + (size_t)s1 * pb,
                     kb + (size_t)s2 * pb, kb + (size_t)s3 * pb, cl, ch, padded,
                     d);
    logits[i] = d[0] * attn_s * kscale[s0];
    if (logits[i] > mx)
      mx = logits[i];
    logits[i + 1] = d[1] * attn_s * kscale[s1];
    if (logits[i + 1] > mx)
      mx = logits[i + 1];
    logits[i + 2] = d[2] * attn_s * kscale[s2];
    if (logits[i + 2] > mx)
      mx = logits[i + 2];
    logits[i + 3] = d[3] * attn_s * kscale[s3];
    if (logits[i + 3] > mx)
      mx = logits[i + 3];
  }
  for (; i < n; ++i) {
    int s = slots[i];
    ADAPTQ_PREFETCH(vb + (size_t)s * pb);
    logits[i] = kdot1<BITS>(qr, kb + (size_t)s * pb, cl, ch, padded) * attn_s *
                kscale[s];
    if (logits[i] > mx)
      mx = logits[i];
  }

  float sv = 0.f;
  for (int j = 0; j < n; ++j) {
    logits[j] = expf(logits[j] - mx);
    sv += logits[j];
  }
  float inv = 1.f / sv;
  for (int j = 0; j < n; ++j)
    logits[j] *= inv;

  memset(acc, 0, padded * sizeof(float));

  if (v_mass_thresh <= 0.f) {
    int ii = 0;
    for (; ii + 3 < n; ii += 4) {
      int s0 = slots[ii], s1 = slots[ii + 1], s2 = slots[ii + 2],
          s3 = slots[ii + 3];
      vaccum4<BITS>(
          acc, vb + (size_t)s0 * pb, vb + (size_t)s1 * pb, vb + (size_t)s2 * pb,
          vb + (size_t)s3 * pb, logits[ii] * vscale[s0] * isp,
          logits[ii + 1] * vscale[s1] * isp, logits[ii + 2] * vscale[s2] * isp,
          logits[ii + 3] * vscale[s3] * isp, cl, ch, padded);
    }
    for (; ii < n; ++ii) {
      int s = slots[ii];
      vaccum1<BITS>(acc, vb + (size_t)s * pb, logits[ii] * vscale[s] * isp, cl,
                    ch, padded);
    }
  } else {
    int *ord = tl_ws.ord.data();
    for (int ii = 0; ii < n; ++ii)
      ord[ii] = ii;
    std::sort(ord, ord + n,
              [&](int a, int b) { return logits[a] > logits[b]; });
    float mass = 0.f;
    int ii = 0;
    for (; ii + 3 < n && mass < v_mass_thresh; ii += 4) {
      int i0 = ord[ii], i1 = ord[ii + 1], i2 = ord[ii + 2], i3 = ord[ii + 3];
      int s0 = slots[i0], s1 = slots[i1], s2 = slots[i2], s3 = slots[i3];
      vaccum4<BITS>(
          acc, vb + (size_t)s0 * pb, vb + (size_t)s1 * pb, vb + (size_t)s2 * pb,
          vb + (size_t)s3 * pb, logits[i0] * vscale[s0] * isp,
          logits[i1] * vscale[s1] * isp, logits[i2] * vscale[s2] * isp,
          logits[i3] * vscale[s3] * isp, cl, ch, padded);
      mass += logits[i0] + logits[i1] + logits[i2] + logits[i3];
    }
    for (; ii < n && mass < v_mass_thresh; ++ii) {
      int s = slots[ord[ii]];
      float w = logits[ord[ii]];
      vaccum1<BITS>(acc, vb + (size_t)s * pb, w * vscale[s] * isp, cl, ch,
                    padded);
      mass += w;
    }
  }
}
#endif

int AttentionHead::compute(const float *q, float *out) const {
  const int n = kv_buf.size, cap = kv_buf.capacity, pb = kv_buf.packed_bytes;
  if (!n) {
    memset(out, 0, dim * sizeof(float));
    return 0;
  }

  // ---- Hybrid path: FP32 attention for small sequences -------------------
  // Below hybrid_thresh, FP32 is faster (data fits in L1/L2, no decode cost).
  // raw_kv stores interleaved [k0...kd, v0...vd, k1...] for the first
  // hybrid_thresh tokens. Zero-overhead check: one integer compare.
  if (hybrid_thresh > 0 && n <= hybrid_thresh &&
      (int)raw_kv.size() == n * 2 * dim) {
    tl_ws.ensure_capacity(n, padded);
    float *logits = tl_ws.logits.data();
    const float scale = 1.f / sqrtf((float)dim);
    float mx = -1e30f;
    // Compute all logits in one pass, track max
    for (int i = 0; i < n; ++i) {
      const float *k = raw_kv.data() + (size_t)i * 2 * dim;
      float d = 0.f;
      for (int j = 0; j < dim; ++j)
        d += q[j] * k[j];
      logits[i] = d * scale;
      if (logits[i] > mx)
        mx = logits[i];
    }
    // Softmax: 2 passes (max known)
    float sv = 0.f;
    for (int i = 0; i < n; ++i) {
      logits[i] = expf(logits[i] - mx);
      sv += logits[i];
    }
    float inv = 1.f / sv;
    // Weighted V accumulation
    memset(out, 0, dim * sizeof(float));
    for (int i = 0; i < n; ++i) {
      float w = logits[i] * inv;
      const float *v = raw_kv.data() + (size_t)i * 2 * dim + dim;
      for (int j = 0; j < dim; ++j)
        out[j] += w * v[j];
    }
    return n;
  }
  // ---- End hybrid path ---------------------------------------------------

  tl_ws.ensure_capacity(n, padded);

  // Rotate query
  float *qr = tl_ws.q_rot.data();
  memcpy(qr, q, dim * sizeof(float));
  for (int i = dim; i < padded; ++i)
    qr[i] = 0.f;
  float qn = 0.f;
  for (int i = 0; i < dim; ++i)
    qn += q[i] * q[i];
  qn = sqrtf(qn + 1e-12f);
  {
    float inv = 1.f / qn;
    for (int i = 0; i < padded; ++i)
      qr[i] *= inv;
  }
  fwht_forward(qr, quant.D.data(), padded);
  float sp = sqrtf((float)padded);
  for (int i = 0; i < padded; ++i)
    qr[i] *= sp * qn;

  const float *cb = get_codebook(bits);
  float *acc = tl_ws.v_accum.data();
  const uint8_t *kb = kv_buf.k_data;
  const uint8_t *vb = kv_buf.v_data;
  float attn_s = 1.f / (sqrtf((float)dim) * (float)padded);
  float isp = 1.f / sqrtf((float)padded);
  float *logits = tl_ws.logits.data();
  int *slots = tl_ws.slots.data();
  for (int i = 0; i < n; ++i)
    slots[i] = (kv_buf.head - n + cap + i) % cap;



  // inside compute():
#ifdef __AVX2__
  if (bits == 4) {
    compute_avx2<4>(qr, acc, cb, kb, vb, kv_buf.k_scale.data(),
                    kv_buf.v_scale.data(), attn_s, isp, slots, n, pb, padded,
                    v_mass_thresh, logits);
    fwht_inverse(acc, quant.D.data(), padded);
    memcpy(out, acc, dim * sizeof(float));
    return n;
  } else if (bits == 3) {
    compute_avx2<3>(qr, acc, cb, kb, vb, kv_buf.k_scale.data(),
                    kv_buf.v_scale.data(), attn_s, isp, slots, n, pb, padded,
                    v_mass_thresh, logits);
    fwht_inverse(acc, quant.D.data(), padded);
    memcpy(out, acc, dim * sizeof(float));
    return n;
  } else if (bits == 2) {
    compute_avx2<2>(qr, acc, cb, kb, vb, kv_buf.k_scale.data(),
                    kv_buf.v_scale.data(), attn_s, isp, slots, n, pb, padded,
                    v_mass_thresh, logits);
    fwht_inverse(acc, quant.D.data(), padded);
    memcpy(out, acc, dim * sizeof(float));
    return n;
  }
#endif

  // Scalar 2/3-bit path
  memset(acc, 0, padded * sizeof(float));
  for (int i = 0; i < n; ++i) {
    int s = slots[i];
    if (i + 4 < n) {
      int ps = slots[i + 4];
      ADAPTQ_PREFETCH(kb + (size_t)ps * pb);
      ADAPTQ_PREFETCH(vb + (size_t)ps * pb);
    }
    logits[i] = kdot_scalar(qr, kb + (size_t)s * pb, cb, pb, bits) * attn_s *
                kv_buf.k_scale[s];
  }
  softmax(logits, n);
  int cb_sz = 1 << bits;
  for (int i = 0; i < n; ++i) {
    int s = slots[i];
    float ecb[8];
    float ew = logits[i] * kv_buf.v_scale[s] * isp;
    for (int k = 0; k < cb_sz; ++k)
      ecb[k] = ew * cb[k];
    vaccum_scalar(acc, vb + (size_t)s * pb, ecb, pb, bits);
  }
  fwht_inverse(acc, quant.D.data(), padded);
  memcpy(out, acc, dim * sizeof(float));
  return n;
}

int AttentionHead::compute_batch(const float *queries, int num_queries,
                                 float *outs) const {
  if (kv_buf.size == 0) {
    memset(outs, 0, num_queries * dim * sizeof(float));
    return 0;
  }

  // Auto-tune batch threading depending on active queries
  // thread_local buffers in compute() ensure OpenMP safety
#pragma omp parallel for if (num_queries > 1)
  for (int q = 0; q < num_queries; ++q) {
    compute(queries + q * dim, outs + q * dim);
  }

  return kv_buf.size;
}

#include "../include/quantizer.h"
#include "../include/codebook.h"
#include "../include/fwht.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

// Maximum padded dimension supported by thread-local scratch buffers.
// padded = next_pow2(head_dim). If head_dim > 512, padded > 1024 at 2-bit.
// Increase this constant (and recompile) if you need larger head dims.
static constexpr int ADAPTQ_TL_BUF_FLOATS = 1024;
static constexpr int ADAPTQ_TL_BUF_BYTES  = 1024;

static_assert(ADAPTQ_TL_BUF_FLOATS >= 1024,
              "Thread-local float buffer must hold at least 1024 elements");

static thread_local float   tl_float_buf[ADAPTQ_TL_BUF_FLOATS];
static thread_local uint8_t tl_idx_buf[ADAPTQ_TL_BUF_BYTES];

#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#pragma GCC push_options
#pragma GCC target("avx2")

static inline __m256 dequant_lup8(__m256i indices, __m256 low,
                                  __m256 high) {
  const __m256i low_mask = _mm256_set1_epi32(7);
  const __m256i high_indices = _mm256_cmpgt_epi32(indices, low_mask);
  const __m256i lane_indices = _mm256_and_si256(indices, low_mask);
  return _mm256_blendv_ps(
      _mm256_permutevar8x32_ps(low, lane_indices),
      _mm256_permutevar8x32_ps(high, lane_indices),
      _mm256_castsi256_ps(high_indices));
}

static inline int unpack_index_scalar(const uint8_t *packed, int index,
                                      int bits) {
  const int bit_offset = index * bits;
  const int byte_offset = bit_offset >> 3;
  const int shift = 24 - bits - (bit_offset & 7);
  const int packed_bytes = (index * bits + bits + 7) / 8;
  uint32_t value = (uint32_t)packed[byte_offset] << 16;
  if (byte_offset + 1 < packed_bytes)
    value |= (uint32_t)packed[byte_offset + 1] << 8;
  if (byte_offset + 2 < packed_bytes)
    value |= (uint32_t)packed[byte_offset + 2];
  return (int)((value >> shift) & ((1u << bits) - 1u));
}

template<int BITS>
static void dequantize_avx2(const uint8_t *packed, int padded, float inv_sq,
                            const float *cb, float *buf) {
  alignas(32) float padded_cb[16] = {};
  const int cb_size = 1 << BITS;
  std::memcpy(padded_cb, cb, (size_t)cb_size * sizeof(float));
  const __m256 low = _mm256_load_ps(padded_cb);
  const __m256 high = _mm256_load_ps(padded_cb + 8);
  const __m256 scale = _mm256_set1_ps(inv_sq);

  constexpr int packed_step = BITS;
  int packed_offset = 0;
  int i = 0;
  for (; i + 7 < padded; i += 8) {
    const __m256i indices =
        BITS == 4 ? [&]() {
          uint32_t value;
          std::memcpy(&value, packed + packed_offset, sizeof(value));
          const __m128i high_nibbles =
              _mm_cvtepu8_epi32(_mm_cvtsi32_si128(
                  (int)((value >> 4) & 0x0F0F0F0F)));
          const __m128i low_nibbles =
              _mm_cvtepu8_epi32(
                  _mm_cvtsi32_si128((int)(value & 0x0F0F0F0F)));
          return _mm256_set_m128i(_mm_unpackhi_epi32(high_nibbles, low_nibbles),
                                  _mm_unpacklo_epi32(high_nibbles, low_nibbles));
        }()
                : BITS == 3 ? [&]() {
          const uint32_t value =
              ((uint32_t)packed[packed_offset] << 16) |
              ((uint32_t)packed[packed_offset + 1] << 8) |
              (uint32_t)packed[packed_offset + 2];
          return _mm256_and_si256(
              _mm256_srlv_epi32(
                  _mm256_set1_epi32(value),
                  _mm256_setr_epi32(21, 18, 15, 12, 9, 6, 3, 0)),
              _mm256_set1_epi32(7));
        }()
                : _mm256_and_si256(
                      _mm256_srlv_epi32(
                          _mm256_set1_epi32(
                              (uint32_t)packed[packed_offset] |
                              ((uint32_t)packed[packed_offset + 1] << 8)),
                          _mm256_setr_epi32(6, 4, 2, 0, 14, 12, 10, 8)),
                      _mm256_set1_epi32(3));

    const __m256 values = _mm256_mul_ps(
        dequant_lup8(indices, low, high), scale);
    _mm256_storeu_ps(buf + i, values);
    packed_offset += packed_step;
  }

  for (; i < padded; ++i)
    buf[i] = cb[unpack_index_scalar(packed, i, BITS)] * inv_sq;
}

static inline void scale_output_avx2(const float *input, int count,
                                      float scale, float *output) {
  const __m256 scale_vec = _mm256_set1_ps(scale);
  int i = 0;
  for (; i + 7 < count; i += 8)
    _mm256_storeu_ps(output + i,
                     _mm256_mul_ps(_mm256_loadu_ps(input + i), scale_vec));
  for (; i < count; ++i)
    output[i] = input[i] * scale;
}

#pragma GCC pop_options
#endif

void Quantizer::init(int d, uint64_t seed) {
  if (d <= 0) {
    throw std::invalid_argument("head_dim must be positive");
  }
  dim = d;
  padded = next_pow2(d);
  if (padded > ADAPTQ_TL_BUF_FLOATS || padded > ADAPTQ_TL_BUF_BYTES) {
    throw std::invalid_argument(
        "head_dim too large: padded dimension exceeds ADAPTQ_TL_BUF scratch buffers. "
        "Increase ADAPTQ_TL_BUF_FLOATS and ADAPTQ_TL_BUF_BYTES in quantizer.cpp.");
  }
  D.resize(padded);
  gen_rademacher(D.data(), padded, seed);
}

// ---------------------------------------------------------------------------
// Shared inner helper: prep buffer, L2-normalise, FWHT, soft-clip (±3σ),
// quantise, return packed indices and effective scale.
// No heap allocs — uses thread_local scratch.
// ---------------------------------------------------------------------------
static inline float _quantize_core(const float *x, int dim, int padded,
                                   const int8_t *D_vec, int bits,
                                   uint8_t *idx_out) {
  float *buf = tl_float_buf;

  // L2-normalise
  float norm = 0.f;
  for (int i = 0; i < dim; ++i) {
    buf[i] = x[i];
    norm += x[i] * x[i];
  }
  for (int i = dim; i < padded; ++i)
    buf[i] = 0.f;
  norm = sqrtf(norm + 1e-12f);
  float inv = 1.f / norm;
  for (int i = 0; i < padded; ++i)
    buf[i] *= inv;

  fwht_forward(buf, D_vec, padded);

  // ±3σ soft-clip — prevents FWHT tail-outliers from saturating codebook edges.
  // The codebooks are empirically tuned for the Rademacher-FWHT distribution,
  // so we preserve the distribution shape and only clip true outliers.
  float sp = sqrtf((float)padded);
  float sum2 = 0.f;
  for (int i = 0; i < padded; ++i) {
    float v = buf[i] * sp;
    sum2 += v * v;
  }
  float sigma = sqrtf(sum2 / (float)padded + 1e-12f);
  float clip = 3.0f * sigma;
  float inv_clip = 1.f / (clip + 1e-12f);

  for (int i = 0; i < padded; ++i) {
    float v = buf[i] * sp;
    if (v > clip)
      v = clip;
    if (v < -clip)
      v = -clip;
    buf[i] = v * inv_clip; // now in [-1, 1] matching codebook range
  }

  // Quantise
  for (int i = 0; i < padded; ++i)
    idx_out[i] = (uint8_t)quantize_fast(buf[i], bits);

  // Effective scale that dequant must multiply by
  return norm * clip;
}

QuantizedVec Quantizer::quantize(const float *x, int bits) const {
  uint8_t *idx = tl_idx_buf;
  float scale = _quantize_core(x, dim, padded, D.data(), bits, idx);

  int packed_bytes = (padded * bits + 7) / 8;
  QuantizedVec q;
  q.data.resize(packed_bytes, 0);
  pack_indices(idx, padded, bits, q.data.data());
  q.scale = scale;
  q.dim = padded;
  q.bits = bits;
  return q;
}

void Quantizer::dequantize(const QuantizedVec &q, float *out) const {
  int d = q.dim;
  const float *cb = get_codebook(q.bits);
  float inv_sq = 1.f / sqrtf((float)d);
  float *buf = tl_float_buf;
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
  if (__builtin_cpu_supports("avx2") && d >= 8 && q.bits >= 2 && q.bits <= 4) {
    if (q.bits == 4)
      dequantize_avx2<4>(q.data.data(), d, inv_sq, cb, buf);
    else if (q.bits == 3)
      dequantize_avx2<3>(q.data.data(), d, inv_sq, cb, buf);
    else
      dequantize_avx2<2>(q.data.data(), d, inv_sq, cb, buf);
  } else
#endif
  {
    uint8_t *idx = tl_idx_buf;
    unpack_indices(q.data.data(), d, q.bits, idx);
    for (int i = 0; i < d; ++i)
      buf[i] = cb[idx[i]] * inv_sq;
  }
  fwht_inverse(buf, D.data(), d);
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
  if (__builtin_cpu_supports("avx2"))
    scale_output_avx2(buf, dim, q.scale, out);
  else
#endif
  {
    for (int i = 0; i < dim; ++i)
      out[i] = buf[i] * q.scale;
  }
}

float Quantizer::quantize_into(const float *x, int bits,
                               uint8_t *packed) const {
  uint8_t *idx = tl_idx_buf;
  float scale = _quantize_core(x, dim, padded, D.data(), bits, idx);
  pack_indices(idx, padded, bits, packed);
  return scale;
}

void Quantizer::dequantize_raw(const uint8_t *packed, float scale, int padded_d,
                               int bits_arg, float *out) const {
  const float *cb = get_codebook(bits_arg);
  float inv_sq = 1.f / sqrtf((float)padded_d);
  float *buf = tl_float_buf;
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
  if (__builtin_cpu_supports("avx2") && padded_d >= 8 && bits_arg >= 2 && bits_arg <= 4) {
    if (bits_arg == 4)
      dequantize_avx2<4>(packed, padded_d, inv_sq, cb, buf);
    else if (bits_arg == 3)
      dequantize_avx2<3>(packed, padded_d, inv_sq, cb, buf);
    else
      dequantize_avx2<2>(packed, padded_d, inv_sq, cb, buf);
  } else
#endif
  {
    uint8_t *idx = tl_idx_buf;
    unpack_indices(packed, padded_d, bits_arg, idx);
    for (int i = 0; i < padded_d; ++i)
      buf[i] = cb[idx[i]] * inv_sq;
  }
  fwht_inverse(buf, D.data(), padded_d);
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
  if (__builtin_cpu_supports("avx2"))
    scale_output_avx2(buf, dim, scale, out);
  else
#endif
  {
    for (int i = 0; i < dim; ++i)
      out[i] = buf[i] * scale;
  }
}

// ---------------------------------------------------------------------------
// Bit packing — specialised fast paths (no division/modulo)
// ---------------------------------------------------------------------------
static void pack4(const uint8_t *idx, int d, uint8_t *dst) {
  int n = d >> 1;
  for (int i = 0; i < n; ++i)
    dst[i] = (uint8_t)((idx[i * 2] << 4) | (idx[i * 2 + 1] & 0xF));
}
static void unpack4(const uint8_t *src, int d, uint8_t *idx) {
  int n = d >> 1;
  for (int i = 0; i < n; ++i) {
    idx[i * 2] = src[i] >> 4;
    idx[i * 2 + 1] = src[i] & 0xF;
  }
}

static void pack2(const uint8_t *idx, int d, uint8_t *dst) {
  int n = d >> 2;
  for (int i = 0; i < n; ++i)
    dst[i] = (uint8_t)((idx[i * 4] << 6) | (idx[i * 4 + 1] << 4) |
                       (idx[i * 4 + 2] << 2) | idx[i * 4 + 3]);
}
static void unpack2(const uint8_t *src, int d, uint8_t *idx) {
  int n = d >> 2;
  for (int i = 0; i < n; ++i) {
    idx[i * 4] = (src[i] >> 6) & 3;
    idx[i * 4 + 1] = (src[i] >> 4) & 3;
    idx[i * 4 + 2] = (src[i] >> 2) & 3;
    idx[i * 4 + 3] = src[i] & 3;
  }
}

static void pack3(const uint8_t *idx, int d, uint8_t *dst) {
  int g = d / 8;
  for (int i = 0; i < g; ++i) {
    const uint8_t *s = idx + i * 8;
    uint8_t *p = dst + i * 3;
    p[0] = (uint8_t)((s[0] << 5) | (s[1] << 2) | (s[2] >> 1));
    p[1] = (uint8_t)((s[2] << 7) | (s[3] << 4) | (s[4] << 1) | (s[5] >> 2));
    p[2] = (uint8_t)((s[5] << 6) | (s[6] << 3) | s[7]);
  }
}
static void unpack3(const uint8_t *src, int d, uint8_t *idx) {
  int g = d / 8;
  for (int i = 0; i < g; ++i) {
    const uint8_t *p = src + i * 3;
    uint8_t *s = idx + i * 8;
    s[0] = (p[0] >> 5) & 7;
    s[1] = (p[0] >> 2) & 7;
    s[2] = ((p[0] & 3) << 1) | (p[1] >> 7);
    s[3] = (p[1] >> 4) & 7;
    s[4] = (p[1] >> 1) & 7;
    s[5] = ((p[1] & 1) << 2) | (p[2] >> 6);
    s[6] = (p[2] >> 3) & 7;
    s[7] = p[2] & 7;
  }
}

void pack_indices(const uint8_t *indices, int d, int bits, uint8_t *dst) {
  switch (bits) {
  case 2:
    pack2(indices, d, dst);
    return;
  case 3:
    pack3(indices, d, dst);
    return;
  case 4:
    pack4(indices, d, dst);
    return;
  default:
    throw std::invalid_argument("Unsupported bit width in pack_indices (expected 2, 3, or 4)");
  }
}
void unpack_indices(const uint8_t *src, int d, int bits, uint8_t *indices) {
  switch (bits) {
  case 2:
    unpack2(src, d, indices);
    return;
  case 3:
    unpack3(src, d, indices);
    return;
  case 4:
    unpack4(src, d, indices);
    return;
  default:
    throw std::invalid_argument("Unsupported bit width in unpack_indices (expected 2, 3, or 4)");
  }
}

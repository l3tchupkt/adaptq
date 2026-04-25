#pragma once
#include <cstdint>
#include <cstddef>

// Pre-computed Max-Lloyd optimal codebooks for unit-variance Gaussian distribution.
// Entries are sorted ascending. All quantization operates on rotated vectors
// scaled to unit variance (i.e., after FWHT + 1/sqrt(d) normalization).

// 2-bit codebook: 4 centroids
extern const float CB2[4];

// 3-bit codebook: 8 centroids
extern const float CB3[8];

// 4-bit codebook: 16 centroids
extern const float CB4[16];

// Returns the codebook for the given bit-width (2, 3, or 4).
const float* get_codebook(int bits);

// Returns codebook size (2^bits).
int codebook_size(int bits);

// Find nearest centroid index via linear scan (used for correctness tests).
int quantize_scalar(float val, const float* cb, int cb_size);

// Branchless nearest-centroid via precomputed midpoint thresholds.
// bits must be 2, 3, or 4.  No branches → no mispredictions.
int quantize_fast(float v, int bits);

// Retrieve centroid value by index.
float dequantize_scalar(int idx, const float* cb);

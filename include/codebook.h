#pragma once
#include <cstdint>
#include <cstddef>

// Pre-computed Lloyd-Max minimum-MSE scalar quantizer codebooks for N(0,1).
// Reference: J. Max, "Quantizing for minimum distortion", IRE Trans. Inf.
// Theory, 1960; Jayant & Noll, "Digital Coding of Waveforms", 1984.
//
// Symmetry invariant: CB[i] == -CB[N-1-i] where N = 2^bits.
// All entries are strictly ascending.
// Domain: values quantized here must be approximately N(0,1) distributed.
//
// After FWHT + L2-normalisation + √p scaling, KV vectors are approximately
// N(0,1) distributed. Soft-clipping to ±3σ handles rare outliers.

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

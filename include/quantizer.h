#pragma once
#include <cstdint>
#include <vector>
#include "fwht.h"
#include "codebook.h"

// Quantized vector: packed bits + scale factor
struct QuantizedVec {
    std::vector<uint8_t> data;   // packed quantized indices
    float                scale;  // L2 norm of original rotated vector (for reconstruction)
    int                  dim;    // original dimension
    int                  bits;   // bits per coordinate (2, 3, or 4)
};

// HAR + MSE vector quantizer.
// One instance per (layer, head) — holds the Rademacher D vector.
struct Quantizer {
    std::vector<int8_t> D;       // Rademacher sign vector, length = padded_dim
    int                  dim;    // original head dimension d_h
    int                  padded; // next power of 2 >= dim

    // Initialize with head dimension and random seed.
    void init(int d, uint64_t seed);

    // Quantize a float vector of length dim. Returns packed QuantizedVec.
    QuantizedVec quantize(const float* x, int bits) const;

    // Quantize and write directly into a caller-allocated byte buffer.
    // packed must have (padded * bits + 7) / 8 bytes. Returns L2 norm.
    float quantize_into(const float* x, int bits, uint8_t* packed) const;

    // Reconstruct float vector from QuantizedVec into out[dim].
    void dequantize(const QuantizedVec& q, float* out) const;

    // Reconstruct directly from raw packed bytes (KVFlatBuffer path).
    void dequantize_raw(const uint8_t* packed, float scale,
                        int padded_d, int bits_arg, float* out) const;
};

// Bit packing / unpacking helpers (public for testing).
// Pack d indices of `bits` bits into a byte array. Dst must have ceil(d*bits/8) bytes.
void pack_indices(const uint8_t* indices, int d, int bits, uint8_t* dst);
void unpack_indices(const uint8_t* src, int d, int bits, uint8_t* indices);

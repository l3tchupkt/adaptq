#pragma once
#include <cstdint>

// Fast Walsh-Hadamard Transform utilities.
// All transforms are in-place on float arrays.

// Next power of two >= n
int next_pow2(int n);

// Generate a random Rademacher sign vector (+1 or -1) of length d.
// Writes d int8_t values.
void gen_rademacher(int8_t* D, int d, uint64_t seed);

// Apply Rademacher diagonal D then normalized FWHT.
// x has length d (must be power of 2 or will be zero-padded internally).
// Operates on a work buffer of size next_pow2(d).
void fwht_forward(float* x, const int8_t* D, int d);

// Inverse: FWHT is self-inverse up to sign; we just apply forward again
// (same butterfly), then reapply D (since D^{-1} = D for ±1 diagonal).
void fwht_inverse(float* x, const int8_t* D, int d);

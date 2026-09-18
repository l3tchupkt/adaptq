#pragma once

// Internal AVX2 softmax entry point used by the attention implementation.
// Falls back to the scalar implementation on builds without AVX2 support.
void softmax_avx2(float *x, int n);

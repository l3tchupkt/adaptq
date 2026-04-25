/* adapter_standalone.c
 *
 * Plain-C adapter example — no C++, no Python, no llama.cpp.
 * Uses the pure C adaptq_* API directly via the C vtable.
 *
 * Compile:
 *   gcc -O2 -Iinclude adapters/adapter_standalone.c -L. -ladaptq -o
 * standalone_test
 *
 * This is the reference adapter showing the minimal integration contract.
 */

#include "../include/adaptq.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- tiny helper ---- */
static void rand_vec(float *v, int n) {
  for (int i = 0; i < n; ++i)
    v[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
}

int main(void) {
  const int N_HEADS = 4;
  const int HEAD_DIM = 128;
  const int BITS = 4;
  const int CAPACITY = 2048;
  const int N_TOKENS = 1024;

  /* Create multi-head context via the C API */
  adaptq_mha_t mha = adaptq_mha_create(N_HEADS, HEAD_DIM, BITS, CAPACITY,
                                       0xDEADBEEFULL, 0.95f, 512);
  if (!mha) {
    fprintf(stderr, "adaptq_mha_create failed\n");
    return 1;
  }

  float *key = malloc(HEAD_DIM * sizeof(float));
  float *val = malloc(HEAD_DIM * sizeof(float));
  float *q = malloc(HEAD_DIM * sizeof(float));
  float *out = malloc(HEAD_DIM * sizeof(float));

  /* Populate KV cache */
  for (int tok = 0; tok < N_TOKENS; ++tok) {
    for (int h = 0; h < N_HEADS; ++h) {
      rand_vec(key, HEAD_DIM);
      rand_vec(val, HEAD_DIM);
      adaptq_mha_append(mha, h, key, val, tok);
    }
  }

  /* Compute attention for each head */
  rand_vec(q, HEAD_DIM);
  for (int h = 0; h < N_HEADS; ++h) {
    adaptq_mha_compute(mha, h, q, out);
    printf("[head %d] out[0]=%.4f  kv_bytes=%zu\n", h, out[0],
           adaptq_mha_total_kv_bytes(mha));
  }

  adaptq_mha_reset(mha);
  adaptq_mha_destroy(mha);
  free(key);
  free(val);
  free(q);
  free(out);
  printf("OK\n");
  return 0;
}

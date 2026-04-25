#include "../include/codebook.h"
#include <cmath>
#include <cfloat>

const float CB2[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f
};

const float CB3[8] = {
    -2.1529f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1529f
};

const float CB4[16] = {
    -2.7326f, -2.0690f, -1.5860f, -1.1880f,
    -0.8491f, -0.5480f, -0.2633f,  0.0000f,
     0.2633f,  0.5480f,  0.8491f,  1.1880f,
     1.5860f,  2.0690f,  2.7326f,  3.5714f
};

// Decision thresholds: midpoints between adjacent centroids.
// Quantize to index i iff THRESH[i-1] <= v < THRESH[i].
static const float THRESH2[3] = {
    (-1.5104f-0.4528f)*0.5f,   // -0.9816
    0.0f,                       //  0.0000
    ( 0.4528f+1.5104f)*0.5f,   //  0.9816
};

static const float THRESH3[7] = {
    (-2.1529f-1.3439f)*0.5f,
    (-1.3439f-0.7560f)*0.5f,
    (-0.7560f-0.2451f)*0.5f,
    0.0f,
    ( 0.2451f+0.7560f)*0.5f,
    ( 0.7560f+1.3439f)*0.5f,
    ( 1.3439f+2.1529f)*0.5f,
};

static const float THRESH4[15] = {
    (-2.7326f-2.0690f)*0.5f,
    (-2.0690f-1.5860f)*0.5f,
    (-1.5860f-1.1880f)*0.5f,
    (-1.1880f-0.8491f)*0.5f,
    (-0.8491f-0.5480f)*0.5f,
    (-0.5480f-0.2633f)*0.5f,
    (-0.2633f+0.0000f)*0.5f,
    ( 0.0000f+0.2633f)*0.5f,
    ( 0.2633f+0.5480f)*0.5f,
    ( 0.5480f+0.8491f)*0.5f,
    ( 0.8491f+1.1880f)*0.5f,
    ( 1.1880f+1.5860f)*0.5f,
    ( 1.5860f+2.0690f)*0.5f,
    ( 2.0690f+2.7326f)*0.5f,
    ( 2.7326f+3.5714f)*0.5f,
};

const float* get_codebook(int bits) {
    switch (bits) {
        case 2: return CB2;
        case 3: return CB3;
        case 4: return CB4;
        default: return CB4;
    }
}

int codebook_size(int bits) { return 1 << bits; }

// Branchless binary-search via conditional adds.
// No branch mispredictions; compiles to cmov chains.
int quantize_fast(float v, int bits) {
    if (bits == 4) {
        // 4-level binary search over 15 thresholds → index in [0,15]
        int i = (v >= THRESH4[7]) ? 8 : 0;
        i    += (v >= THRESH4[i + 3]) ? 4 : 0;
        i    += (v >= THRESH4[i + 1]) ? 2 : 0;
        i    += (v >= THRESH4[i    ]) ? 1 : 0;
        return i;
    }
    if (bits == 3) {
        int i = (v >= THRESH3[3]) ? 4 : 0;
        i    += (v >= THRESH3[i + 1]) ? 2 : 0;
        i    += (v >= THRESH3[i    ]) ? 1 : 0;
        return i;
    }
    // bits == 2
    int i = (v >= THRESH2[1]) ? 2 : 0;
    i    += (v >= THRESH2[i ]) ? 1 : 0;
    return i;
}

// Legacy fallback (used in tests for correctness comparison)
int quantize_scalar(float val, const float* cb, int cb_size) {
    int best = 0;
    float best_dist = fabsf(val - cb[0]);
    for (int i = 1; i < cb_size; ++i) {
        float d = fabsf(val - cb[i]);
        if (d < best_dist) { best_dist = d; best = i; }
    }
    return best;
}

float dequantize_scalar(int idx, const float* cb) { return cb[idx]; }

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

int codebook_size(int bits) {
    if (bits <= 2) return 4;
    if (bits == 3) return 8;
    return 16;
}

// Branchless binary-search via conditional adds.
// No branch mispredictions; compiles to cmov chains.
int quantize_fast(float v, int bits) {
    if (!std::isfinite(v)) {
        // Handle NaN gracefully: map to closest-to-zero centroid
        if (bits <= 2) return 1; // CB2[1] = -0.4528f
        if (bits == 3) return 3; // CB3[3] = -0.2451f
        return 7;                // CB4[7] = 0.0000f
    }
    if (bits >= 4) {
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
    // bits <= 2
    int i = (v >= THRESH2[1]) ? 2 : 0;
    i    += (v >= THRESH2[i ]) ? 1 : 0;
    return i;
}

// Legacy fallback (used in tests for correctness comparison)
int quantize_scalar(float val, const float* cb, int cb_size) {
    if (!cb || cb_size <= 0) return 0;
    if (std::isnan(val)) return 0;
    if (val >= cb[cb_size - 1]) return cb_size - 1;
    if (val <= cb[0]) return 0;
    int best = 0;
    float best_dist = fabsf(val - cb[0]);
    for (int i = 1; i < cb_size; ++i) {
        float d = fabsf(val - cb[i]);
        if (d < best_dist) { best_dist = d; best = i; }
    }
    return best;
}


float dequantize_scalar(int idx, const float* cb) {
    if (!cb) return 0.0f;
    if (idx < 0) idx = 0;
    return cb[idx];
}


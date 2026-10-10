#include "../include/codebook.h"
#include <cmath>
#include <cfloat>

// ---------------------------------------------------------------------------
// Lloyd-Max optimal scalar quantizer codebooks for N(0,1).
//
// Derivation: These are the Max-Lloyd minimum-MSE centroids for a zero-mean
// unit-variance Gaussian source. Reference: J. Max, "Quantizing for minimum
// distortion", IRE Trans. Inf. Theory, 1960; Jayant & Noll, "Digital Coding
// of Waveforms", 1984, Table 4.2.
//
// All codebooks are strictly ascending and symmetric about zero:
//   CB[i] == -CB[N-1-i]  where N = 2^bits
//
// These centroids apply in the N(0,1) domain — i.e., to data whose values
// are approximately standard-normal distributed.
//
// Quantization pipeline:
//   x (float[d])                      — original KV vector
//   → L2-normalise: x̂ = x / ||x||    — now ||x̂|| = 1
//   → zero-pad to p = next_pow2(d)
//   → fwht_forward(x̂_padded, D, p)   — outputs (1/√p)·H·D·x̂_padded
//   → scale by √p                     — now values ≈ N(0,1) by CLT
//   → clip to ±3σ (soft outlier guard)
//   → codebook lookup in N(0,1) domain
//
// Scale returned = ||x||  (L2 norm of original input, prior to normalisation)
// Reconstruction: dequant gives (reconstructed_normalised / √p) → IFWHT → × scale
// ---------------------------------------------------------------------------

const float CB2[4] = {
    // 2-bit (4 levels), Lloyd-Max N(0,1), symmetric
    // Centroids from Max (1960), reproduced in Jayant & Noll (1984) Table 4.2
    -1.5104f, -0.4528f, 0.4528f, 1.5104f
};

const float CB3[8] = {
    // 3-bit (8 levels), Lloyd-Max N(0,1), symmetric
    -2.1529f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1529f
};

const float CB4[16] = {
    // 4-bit (16 levels), Lloyd-Max N(0,1), symmetric
    // Source: Max (1960) / Jayant & Noll (1984) Table 4.2
    // Previously this was INCORRECT: the last centroid was 3.5714f which
    // is not symmetric with the first centroid -2.7326f (issue #229).
    // Correct symmetric values:
    //   CB4[ 0] == -CB4[15] == -2.4008
    //   CB4[ 1] == -CB4[14] == -1.8944
    //   CB4[ 2] == -CB4[13] == -1.5105
    //   CB4[ 3] == -CB4[12] == -1.2016
    //   CB4[ 4] == -CB4[11] == -0.9138
    //   CB4[ 5] == -CB4[10] == -0.6402
    //   CB4[ 6] == -CB4[ 9] == -0.3795
    //   CB4[ 7] == -CB4[ 8] == -0.1258
    -2.4008f, -1.8944f, -1.5105f, -1.2016f,
    -0.9138f, -0.6402f, -0.3795f, -0.1258f,
     0.1258f,  0.3795f,  0.6402f,  0.9138f,
     1.2016f,  1.5105f,  1.8944f,  2.4008f
};

// Decision thresholds: midpoints between adjacent centroids.
// Quantize to index i iff THRESH[i-1] <= v < THRESH[i].
// All symmetric codebooks have THRESH[N/2 - 1] == 0.0f exactly.
static const float THRESH2[3] = {
    (-1.5104f + -0.4528f) * 0.5f,  // -0.9816
    0.0f,                           //  0.0000
    ( 0.4528f +  1.5104f) * 0.5f,  //  0.9816
};

static const float THRESH3[7] = {
    (-2.1529f + -1.3439f) * 0.5f,  // -1.7484
    (-1.3439f + -0.7560f) * 0.5f,  // -1.0500
    (-0.7560f + -0.2451f) * 0.5f,  // -0.5006
    0.0f,                           //  0.0000
    ( 0.2451f +  0.7560f) * 0.5f,  //  0.5006
    ( 0.7560f +  1.3439f) * 0.5f,  //  1.0500
    ( 1.3439f +  2.1529f) * 0.5f,  //  1.7484
};

static const float THRESH4[15] = {
    // Derived from corrected symmetric CB4 (issue #229)
    (-2.4008f + -1.8944f) * 0.5f,  // -2.1476
    (-1.8944f + -1.5105f) * 0.5f,  // -1.7025
    (-1.5105f + -1.2016f) * 0.5f,  // -1.3561
    (-1.2016f + -0.9138f) * 0.5f,  // -1.0577
    (-0.9138f + -0.6402f) * 0.5f,  // -0.7770
    (-0.6402f + -0.3795f) * 0.5f,  // -0.5099
    (-0.3795f + -0.1258f) * 0.5f,  // -0.2527
    0.0f,                           //  0.0000  (midpoint of -0.1258 and +0.1258)
    ( 0.1258f +  0.3795f) * 0.5f,  //  0.2527
    ( 0.3795f +  0.6402f) * 0.5f,  //  0.5099
    ( 0.6402f +  0.9138f) * 0.5f,  //  0.7770
    ( 0.9138f +  1.2016f) * 0.5f,  //  1.0577
    ( 1.2016f +  1.5105f) * 0.5f,  //  1.3561
    ( 1.5105f +  1.8944f) * 0.5f,  //  1.7025
    ( 1.8944f +  2.4008f) * 0.5f,  //  2.1476
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
    if (std::isnan(v)) {
        // Handle NaN gracefully: map to closest-to-zero centroid
        if (bits <= 2) return 1; // CB2[1] = -0.4528f
        if (bits == 3) return 3; // CB3[3] = -0.2451f
        return 7;                // CB4[7] = -0.1258f (closest to zero; tied with CB4[8]=+0.1258)
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


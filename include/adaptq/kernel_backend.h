#pragma once
#include <cstdint>
#include <memory>

// IKernelBackend isolates the hardware-specific execution paths (Scalar, AVX2, etc.)
// from the AttentionHead control flow.
struct IKernelBackend {
    virtual ~IKernelBackend() = default;

    // Executes the K-dot scan, V-accumulation, and softmax fusion.
    // To satisfy "Phase 5 - NO VTABLES IN THE INNER LOOP", this is a coarse-grained
    // virtual method called EXACTLY ONCE per sequence query, passing all slots at once.
    virtual void compute_attention(
        const float* q_rot,
        float* v_accum,
        const float* codebook,
        const uint8_t* k_data,
        const uint8_t* v_data,
        const float* k_scale,
        const float* v_scale,
        int bits,
        int dim,
        int padded_dim,
        int packed_bytes,
        int num_slots,
        const int* slots,
        float v_mass_thresh,
        float* logits_workspace
    ) const = 0;
};

std::unique_ptr<IKernelBackend> create_auto_kernel();

#pragma once
#include "policy.h"
#include "storage_backend.h"
#include "kernel_backend.h"
#include <memory>

// Factory to create the 0.2.3 reference adapters.
struct RuntimeFactory {
    // Returns a policy that statically enforces the configured bit-width and no eviction.
    static std::unique_ptr<IPolicy> create_default_policy(int bits);

    // Returns an IStorageBackend wrapping the legacy KVFlatBuffer allocation logic.
    static std::unique_ptr<IStorageBackend> create_contiguous_storage(int capacity, int padded_dim, int bits);

    // Automatically detects CPU features (e.g. AVX2/FMA) and returns the fastest available kernel.
    static std::unique_ptr<IKernelBackend> create_auto_kernel();
};

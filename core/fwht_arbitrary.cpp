#include "fwht_arbitrary.h"

namespace adaptq {

// Standalone functions for external C interoperability
extern "C" {

void *adaptq_fwht_adapter_create(int orig_dim, uint64_t seed) {
    try {
        return new ArbitraryDimFWHTAdapter(orig_dim, seed);
    } catch (...) {
        return nullptr;
    }
}

void adaptq_fwht_adapter_destroy(void *adapter) {
    if (adapter) {
        delete static_cast<ArbitraryDimFWHTAdapter *>(adapter);
    }
}

int adaptq_fwht_adapter_padded_dim(void *adapter) {
    if (!adapter) return 0;
    return static_cast<ArbitraryDimFWHTAdapter *>(adapter)->padded_dim();
}

int adaptq_fwht_adapter_forward(void *adapter, const float *src, float *dst_pad) {
    if (!adapter || !src || !dst_pad) return -1;
    try {
        static_cast<ArbitraryDimFWHTAdapter *>(adapter)->forward(src, dst_pad);
        return 0;
    } catch (...) {
        return -1;
    }
}

int adaptq_fwht_adapter_inverse(void *adapter, const float *src_pad, float *dst_orig) {
    if (!adapter || !src_pad || !dst_orig) return -1;
    try {
        static_cast<ArbitraryDimFWHTAdapter *>(adapter)->inverse(src_pad, dst_orig);
        return 0;
    } catch (...) {
        return -1;
    }
}

} // extern "C"

} // namespace adaptq

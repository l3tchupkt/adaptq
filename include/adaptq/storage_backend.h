#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

// IStorageBackend abstracts the cache layout, memory allocation, and layout access.
// By avoiding assumption of a paged layout in 0.2.3, it can adapt the existing ContiguousStorage.
struct IStorageBackend {
    virtual ~IStorageBackend() = default;

    virtual int capacity() const = 0;
    virtual int size() const = 0;
    virtual int padded_dim() const = 0;
    virtual int bits() const = 0;
    virtual int packed_bytes() const = 0;
    virtual int head() const = 0;

    // Append a quantized key/value pair. Returns the internal slot/block allocated.
    virtual int insert(const uint8_t* k_packed, float ks, 
                       const uint8_t* v_packed, float vs, 
                       int position) = 0;

    // Accessors for the Kernel Backend
    // These abstract away how the underlying bytes are managed (contiguous vs paged).
    // For 0.2.3, we return raw contiguous pointers to map directly to the existing AVX2 kernel.
    virtual const uint8_t* k_data_ptr() const = 0;
    virtual const uint8_t* v_data_ptr() const = 0;
    virtual const float* k_scale_ptr() const = 0;
    virtual const float* v_scale_ptr() const = 0;

    // Populate a vector with the chronological slots for computing attention
    virtual void get_ordered_slots(std::vector<int>& out_slots) const = 0;

    // Reset the internal memory structures
    virtual void clear() = 0;

    // Memory accounting
    virtual size_t kv_bytes() const = 0;
    virtual size_t k_scan_bytes() const = 0;
};

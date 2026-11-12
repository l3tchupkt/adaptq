
#include "../include/ring_buffer.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace {

size_t checked_multiply(size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        throw std::length_error("KV buffer size overflow");
    return a * b;
}

size_t checked_align_64(size_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max() - 63)
        throw std::length_error("KV buffer alignment overflow");
    return (bytes + 63) & ~size_t(63);
}

size_t packed_size(int padded, int bits) {
    if (padded <= 0 || bits <= 0 || bits > 16)
        throw std::invalid_argument("Invalid dimension or bit width");

    if (bits == 16)
        return checked_multiply(static_cast<size_t>(padded), sizeof(float));

    const size_t total_bits =
        checked_multiply(static_cast<size_t>(padded),
                         static_cast<size_t>(bits));

    if (total_bits > std::numeric_limits<size_t>::max() - 7)
        throw std::length_error("Packed KV size overflow");

    return (total_bits + 7) / 8;
}

} // namespace

// KVRingBuffer

void KVRingBuffer::init(int cap) {
    if (cap <= 0)
        throw std::invalid_argument("Ring buffer capacity must be positive");

    slots.clear();
    slots.resize(static_cast<size_t>(cap));

    capacity = cap;
    head = 0;
    size = 0;
}

int KVRingBuffer::insert(QuantizedVec qk, QuantizedVec qv,
                         int token_pos, float importance) {
    if (capacity <= 0 || slots.empty())
        throw std::logic_error("Ring buffer is not initialized");

    const int idx = head;

    slots[idx].qk = std::move(qk);
    slots[idx].qv = std::move(qv);
    slots[idx].pos = token_pos;
    slots[idx].importance = importance;

    head = (head + 1) % capacity;

    if (size < capacity)
        ++size;

    return idx;
}

void KVRingBuffer::reset() {
    head = 0;
    size = 0;

    for (auto &slot : slots) {
        slot.qk.data.clear();
        slot.qv.data.clear();
        slot.pos = 0;
        slot.importance = 0.0f;
    }
}

size_t KVRingBuffer::memory_bytes() const {
    size_t total = 0;

    if (capacity <= 0 || size <= 0)
        return total;

    for (int i = 0; i < size; ++i) {
        const int idx = (head - size + capacity + i) % capacity;

        const size_t k_size = slots[idx].qk.data.size();
        const size_t v_size = slots[idx].qv.data.size();

        if (k_size > std::numeric_limits<size_t>::max() - total)
            throw std::length_error("Ring buffer memory size overflow");

        total += k_size;

        if (v_size > std::numeric_limits<size_t>::max() - total)
            throw std::length_error("Ring buffer memory size overflow");

        total += v_size;
    }

    return total;
}

// KVFlatBuffer

void KVFlatBuffer::init(int cap, int padded, int k_b, int v_b) {
    if (cap <= 0)
        throw std::invalid_argument("Flat buffer capacity must be positive");

    const size_t new_k_packed_bytes = packed_size(padded, k_b);
    const size_t new_v_packed_bytes = packed_size(padded, v_b);

    const size_t k_total = checked_align_64(
        checked_multiply(static_cast<size_t>(cap), new_k_packed_bytes));

    const size_t v_total = checked_align_64(
        checked_multiply(static_cast<size_t>(cap), new_v_packed_bytes));

    // Release previous allocation.
    free_aligned();

    capacity = 0;
    head = 0;
    size = 0;

    uint8_t *new_k_data = nullptr;
    uint8_t *new_v_data = nullptr;

#if defined(_MSC_VER)
    new_k_data = static_cast<uint8_t *>(_aligned_malloc(k_total, 64));
    if (!new_k_data)
        throw std::bad_alloc();

    new_v_data = static_cast<uint8_t *>(_aligned_malloc(v_total, 64));
    if (!new_v_data) {
        _aligned_free(new_k_data);
        throw std::bad_alloc();
    }
#else
    if (posix_memalign(reinterpret_cast<void **>(&new_k_data), 64, k_total))
        throw std::bad_alloc();

    if (posix_memalign(reinterpret_cast<void **>(&new_v_data), 64, v_total)) {
        std::free(new_k_data);
        throw std::bad_alloc();
    }
#endif

    std::memset(new_k_data, 0, k_total);
    std::memset(new_v_data, 0, v_total);

    try {
        k_scale.assign(static_cast<size_t>(cap), 1.0f);
        v_scale.assign(static_cast<size_t>(cap), 1.0f);
        positions.assign(static_cast<size_t>(cap), 0);
    } catch (...) {
#if defined(_MSC_VER)
        _aligned_free(new_k_data);
        _aligned_free(new_v_data);
#else
        std::free(new_k_data);
        std::free(new_v_data);
#endif
        throw;
    }

    k_data = new_k_data;
    v_data = new_v_data;

    capacity = cap;
    padded_dim = padded;
    k_bits = k_b;
    v_bits = v_b;
    k_packed_bytes = new_k_packed_bytes;
    v_packed_bytes = new_v_packed_bytes;

    head = 0;
    size = 0;
}

void KVFlatBuffer::free_aligned() {
    if (k_data) {
#if defined(_MSC_VER)
        _aligned_free(k_data);
#else
        std::free(k_data);
#endif
        k_data = nullptr;
    }

    if (v_data) {
#if defined(_MSC_VER)
        _aligned_free(v_data);
#else
        std::free(v_data);
#endif
        v_data = nullptr;
    }
}

int KVFlatBuffer::insert(const uint8_t *kp, float ks,
                         const uint8_t *vp, float vs, int pos) {
    if (capacity <= 0 || !k_data || !v_data)
        throw std::logic_error("Flat buffer is not initialized");

    if (!kp || !vp)
        throw std::invalid_argument("KV input pointers cannot be null");

    const int idx = head;

    std::memcpy(
        k_data + static_cast<size_t>(idx) * k_packed_bytes,
        kp, k_packed_bytes);

    std::memcpy(
        v_data + static_cast<size_t>(idx) * v_packed_bytes,
        vp, v_packed_bytes);

    k_scale[idx] = ks;
    v_scale[idx] = vs;
    positions[idx] = pos;

    head = (head + 1) % capacity;

    if (size < capacity)
        ++size;

    return idx;
}


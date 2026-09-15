#pragma once
#include "quantizer.h"
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

// 64-byte aligned allocator for Cache-line boundary mapped K/V storage
template <class T, size_t Align = 64> struct AlignedAllocator {
  using value_type = T;
  T *allocate(size_t n) {
    size_t bytes = n * sizeof(T);
    bytes = (bytes + Align - 1) & ~(Align - 1);
    void *p = nullptr;
#if defined(_MSC_VER)
    p = _aligned_malloc(bytes, Align);
    if (!p)
      throw std::bad_alloc();
#else
    if (posix_memalign(&p, Align, bytes) != 0)
      throw std::bad_alloc();
#endif
    return static_cast<T *>(p);
  }
  void deallocate(T *p, size_t /*n*/) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
  }
  template <class U> struct rebind {
    using other = AlignedAllocator<U, Align>;
  };
  AlignedAllocator() noexcept = default;
  template <class U>
  AlignedAllocator(const AlignedAllocator<U, Align> &) noexcept {}
  bool operator==(const AlignedAllocator &) const { return true; }
  bool operator!=(const AlignedAllocator &) const { return false; }
};

// Original scatter-allocated ring buffer (kept for backward compat).
struct KVSlot {
  QuantizedVec qk;
  QuantizedVec qv;
  int pos;
  float importance;
};

struct KVRingBuffer {
  std::vector<KVSlot> slots;
  int capacity, head, size;
  void init(int cap);
  int insert(QuantizedVec qk, QuantizedVec qv, int token_pos, float importance);
  void reset();
  size_t memory_bytes() const;
};

// Contiguous flat KV storage: K data packed sequentially, V data packed
// sequentially. Makes large-context DRAM reads fully sequential for the K scan.
struct KVFlatBuffer {
  uint8_t *k_data = nullptr; // capacity * packed_bytes
  uint8_t *v_data = nullptr;
  std::vector<float> k_scale;
  std::vector<float> v_scale;
  std::vector<int> positions;

  int capacity;
  int padded_dim;
  int bits;
  int packed_bytes; // (padded_dim * bits + 7) / 8
  int head;
  int size;

  void init(int cap, int padded, int b);
  void free_aligned();
  ~KVFlatBuffer() { free_aligned(); }

  // Returns slot index written.
  int insert(const uint8_t *k_packed, float ks, const uint8_t *v_packed,
             float vs, int pos);

  const uint8_t *k_ptr(int slot) const {
    return k_data + (size_t)slot * packed_bytes;
  }
  const uint8_t *v_ptr(int slot) const {
    return v_data + (size_t)slot * packed_bytes;
  }

  // Bytes consumed by packed KV data only.
  size_t kv_bytes() const { return (size_t)size * packed_bytes * 2; }
  // Bytes for sequential K scan (K data region being read).
  size_t k_scan_bytes() const { return (size_t)size * packed_bytes; }
};

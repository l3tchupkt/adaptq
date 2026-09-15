#include "../include/ring_buffer.h"
#include <cstring>
#include <utility>

// KVRingBuffer (original, scatter-allocated — kept for backward compat)

void KVRingBuffer::init(int cap) {
  capacity = cap;
  head = 0;
  size = 0;
  slots.resize(cap);
}

int KVRingBuffer::insert(QuantizedVec qk, QuantizedVec qv, int token_pos,
                         float importance) {
  int idx = head;
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
  for (auto &s : slots) {
    s.qk.data.clear();
    s.qv.data.clear();
  }
}

size_t KVRingBuffer::memory_bytes() const {
  size_t t = 0;
  for (int i = 0; i < size; ++i) {
    int idx = (head - size + capacity + i) % capacity;
    t += slots[idx].qk.data.size() + slots[idx].qv.data.size();
  }
  return t;
}

// KVFlatBuffer (new — contiguous storage for sequential DRAM access)

void KVFlatBuffer::init(int cap, int padded, int b) {
  capacity = cap;
  padded_dim = padded;
  bits = b;
  packed_bytes = (padded * b + 7) / 8;
  head = 0;
  size = 0;

  free_aligned();

  size_t bytes = (capacity * packed_bytes + 63) & ~63;
#if defined(_MSC_VER)
  k_data = (uint8_t *)_aligned_malloc(bytes, 64);
  if (!k_data)
    throw std::bad_alloc();
  v_data = (uint8_t *)_aligned_malloc(bytes, 64);
  if (!v_data) {
    free_aligned();
    throw std::bad_alloc();
  }
#else
  if (posix_memalign((void **)&k_data, 64, bytes))
    throw std::bad_alloc();
  if (posix_memalign((void **)&v_data, 64, bytes)) {
    free_aligned();
    throw std::bad_alloc();
  }
#endif
  memset(k_data, 0, bytes);
  memset(v_data, 0, bytes);

  k_scale.assign(cap, 1.f);
  v_scale.assign(cap, 1.f);
  positions.assign(cap, 0);
}

void KVFlatBuffer::free_aligned() {
  if (k_data) {
#if defined(_MSC_VER)
    _aligned_free(k_data);
#else
    free(k_data);
#endif
    k_data = nullptr;
  }
  if (v_data) {
#if defined(_MSC_VER)
    _aligned_free(v_data);
#else
    free(v_data);
#endif
    v_data = nullptr;
  }
}

int KVFlatBuffer::insert(const uint8_t *kp, float ks, const uint8_t *vp,
                         float vs, int pos) {
  if (capacity <= 0)
    return -1;
  int idx = head;
  memcpy(k_data + (size_t)idx * packed_bytes, kp, packed_bytes);
  memcpy(v_data + (size_t)idx * packed_bytes, vp, packed_bytes);
  k_scale[idx] = ks;
  v_scale[idx] = vs;
  positions[idx] = pos;
  head = (head + 1) % capacity;
  if (size < capacity)
    ++size;
  return idx;
}

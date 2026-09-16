#pragma once
#include "../include/adaptq/storage.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

/* -------------------------------------------------------------------------
 * storage/segmented_slab.h — SegmentedSlabStorage : IStorageBackend
 *
 * Multi-format slab storage. One 64-byte-aligned slab per unique format_tag.
 * In V1 HARFixedStrategy uses a single tag (degenerates to one slab).
 * In V3 HARAdaptiveStrategy uses three tags (2/3/4-bit per token).
 *
 * StorageSlot encodes (slab_id × MAX_CAP + local_index) for O(1) read().
 * Header-only: all methods inline to allow inclusion in test TUs safely.
 * ----------------------------------------------------------------------- */

namespace adaptq {

static constexpr int SS_MAX_SLABS = 16;
static constexpr int SS_MAX_CAP   = 65536;
static constexpr int SS_ENCODE    = SS_MAX_CAP;

class SegmentedSlabStorage final : public IStorageBackend {
public:
    SegmentedSlabStorage() = default;
    ~SegmentedSlabStorage() override {
        for (int i = 0; i < n_slabs_; ++i)
            if (slabs_[i].data) aligned_free(slabs_[i].data);
    }

    void init(int capacity, int max_slot_bytes) override {
        if (capacity <= 0 || capacity > SS_MAX_CAP || max_slot_bytes <= 0)
            throw std::invalid_argument("SegmentedSlabStorage::init: capacity must be in [1, 65536] and max_slot_bytes > 0");
        for (int i = 0; i < n_slabs_; ++i) {
            if (slabs_[i].data) {
                aligned_free(slabs_[i].data);
                slabs_[i].data = nullptr;
            }
            slabs_[i] = Slab{};
        }
        capacity_       = capacity;
        max_slot_bytes_ = max_slot_bytes;
        n_slabs_        = 0;
        total_used_     = 0;
    }

    void reset() override {
        for (int i = 0; i < n_slabs_; ++i) {
            slabs_[i].size = slabs_[i].head = slabs_[i].next_slot = 0;
            slabs_[i].data_sizes.assign(capacity_, 0);
        }
        total_used_ = 0;
    }

    StorageSlot write(const uint8_t *data, int data_bytes,
                      float scale, uint8_t format_tag) override {
        if (!data || data_bytes <= 0 || data_bytes > max_slot_bytes_ || capacity_ <= 0)
            return (StorageSlot)0;
        Slab *sl = get_or_create_slab(format_tag, data_bytes);
        if (!sl) return (StorageSlot)0;
        int local;
        if (sl->size < capacity_) { local = sl->next_slot++; sl->size++; }
        else { local = sl->head; sl->head = (sl->head + 1) % capacity_; total_used_ -= sl->data_sizes[local]; }
        uint8_t *dst = sl->data + (size_t)local * sl->slot_bytes;
        memcpy(dst, data, data_bytes);
        if (data_bytes < sl->slot_bytes)
            memset(dst + data_bytes, 0, sl->slot_bytes - data_bytes);
        sl->scales[local]     = scale;
        sl->tags[local]       = format_tag;
        sl->data_sizes[local] = data_bytes;
        total_used_          += data_bytes;
        return (StorageSlot)(sl->id * SS_ENCODE + local);
    }

    CompressResult read(StorageSlot slot) const override {
        int slab_id = (int)(slot / SS_ENCODE);
        int local   = (int)(slot % SS_ENCODE);
        if (slab_id < 0 || slab_id >= n_slabs_ || local < 0 || local >= capacity_)
            return CompressResult{nullptr, 0, 0.f, 0, slot};
        const Slab &sl = slabs_[slab_id];
        return CompressResult{
            sl.data + (size_t)local * sl.slot_bytes,
            sl.slot_bytes, sl.scales[local], sl.tags[local], slot
        };
    }

    void free_slot(StorageSlot slot) override {
        int slab_id = (int)(slot / SS_ENCODE);
        int local   = (int)(slot % SS_ENCODE);
        if (slab_id < 0 || slab_id >= n_slabs_ || local < 0 || local >= (int)slabs_[slab_id].data_sizes.size())
            return;
        total_used_ -= slabs_[slab_id].data_sizes[local];
        if (total_used_ < 0) total_used_ = 0;
        slabs_[slab_id].data_sizes[local] = 0;
    }

    size_t bytes_used()     const override { return (size_t)total_used_; }
    size_t bytes_capacity() const override {
        size_t cap = 0;
        for (int i = 0; i < n_slabs_; ++i)
            cap += (size_t)capacity_ * slabs_[i].slot_bytes;
        return cap;
    }
    const char *name() const override { return "segmented_slab"; }

private:
    struct Slab {
        uint8_t             *data       = nullptr;
        std::vector<float>   scales;
        std::vector<uint8_t> tags;
        std::vector<int>      data_sizes;
        int                  id         = 0;
        int                  slot_bytes = 0;
        int                  size       = 0;
        int                  head       = 0;
        int                  next_slot  = 0;
        uint8_t              format_tag = 0xFF;
    };

    Slab slabs_[SS_MAX_SLABS] = {};
    int  n_slabs_       = 0;
    int  capacity_      = 0;
    int  max_slot_bytes_= 0;
    int  total_used_    = 0;

    Slab *get_or_create_slab(uint8_t format_tag, int data_bytes) {
        for (int i = 0; i < n_slabs_; ++i)
            if (slabs_[i].format_tag == format_tag) return &slabs_[i];
        if (n_slabs_ >= SS_MAX_SLABS) return nullptr;
        Slab &sl      = slabs_[n_slabs_];
        sl.id         = n_slabs_++;
        sl.format_tag = format_tag;
        sl.slot_bytes = data_bytes;
        size_t total  = (size_t)capacity_ * data_bytes;
        sl.data = static_cast<uint8_t *>(aligned_alloc_64(total));
        if (!sl.data) throw std::bad_alloc();
        memset(sl.data, 0, total);
        sl.scales.assign(capacity_, 0.f);
        sl.tags.assign(capacity_, format_tag);
        sl.data_sizes.assign(capacity_, 0);
        return &sl;
    }

    static void *aligned_alloc_64(size_t bytes) {
#if defined(_MSC_VER)
        return _aligned_malloc(bytes, 64);
#else
        void *p = nullptr;
        if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
        return p;
#endif
    }
    static void aligned_free(void *p) {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p);
#endif
    }
};

} /* namespace adaptq */

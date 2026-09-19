#pragma once

#include "../include/adaptq/storage.h"
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <atomic>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <memoryapi.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace adaptq {
namespace storage {

/**
 * @brief NUMA node configuration and memory pinning options.
 */
struct NumaAllocationConfig {
    int preferred_numa_node{0};         /// Target NUMA socket/node ID
    bool enable_page_pinning{true};     /// Whether to lock virtual memory pages in RAM (mlock / VirtualLock)
    size_t page_alignment{4096};        /// Page alignment boundary (typically 4096 bytes)
    bool strict_node_binding{false};    /// If true, fails if allocation cannot be fulfilled on target node
};

/**
 * @brief Runtime telemetry for NUMA-pinned storage backend.
 */
struct NumaStorageStats {
    uint64_t total_allocations{0};
    uint64_t total_deallocations{0};
    size_t allocated_bytes{0};
    size_t pinned_bytes{0};
    int active_numa_node{-1};
    bool is_locked{false};
    uint64_t write_count{0};
    uint64_t read_count{0};
    uint64_t release_count{0};
};

/**
 * @brief NUMA-Aware Memory Pool & Pinned Slab Storage Backend.
 *
 * Implements IStorageBackend with hardware NUMA node affinity and physical memory
 * page pinning (mlock / VirtualLock). Prevents operating system swapping and cross-socket
 * interconnect latency overheads for latency-critical KV cache inference.
 */
class NumaPinnedSlabStorage final : public IStorageBackend {
public:
    explicit NumaPinnedSlabStorage(const NumaAllocationConfig& config = NumaAllocationConfig{})
        : config_(config) {}

    ~NumaPinnedSlabStorage() override {
        free_numa_memory();
    }

    NumaPinnedSlabStorage(const NumaPinnedSlabStorage&) = delete;
    NumaPinnedSlabStorage& operator=(const NumaPinnedSlabStorage&) = delete;

    void init(int capacity, int max_slot_bytes) override {
        if (capacity < 2 || (capacity % 2 != 0) || max_slot_bytes <= 0) {
            throw std::invalid_argument("NumaPinnedSlabStorage::init: capacity must be >= 2 and even, max_slot_bytes > 0");
        }

        free_numa_memory();

        capacity_ = capacity;
        slot_bytes_ = max_slot_bytes;
        size_ = 0;
        head_ = 0;
        next_slot_ = 0;
        k_head_ = k_next_slot_ = k_size_ = 0;
        v_head_ = v_next_slot_ = v_size_ = 0;
        write_phase_key_ = true;

        size_t raw_bytes = static_cast<size_t>(capacity) * static_cast<size_t>(max_slot_bytes);
        // Align up to page boundary
        size_t page_size = config_.page_alignment;
        total_allocated_size_ = ((raw_bytes + page_size - 1) / page_size) * page_size;

        allocate_numa_memory(total_allocated_size_);

        scale_.assign(capacity, 0.0f);
        format_tag_.assign(capacity, 0);
        slot_lengths_.assign(capacity, 0);
    }

    void reset() override {
        size_ = head_ = next_slot_ = 0;
        k_head_ = k_next_slot_ = k_size_ = 0;
        v_head_ = v_next_slot_ = v_size_ = 0;
        write_phase_key_ = true;
        if (data_ && total_allocated_size_ > 0) {
            std::memset(data_, 0, total_allocated_size_);
        }
    }

    StorageSlot write(const uint8_t* data, int data_bytes, float scale, uint8_t format_tag) override {
        if (!data_ || capacity_ <= 0) {
            throw std::runtime_error("NumaPinnedSlabStorage::write: buffer not initialized");
        }
        if (data_bytes > slot_bytes_) {
            throw std::invalid_argument("NumaPinnedSlabStorage::write: data_bytes exceeds max_slot_bytes");
        }

        StorageSlot slot = write_role(data, data_bytes, scale, format_tag,
                                      write_phase_key_, role_capacity());
        write_phase_key_ = !write_phase_key_;
        stats_.write_count++;
        return slot;
    }

    CompressResult read(StorageSlot slot) const override {
        if (!data_ || slot >= static_cast<StorageSlot>(capacity_)) {
            return CompressResult{nullptr, 0, 0.0f, 0, ADAPTQ_INVALID_SLOT};
        }

        const uint8_t* slot_ptr = data_ + static_cast<size_t>(slot) * static_cast<size_t>(slot_bytes_);
        int bytes = slot_lengths_[slot];
        if (bytes == 0) {
            bytes = slot_bytes_;
        }

        stats_.read_count++;
        return CompressResult{
            slot_ptr,
            bytes,
            scale_[slot],
            format_tag_[slot],
            slot
        };
    }

    void release(StorageSlot slot) override {
        if (slot < static_cast<StorageSlot>(capacity_)) {
            slot_lengths_[slot] = 0;
            scale_[slot] = 0.0f;
            format_tag_[slot] = 0;
            stats_.release_count++;
        }
    }

    int count() const override {
        return size_;
    }

    int capacity() const override {
        return capacity_;
    }

    const NumaStorageStats& stats() const {
        return stats_;
    }

    const NumaAllocationConfig& config() const {
        return config_;
    }

    int role_capacity() const {
        return capacity_ / 2;
    }

private:
    void allocate_numa_memory(size_t size_bytes) {
        data_ = nullptr;
        bool locked = false;

#if defined(_WIN32)
        HANDLE hProcess = GetCurrentProcess();
        DWORD node_id = static_cast<DWORD>(config_.preferred_numa_node);

        // Attempt NUMA-specific virtual allocation
        data_ = static_cast<uint8_t*>(VirtualAllocExNuma(
            hProcess,
            NULL,
            size_bytes,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
            node_id
        ));

        // Fallback to standard VirtualAlloc if NUMA-specific node is unavailable and strict binding not enforced
        if (!data_ && !config_.strict_node_binding) {
            data_ = static_cast<uint8_t*>(VirtualAlloc(
                NULL,
                size_bytes,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE
            ));
        }

        if (data_ && config_.enable_page_pinning) {
            if (VirtualLock(data_, size_bytes)) {
                locked = true;
            }
        }
#else
        // POSIX implementation with posix_memalign and mlock
        void* ptr = nullptr;
        int rc = posix_memalign(&ptr, config_.page_alignment, size_bytes);
        if (rc == 0 && ptr) {
            data_ = static_cast<uint8_t*>(ptr);
            if (config_.enable_page_pinning) {
                if (mlock(data_, size_bytes) == 0) {
                    locked = true;
                }
            }
        }
#endif

        if (!data_) {
            // Software fallback for portable continuous operation
            if (config_.strict_node_binding) {
                throw std::bad_alloc();
            }
#if defined(_MSC_VER)
            data_ = static_cast<uint8_t*>(_aligned_malloc(size_bytes, 64));
#else
            data_ = static_cast<uint8_t*>(std::aligned_alloc(64, size_bytes));
#endif
            if (!data_) throw std::bad_alloc();
        }

        std::memset(data_, 0, size_bytes);

        stats_.total_allocations++;
        stats_.allocated_bytes = size_bytes;
        stats_.pinned_bytes = locked ? size_bytes : 0;
        stats_.active_numa_node = config_.preferred_numa_node;
        stats_.is_locked = locked;
    }

    void free_numa_memory() {
        if (!data_) return;

#if defined(_WIN32)
        if (stats_.is_locked) {
            VirtualUnlock(data_, total_allocated_size_);
        }
        VirtualFree(data_, 0, MEM_RELEASE);
#else
        if (stats_.is_locked) {
            munlock(data_, total_allocated_size_);
        }
        std::free(data_);
#endif
        data_ = nullptr;
        total_allocated_size_ = 0;
        stats_.total_deallocations++;
        stats_.allocated_bytes = 0;
        stats_.pinned_bytes = 0;
        stats_.is_locked = false;
    }

    StorageSlot write_role(const uint8_t* data, int data_bytes, float scale,
                           uint8_t format_tag, bool is_key, int role_cap) {
        int& r_head = is_key ? k_head_ : v_head_;
        int& r_next = is_key ? k_next_slot_ : v_next_slot_;
        int& r_size = is_key ? k_size_ : v_size_;
        int base_offset = is_key ? 0 : role_cap;

        StorageSlot local_slot;
        if (r_size < role_cap) {
            local_slot = static_cast<StorageSlot>(r_next++);
            r_size++;
        } else {
            local_slot = static_cast<StorageSlot>(r_head);
            r_head = (r_head + 1) % role_cap;
        }

        StorageSlot physical_slot = static_cast<StorageSlot>(base_offset) + local_slot;

        uint8_t* dst = data_ + static_cast<size_t>(physical_slot) * static_cast<size_t>(slot_bytes_);
        if (data && data_bytes > 0) {
            std::memcpy(dst, data, data_bytes);
        }
        slot_lengths_[physical_slot] = data_bytes;
        scale_[physical_slot] = scale;
        format_tag_[physical_slot] = format_tag;

        size_ = k_size_ + v_size_;
        return physical_slot;
    }

    NumaAllocationConfig config_;
    mutable NumaStorageStats stats_;
    uint8_t* data_{nullptr};
    size_t total_allocated_size_{0};
    int capacity_{0};
    int slot_bytes_{0};

    int size_{0};
    int head_{0};
    int next_slot_{0};

    int k_head_{0};
    int k_next_slot_{0};
    int k_size_{0};

    int v_head_{0};
    int v_next_slot_{0};
    int v_size_{0};

    bool write_phase_key_{true};

    std::vector<float> scale_;
    std::vector<uint8_t> format_tag_;
    std::vector<int> slot_lengths_;
};

} // namespace storage
} // namespace adaptq

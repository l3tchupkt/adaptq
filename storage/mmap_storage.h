#pragma once
#include "../include/adaptq/storage.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace adaptq {

class MMapStorage final : public IStorageBackend {
public:
    MMapStorage()
        : capacity_(0),
          max_slot_bytes_(0),
          head_(0),
          size_(0),
          next_slot_(0),
          total_mapped_bytes_(0),
          mapped_ptr_(nullptr),
          use_simulated_mmap_(false) {
#if defined(_WIN32)
        file_handle_ = INVALID_HANDLE_VALUE;
        map_handle_ = nullptr;
#else
        fd_ = -1;
#endif
    }

    ~MMapStorage() override {
        cleanup();
    }

    MMapStorage(const MMapStorage &) = delete;
    MMapStorage &operator=(const MMapStorage &) = delete;

    const char *name() const override { return "mmap"; }

    int capacity() const override { return capacity_; }

    size_t bytes_used() const override {
        return (size_t)size_ * (size_t)max_slot_bytes_;
    }

    void init(int capacity, int max_slot_bytes) override {
        if (capacity <= 0 || max_slot_bytes <= 0) {
            throw std::invalid_argument("MMapStorage::init: capacity and max_slot_bytes must be positive");
        }
        cleanup();

        capacity_ = capacity;
        max_slot_bytes_ = max_slot_bytes;
        head_ = 0;
        size_ = 0;
        next_slot_ = 0;
        total_mapped_bytes_ = (size_t)capacity * (size_t)max_slot_bytes;

        scales_.assign(capacity, 0.0f);
        format_tags_.assign(capacity, 0);
        data_sizes_.assign(capacity, 0);
        is_active_.assign(capacity, false);

        allocate_mapping(total_mapped_bytes_);
    }

    void reset() override {
        head_ = 0;
        size_ = 0;
        next_slot_ = 0;
        std::fill(scales_.begin(), scales_.end(), 0.0f);
        std::fill(format_tags_.begin(), format_tags_.end(), 0);
        std::fill(data_sizes_.begin(), data_sizes_.end(), 0);
        std::fill(is_active_.begin(), is_active_.end(), false);
        if (mapped_ptr_) {
            std::memset(mapped_ptr_, 0, total_mapped_bytes_);
        }
    }

    StorageSlot write(const uint8_t *data,
                      int data_bytes,
                      float scale,
                      uint8_t format_tag) override {
        if (!data || data_bytes <= 0 || data_bytes > max_slot_bytes_) {
            throw std::invalid_argument("MMapStorage::write: invalid data payload or length");
        }
        if (!mapped_ptr_ || capacity_ <= 0) {
            throw std::runtime_error("MMapStorage::write: backend not initialized");
        }

        uint32_t slot;
        if (size_ < capacity_) {
            slot = next_slot_++;
            size_++;
        } else {
            // Ring buffer eviction
            slot = head_;
            head_ = (head_ + 1) % capacity_;
        }

        uint8_t *dst = mapped_ptr_ + (size_t)slot * (size_t)max_slot_bytes_;
        std::memcpy(dst, data, (size_t)data_bytes);

        scales_[slot] = scale;
        format_tags_[slot] = format_tag;
        data_sizes_[slot] = data_bytes;
        is_active_[slot] = true;

        return slot;
    }

    CompressResult read(StorageSlot slot) const override {
        if (slot >= (uint32_t)capacity_ || !is_active_[slot] || !mapped_ptr_) {
            return CompressResult{nullptr, 0, 0.0f, 0xFF, ADAPTQ_INVALID_SLOT};
        }

        const uint8_t *src = mapped_ptr_ + (size_t)slot * (size_t)max_slot_bytes_;
        return CompressResult{
            src,
            data_sizes_[slot],
            scales_[slot],
            format_tags_[slot],
            slot
        };
    }

    void free_slot(StorageSlot slot) override {
        if (slot < (uint32_t)capacity_ && is_active_[slot]) {
            is_active_[slot] = false;
            data_sizes_[slot] = 0;
            if (size_ > 0) size_--;
        }
    }

    bool flush() {
        if (!mapped_ptr_ || total_mapped_bytes_ == 0) return true;
        if (use_simulated_mmap_) return true;

#if defined(_WIN32)
        return FlushViewOfFile(mapped_ptr_, total_mapped_bytes_) != 0;
#else
        return msync(mapped_ptr_, total_mapped_bytes_, MS_SYNC) == 0;
#endif
    }

    bool is_simulated() const { return use_simulated_mmap_; }

private:
    void allocate_mapping(size_t bytes) {
        use_simulated_mmap_ = false;

#if defined(_WIN32)
        map_handle_ = CreateFileMappingA(
            INVALID_HANDLE_VALUE,
            nullptr,
            PAGE_READWRITE,
            (DWORD)(bytes >> 32),
            (DWORD)(bytes & 0xFFFFFFFF),
            nullptr
        );
        if (map_handle_) {
            mapped_ptr_ = static_cast<uint8_t *>(MapViewOfFile(
                map_handle_,
                FILE_MAP_ALL_ACCESS,
                0, 0, bytes
            ));
        }
        if (!mapped_ptr_) {
            // Fallback to anonymous heap simulation if OS mapping limits hit
            fallback_simulated_alloc(bytes);
        }
#else
        fd_ = -1;
        mapped_ptr_ = static_cast<uint8_t *>(mmap(
            nullptr, bytes,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1, 0
        ));
        if (mapped_ptr_ == MAP_FAILED) {
            mapped_ptr_ = nullptr;
            fallback_simulated_alloc(bytes);
        }
#endif
        if (mapped_ptr_) {
            std::memset(mapped_ptr_, 0, bytes);
        }
    }

    void fallback_simulated_alloc(size_t bytes) {
        simulated_buffer_.resize(bytes, 0);
        mapped_ptr_ = simulated_buffer_.data();
        use_simulated_mmap_ = true;
    }

    void cleanup() {
        if (mapped_ptr_) {
            if (!use_simulated_mmap_) {
#if defined(_WIN32)
                UnmapViewOfFile(mapped_ptr_);
                if (map_handle_) {
                    CloseHandle(map_handle_);
                    map_handle_ = nullptr;
                }
#else
                munmap(mapped_ptr_, total_mapped_bytes_);
                if (fd_ >= 0) {
                    close(fd_);
                    fd_ = -1;
                }
#endif
            }
            mapped_ptr_ = nullptr;
        }
        simulated_buffer_.clear();
        use_simulated_mmap_ = false;
        total_mapped_bytes_ = 0;
        capacity_ = 0;
    }

    int capacity_;
    int max_slot_bytes_;
    int head_;
    int size_;
    uint32_t next_slot_;
    size_t total_mapped_bytes_;
    uint8_t *mapped_ptr_;
    bool use_simulated_mmap_;
    std::vector<uint8_t> simulated_buffer_;

    std::vector<float> scales_;
    std::vector<uint8_t> format_tags_;
    std::vector<int> data_sizes_;
    std::vector<bool> is_active_;

#if defined(_WIN32)
    HANDLE file_handle_;
    HANDLE map_handle_;
#else
    int fd_;
#endif
};

} // namespace adaptq

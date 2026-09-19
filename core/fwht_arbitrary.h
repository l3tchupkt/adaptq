#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "../include/fwht.h"

namespace adaptq {

class ArbitraryDimFWHTAdapter {
public:
    explicit ArbitraryDimFWHTAdapter(int orig_dim, uint64_t seed = 12345ULL)
        : orig_dim_(orig_dim), seed_(seed) {
        if (orig_dim <= 0) {
            throw std::invalid_argument("orig_dim must be positive");
        }
        pad_dim_ = next_pow2(orig_dim);
        scale_factor_ = std::sqrt(static_cast<float>(pad_dim_) / static_cast<float>(orig_dim_));
        inv_scale_factor_ = 1.0f / scale_factor_;

        rademacher_d_.resize(pad_dim_);
        gen_rademacher(rademacher_d_.data(), pad_dim_, seed_);
        work_buf_.resize(pad_dim_, 0.0f);
    }

    int original_dim() const { return orig_dim_; }
    int padded_dim() const { return pad_dim_; }
    bool is_power_of_two() const { return orig_dim_ == pad_dim_; }
    float energy_scale() const { return scale_factor_; }

    void forward(const float *src, float *dst_pad) {
        if (!src || !dst_pad) {
            throw std::invalid_argument("forward: null buffer pointers");
        }

        // 1. Copy src and zero-pad remainder
        std::memcpy(work_buf_.data(), src, sizeof(float) * orig_dim_);
        if (pad_dim_ > orig_dim_) {
            std::memset(work_buf_.data() + orig_dim_, 0, sizeof(float) * (pad_dim_ - orig_dim_));
        }

        // 2. Perform normalized FWHT on padded buffer
        fwht_forward(work_buf_.data(), rademacher_d_.data(), pad_dim_);

        // 3. Compensate energy loss due to zero-padding
        if (pad_dim_ > orig_dim_) {
            for (int i = 0; i < pad_dim_; ++i) {
                work_buf_[i] *= scale_factor_;
            }
        }

        std::memcpy(dst_pad, work_buf_.data(), sizeof(float) * pad_dim_);
    }

    void inverse(const float *src_pad, float *dst_orig) {
        if (!src_pad || !dst_orig) {
            throw std::invalid_argument("inverse: null buffer pointers");
        }

        std::memcpy(work_buf_.data(), src_pad, sizeof(float) * pad_dim_);

        // 1. Invert energy compensation
        if (pad_dim_ > orig_dim_) {
            for (int i = 0; i < pad_dim_; ++i) {
                work_buf_[i] *= inv_scale_factor_;
            }
        }

        // 2. Perform inverse FWHT on padded buffer
        fwht_inverse(work_buf_.data(), rademacher_d_.data(), pad_dim_);

        // 3. Slice out original dimensions
        std::memcpy(dst_orig, work_buf_.data(), sizeof(float) * orig_dim_);
    }

    float compute_l2_norm(const float *vec, int len) const {
        float sum = 0.0f;
        for (int i = 0; i < len; ++i) {
            sum += vec[i] * vec[i];
        }
        return std::sqrt(sum);
    }

private:
    int orig_dim_;
    int pad_dim_;
    uint64_t seed_;
    float scale_factor_;
    float inv_scale_factor_;
    std::vector<int8_t> rademacher_d_;
    std::vector<float> work_buf_;
};

} // namespace adaptq

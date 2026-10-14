#pragma once

#include <adaptq/strategy.h>
#include <adaptq/context.h>
#include <adaptq/storage.h>
#include <adaptq/config.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace adaptq {

struct AdaptiveBitrateConfig {
    int sink_tokens = 4;        // Protected initial attention sink tokens
    int recent_window = 64;     // Tokens in recent sliding window
    int mid_window = 256;       // Intermediate window
    int high_bits = 4;          // Precision for sink & recent tokens
    int mid_bits = 3;           // Precision for intermediate tokens
    int low_bits = 2;           // Precision for cold tokens
    float entropy_threshold = 1.8f; // Variance/entropy trigger for bit promotion
};

class AdaptiveBitratePolicy {
public:
    explicit AdaptiveBitratePolicy(AdaptiveBitrateConfig cfg = AdaptiveBitrateConfig{})
        : cfg_(cfg), total_tokens_seen_(0), total_bits_allocated_(0) {
        validate_config(cfg_);
    }

    void validate_config(const AdaptiveBitrateConfig& cfg) {
        if (cfg.sink_tokens < 0) {
            throw std::invalid_argument("sink_tokens must be non-negative");
        }
        if (cfg.recent_window <= 0 || cfg.mid_window <= cfg.recent_window) {
            throw std::invalid_argument("invalid window bounds: must satisfy 0 < recent_window < mid_window");
        }
        if (cfg.low_bits < 2 || cfg.mid_bits < cfg.low_bits || cfg.high_bits < cfg.mid_bits || cfg.high_bits > 4) {
            throw std::invalid_argument("invalid bit widths: must satisfy 2 <= low_bits <= mid_bits <= high_bits <= 4");
        }
    }

    int determine_bits(int token_idx, int current_seq_len, const float* vector_data = nullptr, int dim = 0) {
        int bits = cfg_.low_bits;

        // 1. Attention Sink Protection (always high precision)
        if (token_idx < cfg_.sink_tokens) {
            bits = cfg_.high_bits;
        }
        // 2. Recent Token Window (local context)
        else if (current_seq_len - token_idx <= cfg_.recent_window) {
            bits = cfg_.high_bits;
        }
        // 3. Intermediate Context Window
        else if (current_seq_len - token_idx <= cfg_.mid_window) {
            bits = cfg_.mid_bits;
        }
        // 4. Cold Context (low bits default)
        else {
            bits = cfg_.low_bits;
        }

        // 5. Activation Entropy / Variance Check for bit promotion
        if (vector_data && dim > 0) {
            float mean = 0.0f;
            for (int i = 0; i < dim; ++i) mean += vector_data[i];
            mean /= dim;

            float var = 0.0f;
            for (int i = 0; i < dim; ++i) {
                float diff = vector_data[i] - mean;
                var += diff * diff;
            }
            var /= dim;

            if (var > cfg_.entropy_threshold && bits < cfg_.high_bits) {
                bits++; // Promote to next higher precision
            }
        }

        total_tokens_seen_++;
        total_bits_allocated_ += bits;
        return bits;
    }

    double average_bits_per_token() const {
        if (total_tokens_seen_ == 0) return 0.0;
        return static_cast<double>(total_bits_allocated_) / total_tokens_seen_;
    }

    uint64_t total_tokens_seen() const { return total_tokens_seen_; }
    uint64_t total_bits_allocated() const { return total_bits_allocated_; }
    const AdaptiveBitrateConfig& config() const { return cfg_; }

    void reset() {
        total_tokens_seen_ = 0;
        total_bits_allocated_ = 0;
    }

private:
    AdaptiveBitrateConfig cfg_;
    uint64_t total_tokens_seen_;
    uint64_t total_bits_allocated_;
};

class AdaptiveBitrateStrategy final : public IKVStrategy {
public:
    explicit AdaptiveBitrateStrategy(AdaptiveBitrateConfig cfg = AdaptiveBitrateConfig{})
        : policy_(cfg), dim_(64), capacity_(1024), current_token_pos_(0) {}

    const char* name() const override { return "adaptive_bitrate"; }

    void init(const StrategyConfig& cfg) override {
        dim_ = cfg.dim;
        capacity_ = cfg.capacity;
        current_token_pos_ = 0;
        policy_.reset();
    }

    void reset() override {
        current_token_pos_ = 0;
        policy_.reset();
    }

    int determine_bit_allocation(int token_idx, const float* k_vec, int dim) {
        return policy_.determine_bits(token_idx, current_token_pos_, k_vec, dim);
    }

    void on_append(int token_count) {
        current_token_pos_ += token_count;
    }

    double average_bitrate() const {
        return policy_.average_bits_per_token();
    }

    int current_token_position() const {
        return current_token_pos_;
    }

    AdaptiveBitratePolicy& policy() {
        return policy_;
    }

private:
    AdaptiveBitratePolicy policy_;
    int dim_;
    int capacity_;
    int current_token_pos_;
};

} // namespace adaptq

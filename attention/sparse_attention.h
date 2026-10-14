#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace adaptq {

struct SparseAttentionConfig {
    int window_size = 512;      // Sliding window size
    int sink_tokens = 4;        // Attention sink tokens preserved at start
    int top_k = 128;            // Maximum tokens to retain if Top-K enabled (>0)
    float prune_threshold = 1e-5f; // Softmax probability pruning threshold
    bool enable_top_k = false;  // Toggle Top-K sparsity
};

class SparseAttentionKernel {
public:
    explicit SparseAttentionKernel(SparseAttentionConfig cfg = SparseAttentionConfig{})
        : cfg_(cfg) {
        validate_config(cfg_);
    }

    void validate_config(const SparseAttentionConfig& cfg) {
        if (cfg.window_size <= 0) {
            throw std::invalid_argument("window_size must be positive");
        }
        if (cfg.sink_tokens < 0) {
            throw std::invalid_argument("sink_tokens must be non-negative");
        }
        if (cfg.enable_top_k && cfg.top_k <= 0) {
            throw std::invalid_argument("top_k must be positive when enabled");
        }
        if (cfg.prune_threshold < 0.0f || cfg.prune_threshold >= 1.0f) {
            throw std::invalid_argument("prune_threshold must be in [0, 1)");
        }
    }

    void select_active_indices(int n_tokens, int query_pos, std::vector<int>& active_indices) const {
        if (n_tokens <= 0 || query_pos < 0) {
            active_indices.clear();
            return;
        }

        active_indices.clear();
        int available = std::min(n_tokens, query_pos + 1);
        active_indices.reserve(available);

        // Always include initial attention sinks
        int sinks = std::min(cfg_.sink_tokens, available);
        for (int i = 0; i < sinks; ++i) {
            active_indices.push_back(i);
        }

        // Sliding window: tokens in [query_pos - window_size + 1, query_pos]
        int window_start = std::max(sinks, query_pos - cfg_.window_size + 1);
        for (int i = window_start; i < available; ++i) {
            active_indices.push_back(i);
        }

        // Deduplicate and sort
        std::sort(active_indices.begin(), active_indices.end());
        active_indices.erase(std::unique(active_indices.begin(), active_indices.end()), active_indices.end());
    }

    void apply_top_k(const std::vector<float>& full_logits,
                     std::vector<int>& active_indices) const {
        if (!cfg_.enable_top_k || static_cast<int>(active_indices.size()) <= cfg_.top_k) {
            return;
        }

        // Preserve sinks, Top-K on remaining window candidates
        std::vector<int> sinks;
        std::vector<int> candidates;
        for (int idx : active_indices) {
            if (idx < cfg_.sink_tokens) {
                sinks.push_back(idx);
            } else {
                candidates.push_back(idx);
            }
        }

        int k_remaining = cfg_.top_k - static_cast<int>(sinks.size());
        if (k_remaining > 0 && k_remaining < static_cast<int>(candidates.size())) {
            std::partial_sort(
                candidates.begin(),
                candidates.begin() + k_remaining,
                candidates.end(),
                [&full_logits](int a, int b) {
                    return full_logits[a] > full_logits[b];
                }
            );
            candidates.resize(k_remaining);
        }

        active_indices = std::move(sinks);
        active_indices.insert(active_indices.end(), candidates.begin(), candidates.end());
        std::sort(active_indices.begin(), active_indices.end());
    }

    void compute_sparse_softmax(const std::vector<float>& logits,
                                const std::vector<int>& active_indices,
                                std::vector<float>& out_probs) const {
        if (active_indices.empty()) {
            out_probs.clear();
            return;
        }

        out_probs.resize(active_indices.size());
        float max_val = -1e30f;
        for (size_t i = 0; i < active_indices.size(); ++i) {
            float val = logits[active_indices[i]];
            if (val > max_val) max_val = val;
        }

        float sum = 0.0f;
        for (size_t i = 0; i < active_indices.size(); ++i) {
            float exp_val = std::exp(logits[active_indices[i]] - max_val);
            out_probs[i] = exp_val;
            sum += exp_val;
        }

        float inv_sum = (sum > 1e-12f) ? (1.0f / sum) : 0.0f;
        for (size_t i = 0; i < active_indices.size(); ++i) {
            out_probs[i] *= inv_sum;
            if (out_probs[i] < cfg_.prune_threshold) {
                out_probs[i] = 0.0f; // Sparsify negligible attention weights
            }
        }
    }

    void accumulate_sparse_v(const std::vector<float>& probs,
                            const std::vector<int>& active_indices,
                            const float* v_matrix, // shape: [n_tokens, dim]
                            int dim,
                            float* out_acc) const {
        if (!v_matrix || !out_acc || dim <= 0) return;
        std::fill(out_acc, out_acc + dim, 0.0f);

        for (size_t i = 0; i < active_indices.size(); ++i) {
            float w = probs[i];
            if (w <= 0.0f) continue; // Skip pruned tokens

            int token_idx = active_indices[i];
            const float* v_row = v_matrix + static_cast<size_t>(token_idx) * dim;
            for (int d = 0; d < dim; ++d) {
                out_acc[d] += w * v_row[d];
            }
        }
    }

    const SparseAttentionConfig& config() const { return cfg_; }

private:
    SparseAttentionConfig cfg_;
};

} // namespace adaptq

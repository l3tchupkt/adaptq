#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace adaptq {

struct OnlineSoftmaxState {
    float max_logit = -1e30f;
    float sum_exp = 0.0f;
    std::vector<float> acc; // shape: [dim]
};

class ChunkedPrefillEngine {
public:
    ChunkedPrefillEngine(int dim, int chunk_size = 256)
        : dim_(dim), chunk_size_(chunk_size), total_tokens_ingested_(0) {
        if (dim <= 0) {
            throw std::invalid_argument("dim must be positive");
        }
        if (chunk_size <= 0) {
            throw std::invalid_argument("chunk_size must be positive");
        }
        state_.acc.assign(dim, 0.0f);
    }

    int dim() const { return dim_; }
    int chunk_size() const { return chunk_size_; }
    int total_tokens_ingested() const { return total_tokens_ingested_; }

    void reset() {
        state_.max_logit = -1e30f;
        state_.sum_exp = 0.0f;
        std::fill(state_.acc.begin(), state_.acc.end(), 0.0f);
        total_tokens_ingested_ = 0;
    }

    /**
     * Process a chunk of K and V vectors against query using Online Softmax (FlashAttention formulation).
     * @param k_chunk Float array of size [chunk_tokens * dim]
     * @param v_chunk Float array of size [chunk_tokens * dim]
     * @param chunk_tokens Number of tokens in this chunk (<= chunk_size)
     * @param query Float array of size [dim]
     */
    void process_chunk(
        const float *k_chunk,
        const float *v_chunk,
        int chunk_tokens,
        const float *query
    ) {
        if (!k_chunk || !v_chunk || !query) {
            throw std::invalid_argument("process_chunk: null pointer passed");
        }
        if (chunk_tokens <= 0) return;

        float scale = 1.0f / std::sqrt(static_cast<float>(dim_));

        // 1. Compute dot products for this chunk
        std::vector<float> chunk_logits(chunk_tokens, 0.0f);
        float chunk_max = -1e30f;

        for (int i = 0; i < chunk_tokens; ++i) {
            const float *k_vec = k_chunk + static_cast<size_t>(i) * dim_;
            float dot = 0.0f;
            for (int d = 0; d < dim_; ++d) {
                dot += query[d] * k_vec[d];
            }
            dot *= scale;
            chunk_logits[i] = dot;
            if (dot > chunk_max) {
                chunk_max = dot;
            }
        }

        // 2. Compute local chunk exp and V accumulation
        float chunk_sum = 0.0f;
        std::vector<float> chunk_weights(chunk_tokens, 0.0f);
        for (int i = 0; i < chunk_tokens; ++i) {
            float exp_val = std::exp(chunk_logits[i] - chunk_max);
            chunk_weights[i] = exp_val;
            chunk_sum += exp_val;
        }

        std::vector<float> chunk_acc(dim_, 0.0f);
        for (int i = 0; i < chunk_tokens; ++i) {
            float w = chunk_weights[i];
            const float *v_vec = v_chunk + static_cast<size_t>(i) * dim_;
            for (int d = 0; d < dim_; ++d) {
                chunk_acc[d] += w * v_vec[d];
            }
        }

        // 3. Online Softmax Merge with global state
        if (total_tokens_ingested_ == 0) {
            // First chunk initialization
            state_.max_logit = chunk_max;
            state_.sum_exp = chunk_sum;
            state_.acc = std::move(chunk_acc);
        } else {
            // Merge previous accumulator and new chunk
            float new_max = std::max(state_.max_logit, chunk_max);
            float alpha = std::exp(state_.max_logit - new_max);
            float beta = std::exp(chunk_max - new_max);

            float new_sum = state_.sum_exp * alpha + chunk_sum * beta;

            for (int d = 0; d < dim_; ++d) {
                state_.acc[d] = state_.acc[d] * alpha + chunk_acc[d] * beta;
            }

            state_.max_logit = new_max;
            state_.sum_exp = new_sum;
        }

        total_tokens_ingested_ += chunk_tokens;
    }

    /**
     * Finalize and normalize the output accumulator.
     * @param out_attention_vector Output buffer of length [dim]
     */
    void finalize(float *out_attention_vector) const {
        if (!out_attention_vector) {
            throw std::invalid_argument("finalize: null output buffer");
        }
        if (total_tokens_ingested_ == 0 || state_.sum_exp <= 1e-12f) {
            std::fill(out_attention_vector, out_attention_vector + dim_, 0.0f);
            return;
        }

        float inv_sum = 1.0f / state_.sum_exp;
        for (int d = 0; d < dim_; ++d) {
            out_attention_vector[d] = state_.acc[d] * inv_sum;
        }
    }

private:
    int dim_;
    int chunk_size_;
    int total_tokens_ingested_;
    OnlineSoftmaxState state_;
};

} // namespace adaptq

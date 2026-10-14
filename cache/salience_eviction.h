#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <cstdint>

namespace adaptq {
namespace cache {

/**
 * @brief Metadata tracking attention salience and access statistics for a cached token.
 */
struct TokenSalienceMetadata {
    uint64_t token_id{0};
    size_t sequence_index{0};
    uint64_t insertion_timestamp_ns{0};
    double cumulative_salience{0.0};
    uint64_t access_count{0};
    bool is_sink{false};
    bool is_recent{false};
};

/**
 * @brief Configuration parameters for the heavy-hitter salience eviction policy.
 */
struct SalienceEvictionConfig {
    size_t max_cache_capacity{2048};      /// Total token slots in cache
    size_t sink_token_count{4};           /// Number of initial prompt tokens (attention sinks) to permanently pin
    size_t recent_window_size{64};        /// Sliding window of latest tokens permanently retained
    double decay_factor{0.95};            /// Exponential decay factor gamma applied to prior salience
    double min_salience_threshold{1e-6};  /// Lower clamp limit for cumulative salience
};

/**
 * @brief Operational runtime statistics for eviction policy auditing.
 */
struct SalienceEvictionStats {
    uint64_t total_tokens_inserted{0};
    uint64_t total_evictions{0};
    uint64_t sink_retentions{0};
    uint64_t heavy_hitters_retained{0};
    double total_salience_decay_accumulated{0.0};
    double avg_eviction_latency_us{0.0};
    uint64_t eviction_pass_count{0};
};

/**
 * @brief Salience-Aware Heavy-Hitter KV Cache Eviction Policy.
 *
 * Implements H2O-style attention-driven eviction combining:
 * 1. Initial attention sink preservation (e.g. first N prompt tokens).
 * 2. Sliding window recency preservation (e.g. most recent W tokens).
 * 3. Dynamic heavy-hitter selection via exponentially-decayed cumulative attention weights.
 */
class SalienceEvictionPolicy {
public:
    explicit SalienceEvictionPolicy(const SalienceEvictionConfig& config = SalienceEvictionConfig{})
        : config_(config) {
        if (config_.sink_token_count + config_.recent_window_size >= config_.max_cache_capacity) {
            throw std::invalid_argument(
                "Sink token count + recent window size must be strictly less than max cache capacity");
        }
        if (config_.decay_factor <= 0.0 || config_.decay_factor > 1.0) {
            throw std::invalid_argument("Decay factor must be in (0.0, 1.0]");
        }
    }

    /**
     * @brief Record insertion of a newly generated or prefilled token into the cache.
     * @param token_id Unique identifier for the token.
     * @param sequence_index Logical position in the autoregressive sequence.
     */
    void record_insertion(uint64_t token_id, size_t sequence_index) {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        uint64_t now_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());

        TokenSalienceMetadata meta;
        meta.token_id = token_id;
        meta.sequence_index = sequence_index;
        meta.insertion_timestamp_ns = now_ns;
        meta.cumulative_salience = 1.0; // Initial non-zero prior
        meta.access_count = 1;
        meta.is_sink = (sequence_index < config_.sink_token_count);
        meta.is_recent = true;

        tokens_[token_id] = meta;
        active_token_ids_.push_back(token_id);
        stats_.total_tokens_inserted++;

        recompute_recent_and_sink_tags();
    }

    /**
     * @brief Update cumulative attention weights for active tokens across an attention layer.
     * @param attention_weights Map of token_id to current attention probability/weight.
     */
    void update_salience_scores(const std::unordered_map<uint64_t, double>& attention_weights) {
        // Apply exponential decay to all active non-sink tokens
        for (auto& pair : tokens_) {
            TokenSalienceMetadata& meta = pair.second;
            if (!meta.is_sink) {
                meta.cumulative_salience *= config_.decay_factor;
                if (meta.cumulative_salience < config_.min_salience_threshold) {
                    meta.cumulative_salience = config_.min_salience_threshold;
                }
            }
        }

        // Add newly observed attention salience
        for (const auto& kv : attention_weights) {
            auto it = tokens_.find(kv.first);
            if (it != tokens_.end()) {
                it->second.cumulative_salience += kv.second;
                it->second.access_count++;
            }
        }
    }

    /**
     * @brief Determine which tokens should be evicted to reduce cache occupancy below capacity.
     * @param target_eviction_count Desired number of tokens to evict.
     * @return Vector of token_ids chosen for eviction, ordered by lowest salience first.
     */
    std::vector<uint64_t> select_eviction_candidates(size_t target_eviction_count) {
        auto start_time = std::chrono::steady_clock::now();

        if (active_token_ids_.empty() || target_eviction_count == 0) {
            return {};
        }

        recompute_recent_and_sink_tags();

        // Separate candidates that are eligible for eviction (neither sink nor recent)
        std::vector<uint64_t> eligible_candidates;
        eligible_candidates.reserve(active_token_ids_.size());

        for (uint64_t tid : active_token_ids_) {
            auto it = tokens_.find(tid);
            if (it != tokens_.end()) {
                if (!it->second.is_sink && !it->second.is_recent) {
                    eligible_candidates.push_back(tid);
                }
            }
        }

        // Sort eligible candidates by cumulative salience ascending (lowest salience evicted first)
        std::sort(eligible_candidates.begin(), eligible_candidates.end(),
            [this](uint64_t a, uint64_t b) {
                const auto& ma = tokens_.at(a);
                const auto& mb = tokens_.at(b);
                if (std::abs(ma.cumulative_salience - mb.cumulative_salience) > 1e-9) {
                    return ma.cumulative_salience < mb.cumulative_salience;
                }
                return ma.sequence_index < mb.sequence_index; // Tie breaker: older first
            });

        size_t count_to_evict = std::min(target_eviction_count, eligible_candidates.size());
        std::vector<uint64_t> evicted(eligible_candidates.begin(), eligible_candidates.begin() + count_to_evict);

        auto end_time = std::chrono::steady_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();

        stats_.eviction_pass_count++;
        stats_.avg_eviction_latency_us = (stats_.avg_eviction_latency_us * (stats_.eviction_pass_count - 1) + elapsed_us) / stats_.eviction_pass_count;

        return evicted;
    }

    /**
     * @brief Confirm eviction of tokens, removing them from active tracking.
     * @param evicted_token_ids Tokens that were evicted from underlying KV store.
     */
    void commit_eviction(const std::vector<uint64_t>& evicted_token_ids) {
        std::unordered_set<uint64_t> to_remove(evicted_token_ids.begin(), evicted_token_ids.end());

        for (uint64_t tid : evicted_token_ids) {
            tokens_.erase(tid);
        }

        active_token_ids_.erase(
            std::remove_if(active_token_ids_.begin(), active_token_ids_.end(),
                [&to_remove](uint64_t tid) { return to_remove.count(tid) > 0; }),
            active_token_ids_.end());

        stats_.total_evictions += evicted_token_ids.size();
    }

    /**
     * @brief Automatically enforce cache capacity by selecting and committing evictions.
     * @return Number of tokens evicted.
     */
    size_t enforce_capacity() {
        if (active_token_ids_.size() <= config_.max_cache_capacity) {
            return 0;
        }

        size_t deficit = active_token_ids_.size() - config_.max_cache_capacity;
        auto candidates = select_eviction_candidates(deficit);
        commit_eviction(candidates);
        return candidates.size();
    }

    /**
     * @brief Get current number of active tokens in cache.
     */
    size_t size() const {
        return active_token_ids_.size();
    }

    /**
     * @brief Check if a token ID exists in tracking.
     */
    bool contains(uint64_t token_id) const {
        return tokens_.find(token_id) != tokens_.end();
    }

    /**
     * @brief Retrieve metadata for a specific token.
     */
    const TokenSalienceMetadata& get_metadata(uint64_t token_id) const {
        auto it = tokens_.find(token_id);
        if (it == tokens_.end()) {
            throw std::out_of_range("Token ID not found in salience eviction tracker");
        }
        return it->second;
    }

    /**
     * @brief Get operational statistics.
     */
    const SalienceEvictionStats& stats() const {
        return stats_;
    }

    /**
     * @brief Reset all internal state and statistics.
     */
    void clear() {
        tokens_.clear();
        active_token_ids_.clear();
        stats_ = SalienceEvictionStats{};
    }

private:
    void recompute_recent_and_sink_tags() {
        if (active_token_ids_.empty()) {
            return;
        }

        size_t total = active_token_ids_.size();
        size_t recent_threshold_index = (total > config_.recent_window_size)
                                      ? (total - config_.recent_window_size)
                                      : 0;

        for (size_t i = 0; i < total; ++i) {
            uint64_t tid = active_token_ids_[i];
            auto it = tokens_.find(tid);
            if (it != tokens_.end()) {
                it->second.is_sink = (it->second.sequence_index < config_.sink_token_count);
                it->second.is_recent = (i >= recent_threshold_index);
            }
        }
    }

    SalienceEvictionConfig config_;
    SalienceEvictionStats stats_;
    std::unordered_map<uint64_t, TokenSalienceMetadata> tokens_;
    std::vector<uint64_t> active_token_ids_;
};

} // namespace cache
} // namespace adaptq

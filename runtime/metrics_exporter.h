#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace adaptq {

class HistogramMetric {
public:
    explicit HistogramMetric(std::vector<double> buckets)
        : buckets_(std::move(buckets)), count_(0), sum_(0.0) {
        std::sort(buckets_.begin(), buckets_.end());
        bucket_counts_.resize(buckets_.size() + 1, 0);
    }

    void observe(double val) {
        sum_ += val;
        count_++;
        size_t idx = 0;
        for (; idx < buckets_.size(); ++idx) {
            if (val <= buckets_[idx]) {
                bucket_counts_[idx]++;
                return;
            }
        }
        bucket_counts_[idx]++; // +Inf bucket
    }

    uint64_t count() const { return count_; }
    double sum() const { return sum_; }
    const std::vector<double>& buckets() const { return buckets_; }
    const std::vector<uint64_t>& bucket_counts() const { return bucket_counts_; }

    void reset() {
        count_ = 0;
        sum_ = 0.0;
        std::fill(bucket_counts_.begin(), bucket_counts_.end(), 0);
    }

private:
    std::vector<double> buckets_;
    uint64_t count_;
    double sum_;
    std::vector<uint64_t> bucket_counts_;
};

class PrometheusMetricsExporter {
public:
    PrometheusMetricsExporter()
        : tokens_appended_total_(0),
          cache_bytes_current_(0),
          cache_evictions_total_(0),
          compression_ratio_(1.0),
          quant_latency_hist_({0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5}),
          compute_latency_hist_({0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5}) {}

    void record_append(int tokens, size_t bytes_allocated, double latency_seconds) {
        std::lock_guard<std::mutex> lock(mu_);
        if (tokens > 0) {
            tokens_appended_total_ += tokens;
        }
        cache_bytes_current_ = bytes_allocated;
        if (latency_seconds >= 0.0) {
            quant_latency_hist_.observe(latency_seconds);
        }
    }

    void record_attention_compute(double latency_seconds, int tokens_scanned) {
        std::lock_guard<std::mutex> lock(mu_);
        (void)tokens_scanned;
        if (latency_seconds >= 0.0) {
            compute_latency_hist_.observe(latency_seconds);
        }
    }

    void record_eviction(int count = 1) {
        std::lock_guard<std::mutex> lock(mu_);
        if (count > 0) {
            cache_evictions_total_ += count;
        }
    }

    void update_compression_ratio(double ratio) {
        std::lock_guard<std::mutex> lock(mu_);
        if (ratio > 0.0) {
            compression_ratio_ = ratio;
        }
    }

    void set_custom_gauge(const std::string& name, double val) {
        std::lock_guard<std::mutex> lock(mu_);
        custom_gauges_[name] = val;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mu_);
        tokens_appended_total_ = 0;
        cache_bytes_current_ = 0;
        cache_evictions_total_ = 0;
        compression_ratio_ = 1.0;
        quant_latency_hist_.reset();
        compute_latency_hist_.reset();
        custom_gauges_.clear();
    }

    std::string to_prometheus_text(const std::string& model_id = "default") const {
        std::lock_guard<std::mutex> lock(mu_);
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(6);

        // Header and counter
        ss << "# HELP adaptq_tokens_appended_total Total tokens appended to KV cache.\n";
        ss << "# TYPE adaptq_tokens_appended_total counter\n";
        ss << "adaptq_tokens_appended_total{model=\"" << model_id << "\"} " << tokens_appended_total_ << "\n\n";

        // Gauge: cache bytes
        ss << "# HELP adaptq_cache_bytes_current Current allocated KV cache size in bytes.\n";
        ss << "# TYPE adaptq_cache_bytes_current gauge\n";
        ss << "adaptq_cache_bytes_current{model=\"" << model_id << "\"} " << cache_bytes_current_ << "\n\n";

        // Gauge: compression ratio
        ss << "# HELP adaptq_compression_ratio Effective KV memory compression factor.\n";
        ss << "# TYPE adaptq_compression_ratio gauge\n";
        ss << "adaptq_compression_ratio{model=\"" << model_id << "\"} " << compression_ratio_ << "\n\n";

        // Counter: evictions
        ss << "# HELP adaptq_cache_evictions_total Total number of token eviction cycles.\n";
        ss << "# TYPE adaptq_cache_evictions_total counter\n";
        ss << "adaptq_cache_evictions_total{model=\"" << model_id << "\"} " << cache_evictions_total_ << "\n\n";

        // Histogram: quantization latency
        emit_histogram(ss, "adaptq_quantization_latency_seconds",
                       "Latency of vector quantization and FWHT per append batch in seconds.",
                       model_id, quant_latency_hist_);

        // Histogram: compute latency
        emit_histogram(ss, "adaptq_attention_compute_latency_seconds",
                       "Latency of quantized attention kernel execution in seconds.",
                       model_id, compute_latency_hist_);

        // Custom gauges
        for (const auto& kv : custom_gauges_) {
            ss << "# TYPE adaptq_" << kv.first << " gauge\n";
            ss << "adaptq_" << kv.first << "{model=\"" << model_id << "\"} " << kv.second << "\n";
        }

        return ss.str();
    }

    std::string to_json() const {
        std::lock_guard<std::mutex> lock(mu_);
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"tokens_appended_total\": " << tokens_appended_total_ << ",\n";
        ss << "  \"cache_bytes_current\": " << cache_bytes_current_ << ",\n";
        ss << "  \"cache_evictions_total\": " << cache_evictions_total_ << ",\n";
        ss << "  \"compression_ratio\": " << compression_ratio_ << ",\n";
        ss << "  \"quant_latency_count\": " << quant_latency_hist_.count() << ",\n";
        ss << "  \"quant_latency_sum_seconds\": " << quant_latency_hist_.sum() << ",\n";
        ss << "  \"compute_latency_count\": " << compute_latency_hist_.count() << ",\n";
        ss << "  \"compute_latency_sum_seconds\": " << compute_latency_hist_.sum() << "\n";
        ss << "}";
        return ss.str();
    }

    uint64_t get_tokens_appended() const {
        std::lock_guard<std::mutex> lock(mu_);
        return tokens_appended_total_;
    }

    size_t get_cache_bytes() const {
        std::lock_guard<std::mutex> lock(mu_);
        return cache_bytes_current_;
    }

private:
    void emit_histogram(std::ostringstream& ss, const std::string& name,
                        const std::string& help, const std::string& model_id,
                        const HistogramMetric& hist) const {
        ss << "# HELP " << name << " " << help << "\n";
        ss << "# TYPE " << name << " histogram\n";
        uint64_t cumulative = 0;
        for (size_t i = 0; i < hist.buckets().size(); ++i) {
            cumulative += hist.bucket_counts()[i];
            ss << name << "_bucket{model=\"" << model_id << "\",le=\""
               << hist.buckets()[i] << "\"} " << cumulative << "\n";
        }
        cumulative += hist.bucket_counts().back();
        ss << name << "_bucket{model=\"" << model_id << "\",le=\"+Inf\"} " << cumulative << "\n";
        ss << name << "_sum{model=\"" << model_id << "\"} " << hist.sum() << "\n";
        ss << name << "_count{model=\"" << model_id << "\"} " << hist.count() << "\n\n";
    }

    mutable std::mutex mu_;
    uint64_t tokens_appended_total_;
    size_t cache_bytes_current_;
    uint64_t cache_evictions_total_;
    double compression_ratio_;
    HistogramMetric quant_latency_hist_;
    HistogramMetric compute_latency_hist_;
    std::map<std::string, double> custom_gauges_;
};

} // namespace adaptq

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <thread>
#include <vector>

#include "../../runtime/metrics_exporter.h"

using namespace adaptq;

TEST_CASE("HistogramMetric: basic observations and inf bucket", "[metrics][histogram]") {
    std::vector<double> buckets = {0.001, 0.01, 0.1, 1.0};
    HistogramMetric hist(buckets);

    REQUIRE(hist.count() == 0);
    REQUIRE(hist.sum() == 0.0);

    hist.observe(0.0005);
    hist.observe(0.005);
    hist.observe(0.05);
    hist.observe(0.5);
    hist.observe(2.5); // Falls into +Inf

    REQUIRE(hist.count() == 5);
    REQUIRE(hist.sum() > 3.0);

    const auto& counts = hist.bucket_counts();
    REQUIRE(counts.size() == 5);
    REQUIRE(counts[0] == 1); // <= 0.001
    REQUIRE(counts[1] == 1); // <= 0.01
    REQUIRE(counts[2] == 1); // <= 0.1
    REQUIRE(counts[3] == 1); // <= 1.0
    REQUIRE(counts[4] == 1); // +Inf

    hist.reset();
    REQUIRE(hist.count() == 0);
    REQUIRE(hist.sum() == 0.0);
    for (auto c : hist.bucket_counts()) {
        REQUIRE(c == 0);
    }
}

TEST_CASE("PrometheusMetricsExporter: append and eviction tracking", "[metrics][exporter]") {
    PrometheusMetricsExporter exporter;

    exporter.record_append(128, 65536, 0.0025);
    exporter.record_append(256, 131072, 0.0040);
    exporter.record_eviction(3);
    exporter.update_compression_ratio(7.85);

    REQUIRE(exporter.get_tokens_appended() == 384);
    REQUIRE(exporter.get_cache_bytes() == 131072);

    std::string text = exporter.to_prometheus_text("llama-3-8b");
    REQUIRE(text.find("adaptq_tokens_appended_total{model=\"llama-3-8b\"} 384") != std::string::npos);
    REQUIRE(text.find("adaptq_cache_bytes_current{model=\"llama-3-8b\"} 131072") != std::string::npos);
    REQUIRE(text.find("adaptq_cache_evictions_total{model=\"llama-3-8b\"} 3") != std::string::npos);
    REQUIRE(text.find("adaptq_compression_ratio{model=\"llama-3-8b\"} 7.850000") != std::string::npos);
    REQUIRE(text.find("adaptq_quantization_latency_seconds_count{model=\"llama-3-8b\"} 2") != std::string::npos);

    std::string json_str = exporter.to_json();
    REQUIRE(json_str.find("\"tokens_appended_total\": 384") != std::string::npos);
    REQUIRE(json_str.find("\"cache_bytes_current\": 131072") != std::string::npos);
    REQUIRE(json_str.find("\"cache_evictions_total\": 3") != std::string::npos);
}

TEST_CASE("PrometheusMetricsExporter: attention compute and custom gauges", "[metrics][exporter]") {
    PrometheusMetricsExporter exporter;

    exporter.record_attention_compute(0.0008, 1024);
    exporter.record_attention_compute(0.0012, 2048);
    exporter.set_custom_gauge("active_sessions", 4.0);

    std::string text = exporter.to_prometheus_text("mistral-7b");
    REQUIRE(text.find("adaptq_attention_compute_latency_seconds_count{model=\"mistral-7b\"} 2") != std::string::npos);
    REQUIRE(text.find("adaptq_active_sessions{model=\"mistral-7b\"} 4.000000") != std::string::npos);

    exporter.reset();
    REQUIRE(exporter.get_tokens_appended() == 0);
    REQUIRE(exporter.get_cache_bytes() == 0);

    std::string reset_text = exporter.to_prometheus_text("mistral-7b");
    REQUIRE(reset_text.find("adaptq_tokens_appended_total{model=\"mistral-7b\"} 0") != std::string::npos);
    REQUIRE(reset_text.find("adaptq_active_sessions") == std::string::npos);
}

TEST_CASE("PrometheusMetricsExporter: multithreaded recording safety", "[metrics][concurrency]") {
    PrometheusMetricsExporter exporter;
    const int NUM_THREADS = 4;
    const int ITERS_PER_THREAD = 100;

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&exporter, t]() {
            for (int i = 0; i < ITERS_PER_THREAD; ++i) {
                exporter.record_append(1, 1024 * (t + 1), 0.001);
                exporter.record_attention_compute(0.0005, 512);
                if (i % 25 == 0) {
                    exporter.record_eviction(1);
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    REQUIRE(exporter.get_tokens_appended() == NUM_THREADS * ITERS_PER_THREAD);
    std::string text = exporter.to_prometheus_text();
    REQUIRE(!text.empty());
}

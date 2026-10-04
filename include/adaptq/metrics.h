#pragma once
#include <cstddef>
#include <cstdint>

/* -------------------------------------------------------------------------
 * adaptq/metrics.h — BenchmarkResult, ComputeMetrics, TokenMetrics
 *
 * All measurement types used by the metrics collector, benchmark runner,
 * and profiler. NaN is used for fields that are not measured in a given
 * context (e.g., perplexity when no language model is available).
 * ----------------------------------------------------------------------- */

#include <cmath>  /* for NAN */

namespace adaptq {

/* ---- Per-compute-call metrics ----------------------------------------- */
struct ComputeMetrics {
    int   layer;
    int   head;
    int   n_tokens_used;      /* tokens scanned in this compute()      */
    int   n_tokens_evicted;   /* tokens evicted during this call       */
    float latency_us;         /* wall-clock time for compute()          */
    float quality;            /* IQualityOracle estimate, -1 if absent */
    float avg_bits_per_dim;   /* from IQuality, -1 if absent           */
    float logit_max;
    float logit_min;
};

/* ---- Per-token metrics ------------------------------------------------- */
struct TokenMetrics {
    int   layer;
    int   head;
    int   slot;
    int   token_pos;
    float quant_error_mse;    /* MSE of decompressed vs FP32 vector    */
    float attention_weight;   /* weight assigned in the last compute()  */
    bool  was_evicted;
    uint8_t bits_used;        /* actual bits used for this token        */
};

/* ---- Full benchmark result --------------------------------------------- */
struct BenchmarkResult {
    /* Throughput */
    double tokens_per_sec   = 0.0;
    double ttft_ms          = NAN;  /* time to first token                   */
    double decode_p50_us    = NAN;
    double decode_p95_us    = NAN;
    double decode_p99_us    = NAN;

    /* Memory */
    double peak_rss_mb       = NAN;
    double kv_cache_mb       = NAN;
    double compression_ratio = NAN;  /* vs FP16 KV                           */

    /* Quality */
    double cosine_sim_mean    = NAN;
    double cosine_sim_min     = NAN;
    double output_mse         = NAN;
    double perplexity         = NAN;  /* NaN if no LM available               */
    double ppl_delta_vs_fp16  = NAN;  /* NaN if no baseline available         */

    /* Hardware */
    double memory_bandwidth_gbps = NAN;
    double l3_cache_miss_rate    = NAN;  /* Linux perf counters; NaN otherwise */
    double cpu_utilization       = NAN;
    double energy_joules         = NAN;  /* RAPL / Apple PMU; NaN if unavailable */

    /* Cost model */
    double total_cost = NAN;  /* scalar cost under the active ICostFunction   */

    /* Metadata */
    const char *strategy_name = nullptr;
    const char *policy_name   = nullptr;
    const char *storage_name  = nullptr;
    const char *kernel_name   = nullptr;
    const char *cost_fn_name  = nullptr;
};

} /* namespace adaptq */

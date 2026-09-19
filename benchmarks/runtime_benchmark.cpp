/*
 * benchmarks/runtime_benchmark.cpp
 *
 * v0.2.3 runtime benchmark harness.
 *
 * The benchmark compares the legacy AttentionHead execution path with the
 * RuntimeContext path, using identical deterministic K/V and query inputs.
 * It also compares the selected runtime backend against a forced scalar
 * backend when the host provides a distinct accelerated backend.
 *
 * Results are emitted as JSON so the same command can be run on different
 * commits/releases and the outputs can be compared mechanically.
 */

#include "attention.h"
#include "runtime/runtime_context.h"
#include "adaptq/kernel.h"
#include "adaptq/storage.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kDefaultDim = 128;
constexpr int kDefaultBits = 4;
constexpr int kDefaultWarmup = 10;
constexpr int kDefaultRepetitions = 50;
constexpr int kMaxTokens = 4096;

const int kWorkloads[] = {128, 256, 512, 1024, 2048, 4096};

struct Options {
    std::string output_path;
    int dim = kDefaultDim;
    int bits = kDefaultBits;
    int warmup = kDefaultWarmup;
    int repetitions = kDefaultRepetitions;
};

struct Timing {
    double p50_us = 0.0;
    double p95_us = 0.0;
    double tokens_per_sec = 0.0;
};

struct RunResult {
    std::string backend;
    Timing timing;
    std::size_t kv_bytes = 0;
    double kv_cache_mb = 0.0;
    double compression_ratio_vs_fp16 = 0.0;
    double output_checksum = 0.0;
};

struct NumericResult {
    double mse = 0.0;
    double cosine_similarity = 0.0;
    double max_error = 0.0;
};

struct SelectionResult {
    std::string backend;
    Timing timing;
};

struct WorkloadResult {
    int tokens = 0;
    RunResult legacy;
    RunResult runtime_auto;
    RunResult runtime_scalar;
    bool accelerated_backend_available = false;
    NumericResult runtime_vs_legacy;
    NumericResult scalar_vs_auto;
    double runtime_latency_delta_pct_vs_legacy = 0.0;
    double runtime_throughput_ratio_vs_legacy = 0.0;
    double auto_speedup_vs_scalar = 0.0;
};

struct BackendEnvState {
    bool force_scalar_present = false;
    std::string force_scalar;
    bool backend_present = false;
    std::string backend;
    bool disable_avx2_present = false;
    std::string disable_avx2;
};

static void fail(const std::string &message) {
    throw std::runtime_error(message);
}

static void set_env_value(const char *key, const char *value) {
#if defined(_WIN32)
    if (_putenv_s(key, value) != 0)
        fail(std::string("failed to set environment variable ") + key);
#else
    if (setenv(key, value, 1) != 0)
        fail(std::string("failed to set environment variable ") + key);
#endif
}

static void unset_env_value(const char *key) {
#if defined(_WIN32)
    if (_putenv_s(key, "") != 0)
        fail(std::string("failed to clear environment variable ") + key);
#else
    if (unsetenv(key) != 0)
        fail(std::string("failed to clear environment variable ") + key);
#endif
}

static BackendEnvState save_backend_env() {
    BackendEnvState state;

    if (const char *value = std::getenv("ADAPTQ_FORCE_SCALAR")) {
        state.force_scalar_present = true;
        state.force_scalar = value;
    }
    if (const char *value = std::getenv("ADAPTQ_BACKEND")) {
        state.backend_present = true;
        state.backend = value;
    }
    if (const char *value = std::getenv("ADAPTQ_DISABLE_AVX2")) {
        state.disable_avx2_present = true;
        state.disable_avx2 = value;
    }

    return state;
}

static void restore_backend_env(const BackendEnvState &state) {
    if (state.force_scalar_present)
        set_env_value("ADAPTQ_FORCE_SCALAR", state.force_scalar.c_str());
    else
        unset_env_value("ADAPTQ_FORCE_SCALAR");

    if (state.backend_present)
        set_env_value("ADAPTQ_BACKEND", state.backend.c_str());
    else
        unset_env_value("ADAPTQ_BACKEND");

    if (state.disable_avx2_present)
        set_env_value("ADAPTQ_DISABLE_AVX2", state.disable_avx2.c_str());
    else
        unset_env_value("ADAPTQ_DISABLE_AVX2");
}

class ScopedBackendEnv {
public:
    enum class Mode {
        Auto,
        Scalar
    };

    explicit ScopedBackendEnv(Mode mode)
        : saved_(save_backend_env()) {
        unset_env_value("ADAPTQ_BACKEND");
        unset_env_value("ADAPTQ_DISABLE_AVX2");

        if (mode == Mode::Scalar)
            set_env_value("ADAPTQ_FORCE_SCALAR", "1");
        else
            unset_env_value("ADAPTQ_FORCE_SCALAR");
    }

    ~ScopedBackendEnv() {
        try {
            restore_backend_env(saved_);
        } catch (...) {
            /* Destructors must not throw. */
        }
    }

    ScopedBackendEnv(const ScopedBackendEnv &) = delete;
    ScopedBackendEnv &operator=(const ScopedBackendEnv &) = delete;

private:
    BackendEnvState saved_;
};

template <typename Fn>
static Timing measure(Fn &&fn, int warmup, int repetitions, int tokens) {
    if (warmup < 0 || repetitions <= 0 || tokens <= 0)
        fail("invalid benchmark timing parameters");

    for (int i = 0; i < warmup; ++i)
        fn();

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repetitions));

    for (int i = 0; i < repetitions; ++i) {
        const auto start = Clock::now();
        fn();
        const auto stop = Clock::now();

        const double elapsed_us =
            std::chrono::duration<double, std::micro>(stop - start).count();
        samples.push_back(elapsed_us);
    }

    std::sort(samples.begin(), samples.end());

    auto nearest_rank = [&samples](double quantile) {
        const double scaled = quantile * static_cast<double>(samples.size());
        std::size_t index =
            scaled <= 1.0
                ? 0
                : static_cast<std::size_t>(std::ceil(scaled)) - 1;
        if (index >= samples.size())
            index = samples.size() - 1;
        return samples[index];
    };

    Timing result;
    result.p50_us = nearest_rank(0.50);
    result.p95_us = nearest_rank(0.95);
    result.tokens_per_sec =
        result.p50_us > 0.0
            ? static_cast<double>(tokens) * 1000000.0 / result.p50_us
            : std::numeric_limits<double>::infinity();
    return result;
}

static double checksum(const std::vector<float> &values) {
    double sum = 0.0;
    for (float value : values)
        sum += static_cast<double>(value);
    return sum;
}

static NumericResult compare_outputs(const std::vector<float> &lhs,
                                      const std::vector<float> &rhs) {
    if (lhs.size() != rhs.size())
        fail("cannot compare output vectors with different sizes");

    if (lhs.empty())
        return {};

    long double squared_error = 0.0L;
    long double lhs_norm = 0.0L;
    long double rhs_norm = 0.0L;
    float max_error = 0.0f;

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        const long double a = lhs[i];
        const long double b = rhs[i];
        const long double diff = a - b;

        squared_error += diff * diff;
        lhs_norm += a * a;
        rhs_norm += b * b;
        max_error = std::max(max_error, static_cast<float>(std::fabs(diff)));
    }

    const long double cosine_denominator =
        std::sqrt(lhs_norm) * std::sqrt(rhs_norm);

    double cosine = 1.0;
    if (cosine_denominator > 1e-18L) {
        long double dot = 0.0L;
        for (std::size_t i = 0; i < lhs.size(); ++i)
            dot += static_cast<long double>(lhs[i]) *
                   static_cast<long double>(rhs[i]);
        cosine = static_cast<double>(dot / cosine_denominator);
        cosine = std::max(-1.0, std::min(1.0, cosine));
    } else if (lhs_norm > 1e-18L || rhs_norm > 1e-18L) {
        cosine = 0.0;
    }

    NumericResult result;
    result.mse = static_cast<double>(
        squared_error / static_cast<long double>(lhs.size()));
    result.cosine_similarity = cosine;
    result.max_error = static_cast<double>(max_error);
    return result;
}

static double fp16_kv_bytes(int tokens, int dim) {
    /* K + V, two FP16 values per vector dimension. */
    return static_cast<double>(tokens) * static_cast<double>(dim) * 2.0 * 2.0;
}

static std::string json_escape(const std::string &value) {
    std::ostringstream escaped;
    for (char ch : value) {
        switch (ch) {
        case '"':  escaped << "\\\""; break;
        case '\\': escaped << "\\\\"; break;
        case '\b': escaped << "\\b"; break;
        case '\f': escaped << "\\f"; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                escaped << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch))
                        << std::dec << std::setfill(' ');
            } else {
                escaped << ch;
            }
        }
    }
    return escaped.str();
}

static void json_string(std::ostream &out, const std::string &value) {
    out << '"' << json_escape(value) << '"';
}

static void json_number(std::ostream &out, double value) {
    if (!std::isfinite(value)) {
        out << "null";
        return;
    }
    out << std::setprecision(12) << value;
}

static void write_timing(std::ostream &out, const Timing &timing) {
    out << "{\n"
        << "        \"p50_us\": ";
    json_number(out, timing.p50_us);
    out << ",\n"
        << "        \"p95_us\": ";
    json_number(out, timing.p95_us);
    out << ",\n"
        << "        \"tokens_per_sec\": ";
    json_number(out, timing.tokens_per_sec);
    out << "\n      }";
}

static void write_numeric(std::ostream &out, const NumericResult &result) {
    out << "{\n"
        << "        \"mse\": ";
    json_number(out, result.mse);
    out << ",\n"
        << "        \"cosine_similarity\": ";
    json_number(out, result.cosine_similarity);
    out << ",\n"
        << "        \"max_error\": ";
    json_number(out, result.max_error);
    out << "\n      }";
}

static void write_run(std::ostream &out, const RunResult &result) {
    out << "{\n"
        << "      \"backend\": ";
    json_string(out, result.backend);
    out << ",\n"
        << "      \"latency\": ";
    write_timing(out, result.timing);
    out << ",\n"
        << "      \"kv_bytes\": " << result.kv_bytes
        << ",\n"
        << "      \"kv_cache_mb\": ";
    json_number(out, result.kv_cache_mb);
    out << ",\n"
        << "      \"compression_ratio_vs_fp16\": ";
    json_number(out, result.compression_ratio_vs_fp16);
    out << ",\n"
        << "      \"output_checksum\": ";
    json_number(out, result.output_checksum);
    out << "\n    }";
}

static std::vector<float> make_vector(std::mt19937 &rng, int dim) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> values(static_cast<std::size_t>(dim));
    for (float &value : values)
        value = dist(rng);
    return values;
}

static void populate_legacy(AttentionHead &head,
                            const std::vector<std::vector<float>> &keys,
                            const std::vector<std::vector<float>> &values) {
    if (keys.size() != values.size())
        fail("legacy key/value workload size mismatch");

    for (std::size_t i = 0; i < keys.size(); ++i)
        head.append_kv(keys[i].data(), values[i].data(),
                       static_cast<int>(i));
}

static void populate_runtime(adaptq::RuntimeContext &context,
                             const std::vector<std::vector<float>> &keys,
                             const std::vector<std::vector<float>> &values) {
    if (keys.size() != values.size())
        fail("runtime key/value workload size mismatch");

    for (std::size_t i = 0; i < keys.size(); ++i)
        context.append(0, 0, keys[i].data(), values[i].data());
}

static RunResult benchmark_legacy(
    AttentionHead &head,
    const std::vector<float> &query,
    std::vector<float> &output,
    const Options &options,
    int tokens) {
    RunResult result;
    result.backend = "legacy_attention_head";

    result.timing = measure(
        [&] { head.compute(query.data(), output.data()); },
        options.warmup, options.repetitions, tokens);

    result.kv_bytes = head.kv_bytes();
    result.kv_cache_mb =
        static_cast<double>(result.kv_bytes) / (1024.0 * 1024.0);

    const double fp16_bytes = fp16_kv_bytes(tokens, options.dim);
    result.compression_ratio_vs_fp16 =
        result.kv_bytes > 0
            ? fp16_bytes / static_cast<double>(result.kv_bytes)
            : 0.0;
    result.output_checksum = checksum(output);
    return result;
}

static RunResult benchmark_runtime(
    adaptq::RuntimeContext &context,
    const std::vector<float> &query,
    std::vector<float> &output,
    const Options &options,
    int tokens) {
    RunResult result;
    const adaptq::IKernelBackend *backend = context.get_kernel();
    result.backend = backend && backend->name() ? backend->name() : "unknown";

    result.timing = measure(
        [&] {
            context.compute(0, 0, query.data(), output.data());
        },
        options.warmup, options.repetitions, tokens);

    adaptq::IStorageBackend *storage = context.get_storage(0, 0);
    if (!storage)
        fail("RuntimeContext did not expose a storage backend");

    result.kv_bytes = storage->bytes_used();
    result.kv_cache_mb =
        static_cast<double>(result.kv_bytes) / (1024.0 * 1024.0);

    const double fp16_bytes = fp16_kv_bytes(tokens, options.dim);
    result.compression_ratio_vs_fp16 =
        result.kv_bytes > 0
            ? fp16_bytes / static_cast<double>(result.kv_bytes)
            : 0.0;
    result.output_checksum = checksum(output);
    return result;
}

static SelectionResult benchmark_backend_selection(ScopedBackendEnv::Mode mode,
                                                    const Options &options) {
    ScopedBackendEnv env(mode);

    adaptq::IKernelBackend *backend = adaptq::select_kernel_backend();
    if (!backend)
        fail("select_kernel_backend() returned null");

    SelectionResult result;
    result.backend = backend->name() ? backend->name() : "unknown";
    result.timing = measure(
        [] {
            adaptq::IKernelBackend *selected =
                adaptq::select_kernel_backend();
            if (!selected)
                fail("select_kernel_backend() returned null during timing");
        },
        options.warmup, options.repetitions, 1);
    return result;
}

static WorkloadResult benchmark_workload(int tokens,
                                         const Options &options) {
    if (tokens <= 0 || tokens > kMaxTokens)
        fail("workload token count out of supported range");

    std::mt19937 rng(0xC0FFEEu + static_cast<unsigned>(tokens));
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
    keys.reserve(static_cast<std::size_t>(tokens));
    values.reserve(static_cast<std::size_t>(tokens));

    for (int i = 0; i < tokens; ++i) {
        keys.push_back(make_vector(rng, options.dim));
        values.push_back(make_vector(rng, options.dim));
    }
    std::vector<float> query = make_vector(rng, options.dim);

    WorkloadResult result;
    result.tokens = tokens;

    std::vector<float> legacy_output(static_cast<std::size_t>(options.dim));
    std::vector<float> auto_output(static_cast<std::size_t>(options.dim));
    std::vector<float> scalar_output(static_cast<std::size_t>(options.dim));

    /*
     * The legacy AttentionHead is the in-tree pre-runtime execution path.
     * Keep its hybrid path disabled so both paths benchmark the quantized
     * KV-cache behavior at every requested sequence length.
     */
    {
        AttentionHead legacy;
        legacy.init(options.dim, options.bits, tokens, 0, 0.0f, 0);
        populate_legacy(legacy, keys, values);
        result.legacy = benchmark_legacy(
            legacy, query, legacy_output, options, tokens);
    }

    /*
     * Auto-selected v0.2.3 runtime.
     */
    {
        ScopedBackendEnv env(ScopedBackendEnv::Mode::Auto);

        adaptq::RuntimeContextConfig config;
        config.n_layers = 1;
        config.n_heads = 1;
        config.dim = options.dim;
        config.bits = options.bits;
        config.capacity = tokens;
        config.v_mass = 0.0f;

        adaptq::RuntimeContext context;
        context.init(config);
        populate_runtime(context, keys, values);

        result.runtime_auto = benchmark_runtime(
            context, query, auto_output, options, tokens);
    }

    /*
     * Forced scalar v0.2.3 runtime.
     */
    {
        ScopedBackendEnv env(ScopedBackendEnv::Mode::Scalar);

        adaptq::RuntimeContextConfig config;
        config.n_layers = 1;
        config.n_heads = 1;
        config.dim = options.dim;
        config.bits = options.bits;
        config.capacity = tokens;
        config.v_mass = 0.0f;

        adaptq::RuntimeContext context;
        context.init(config);
        populate_runtime(context, keys, values);

        result.runtime_scalar = benchmark_runtime(
            context, query, scalar_output, options, tokens);
    }

    result.accelerated_backend_available =
        result.runtime_auto.backend != result.runtime_scalar.backend;

    result.runtime_vs_legacy =
        compare_outputs(auto_output, legacy_output);

    if (result.accelerated_backend_available) {
        result.scalar_vs_auto =
            compare_outputs(scalar_output, auto_output);
        result.auto_speedup_vs_scalar =
            result.runtime_scalar.timing.p50_us /
            std::max(result.runtime_auto.timing.p50_us, 1e-12);
    } else {
        result.scalar_vs_auto = {};
        result.auto_speedup_vs_scalar = 0.0;
    }

    if (result.legacy.timing.p50_us > 0.0) {
        result.runtime_latency_delta_pct_vs_legacy =
            ((result.runtime_auto.timing.p50_us /
              result.legacy.timing.p50_us) - 1.0) * 100.0;

        result.runtime_throughput_ratio_vs_legacy =
            result.runtime_auto.timing.tokens_per_sec /
            std::max(result.legacy.timing.tokens_per_sec, 1e-12);
    }

    return result;
}

static void write_selection(std::ostream &out,
                            const SelectionResult &result) {
    out << "{\n"
        << "      \"backend\": ";
    json_string(out, result.backend);
    out << ",\n"
        << "      \"selection_latency\": {\n"
        << "        \"p50_us\": ";
    json_number(out, result.timing.p50_us);
    out << ",\n"
        << "        \"p95_us\": ";
    json_number(out, result.timing.p95_us);
    out << "\n      }\n"
        << "    }";
}

static void write_workload(std::ostream &out,
                           const WorkloadResult &result) {
    out << "    {\n"
        << "      \"tokens\": " << result.tokens << ",\n"
        << "      \"legacy_0_2_2_proxy\": ";
    write_run(out, result.legacy);
    out << ",\n"
        << "      \"runtime_0_2_3_auto\": ";
    write_run(out, result.runtime_auto);
    out << ",\n"
        << "      \"runtime_0_2_3_scalar\": ";
    write_run(out, result.runtime_scalar);
    out << ",\n"
        << "      \"runtime_vs_legacy\": {\n"
        << "        \"latency_delta_pct\": ";
    json_number(out, result.runtime_latency_delta_pct_vs_legacy);
    out << ",\n"
        << "        \"throughput_ratio\": ";
    json_number(out, result.runtime_throughput_ratio_vs_legacy);
    out << ",\n"
        << "        \"numerical_difference\": ";
    write_numeric(out, result.runtime_vs_legacy);
    out << "\n      },\n"
        << "      \"scalar_vs_auto\": {\n"
        << "        \"available\": "
        << (result.accelerated_backend_available ? "true" : "false");

    if (result.accelerated_backend_available) {
        out << ",\n"
            << "        \"auto_speedup_vs_scalar\": ";
        json_number(out, result.auto_speedup_vs_scalar);
        out << ",\n"
            << "        \"numerical_difference\": ";
        write_numeric(out, result.scalar_vs_auto);
    }

    out << "\n      }\n"
        << "    }";
}

static Options parse_options(int argc, char **argv) {
    Options options;

    auto require_value = [argc, argv](int &index,
                                      const char *name) -> std::string {
        if (index + 1 >= argc)
            fail(std::string(name) + " requires a value");
        return argv[++index];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--output") {
            options.output_path = require_value(i, "--output");
        } else if (arg == "--dim") {
            options.dim = std::stoi(require_value(i, "--dim"));
        } else if (arg == "--bits") {
            options.bits = std::stoi(require_value(i, "--bits"));
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(require_value(i, "--warmup"));
        } else if (arg == "--repetitions") {
            options.repetitions = std::stoi(require_value(i, "--repetitions"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "AdapTQ v0.2.3 runtime benchmark\n\n"
                << "Usage:\n"
                << "  adaptq_runtime_bench [options]\n\n"
                << "Options:\n"
                << "  --output PATH       Write JSON results to PATH\n"
                << "  --dim N             Head dimension (default: 128)\n"
                << "  --bits N            Quantization bits (default: 4)\n"
                << "  --warmup N          Warmup iterations per measurement (default: 10)\n"
                << "  --repetitions N     Measured iterations (default: 50)\n"
                << "  --help              Show this help message\n";
            std::exit(0);
        } else {
            fail("unknown argument: " + arg);
        }
    }

    if (options.dim <= 0)
        fail("--dim must be positive");
    if (options.bits < 2 || options.bits > 4)
        fail("--bits must be 2, 3, or 4");
    if (options.warmup < 0)
        fail("--warmup must be non-negative");
    if (options.repetitions <= 0)
        fail("--repetitions must be positive");

    return options;
}

static void write_json(std::ostream &out,
                       const Options &options,
                       const SelectionResult &auto_selection,
                       const SelectionResult &scalar_selection,
                       const std::vector<WorkloadResult> &workloads) {
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"benchmark\": \"adaptq-runtime-v0.2.3\",\n"
        << "  \"description\": "
        << "\"Compares the legacy AttentionHead path with RuntimeContext "
        << "and measures runtime backend selection.\",\n"
        << "  \"configuration\": {\n"
        << "    \"dimension\": " << options.dim << ",\n"
        << "    \"bits\": " << options.bits << ",\n"
        << "    \"warmup_iterations\": " << options.warmup << ",\n"
        << "    \"repetitions\": " << options.repetitions << ",\n"
        << "    \"workloads\": [128, 256, 512, 1024, 2048, 4096]\n"
        << "  },\n"
        << "  \"backend_selection\": {\n"
        << "    \"auto\": ";
    write_selection(out, auto_selection);
    out << ",\n"
        << "    \"forced_scalar\": ";
    write_selection(out, scalar_selection);
    out << "\n  },\n"
        << "  \"workloads\": [\n";

    for (std::size_t i = 0; i < workloads.size(); ++i) {
        if (i != 0)
            out << ",\n";
        write_workload(out, workloads[i]);
    }

    out << "\n  ],\n"
        << "  \"notes\": [\n"
        << "    \"The legacy AttentionHead path is used as an in-tree "
        << "0.2.2 behavior proxy; no claim is made that it is a "
        << "byte-for-byte reconstruction of a release artifact.\",\n"
        << "    \"The legacy hybrid warm-up path is disabled so every "
        << "requested workload exercises the quantized KV-cache path.\",\n"
        << "    \"Selection latency measures select_kernel_backend() itself; "
        << "it is not part of the steady-state compute timing.\",\n"
        << "    \"KV cache bytes exclude storage metadata overhead, matching "
        << "IStorageBackend::bytes_used().\"\n"
        << "  ]\n"
        << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);

        std::vector<WorkloadResult> workloads;
        workloads.reserve(sizeof(kWorkloads) / sizeof(kWorkloads[0]));

        const SelectionResult auto_selection =
            benchmark_backend_selection(
                ScopedBackendEnv::Mode::Auto, options);
        const SelectionResult scalar_selection =
            benchmark_backend_selection(
                ScopedBackendEnv::Mode::Scalar, options);

        std::cerr << "AdapTQ v0.2.3 runtime benchmark\n";
        std::cerr << "  dimension: " << options.dim
                  << "  bits: " << options.bits
                  << "  warmup: " << options.warmup
                  << "  repetitions: " << options.repetitions << "\n";
        std::cerr << "  auto backend: " << auto_selection.backend << "\n";
        std::cerr << "  scalar backend: " << scalar_selection.backend << "\n";

        for (int tokens : kWorkloads) {
            std::cerr << "  running " << tokens << " tokens...\n";
            workloads.push_back(benchmark_workload(tokens, options));
        }

        if (!options.output_path.empty()) {
            const std::filesystem::path output_path(options.output_path);
            if (output_path.has_parent_path()) {
                std::error_code ec;
                std::filesystem::create_directories(
                    output_path.parent_path(), ec);
                if (ec)
                    fail("failed to create output directory: " + ec.message());
            }

            std::ofstream file(output_path);
            if (!file)
                fail("failed to open output path: " + options.output_path);

            write_json(file, options, auto_selection, scalar_selection,
                       workloads);
            std::cerr << "  JSON written to " << options.output_path << "\n";
        } else {
            write_json(std::cout, options, auto_selection, scalar_selection,
                       workloads);
        }

        return 0;
    } catch (const std::exception &error) {
        std::cerr << "benchmark error: " << error.what() << "\n";
        return 1;
    }
}

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

/* -------------------------------------------------------------------------
 * cli/cmd_create_strategy.cpp — adaptq create-strategy <Name>
 *
 * Generates the scaffolding directory for a new IKVStrategy implementation:
 *
 *   strategies/<lowercase_name>/
 *   ├── strategy.cpp      — IKVStrategy template (compiles as FP passthrough)
 *   ├── CMakeLists.txt    — Links against libadaptq_core
 *   ├── benchmark.json    — Default benchmark parameters
 *   ├── README.md         — Template: description, algorithm, expected results
 *   └── tests.cpp         — Conformance instantiation
 *
 * Usage: adaptq create-strategy MyKV2027
 * ----------------------------------------------------------------------- */

namespace fs = std::filesystem;

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

static void write_file(const fs::path &path, const std::string &content) {
    std::ofstream f(path);
    if (!f) { std::cerr << "ERROR: cannot write " << path << "\n"; std::exit(1); }
    f << content;
}

/* ========================================================================
 * Template bodies
 * ====================================================================== */

static std::string tpl_strategy_cpp(const std::string &name,
                                     const std::string &lower) {
    return R"(#include "../../include/adaptq/strategy.h"
#include "../../include/adaptq/context.h"
#include "../../include/adaptq/storage.h"
#include <cstring>
#include <ostream>
#include <istream>

/* =========================================================================
 * )" + name + R"( : IKVStrategy
 *
 * TODO: Replace this description with a summary of your algorithm.
 *
 * Algorithm:
 *   Append: TODO
 *   Compute: TODO
 *   Eviction: TODO
 *
 * References:
 *   [1] TODO: paper or algorithm description
 * ======================================================================= */

namespace adaptq {

/* ---- Capability: Compression ----------------------------------------- */
class )" + name + R"(Compression final : public ICompression {
public:
    int dim    = 0;
    int padded = 0;

    CompressResult compress(const float           *x,
                            int                    dim,
                            bool                   is_key,
                            const ExecutionContext &ctx) override {
        /* TODO: Implement your compression algorithm here.
         * Call ctx.storage->write() with your packed data and return the slot. */
        int bytes = dim * sizeof(float);
        StorageSlot slot = ctx.storage->write(
            reinterpret_cast<const uint8_t *>(x),
            bytes, 1.0f, 0xFF  /* format_tag 0xFF = FP32 passthrough */
        );
        return ctx.storage->read(slot);
    }

    void decompress(const CompressResult &r,
                    int                   dim,
                    float                *out) const override {
        /* TODO: Implement decompression matching your compress() above. */
        memcpy(out, r.data, dim * sizeof(float));
    }
};

/* ---- Capability: Eviction -------------------------------------------- */
class )" + name + R"(Eviction final : public IEviction {
public:
    EvictionDecision on_append(const ExecutionContext &ctx) override {
        /* TODO: Implement your eviction policy.
         * Default: FIFO (evict oldest when at capacity). */
        if (ctx.cache_size >= ctx.cache_capacity)
            return {true, ctx.cache_size % ctx.cache_capacity};
        return {false, -1};
    }

    void on_attention(const AttentionFeedback &fb,
                      const ExecutionContext   &ctx) override {
        /* TODO: Update importance tracking from attention weights if needed.
         * fb.weights[i] is the softmax attention weight for slot fb.slots[i]. */
        (void)fb; (void)ctx;
    }
};

/* ---- Capability: Quality (optional — remove if not implementing) ------ */
// class )" + name + R"(Quality final : public IQuality { ... };

/* ---- Capability: Replay (optional — remove if not implementing) ------- */
// class )" + name + R"(ReplayHooks final : public IReplayHooks { ... };

/* ---- )" + name + R"( : IKVStrategy ------------------------------------ */
class )" + name + R"( final : public IKVStrategy {
public:
    void init(const HeadConfig &cfg) override {
        /* TODO: Initialise your internal state from cfg. */
        compress_.dim    = cfg.dim;
        compress_.padded = cfg.dim; /* set to next_pow2(dim) if needed */
        (void)cfg;
    }

    void reset() override {
        /* TODO: Clear per-session state. Keep configuration. */
    }

    const char *name() const override { return ")" + lower + R"("; }

    ICompression *compression() override { return &compress_; }
    IEviction    *eviction()    override { return &evict_;    }

    /* Uncomment when IQuality is implemented:
    IQuality     *quality()     override { return &quality_; } */

    /* Uncomment when IReplayHooks is implemented:
    IReplayHooks *replay_hooks()override { return &replay_; } */

private:
    )" + name + R"(Compression compress_;
    )" + name + R"(Eviction    evict_;
};

} /* namespace adaptq */
)";
}

static std::string tpl_cmakelists(const std::string &name,
                                   const std::string &lower) {
    return "# CMakeLists.txt for " + name + " strategy\n"
           "cmake_minimum_required(VERSION 3.18)\n"
           "project(" + lower + "_strategy)\n\n"
           "# Link against the AdapTQ core library.\n"
           "# Assumes this directory is added via add_subdirectory() from the root CMakeLists.txt,\n"
           "# or that adapTQ_core is found via find_package().\n\n"
           "add_library(" + lower + "_strategy SHARED strategy.cpp)\n"
           "target_include_directories(" + lower + "_strategy\n"
           "    PRIVATE ${CMAKE_SOURCE_DIR}/include)\n"
           "target_link_libraries(" + lower + "_strategy PRIVATE adapTQ_core)\n"
           "set_target_properties(" + lower + "_strategy PROPERTIES\n"
           "    CXX_STANDARD 17\n"
           "    CXX_STANDARD_REQUIRED ON)\n\n"
           "# Unit tests\n"
           "add_executable(" + lower + "_tests tests.cpp)\n"
           "target_include_directories(" + lower + "_tests\n"
           "    PRIVATE ${CMAKE_SOURCE_DIR}/include)\n"
           "target_link_libraries(" + lower + "_tests PRIVATE adapTQ_core Catch2::Catch2)\n"
           "add_test(NAME " + lower + "_conformance COMMAND " + lower + "_tests \"[conformance]\")\n";
}

static std::string tpl_benchmark_json(const std::string &name,
                                       const std::string &lower) {
    return "{\n"
           "    \"strategy\": \"" + lower + "\",\n"
           "    \"description\": \"" + name + " benchmark parameters\",\n"
           "    \"seq_lens\": [512, 1024, 2048, 4096, 8192],\n"
           "    \"head_dim\": 128,\n"
           "    \"n_heads\": 8,\n"
           "    \"bits\": 4,\n"
           "    \"capacity\": 8192,\n"
           "    \"metrics\": [\"latency_p50\", \"latency_p99\", \"kv_cache_mb\",\n"
           "                  \"compression_ratio\", \"cosine_sim_mean\"]\n"
           "}\n";
}

static std::string tpl_readme(const std::string &name,
                               const std::string &lower) {
    return "# " + name + "\n\n"
           "## Description\n\n"
           "TODO: One-paragraph description of the algorithm.\n\n"
           "## Algorithm\n\n"
           "TODO: Describe the compression, eviction, and quality estimation approach.\n\n"
           "### Append\n"
           "TODO\n\n"
           "### Compute\n"
           "TODO\n\n"
           "### Eviction\n"
           "TODO\n\n"
           "## Expected Performance\n\n"
           "| Metric | Expected Value |\n"
           "|---|---|\n"
           "| Bits per dim | TODO |\n"
           "| Cosine sim (vs FP32) | TODO |\n"
           "| Latency (p50, 4k ctx) | TODO µs |\n"
           "| Memory (4k ctx, 8h×128d) | TODO MB |\n\n"
           "## References\n\n"
           "- [1] TODO: Paper or algorithm reference\n\n"
           "## Usage\n\n"
           "```python\n"
           "from adaptq import Runtime\n"
           "rt = Runtime(strategy=\"" + lower + "\", bits=4, capacity=4096)\n"
           "```\n\n"
           "```bash\n"
           "adaptq bench --strategy " + lower + " --seq-lens 512,1024,4096\n"
           "```\n";
}

static std::string tpl_tests(const std::string &name) {
    return R"(#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "strategy.cpp"
#include "../../tests/conformance/strategy_conformance.cpp"

/* =========================================================================
 * )" + name + R"( — Conformance + Unit Tests
 *
 * The conformance block automatically runs all IKVStrategy contract tests.
 * Add your algorithm-specific tests below.
 * ======================================================================= */

/* Conformance: must pass with zero modifications. */
TEST_CASE(")" + name + R"( conformance", "[conformance]") {
    run_conformance<adaptq::)" + name + R"(>(")" + name + R"(");
}

/* ---- Algorithm-specific tests ---------------------------------------- */
/* TODO: Add tests specific to your algorithm's behavior.
 *
 * Example:
 *   TEST_CASE(")" + name + R"( 4-bit MSE within bounds", "[unit]") {
 *       // Test that your quantization error is within expected bounds
 *   }
 *
 *   TEST_CASE(")" + name + R"( eviction selects lowest-weight token", "[unit]") {
 *       // Test your eviction policy with known attention weights
 *   }
 */
)";
}

/* ========================================================================
 * Entry point called from main CLI
 * ====================================================================== */
namespace adaptq {
int cmd_create_strategy(int argc, char **argv) {
    if (argc < 1) {
        std::cerr << "Usage: adaptq create-strategy <StrategyName>\n"
                     "       Name should be PascalCase (e.g. MyKV2027)\n";
        return 1;
    }

    std::string name  = argv[0];
    std::string lower = to_lower(name);

    /* Validate: alphanumeric only */
    for (char c : name) {
        if (!std::isalnum((unsigned char)c)) {
            std::cerr << "ERROR: strategy name must be alphanumeric, got '" << name << "'\n";
            return 1;
        }
    }

    fs::path dir = fs::path("strategies") / lower;
    if (fs::exists(dir)) {
        std::cerr << "ERROR: directory '" << dir << "' already exists.\n";
        return 1;
    }

    try {
        fs::create_directories(dir);
    } catch (const std::exception &e) {
        std::cerr << "ERROR: cannot create directory: " << e.what() << "\n";
        return 1;
    }

    write_file(dir / "strategy.cpp",     tpl_strategy_cpp(name, lower));
    write_file(dir / "CMakeLists.txt",   tpl_cmakelists(name, lower));
    write_file(dir / "benchmark.json",   tpl_benchmark_json(name, lower));
    write_file(dir / "README.md",        tpl_readme(name, lower));
    write_file(dir / "tests.cpp",        tpl_tests(name));

    std::cout << "Created strategy scaffolding in strategies/" << lower << "/\n\n"
              << "Files generated:\n"
              << "  strategy.cpp     — IKVStrategy template (runs as FP passthrough)\n"
              << "  CMakeLists.txt   — Build config\n"
              << "  benchmark.json   — Default benchmark parameters\n"
              << "  README.md        — Documentation template\n"
              << "  tests.cpp        — Conformance + unit tests\n\n"
              << "Next steps:\n"
              << "  1. Edit strategies/" << lower << "/strategy.cpp\n"
              << "  2. Run: adaptq test-strategy strategies/" << lower << "/\n"
              << "  3. Run: adaptq bench --strategy " << lower << "\n";

    return 0;
}

} /* namespace adaptq */

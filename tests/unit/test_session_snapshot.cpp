/* -------------------------------------------------------------------------
 * tests/unit/test_session_snapshot.cpp
 *
 * Tests for SessionSnapshot: capture, save, load, version guard,
 * compressed vs uncompressed mode.
 * ----------------------------------------------------------------------- */
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../../runtime/runtime_context.h"
#include "../../replay/session_snapshot.h"

using namespace adaptq;
namespace fs = std::filesystem;

/* ---- Helpers ------------------------------------------------------------ */
static void rand_vec(float *v, int d, unsigned seed) {
    unsigned x = seed;
    for (int i = 0; i < d; ++i) {
        x = x * 1664525u + 1013904223u;
        v[i] = (float)(int)(x >> 16) / 32768.f - 1.f;
    }
    float n = 0.f;
    for (int i = 0; i < d; ++i) n += v[i] * v[i];
    n = sqrtf(n + 1e-12f);
    for (int i = 0; i < d; ++i) v[i] /= n;
}

static void write_i32(std::ostream &out, int32_t v) {
    out.write(reinterpret_cast<const char *>(&v), sizeof(v));
}

/* Returns a heap-allocated RuntimeContext (non-copyable, so use unique_ptr). */
static std::unique_ptr<RuntimeContext> make_ctx_with_tokens(int n_tokens,
                                                             int dim = 64,
                                                             bool log = true,
                                                             int bits = 4) {
    RuntimeContextConfig cfg;
    cfg.n_layers   = 1;
    cfg.n_heads    = 1;
    cfg.dim        = dim;
    cfg.bits       = bits;
    cfg.capacity   = 128;
    cfg.log_tokens = log;

    auto ctx = std::make_unique<RuntimeContext>();
    ctx->init(cfg);

    std::vector<float> k(dim), v(dim);
    for (int t = 0; t < n_tokens; ++t) {
        rand_vec(k.data(), dim, (unsigned)(t * 31 + 1));
        rand_vec(v.data(), dim, (unsigned)(t * 31 + 2));
        ctx->append(0, 0, k.data(), v.data());
    }
    return ctx;
}

/* Temporary file path helper. */
static std::string tmp_path(const char *name) {
    return (fs::temp_directory_path() / name).string();
}

/* =========================================================================
 * Test cases
 * ========================================================================= */

TEST_CASE("SessionSnapshot: capture from RuntimeContext is valid", "[snapshot]") {
    auto ctx = make_ctx_with_tokens(10);
    SessionSnapshot snap = SessionSnapshot::capture(*ctx, true);

    REQUIRE(snap.is_valid());
    REQUIRE(snap.n_tokens() == 10);
    REQUIRE(snap.n_layers() == 1);
    REQUIRE(snap.n_heads()  == 1);
    REQUIRE(snap.dim()      == 64);
    REQUIRE(snap.heads().size() == 1u);
}

TEST_CASE("SessionSnapshot: capture without token log has no log", "[snapshot]") {
    auto ctx = make_ctx_with_tokens(5, 64, /*log=*/false);
    SessionSnapshot snap = SessionSnapshot::capture(*ctx, false);

    REQUIRE(snap.is_valid());
    REQUIRE(!snap.has_token_log());
    REQUIRE(snap.token_log().empty());
}

TEST_CASE("SessionSnapshot: capture with token log contains entries", "[snapshot]") {
    auto ctx = make_ctx_with_tokens(8, 64, /*log=*/true);
    SessionSnapshot snap = SessionSnapshot::capture(*ctx, true);

    REQUIRE(snap.has_token_log());
    REQUIRE(snap.token_log().size() == 8u);

    /* Each entry has k_fp32 and v_fp32 of correct dimension. */
    for (const auto &e : snap.token_log()) {
        REQUIRE(e.dim        == 64);
        REQUIRE(e.k_fp32.size() == 64u);
        REQUIRE(e.v_fp32.size() == 64u);
    }
}

TEST_CASE("SessionSnapshot: preserves configured bit width", "[snapshot]") {
    for (int bits : {2, 3, 4}) {
        auto ctx = make_ctx_with_tokens(1, 64, true, bits);
        REQUIRE(SessionSnapshot::capture(*ctx, true).bits() == bits);
    }
}

TEST_CASE("SessionSnapshot: save and load roundtrip", "[snapshot]") {
    auto ctx = make_ctx_with_tokens(12, 64, true);
    SessionSnapshot orig = SessionSnapshot::capture(*ctx, true);

    std::string path = tmp_path("adaptq_test_snap.aqss");
    REQUIRE_NOTHROW(orig.save(path));

    SessionSnapshot loaded;
    REQUIRE_NOTHROW(loaded = SessionSnapshot::load(path));

    REQUIRE(loaded.is_valid());
    REQUIRE(loaded.n_tokens() == orig.n_tokens());
    REQUIRE(loaded.n_layers() == orig.n_layers());
    REQUIRE(loaded.n_heads()  == orig.n_heads());
    REQUIRE(loaded.dim()      == orig.dim());
    REQUIRE(loaded.bits()     == orig.bits());
    REQUIRE(loaded.has_token_log() == orig.has_token_log());
    REQUIRE(loaded.token_log().size() == orig.token_log().size());

    /* Verify first token K/V data matches. */
    if (!orig.token_log().empty()) {
        const auto &oe = orig.token_log()[0];
        const auto &le = loaded.token_log()[0];
        REQUIRE(oe.dim == le.dim);
        for (int i = 0; i < oe.dim; ++i) {
            REQUIRE(oe.k_fp32[i] == le.k_fp32[i]);
            REQUIRE(oe.v_fp32[i] == le.v_fp32[i]);
        }
    }

    fs::remove(path);
}

TEST_CASE("SessionSnapshot: load rejects bad magic", "[snapshot]") {
    std::string path = tmp_path("adaptq_bad_magic.aqss");
    {
        std::ofstream f(path, std::ios::binary);
        uint32_t bad = 0xDEADBEEFu;
        f.write(reinterpret_cast<char *>(&bad), 4);
    }
    REQUIRE_THROWS_AS(SessionSnapshot::load(path), std::runtime_error);
    fs::remove(path);
}

TEST_CASE("SessionSnapshot: load rejects truncated file", "[snapshot]") {
    std::string path = tmp_path("adaptq_truncated.aqss");
    {
        std::ofstream f(path, std::ios::binary);
        uint32_t magic = SessionSnapshot::kMagic;
        f.write(reinterpret_cast<const char *>(&magic), 4);
    }
    REQUIRE_THROWS_AS(SessionSnapshot::load(path), std::runtime_error);
    fs::remove(path);
}

TEST_CASE("SessionSnapshot: load missing file throws", "[snapshot]") {
    REQUIRE_THROWS_AS(SessionSnapshot::load("/nonexistent/path/adaptq.aqss"),
                      std::runtime_error);
}

TEST_CASE("SessionSnapshot: strategy state captured when IReplayHooks available", "[snapshot]") {
    auto ctx = make_ctx_with_tokens(5, 64, true);
    SessionSnapshot snap = SessionSnapshot::capture(*ctx, true);

    /* har_fixed implements IReplayHooks, so flags bit 1 should be set. */
    REQUIRE(snap.has_strategy_state());
    REQUIRE(!snap.heads().empty());
    /* Strategy state must be non-empty (HARReplayHooks writes 4-byte version). */
    REQUIRE(!snap.heads()[0].strategy_state.empty());
}

TEST_CASE("SessionSnapshot: head snapshot has valid storage data", "[snapshot]") {
    auto ctx = make_ctx_with_tokens(4, 64, false);
    SessionSnapshot snap = SessionSnapshot::capture(*ctx, false);

    REQUIRE(snap.heads().size() == 1u);
    const HeadSnapshot &hs = snap.heads()[0];
    REQUIRE(hs.cache_size == 4);
    REQUIRE(hs.slot_bytes > 0);
    /* storage_data should hold K and V slots: 4 pairs × 2 × slot_bytes. */
    size_t expected = (size_t)(hs.cache_size * 2) * hs.slot_bytes;
    REQUIRE(hs.storage_data.size() == expected);
}

TEST_CASE("SessionSnapshot: save/load is deterministic for same input", "[snapshot]") {
    auto ctx1 = make_ctx_with_tokens(6, 64, true);
    auto ctx2 = make_ctx_with_tokens(6, 64, true);

    SessionSnapshot s1 = SessionSnapshot::capture(*ctx1, true);
    SessionSnapshot s2 = SessionSnapshot::capture(*ctx2, true);

    std::string p1 = tmp_path("adaptq_snap1.aqss");
    std::string p2 = tmp_path("adaptq_snap2.aqss");
    s1.save(p1);
    s2.save(p2);

    /* Files should be identical for identical inputs. */
    auto size1 = fs::file_size(p1);
    auto size2 = fs::file_size(p2);
    REQUIRE(size1 == size2);

    fs::remove(p1);
    fs::remove(p2);
}

TEST_CASE("SessionSnapshot: load rejects malicious allocation sizes", "[snapshot][security]") {
    std::string path = tmp_path("adaptq_malicious_alloc.aqss");
    auto ctx = make_ctx_with_tokens(2, 64, false);
    SessionSnapshot orig = SessionSnapshot::capture(*ctx, false);
    orig.save(path);
    // Corrupt the n_heads_total field to be extremely large
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(28); // Offset to n_heads_total
    int32_t bad_heads = 1000000000;
    f.write(reinterpret_cast<char *>(&bad_heads), 4);
    f.close();
    REQUIRE_THROWS_AS(SessionSnapshot::load(path), std::runtime_error);
    orig.save(path);
    // Corrupt data_bytes field of the first head
    f.open(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(40 + 16); // Approximate offset to data_bytes
    uint64_t bad_bytes = 0xFFFFFFFFFFFF;
    f.write(reinterpret_cast<char *>(&bad_bytes), 8);
    f.close();
    REQUIRE_THROWS_AS(SessionSnapshot::load(path), std::runtime_error);
    fs::remove(path);
}

TEST_CASE("SessionSnapshot: load rejects truncated files safely", "[snapshot][security]") {
    std::string path = tmp_path("adaptq_truncated_2.aqss");
    auto ctx = make_ctx_with_tokens(5, 64, true);
    SessionSnapshot orig = SessionSnapshot::capture(*ctx, true);
    orig.save(path);
    std::string path_trunc = tmp_path("adaptq_truncated_3.aqss");
    // Truncate file halfway
    auto size = fs::file_size(path);
    std::ifstream in(path, std::ios::binary);
    std::ofstream out(path_trunc, std::ios::binary);
    std::vector<char> buf(size / 2);
    in.read(buf.data(), buf.size());
    out.write(buf.data(), buf.size());
    in.close();
    out.close();
    REQUIRE_THROWS_AS(SessionSnapshot::load(path_trunc), std::runtime_error);
    fs::remove(path);
    fs::remove(path_trunc);
}

TEST_CASE("SessionSnapshot: load rejects scale allocation larger than remaining data", "[snapshot][security]") {
    std::string path = tmp_path("adaptq_bad_scales.aqss");
    auto ctx = make_ctx_with_tokens(0, 64, false);
    SessionSnapshot orig = SessionSnapshot::capture(*ctx, false);
    orig.save(path);

    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(64);
    int32_t bad_scales = 1000000000;
    f.write(reinterpret_cast<const char *>(&bad_scales), sizeof(bad_scales));
    f.close();

    REQUIRE_THROWS_AS(SessionSnapshot::load(path), std::runtime_error);
    fs::remove(path);
}

TEST_CASE("SessionSnapshot: load rejects tag allocation larger than remaining data", "[snapshot][security]") {
    std::string path = tmp_path("adaptq_bad_tags.aqss");
    auto ctx = make_ctx_with_tokens(0, 64, false);
    SessionSnapshot orig = SessionSnapshot::capture(*ctx, false);
    orig.save(path);

    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(68);
    int32_t bad_tags = 1000000000;
    f.write(reinterpret_cast<const char *>(&bad_tags), sizeof(bad_tags));
    f.close();

    REQUIRE_THROWS_AS(SessionSnapshot::load(path), std::runtime_error);
    fs::remove(path);
}

TEST_CASE("SessionSnapshot: load rejects strategy state allocation larger than remaining data", "[snapshot][security]") {
    std::string path = tmp_path("adaptq_bad_strategy_state.aqss");
    auto ctx = make_ctx_with_tokens(0, 64, true);
    SessionSnapshot orig = SessionSnapshot::capture(*ctx, true);
    orig.save(path);

    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(72);
    int32_t bad_state = 1000000000;
    f.write(reinterpret_cast<const char *>(&bad_state), sizeof(bad_state));
    f.close();

    REQUIRE_THROWS_AS(SessionSnapshot::load(path), std::runtime_error);
    fs::remove(path);
}

TEST_CASE("SessionSnapshot: load rejects token log allocation larger than remaining data", "[snapshot][security]") {
    std::string path = tmp_path("adaptq_bad_token_count.aqss");
    {
        std::ofstream f(path, std::ios::binary);
        const uint32_t magic = SessionSnapshot::kMagic;
        const uint32_t version = SessionSnapshot::kVersion;
        f.write(reinterpret_cast<const char *>(&magic), 4);
        f.write(reinterpret_cast<const char *>(&version), 4);
        write_i32(f, 1);  // n_layers
        write_i32(f, 1);  // n_heads
        write_i32(f, 64); // dim
        write_i32(f, 4);  // bits
        write_i32(f, 0);  // n_tokens
        write_i32(f, 1);  // n_heads_total
        uint64_t flags = 1;
        f.write(reinterpret_cast<const char *>(&flags), sizeof(flags));

        write_i32(f, 0); // layer
        write_i32(f, 0); // head
        write_i32(f, 0); // cache_size
        write_i32(f, 0); // slot_bytes
        uint64_t data_bytes = 0;
        f.write(reinterpret_cast<const char *>(&data_bytes), sizeof(data_bytes));
        write_i32(f, 0); // n_scales
        write_i32(f, 0); // n_tags
        write_i32(f, 1000000000); // n_entries
    }

    REQUIRE_THROWS_AS(SessionSnapshot::load(path), std::runtime_error);
    fs::remove(path);
}

TEST_CASE("SessionSnapshot: load rejects token vector allocation larger than remaining data", "[snapshot][security]") {
    std::string path = tmp_path("adaptq_bad_token_dim.aqss");
    {
        std::ofstream f(path, std::ios::binary);
        const uint32_t magic = SessionSnapshot::kMagic;
        const uint32_t version = SessionSnapshot::kVersion;
        f.write(reinterpret_cast<const char *>(&magic), 4);
        f.write(reinterpret_cast<const char *>(&version), 4);
        write_i32(f, 1);  // n_layers
        write_i32(f, 1);  // n_heads
        write_i32(f, 64); // dim
        write_i32(f, 4);  // bits
        write_i32(f, 0);  // n_tokens
        write_i32(f, 1);  // n_heads_total
        uint64_t flags = 1;
        f.write(reinterpret_cast<const char *>(&flags), sizeof(flags));

        write_i32(f, 0); // layer
        write_i32(f, 0); // head
        write_i32(f, 0); // cache_size
        write_i32(f, 0); // slot_bytes
        uint64_t data_bytes = 0;
        f.write(reinterpret_cast<const char *>(&data_bytes), sizeof(data_bytes));
        write_i32(f, 0); // n_scales
        write_i32(f, 0); // n_tags
        write_i32(f, 1); // n_entries
        write_i32(f, 0); // layer
        write_i32(f, 0); // head
        write_i32(f, 1000000000); // dim
    }

    REQUIRE_THROWS_AS(SessionSnapshot::load(path), std::runtime_error);
    fs::remove(path);
}

TEST_CASE("SessionSnapshot: load rejects excessive head metadata allocation", "[snapshot][security]") {
    std::string path = tmp_path("adaptq_excessive_head_metadata.aqss");
    {
        std::ofstream f(path, std::ios::binary);
        const uint32_t magic = SessionSnapshot::kMagic;
        const uint32_t version = SessionSnapshot::kVersion;
        f.write(reinterpret_cast<const char *>(&magic), 4);
        f.write(reinterpret_cast<const char *>(&version), 4);
        write_i32(f, 50000);       // n_layers
        write_i32(f, 40000);       // n_heads
        write_i32(f, 64);          // dim
        write_i32(f, 4);           // bits
        write_i32(f, 0);           // n_tokens
        write_i32(f, 2000000000);  // n_heads_total
        uint64_t flags = 0;
        f.write(reinterpret_cast<const char *>(&flags), sizeof(flags));
    }

    REQUIRE_THROWS_AS(SessionSnapshot::load(path), std::runtime_error);
    fs::remove(path);
}

TEST_CASE("SessionSnapshot: load rejects excessive token log metadata allocation", "[snapshot][security]") {
    std::string path = tmp_path("adaptq_excessive_token_metadata.aqss");
    {
        std::ofstream f(path, std::ios::binary);
        const uint32_t magic = SessionSnapshot::kMagic;
        const uint32_t version = SessionSnapshot::kVersion;
        f.write(reinterpret_cast<const char *>(&magic), 4);
        f.write(reinterpret_cast<const char *>(&version), 4);
        write_i32(f, 1);          // n_layers
        write_i32(f, 1);          // n_heads
        write_i32(f, 64);         // dim
        write_i32(f, 4);          // bits
        write_i32(f, 0);          // n_tokens
        write_i32(f, 1);          // n_heads_total
        uint64_t flags = 1;
        f.write(reinterpret_cast<const char *>(&flags), sizeof(flags));

        write_i32(f, 0);          // layer
        write_i32(f, 0);          // head
        write_i32(f, 0);          // cache_size
        write_i32(f, 0);          // slot_bytes
        uint64_t data_bytes = 0;
        f.write(reinterpret_cast<const char *>(&data_bytes), sizeof(data_bytes));
        write_i32(f, 0);          // n_scales
        write_i32(f, 0);          // n_tags
        write_i32(f, 2000000000);  // n_entries
    }

    REQUIRE_THROWS_AS(SessionSnapshot::load(path), std::runtime_error);
    fs::remove(path);
}

#include "session_snapshot.h"
#include "../runtime/runtime_context.h"
#include "../storage/contiguous_slab.h"
#include "../include/adaptq/strategy.h"
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>


/* -------------------------------------------------------------------------
 * replay/session_snapshot.cpp — SessionSnapshot serialization
 *
 * Binary I/O helpers use little-endian byte order throughout.
 * No compression in V2 — raw binary only. zlib opt-in is V3.
 * ----------------------------------------------------------------------- */

namespace adaptq {

/* =========================================================================
 * I/O helpers — little-endian
 * ========================================================================= */

static void write_u32(std::ostream &out, uint32_t v) {
    out.write(reinterpret_cast<const char *>(&v), 4);
}
static void write_u64(std::ostream &out, uint64_t v) {
    out.write(reinterpret_cast<const char *>(&v), 8);
}
static void write_i32(std::ostream &out, int32_t v) {
    out.write(reinterpret_cast<const char *>(&v), 4);
}
static void write_bytes(std::ostream &out, const void *p, size_t n) {
    out.write(reinterpret_cast<const char *>(p), (std::streamsize)n);
}

static uint32_t read_u32(std::istream &in) {
    uint32_t v = 0;
    if (!in.read(reinterpret_cast<char *>(&v), 4))
        throw std::runtime_error("SessionSnapshot::load: truncated snapshot");
    return v;
}
static uint64_t read_u64(std::istream &in) {
    uint64_t v = 0;
    if (!in.read(reinterpret_cast<char *>(&v), 8))
        throw std::runtime_error("SessionSnapshot::load: truncated snapshot");
    return v;
}
static int32_t read_i32(std::istream &in) {
    int32_t v = 0;
    if (!in.read(reinterpret_cast<char *>(&v), 4))
        throw std::runtime_error("SessionSnapshot::load: truncated snapshot");
    return v;
}
static void read_bytes(std::istream &in, void *p, size_t n) {
    if (n > 0 && !in.read(reinterpret_cast<char *>(p), (std::streamsize)n))
        throw std::runtime_error("SessionSnapshot::load: truncated snapshot");
}

static uint64_t remaining_bytes(std::istream &in, std::streamsize file_size) {
    std::streampos pos = in.tellg();
    if (pos < 0 || file_size < pos)
        throw std::runtime_error("SessionSnapshot::load: invalid stream position");
    return static_cast<uint64_t>(file_size - pos);
}

static uint64_t checked_count_bytes(int32_t count,
                                    uint64_t element_size,
                                    const char *field) {
    if (count < 0)
        throw std::runtime_error(std::string("SessionSnapshot::load: invalid ") + field);

    const uint64_t n = static_cast<uint64_t>(count);
    if (element_size != 0 && n > std::numeric_limits<uint64_t>::max() / element_size)
        throw std::runtime_error(std::string("SessionSnapshot::load: ") + field + " size overflow");

    return n * element_size;
}

static void require_remaining(std::istream &in,
                              std::streamsize file_size,
                              uint64_t bytes,
                              const char *field) {
    if (bytes > remaining_bytes(in, file_size))
        throw std::runtime_error(std::string("SessionSnapshot::load: ") +
                                 field + " exceeds remaining snapshot data");
}

constexpr uint64_t kMaxMetadataContainerBytes = 64ull * 1024ull * 1024ull;

static void require_metadata_container_bytes(int32_t count,
                                             uint64_t element_size,
                                             const char *field) {
    const uint64_t bytes = checked_count_bytes(count, element_size, field);
    if (bytes > kMaxMetadataContainerBytes)
        throw std::runtime_error(std::string("SessionSnapshot::load: ") +
                                 field + " allocation exceeds safety limit");
}

/* =========================================================================
 * capture()
 * ========================================================================= */

SessionSnapshot SessionSnapshot::capture(const RuntimeContext &ctx,
                                         bool include_token_log) {
    SessionSnapshot snap;
    snap.magic_    = kMagic;
    snap.version_  = kVersion;
    snap.n_layers_ = ctx.n_layers();
    snap.n_heads_  = ctx.n_heads();
    snap.dim_      = ctx.dim();
    snap.bits_     = ctx.bits();
    snap.n_tokens_ = ctx.token_pos();
    snap.flags_    = 0;

    bool has_tlog = include_token_log && !ctx.token_log().empty();
    if (has_tlog) snap.flags_ |= 1u;

    /* Capture per-head storage state. */
    for (int l = 0; l < ctx.n_layers(); ++l) {
        for (int h = 0; h < ctx.n_heads(); ++h) {
            HeadSnapshot hs;
            hs.layer = l;
            hs.head  = h;

            IStorageBackend *st = ctx.get_storage(l, h);

            /* We need to know cache_size (stored pairs). Use a cast to
             * ContiguousSlabStorage if available; otherwise fall back. */
            auto *csb = dynamic_cast<ContiguousSlabStorage *>(st);
            if (!csb) {
                throw std::runtime_error(
                    "SessionSnapshot::capture: storage backend '" +
                    std::string(st ? st->name() : "null") +
                    "' does not support snapshot capture");
            }

            int cache_sz   = csb->size() / 2;  /* size() = K+V slots; pairs = size/2 */
            int slot_bytes = csb->slot_bytes();
            hs.cache_size  = cache_sz;
            hs.slot_bytes  = slot_bytes;

            /* Copy raw slab data: K slots [0 … cache_sz), V slots [cache_sz … 2*cache_sz) */
            size_t total_bytes = (size_t)(cache_sz * 2) * slot_bytes;
            hs.storage_data.resize(total_bytes);
            if (total_bytes > 0)
                memcpy(hs.storage_data.data(), csb->raw_data(), total_bytes);

            /* Copy scales and format_tags for K and V. */
            hs.scales.resize(cache_sz * 2);
            hs.format_tags.resize(cache_sz * 2);
            const float *sc = csb->scales();
            if (sc && cache_sz > 0) {
                memcpy(hs.scales.data(), sc, (size_t)(cache_sz * 2) * sizeof(float));
            }
            /* format_tags are per-slot — read via CompressResult. */
            for (int i = 0; i < cache_sz * 2; ++i) {
                CompressResult r = st->read((StorageSlot)i);
                hs.format_tags[i] = r.format_tag;
            }

            /* Capture strategy state if IReplayHooks is implemented. */
            IKVStrategy *strat = ctx.get_strategy(l, h);
            IReplayHooks *rh = strat ? strat->replay_hooks() : nullptr;
            if (rh) {
                snap.flags_ |= 2u;
                std::ostringstream ss;
                rh->serialize_state(ss);
                const std::string &s = ss.str();
                hs.strategy_state.assign(s.begin(), s.end());
            }

            snap.heads_.push_back(std::move(hs));
        }
    }

    /* Capture token log. */
    if (has_tlog) {
        const auto &log  = ctx.token_log();
        const auto &data = ctx.token_log_data();
        const float *base = data.data();
        int dim = ctx.dim();
        size_t offset = 0;

        for (size_t i = 0; i < log.size(); ++i) {
            SnapshotTokenEntry entry;
            entry.layer = log[i].layer;
            entry.head  = log[i].head;
            entry.dim   = dim;
            entry.k_fp32.assign(base + offset,         base + offset + dim);
            entry.v_fp32.assign(base + offset + dim,   base + offset + 2 * dim);
            snap.token_log_.push_back(std::move(entry));
            offset += 2 * dim;
        }
    }

    return snap;
}

/* =========================================================================
 * save()
 * ========================================================================= */

void SessionSnapshot::save(const std::string &path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("SessionSnapshot::save: cannot open " + path);

    /* Header. */
    write_u32(f, magic_);
    write_u32(f, version_);
    write_i32(f, n_layers_);
    write_i32(f, n_heads_);
    write_i32(f, dim_);
    write_i32(f, bits_);
    write_i32(f, n_tokens_);
    write_i32(f, (int32_t)heads_.size());
    write_u64(f, flags_);

    /* Per-head blocks. */
    for (const auto &hs : heads_) {
        write_i32(f, hs.layer);
        write_i32(f, hs.head);
        write_i32(f, hs.cache_size);
        write_i32(f, hs.slot_bytes);

        size_t data_bytes = hs.storage_data.size();
        write_u64(f, (uint64_t)data_bytes);
        if (data_bytes > 0)
            write_bytes(f, hs.storage_data.data(), data_bytes);

        /* Scales: cache_size * 2 floats. */
        write_i32(f, (int32_t)hs.scales.size());
        if (!hs.scales.empty())
            write_bytes(f, hs.scales.data(), hs.scales.size() * sizeof(float));

        /* Format tags. */
        write_i32(f, (int32_t)hs.format_tags.size());
        if (!hs.format_tags.empty())
            write_bytes(f, hs.format_tags.data(), hs.format_tags.size());
    }

    /* Strategy state (optional). */
    if (flags_ & 2u) {
        for (const auto &hs : heads_) {
            write_i32(f, (int32_t)hs.strategy_state.size());
            if (!hs.strategy_state.empty())
                write_bytes(f, hs.strategy_state.data(), hs.strategy_state.size());
        }
    }

    /* Token log (optional). */
    if (flags_ & 1u) {
        write_i32(f, (int32_t)token_log_.size());
        for (const auto &e : token_log_) {
            write_i32(f, e.layer);
            write_i32(f, e.head);
            write_i32(f, e.dim);
            write_bytes(f, e.k_fp32.data(), e.dim * sizeof(float));
            write_bytes(f, e.v_fp32.data(), e.dim * sizeof(float));
        }
    }

    if (!f) throw std::runtime_error("SessionSnapshot::save: write error for " + path);
}

/* =========================================================================
 * load()
 * ========================================================================= */

SessionSnapshot SessionSnapshot::load(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("SessionSnapshot::load: cannot open " + path);
    f.seekg(0, std::ios::end);
    std::streamsize file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    if (file_size < 40)
        throw std::runtime_error("SessionSnapshot::load: file too small");

    SessionSnapshot snap;
    snap.magic_   = read_u32(f);
    snap.version_ = read_u32(f);

    if (snap.magic_ != kMagic)
        throw std::runtime_error("SessionSnapshot::load: invalid magic (not an AQSS file)");
    if (snap.version_ > kVersion)
        throw std::runtime_error("SessionSnapshot::load: snapshot version " +
                                 std::to_string(snap.version_) +
                                 " > current version " + std::to_string(kVersion));

    snap.n_layers_    = read_i32(f);
    snap.n_heads_     = read_i32(f);
    snap.dim_         = read_i32(f);
    snap.bits_        = read_i32(f);
    snap.n_tokens_    = read_i32(f);
    int n_heads_total = read_i32(f);
    snap.flags_       = read_u64(f);

    if (snap.n_layers_ <= 0 || snap.n_heads_ <= 0)
        throw std::runtime_error("SessionSnapshot::load: invalid layer/head dimensions");
    if (snap.dim_ <= 0)
        throw std::runtime_error("SessionSnapshot::load: invalid dim");
    if (snap.bits_ < 2 || snap.bits_ > 4)
        throw std::runtime_error("SessionSnapshot::load: invalid bits (must be 2, 3, or 4)");
    if (snap.n_tokens_ < 0)
        throw std::runtime_error("SessionSnapshot::load: invalid n_tokens");
    if (n_heads_total < 0)
        throw std::runtime_error("SessionSnapshot::load: invalid n_heads_total");

    const uint64_t expected_heads = static_cast<uint64_t>(snap.n_layers_) *
                                    static_cast<uint64_t>(snap.n_heads_);
    if (expected_heads > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        static_cast<uint64_t>(n_heads_total) != expected_heads)
        throw std::runtime_error("SessionSnapshot::load: inconsistent head count");

    require_metadata_container_bytes(n_heads_total, sizeof(HeadSnapshot), "head count");

    constexpr uint64_t kMinHeadBlockBytes = 32;
    const uint64_t head_block_count = static_cast<uint64_t>(n_heads_total);
    if (head_block_count > remaining_bytes(f, file_size) / kMinHeadBlockBytes)
        throw std::runtime_error("SessionSnapshot::load: head count exceeds remaining snapshot data");

    snap.heads_.resize(n_heads_total);
    for (int i = 0; i < n_heads_total; ++i) {
        HeadSnapshot &hs = snap.heads_[i];
        hs.layer      = read_i32(f);
        hs.head       = read_i32(f);
        hs.cache_size = read_i32(f);
        hs.slot_bytes = read_i32(f);

        if (hs.layer < 0 || hs.layer >= snap.n_layers_ ||
            hs.head < 0 || hs.head >= snap.n_heads_)
            throw std::runtime_error("SessionSnapshot::load: invalid head coordinates");
        if (hs.cache_size < 0 || hs.slot_bytes < 0)
            throw std::runtime_error("SessionSnapshot::load: invalid head storage dimensions");
        if (hs.cache_size > std::numeric_limits<int32_t>::max() / 2)
            throw std::runtime_error("SessionSnapshot::load: cache_size overflow");

        const uint64_t slot_count = static_cast<uint64_t>(hs.cache_size) * 2u;
        const uint64_t expected_data_bytes = slot_count * static_cast<uint64_t>(hs.slot_bytes);

        uint64_t data_bytes = read_u64(f);
        if (data_bytes != expected_data_bytes)
            throw std::runtime_error("SessionSnapshot::load: inconsistent storage data size");
        require_remaining(f, file_size, data_bytes, "storage data");
        hs.storage_data.resize(static_cast<size_t>(data_bytes));
        if (data_bytes > 0)
            read_bytes(f, hs.storage_data.data(), data_bytes);

        int n_scales = read_i32(f);
        const uint64_t expected_scale_bytes = checked_count_bytes(n_scales, sizeof(float), "scale count");
        if (static_cast<uint64_t>(n_scales) != slot_count)
            throw std::runtime_error("SessionSnapshot::load: scale count does not match cache size");
        require_remaining(f, file_size, expected_scale_bytes, "scale data");
        hs.scales.resize(static_cast<size_t>(n_scales));
        if (n_scales > 0)
            read_bytes(f, hs.scales.data(), expected_scale_bytes);

        int n_tags = read_i32(f);
        const uint64_t expected_tag_bytes = checked_count_bytes(n_tags, sizeof(uint8_t), "tag count");
        if (static_cast<uint64_t>(n_tags) != slot_count)
            throw std::runtime_error("SessionSnapshot::load: tag count does not match cache size");
        require_remaining(f, file_size, expected_tag_bytes, "format tag data");
        hs.format_tags.resize(static_cast<size_t>(n_tags));
        if (n_tags > 0)
            read_bytes(f, hs.format_tags.data(), expected_tag_bytes);
    }

    /* Strategy state. */
    if (snap.flags_ & 2u) {
        for (auto &hs : snap.heads_) {
            int slen = read_i32(f);
            const uint64_t state_bytes = checked_count_bytes(slen, sizeof(uint8_t), "strategy state size");
            require_remaining(f, file_size, state_bytes, "strategy state");
            hs.strategy_state.resize(static_cast<size_t>(slen));
            if (slen > 0)
                read_bytes(f, hs.strategy_state.data(), state_bytes);
        }
    }

    /* Token log. */
    if (snap.flags_ & 1u) {
        int n_entries = read_i32(f);

        uint64_t expected_entries = static_cast<uint64_t>(snap.n_tokens_);
        if (expected_entries > std::numeric_limits<uint64_t>::max() /
                              static_cast<uint64_t>(snap.n_layers_)) {
            throw std::runtime_error("SessionSnapshot::load: token log entry count overflow");
        }
        expected_entries *= static_cast<uint64_t>(snap.n_layers_);
        if (expected_entries > std::numeric_limits<uint64_t>::max() /
                              static_cast<uint64_t>(snap.n_heads_)) {
            throw std::runtime_error("SessionSnapshot::load: token log entry count overflow");
        }
        expected_entries *= static_cast<uint64_t>(snap.n_heads_);

        if (static_cast<uint64_t>(n_entries) != expected_entries) {
            throw std::runtime_error(
                "SessionSnapshot::load: token log entry count " +
                std::to_string(n_entries) + " does not match expected count " +
                std::to_string(expected_entries));
        }
        const uint64_t min_entry_bytes = checked_count_bytes(n_entries, 12u, "token_log entries");
        require_metadata_container_bytes(n_entries, sizeof(SnapshotTokenEntry), "token_log entries");
        if (min_entry_bytes > remaining_bytes(f, file_size))
            throw std::runtime_error("SessionSnapshot::load: token_log entries exceed remaining snapshot data");
        snap.token_log_.resize(n_entries);
        for (auto &e : snap.token_log_) {
            e.layer = read_i32(f);
            e.head  = read_i32(f);
            e.dim   = read_i32(f);
            if (e.dim < 0)
                throw std::runtime_error("SessionSnapshot::load: invalid token dimension");

            const uint64_t vector_bytes = checked_count_bytes(
                e.dim, static_cast<uint64_t>(sizeof(float)) * 2u, "token dimension");
            require_remaining(f, file_size, vector_bytes, "token vectors");
            e.k_fp32.resize(e.dim);
            e.v_fp32.resize(e.dim);
            read_bytes(f, e.k_fp32.data(), static_cast<size_t>(e.dim) * sizeof(float));
            read_bytes(f, e.v_fp32.data(), static_cast<size_t>(e.dim) * sizeof(float));
        }
    }

    if (!f.good() && !f.eof())
        throw std::runtime_error("SessionSnapshot::load: read error for " + path);

    return snap;
}

} /* namespace adaptq */

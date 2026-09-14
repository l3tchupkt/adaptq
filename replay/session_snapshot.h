#pragma once
#include <cstdint>
#include <string>
#include <vector>

/* -------------------------------------------------------------------------
 * replay/session_snapshot.h — SessionSnapshot
 *
 * Versioned binary capture of a RuntimeContext at any token position.
 * Enables deterministic replay, branching, and cross-strategy comparison.
 *
 * Binary format (little-endian, self-describing, no external dependencies):
 *
 *   Header:
 *     [4B] magic    = 0x41515353 ("AQSS")
 *     [4B] version  = 2
 *     [4B] n_layers
 *     [4B] n_heads
 *     [4B] dim
 *     [4B] bits
 *     [4B] n_tokens     (token count at capture time)
 *     [4B] n_heads_total (n_layers × n_heads)
 *     [8B] flags        (bit 0 = has_token_log, bit 1 = has_strategy_state)
 *
 *   Per-head blocks (n_heads_total entries):
 *     [4B] layer
 *     [4B] head
 *     [4B] cache_size
 *     [4B] slot_bytes
 *     [N]  storage data   (cache_size * 2 * slot_bytes — K then V)
 *     [N]  scales         (cache_size * 2 * sizeof(float))
 *     [N]  format_tags    (cache_size * 2 * sizeof(uint8_t))
 *
 *   Optional: strategy state (if flags & 2):
 *     Per-head: [4B] state_len, [state_len B] state blob
 *
 *   Optional: token log (if flags & 1):
 *     [4B] n_log_entries
 *     Per-entry:
 *       [4B] layer
 *       [4B] head
 *       [4B] dim
 *       [dim*4B] k_fp32
 *       [dim*4B] v_fp32
 * ----------------------------------------------------------------------- */

namespace adaptq {

class RuntimeContext;  /* forward */

/* ---- Per-head storage snapshot ----------------------------------------- */
struct HeadSnapshot {
    int layer;
    int head;
    int cache_size;
    int slot_bytes;
    std::vector<uint8_t> storage_data;   /* cache_size * 2 * slot_bytes */
    std::vector<float>   scales;         /* cache_size * 2              */
    std::vector<uint8_t> format_tags;    /* cache_size * 2              */
    std::vector<uint8_t> strategy_state; /* IReplayHooks output; empty if none */
};

/* ---- Token log entry (for compressed replay) --------------------------- */
struct SnapshotTokenEntry {
    int               layer;
    int               head;
    int               dim;
    std::vector<float> k_fp32;
    std::vector<float> v_fp32;
};

/* ---- SessionSnapshot --------------------------------------------------- */
class SessionSnapshot {
public:
    /* Snapshot file magic and current version. */
    static constexpr uint32_t kMagic   = 0x41515353u;
    static constexpr uint32_t kVersion = 2u;

    /* ---- Construction -------------------------------------------------- */

    /**
     * Capture a snapshot of ctx at the current token position.
     *
     * @param ctx             The RuntimeContext to snapshot.
     * @param include_token_log  Include K/V FP32 token log for compressed replay.
     *                           Requires RuntimeContextConfig::log_tokens == true.
     */
    static SessionSnapshot capture(const RuntimeContext &ctx,
                                   bool include_token_log = true);

    /* Default-constructed snapshot is empty (for load() use). */
    SessionSnapshot() = default;

    /* ---- Persistence --------------------------------------------------- */

    /** Serialize to a binary file at path. Throws std::runtime_error on I/O error. */
    void save(const std::string &path) const;

    /** Deserialize from a binary file. Throws on I/O error or format mismatch. */
    static SessionSnapshot load(const std::string &path);

    /* ---- Accessors ----------------------------------------------------- */

    bool is_valid()       const { return magic_ == kMagic; }
    bool has_token_log()  const { return flags_ & 1u; }
    bool has_strategy_state() const { return flags_ & 2u; }

    int n_tokens()  const { return n_tokens_; }
    int n_layers()  const { return n_layers_; }
    int n_heads()   const { return n_heads_;  }
    int dim()       const { return dim_;      }
    int bits()      const { return bits_;     }

    const std::vector<HeadSnapshot>         &heads()     const { return heads_;     }
    const std::vector<SnapshotTokenEntry>   &token_log() const { return token_log_; }

private:
    uint32_t magic_    = 0;
    uint32_t version_  = 0;
    int      n_layers_ = 0;
    int      n_heads_  = 0;
    int      dim_      = 0;
    int      bits_     = 0;
    int      n_tokens_ = 0;
    uint64_t flags_    = 0;

    std::vector<HeadSnapshot>       heads_;
    std::vector<SnapshotTokenEntry> token_log_;
};

} /* namespace adaptq */

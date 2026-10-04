#pragma once
#include <cstdint>
#include <cstddef>
#include "context.h"

/* -------------------------------------------------------------------------
 * adaptq/storage.h — IStorageBackend interface
 *
 * Storage is infrastructure. Strategies do not own it.
 * The runtime owns storage backends; strategies write into them via the
 * IStorageBackend* in ExecutionContext.
 *
 * A CompressResult is an opaque handle to one compressed K or V vector.
 * The format_tag field identifies which decode path the attention kernel
 * should use — strategies define their own tag values.
 * ----------------------------------------------------------------------- */

namespace adaptq {

/* Opaque index into a storage backend. */
using StorageSlot = uint32_t;
static constexpr StorageSlot ADAPTQ_INVALID_SLOT = ~StorageSlot(0);

/* ---- CompressResult — returned by ICompression::compress() ------------- */
struct CompressResult {
    const uint8_t *data;       /* packed bytes, owned by the storage backend */
    int            data_bytes; /* length of data                             */
    float          scale;      /* reconstruction scale factor                */
    uint8_t        format_tag; /* strategy-defined format ID for kdot dispatch */
    StorageSlot    slot;       /* backend slot index; ADAPTQ_INVALID_SLOT if not stored */
};

/* ---- IStorageBackend --------------------------------------------------- */
struct IStorageBackend {
    virtual ~IStorageBackend() = default;

    /**
     * Allocate storage for up to `capacity` slots.
     * `max_slot_bytes` is the maximum packed bytes for any single slot.
     * Called once per head at context creation.
     */
    virtual void init(int capacity, int max_slot_bytes) = 0;

    /** Reset all slots. Keeps configuration. */
    virtual void reset() = 0;

    /**
     * Write packed data into the next available slot.
     * Returns the slot index. The caller stores this alongside the slot list.
     * Thread-safety: each backend instance is single-threaded (one head).
     */
    virtual StorageSlot write(const uint8_t *data,
                              int            data_bytes,
                              float          scale,
                              uint8_t        format_tag) = 0;

    /** Read a previously written slot. Returned pointer valid until next write(). */
    virtual CompressResult read(StorageSlot slot) const = 0;

    /** Release a slot (eviction). The slot may be reused by the next write(). */
    virtual void free_slot(StorageSlot slot) = 0;

    /** Bytes currently occupied by compressed data (excludes metadata overhead). */
    virtual size_t bytes_used()     const = 0;

    /** Maximum capacity in bytes. */
    virtual size_t bytes_capacity() const = 0;

    /** Human-readable name for diagnostics and benchmark output. */
    virtual const char *name() const = 0;
};

} /* namespace adaptq */

/* Catch2 v3 — link against Catch2::Catch2WithMain */
#include <catch2/catch_test_macros.hpp>
/* Storage backends: include .h (inline classes), not .cpp */
#include "../../storage/contiguous_slab.h"
#include "../../storage/segmented_slab.h"
#include <cstring>
#include <string>

/* -------------------------------------------------------------------------
 * tests/unit/test_storage.cpp
 *
 * Contract-tests both IStorageBackend implementations.
 * ----------------------------------------------------------------------- */

using namespace adaptq;

static std::vector<uint8_t> make_data(int n, uint8_t fill) {
    return std::vector<uint8_t>(n, fill);
}

/* ======================================================================
 * ContiguousSlabStorage
 * ==================================================================== */
TEST_CASE("ContiguousSlabStorage: write/read round-trip", "[storage][contiguous]") {
    ContiguousSlabStorage st;
    st.init(16, 8);
    for (uint8_t i = 0; i < 8; ++i) {
        auto d = make_data(8, i);
        StorageSlot slot = st.write(d.data(), 8, (float)i * 0.1f, 0x04);
        CompressResult r = st.read(slot);
        REQUIRE(r.data        != nullptr);
        REQUIRE(r.data_bytes  == 8);
        REQUIRE(std::abs(r.scale - (float)i * 0.1f) < 1e-6f);
        REQUIRE(r.data[0]     == i);
    }
}

TEST_CASE("ContiguousSlabStorage: ring eviction at capacity", "[storage][contiguous]") {
    ContiguousSlabStorage st;
    st.init(4, 4);
    for (uint8_t i = 0; i < 4; ++i) {
        auto d = make_data(4, i);
        st.write(d.data(), 4, 0.f, 0x04);
    }
    REQUIRE(st.size() == 4);
    auto extra = make_data(4, 99);
    StorageSlot s5 = st.write(extra.data(), 4, 1.f, 0x04);
    REQUIRE(st.size() == 4);
    CompressResult r = st.read(s5);
    REQUIRE(r.data[0] == 99);
}

TEST_CASE("ContiguousSlabStorage: capacity boundaries and validation", "[storage][contiguous]") {
    ContiguousSlabStorage st;
    REQUIRE_THROWS_AS(st.init(0, 8), std::invalid_argument);
    REQUIRE_THROWS_AS(st.init(1, 8), std::invalid_argument);
    REQUIRE_THROWS_AS(st.init(3, 8), std::invalid_argument);
    REQUIRE_THROWS_AS(st.init(4, 0), std::invalid_argument);

    st.init(4, 8);
    REQUIRE(st.capacity() == 4);
}

TEST_CASE("ContiguousSlabStorage: KV writes stay paired across wrap", "[storage][contiguous][kv]") {
    ContiguousSlabStorage st;
    st.init(4, 4); /* two logical K/V pairs */

    auto k0 = make_data(4, 10);
    auto v0 = make_data(4, 20);
    auto k1 = make_data(4, 11);
    auto v1 = make_data(4, 21);

    StorageSlot sk0 = st.write(k0.data(), 4, 1.f, 0x04);
    StorageSlot sv0 = st.write(v0.data(), 4, 2.f, 0x04);
    StorageSlot sk1 = st.write(k1.data(), 4, 3.f, 0x04);
    StorageSlot sv1 = st.write(v1.data(), 4, 4.f, 0x04);

    REQUIRE(sk0 == 0);
    REQUIRE(sv0 == 2);
    REQUIRE(sk1 == 1);
    REQUIRE(sv1 == 3);
    REQUIRE(st.read(0).data[0] == 10);
    REQUIRE(st.read(1).data[0] == 11);
    REQUIRE(st.read(2).data[0] == 20);
    REQUIRE(st.read(3).data[0] == 21);

    auto k2 = make_data(4, 12);
    auto v2 = make_data(4, 22);
    StorageSlot sk2 = st.write(k2.data(), 4, 5.f, 0x04);
    StorageSlot sv2 = st.write(v2.data(), 4, 6.f, 0x04);

    /* The oldest K/V pair is replaced together. */
    REQUIRE(sk2 == 0);
    REQUIRE(sv2 == 2);
    REQUIRE(st.read(0).data[0] == 12);
    REQUIRE(st.read(2).data[0] == 22);
    REQUIRE(st.read(1).data[0] == 11);
    REQUIRE(st.read(3).data[0] == 21);
}

TEST_CASE("ContiguousSlabStorage: reset zeroes size", "[storage][contiguous]") {
    ContiguousSlabStorage st;
    st.init(8, 8);
    for (int i = 0; i < 5; ++i) {
        auto d = make_data(8, (uint8_t)i);
        st.write(d.data(), 8, 0.f, 0x04);
    }
    REQUIRE(st.size() == 5);
    st.reset();
    REQUIRE(st.size() == 0);
    REQUIRE(st.bytes_used() == 0);
}

TEST_CASE("ContiguousSlabStorage: bytes_used tracks written data", "[storage][contiguous]") {
    ContiguousSlabStorage st;
    st.init(32, 16);
    REQUIRE(st.bytes_used() == 0);
    for (int i = 0; i < 4; ++i) {
        auto d = make_data(16, (uint8_t)i);
        st.write(d.data(), 16, 0.f, 0x04);
        REQUIRE(st.bytes_used() == (size_t)(i + 1) * 16);
    }
}

/* ======================================================================
 * SegmentedSlabStorage
 * ==================================================================== */
TEST_CASE("SegmentedSlabStorage: write/read round-trip single tag", "[storage][segmented]") {
    SegmentedSlabStorage st;
    st.init(16, 8);
    for (uint8_t i = 0; i < 8; ++i) {
        auto d = make_data(8, i);
        StorageSlot slot = st.write(d.data(), 8, (float)i, 0x04);
        CompressResult r = st.read(slot);
        REQUIRE(r.data[0]    == i);
        REQUIRE(std::abs(r.scale - (float)i) < 1e-6f);
        REQUIRE(r.format_tag == 0x04);
    }
}

TEST_CASE("SegmentedSlabStorage: multiple format_tags are independent", "[storage][segmented]") {
    SegmentedSlabStorage st;
    st.init(64, 64);
    auto d4 = make_data(8, 0xAA);
    auto d2 = make_data(4, 0xBB);
    StorageSlot s4  = st.write(d4.data(), 8, 1.f, 0x04);
    StorageSlot s2  = st.write(d2.data(), 4, 2.f, 0x02);
    StorageSlot s4b = st.write(d4.data(), 8, 3.f, 0x04);
    REQUIRE(s4 != s2);
    CompressResult r4  = st.read(s4);
    CompressResult r2  = st.read(s2);
    CompressResult r4b = st.read(s4b);
    REQUIRE(r4.format_tag  == 0x04);
    REQUIRE(r2.format_tag  == 0x02);
    REQUIRE(r4b.format_tag == 0x04);
    REQUIRE(r4.data[0]     == 0xAA);
    REQUIRE(r2.data[0]     == 0xBB);
    REQUIRE(std::abs(r4b.scale - 3.f) < 1e-6f);
}

TEST_CASE("SegmentedSlabStorage: reset clears all slabs", "[storage][segmented]") {
    SegmentedSlabStorage st;
    st.init(16, 16);
    auto d = make_data(8, 0xFF);
    st.write(d.data(), 8, 0.f, 0x04);
    st.write(d.data(), 8, 0.f, 0x02);
    REQUIRE(st.bytes_used() > 0);
    st.reset();
    REQUIRE(st.bytes_used() == 0);
}

TEST_CASE("SegmentedSlabStorage: free_slot tracks actual data bytes", "[storage][segmented]") {
    SegmentedSlabStorage st;
    st.init(4, 8);
    auto full = make_data(8, 0xAA);
    auto partial = make_data(4, 0xBB);
    st.write(full.data(), 8, 1.f, 0x04);
    StorageSlot slot = st.write(partial.data(), 4, 2.f, 0x04);
    REQUIRE(st.bytes_used() == 12);
    st.free_slot(slot);
    REQUIRE(st.bytes_used() == 8);
}

TEST_CASE("SegmentedSlabStorage: reinitialization safely frees previous slabs", "[storage][segmented]") {
    SegmentedSlabStorage st;
    st.init(8, 8);
    auto d1 = make_data(8, 0x11);
    auto d2 = make_data(8, 0x22);
    st.write(d1.data(), 8, 1.f, 0x01);
    st.write(d2.data(), 8, 2.f, 0x02);
    REQUIRE(st.bytes_used() == 16);

    // Re-initialize: should safely free earlier slabs without leaking or corrupting
    st.init(16, 16);
    REQUIRE(st.bytes_used() == 0);
    auto d3 = make_data(16, 0x33);
    StorageSlot slot = st.write(d3.data(), 16, 3.f, 0x03);
    CompressResult r = st.read(slot);
    REQUIRE(r.data[0] == 0x33);
    REQUIRE(r.format_tag == 0x03);
    REQUIRE(st.bytes_used() == 16);
}

TEST_CASE("SegmentedSlabStorage: capacity boundaries and out-of-bounds guards", "[storage][segmented]") {
    SegmentedSlabStorage st;
    // Reject non-positive capacity or oversized capacity > 65536
    REQUIRE_THROWS_AS(st.init(0, 8), std::invalid_argument);
    REQUIRE_THROWS_AS(st.init(-1, 8), std::invalid_argument);
    REQUIRE_THROWS_AS(st.init(65537, 8), std::invalid_argument);
    REQUIRE_THROWS_AS(st.init(8, 0), std::invalid_argument);

    // Valid init
    st.init(4, 8);
    auto d = make_data(8, 0x42);
    StorageSlot slot = st.write(d.data(), 8, 1.f, 0x01);

    // Safe read on out-of-bounds slot
    CompressResult r_bad = st.read((StorageSlot)999999);
    REQUIRE(r_bad.data == nullptr);

    // Safe free on invalid slot
    st.free_slot((StorageSlot)999999);
    REQUIRE(st.bytes_used() == 8);

    CompressResult r_good = st.read(slot);
    REQUIRE(r_good.data != nullptr);
    REQUIRE(r_good.data[0] == 0x42);
}

/* ======================================================================
 * IStorageBackend contract template
 * ==================================================================== */
template <typename T>
static void test_contract(int capacity, int slot_bytes) {
    T st;
    st.init(capacity, slot_bytes);

    std::vector<StorageSlot> slots;
    for (int i = 0; i < capacity; ++i) {
        auto d = make_data(slot_bytes, (uint8_t)i);
        slots.push_back(st.write(d.data(), slot_bytes, (float)i, 0x04));
    }
    REQUIRE((int)slots.size() == capacity);

    for (int i = 0; i < capacity; ++i) {
        CompressResult r = st.read(slots[i]);
        REQUIRE(r.data != nullptr);
        REQUIRE(r.data[0] == (uint8_t)i);
    }

    REQUIRE(st.name() != nullptr);
    REQUIRE(std::string(st.name()).size() > 0);
}

TEST_CASE("Contract: ContiguousSlabStorage", "[storage][contract]") {
    test_contract<ContiguousSlabStorage>(32, 8);
}
TEST_CASE("Contract: SegmentedSlabStorage", "[storage][contract]") {
    test_contract<SegmentedSlabStorage>(32, 8);
}

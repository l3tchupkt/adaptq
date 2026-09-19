#include <catch2/catch_test_macros.hpp>
#include "../../storage/numa_allocator.h"
#include <vector>
#include <cstring>
#include <cmath>

using namespace adaptq;
using namespace adaptq::storage;

static std::vector<uint8_t> generate_test_payload(size_t len, uint8_t seed) {
    std::vector<uint8_t> buf(len);
    for (size_t i = 0; i < len; ++i) {
        buf[i] = static_cast<uint8_t>((seed + i) & 0xFF);
    }
    return buf;
}

TEST_CASE("NumaPinnedSlabStorage: initialization and config validation", "[storage][numa]") {
    NumaAllocationConfig cfg;
    cfg.preferred_numa_node = 0;
    cfg.enable_page_pinning = true;

    NumaPinnedSlabStorage storage(cfg);

    SECTION("Valid initialization") {
        REQUIRE_NOTHROW(storage.init(16, 64));
        REQUIRE(storage.capacity() == 16);
        REQUIRE(storage.count() == 0);
        REQUIRE(storage.stats().allocated_bytes > 0);
        REQUIRE(storage.stats().active_numa_node == 0);
    }

    SECTION("Invalid capacity - odd number") {
        REQUIRE_THROWS_AS(storage.init(15, 64), std::invalid_argument);
    }

    SECTION("Invalid capacity - less than 2") {
        REQUIRE_THROWS_AS(storage.init(0, 64), std::invalid_argument);
    }

    SECTION("Invalid max_slot_bytes") {
        REQUIRE_THROWS_AS(storage.init(16, 0), std::invalid_argument);
        REQUIRE_THROWS_AS(storage.init(16, -10), std::invalid_argument);
    }
}

TEST_CASE("NumaPinnedSlabStorage: key-value write and read roundtrip", "[storage][numa]") {
    NumaAllocationConfig cfg;
    cfg.preferred_numa_node = 0;
    cfg.enable_page_pinning = false; // test unpinned code path as well

    NumaPinnedSlabStorage storage(cfg);
    storage.init(8, 32);

    auto key_data = generate_test_payload(32, 0x10);
    auto val_data = generate_test_payload(32, 0x20);

    // Write Key
    StorageSlot slot_k = storage.write(key_data.data(), 32, 1.5f, 0x01);
    REQUIRE(slot_k != ADAPTQ_INVALID_SLOT);
    REQUIRE(slot_k < 4); // Key partition in [0, 4)

    // Write Value
    StorageSlot slot_v = storage.write(val_data.data(), 32, 2.5f, 0x02);
    REQUIRE(slot_v != ADAPTQ_INVALID_SLOT);
    REQUIRE(slot_v >= 4); // Value partition in [4, 8)
    REQUIRE(slot_v < 8);

    REQUIRE(storage.count() == 2);

    // Read Key
    CompressResult res_k = storage.read(slot_k);
    REQUIRE(res_k.data != nullptr);
    REQUIRE(res_k.data_bytes == 32);
    REQUIRE(std::abs(res_k.scale - 1.5f) < 1e-6f);
    REQUIRE(res_k.format_tag == 0x01);
    REQUIRE(std::memcmp(res_k.data, key_data.data(), 32) == 0);

    // Read Value
    CompressResult res_v = storage.read(slot_v);
    REQUIRE(res_v.data != nullptr);
    REQUIRE(res_v.data_bytes == 32);
    REQUIRE(std::abs(res_v.scale - 2.5f) < 1e-6f);
    REQUIRE(res_v.format_tag == 0x02);
    REQUIRE(std::memcmp(res_v.data, val_data.data(), 32) == 0);

    const auto& stats = storage.stats();
    REQUIRE(stats.write_count == 2);
    REQUIRE(stats.read_count == 2);
}

TEST_CASE("NumaPinnedSlabStorage: size bounds and invalid writes", "[storage][numa]") {
    NumaPinnedSlabStorage storage;
    storage.init(4, 16);

    auto oversized = generate_test_payload(32, 0xAA);
    REQUIRE_THROWS_AS(storage.write(oversized.data(), 32, 1.0f, 0x00), std::invalid_argument);

    // Out of bound slot read
    CompressResult invalid_res = storage.read(999);
    REQUIRE(invalid_res.data == nullptr);
    REQUIRE(invalid_res.slot == ADAPTQ_INVALID_SLOT);
}

TEST_CASE("NumaPinnedSlabStorage: ring buffer eviction and reset", "[storage][numa]") {
    NumaPinnedSlabStorage storage;
    storage.init(4, 16); // 2 keys, 2 values

    for (int i = 0; i < 6; ++i) {
        auto p = generate_test_payload(16, static_cast<uint8_t>(i));
        storage.write(p.data(), 16, static_cast<float>(i), 0x01);
    }

    REQUIRE(storage.count() == 4);

    // Release slot
    storage.release(0);
    REQUIRE(storage.stats().release_count == 1);

    // Reset storage
    storage.reset();
    REQUIRE(storage.count() == 0);
}

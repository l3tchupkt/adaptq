#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstring>
#include <vector>

#include "../../storage/mmap_storage.h"

using namespace adaptq;

TEST_CASE("MMapStorage: initialization and bounds", "[storage][mmap]") {
    MMapStorage storage;
    REQUIRE(storage.capacity() == 0);
    REQUIRE(storage.bytes_used() == 0);
    REQUIRE(std::string(storage.name()) == "mmap");

    REQUIRE_THROWS_AS(storage.init(0, 64), std::invalid_argument);
    REQUIRE_THROWS_AS(storage.init(-10, 64), std::invalid_argument);
    REQUIRE_THROWS_AS(storage.init(64, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(storage.init(64, -5), std::invalid_argument);

    storage.init(32, 64);
    REQUIRE(storage.capacity() == 32);
    REQUIRE(storage.bytes_used() == 0);
}

TEST_CASE("MMapStorage: write and read payload verification", "[storage][mmap]") {
    MMapStorage storage;
    const int CAP = 16;
    const int SLOT_BYTES = 32;
    storage.init(CAP, SLOT_BYTES);

    std::vector<uint8_t> payload1(SLOT_BYTES, 0xAB);
    std::vector<uint8_t> payload2(SLOT_BYTES, 0xCD);

    StorageSlot s0 = storage.write(payload1.data(), SLOT_BYTES, 1.25f, 0x01);
    REQUIRE(s0 == 0);

    StorageSlot s1 = storage.write(payload2.data(), SLOT_BYTES, 2.50f, 0x02);
    REQUIRE(s1 == 1);

    CompressResult r0 = storage.read(s0);
    REQUIRE(r0.slot == s0);
    REQUIRE(r0.data_bytes == SLOT_BYTES);
    REQUIRE(r0.format_tag == 0x01);
    REQUIRE_THAT(r0.scale, Catch::Matchers::WithinRel(1.25f, 1e-4f));
    REQUIRE(r0.data != nullptr);
    REQUIRE(std::memcmp(r0.data, payload1.data(), SLOT_BYTES) == 0);

    CompressResult r1 = storage.read(s1);
    REQUIRE(r1.slot == s1);
    REQUIRE(r1.data_bytes == SLOT_BYTES);
    REQUIRE(r1.format_tag == 0x02);
    REQUIRE_THAT(r1.scale, Catch::Matchers::WithinRel(2.50f, 1e-4f));
    REQUIRE(std::memcmp(r1.data, payload2.data(), SLOT_BYTES) == 0);

    // Read invalid slot
    CompressResult r_invalid = storage.read(999);
    REQUIRE(r_invalid.slot == ADAPTQ_INVALID_SLOT);
    REQUIRE(r_invalid.data == nullptr);
}

TEST_CASE("MMapStorage: ring buffer eviction upon reaching capacity", "[storage][mmap]") {
    MMapStorage storage;
    const int CAP = 4;
    const int SLOT_BYTES = 16;
    storage.init(CAP, SLOT_BYTES);

    for (int i = 0; i < CAP; ++i) {
        std::vector<uint8_t> p(SLOT_BYTES, (uint8_t)(i + 1));
        StorageSlot s = storage.write(p.data(), SLOT_BYTES, (float)(i + 1), 0);
        REQUIRE(s == (uint32_t)i);
    }
    REQUIRE(storage.bytes_used() == (size_t)CAP * SLOT_BYTES);

    // Writing 5th item should evict slot 0 in FIFO order
    std::vector<uint8_t> p_evict(SLOT_BYTES, 0xFF);
    StorageSlot s_new = storage.write(p_evict.data(), SLOT_BYTES, 99.0f, 0xAA);
    REQUIRE(s_new == 0);

    CompressResult res = storage.read(0);
    REQUIRE(res.format_tag == 0xAA);
    REQUIRE_THAT(res.scale, Catch::Matchers::WithinRel(99.0f, 1e-4f));
    REQUIRE(std::memcmp(res.data, p_evict.data(), SLOT_BYTES) == 0);
}

TEST_CASE("MMapStorage: free slot and reset behavior", "[storage][mmap]") {
    MMapStorage storage;
    storage.init(8, 16);

    std::vector<uint8_t> p(16, 0x42);
    StorageSlot s = storage.write(p.data(), 16, 1.0f, 0);
    REQUIRE(storage.read(s).data != nullptr);

    storage.free_slot(s);
    REQUIRE(storage.read(s).data == nullptr);

    // Freeing non-existent slot should be safe
    storage.free_slot(999);

    // Reset clears everything
    storage.write(p.data(), 16, 1.0f, 0);
    storage.reset();
    REQUIRE(storage.read(0).data == nullptr);
    REQUIRE(storage.bytes_used() == 0);

    // Flush
    REQUIRE(storage.flush() == true);
}

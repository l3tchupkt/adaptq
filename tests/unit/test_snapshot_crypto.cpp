#include <catch2/catch_test_macros.hpp>
#include <array>
#include <string>
#include <vector>

#include "../../replay/snapshot_crypto.h"

using namespace adaptq;
using namespace adaptq::crypto;

TEST_CASE("Crypto: SHA256 standard vectors", "[crypto][sha256]") {
    SHA256 hasher;
    // Empty string SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    auto hash = hasher.finalize();
    std::string expected_hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    std::ostringstream ss;
    for (uint8_t b : hash) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    REQUIRE(ss.str() == expected_hex);

    // "abc" vector: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    hasher.reset();
    std::string abc = "abc";
    hasher.update(reinterpret_cast<const uint8_t*>(abc.data()), abc.size());
    auto hash_abc = hasher.finalize();

    std::ostringstream ss_abc;
    for (uint8_t b : hash_abc) {
        ss_abc << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    REQUIRE(ss_abc.str() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST_CASE("Crypto: HMAC-SHA256 calculation and verification", "[crypto][hmac]") {
    std::string key = "super_secret_adaptq_key";
    std::string msg = "KV_CACHE_SESSION_DATA_FOR_LLAMA3";

    auto mac1 = hmac_sha256(reinterpret_cast<const uint8_t*>(key.data()), key.size(),
                            reinterpret_cast<const uint8_t*>(msg.data()), msg.size());

    auto mac2 = hmac_sha256(reinterpret_cast<const uint8_t*>(key.data()), key.size(),
                            reinterpret_cast<const uint8_t*>(msg.data()), msg.size());

    REQUIRE(constant_time_compare(mac1.data(), mac2.data(), 32) == true);

    // Altered message should fail
    std::string altered = "KV_CACHE_SESSION_DATA_FOR_LLAMA4";
    auto mac_altered = hmac_sha256(reinterpret_cast<const uint8_t*>(key.data()), key.size(),
                                   reinterpret_cast<const uint8_t*>(altered.data()), altered.size());
    REQUIRE(constant_time_compare(mac1.data(), mac_altered.data(), 32) == false);
}

TEST_CASE("AuthenticatedSnapshotEnvelope: pack and unpack roundtrip", "[replay][crypto]") {
    std::string key = "production_inference_auth_token_99";
    std::vector<uint8_t> raw_snapshot = {
        0x41, 0x51, 0x53, 0x53, // AQSS
        0x02, 0x00, 0x00, 0x00, // v2
        0x01, 0x00, 0x00, 0x00, // 1 layer
        0x02, 0x00, 0x00, 0x00, // 2 heads
        0x40, 0x00, 0x00, 0x00  // dim 64
    };

    std::vector<uint8_t> envelope = AuthenticatedSnapshotEnvelope::create(
        raw_snapshot, key, "key_v1");

    REQUIRE(!envelope.empty());
    REQUIRE(envelope.size() > raw_snapshot.size() + 32);

    std::vector<uint8_t> unpacked = AuthenticatedSnapshotEnvelope::verify_and_unpack(
        envelope, key);

    REQUIRE(unpacked.size() == raw_snapshot.size());
    REQUIRE(unpacked == raw_snapshot);
}

TEST_CASE("AuthenticatedSnapshotEnvelope: rejection of tampered data and bad keys", "[replay][crypto]") {
    std::string correct_key = "correct_secret_key";
    std::string wrong_key = "wrong_attacker_key";

    std::vector<uint8_t> raw_payload(128, 0xEE);
    std::vector<uint8_t> envelope = AuthenticatedSnapshotEnvelope::create(
        raw_payload, correct_key, "llama_cache");

    // Unpack with wrong key must throw
    REQUIRE_THROWS_AS(AuthenticatedSnapshotEnvelope::verify_and_unpack(envelope, wrong_key),
                      std::runtime_error);

    // Tamper with single byte in payload
    std::vector<uint8_t> tampered_envelope = envelope;
    tampered_envelope[40] ^= 0xFF;
    REQUIRE_THROWS_AS(AuthenticatedSnapshotEnvelope::verify_and_unpack(tampered_envelope, correct_key),
                      std::runtime_error);

    // Truncated envelope must throw
    std::vector<uint8_t> truncated_envelope(envelope.begin(), envelope.begin() + 20);
    REQUIRE_THROWS_AS(AuthenticatedSnapshotEnvelope::verify_and_unpack(truncated_envelope, correct_key),
                      std::runtime_error);

    // Empty arguments
    REQUIRE_THROWS_AS(AuthenticatedSnapshotEnvelope::create({}, correct_key), std::invalid_argument);
    REQUIRE_THROWS_AS(AuthenticatedSnapshotEnvelope::create(raw_payload, ""), std::invalid_argument);
}

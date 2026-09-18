#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace adaptq {

namespace crypto {

/* ---- Portable SHA-256 Implementation ----------------------------------- */
class SHA256 {
public:
    SHA256() { reset(); }

    void reset() {
        state_[0] = 0x6a09e667;
        state_[1] = 0xbb67ae85;
        state_[2] = 0x3c6ef372;
        state_[3] = 0xa54ff53a;
        state_[4] = 0x510e527f;
        state_[5] = 0x9b05688c;
        state_[6] = 0x1f83d9ab;
        state_[7] = 0x5be0cd19;
        bitlen_ = 0;
        datalen_ = 0;
    }

    void update(const uint8_t *data, size_t length) {
        for (size_t i = 0; i < length; ++i) {
            data_[datalen_++] = data[i];
            if (datalen_ == 64) {
                transform(data_);
                bitlen_ += 512;
                datalen_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> finalize() {
        size_t i = datalen_;
        if (datalen_ < 56) {
            data_[i++] = 0x80;
            while (i < 56) data_[i++] = 0x00;
        } else {
            data_[i++] = 0x80;
            while (i < 64) data_[i++] = 0x00;
            transform(data_);
            std::memset(data_, 0, 56);
        }

        bitlen_ += datalen_ * 8;
        data_[63] = static_cast<uint8_t>(bitlen_);
        data_[62] = static_cast<uint8_t>(bitlen_ >> 8);
        data_[61] = static_cast<uint8_t>(bitlen_ >> 16);
        data_[60] = static_cast<uint8_t>(bitlen_ >> 24);
        data_[59] = static_cast<uint8_t>(bitlen_ >> 32);
        data_[58] = static_cast<uint8_t>(bitlen_ >> 40);
        data_[57] = static_cast<uint8_t>(bitlen_ >> 48);
        data_[56] = static_cast<uint8_t>(bitlen_ >> 56);
        transform(data_);

        std::array<uint8_t, 32> hash{};
        for (i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 8; ++j) {
                hash[j * 4 + i] = static_cast<uint8_t>((state_[j] >> (24 - i * 8)) & 0x000000ff);
            }
        }
        return hash;
    }

private:
    static uint32_t rotr(uint32_t a, uint32_t b) { return (a >> b) | (a << (32 - b)); }
    static uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
    static uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
    static uint32_t ep0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
    static uint32_t ep1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
    static uint32_t sig0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
    static uint32_t sig1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

    void transform(const uint8_t *chunk) {
        uint32_t m[64];
        for (size_t i = 0, j = 0; i < 16; ++i, j += 4) {
            m[i] = (static_cast<uint32_t>(chunk[j]) << 24) |
                   (static_cast<uint32_t>(chunk[j + 1]) << 16) |
                   (static_cast<uint32_t>(chunk[j + 2]) << 8) |
                   (static_cast<uint32_t>(chunk[j + 3]));
        }
        for (size_t i = 16; i < 64; ++i) {
            m[i] = sig1(m[i - 2]) + m[i - 7] + sig0(m[i - 15]) + m[i - 16];
        }

        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];

        static const uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
        };

        for (size_t i = 0; i < 64; ++i) {
            uint32_t t1 = h + ep1(e) + ch(e, f, g) + k[i] + m[i];
            uint32_t t2 = ep0(a) + maj(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    uint32_t state_[8];
    uint64_t bitlen_;
    uint8_t data_[64];
    size_t datalen_;
};

/* ---- HMAC-SHA256 Implementation --------------------------------------- */
inline std::array<uint8_t, 32> hmac_sha256(const uint8_t *key, size_t key_len,
                                           const uint8_t *msg, size_t msg_len) {
    uint8_t k[64] = {0};
    if (key_len > 64) {
        SHA256 h;
        h.update(key, key_len);
        auto key_hash = h.finalize();
        std::memcpy(k, key_hash.data(), 32);
    } else {
        std::memcpy(k, key, key_len);
    }

    uint8_t o_key_pad[64];
    uint8_t i_key_pad[64];
    for (int i = 0; i < 64; ++i) {
        o_key_pad[i] = k[i] ^ 0x5c;
        i_key_pad[i] = k[i] ^ 0x36;
    }

    SHA256 inner;
    inner.update(i_key_pad, 64);
    inner.update(msg, msg_len);
    auto inner_hash = inner.finalize();

    SHA256 outer;
    outer.update(o_key_pad, 64);
    outer.update(inner_hash.data(), 32);
    return outer.finalize();
}

/* Constant-time comparison */
inline bool constant_time_compare(const uint8_t *a, const uint8_t *b, size_t len) {
    uint8_t diff = 0;
    for (size_t i = 0; i < len; ++i) {
        diff |= (a[i] ^ b[i]);
    }
    return diff == 0;
}

} // namespace crypto

class AuthenticatedSnapshotEnvelope {
public:
    static constexpr uint32_t MAGIC_ENCRYPTED = 0x41515345; // "AQSE"
    static constexpr uint32_t VERSION = 1;

    static std::vector<uint8_t> create(const std::vector<uint8_t>& raw_payload,
                                       const std::string& secret_key,
                                       const std::string& key_id = "default") {
        if (raw_payload.empty()) {
            throw std::invalid_argument("create: raw_payload cannot be empty");
        }
        if (secret_key.empty()) {
            throw std::invalid_argument("create: secret_key cannot be empty");
        }

        uint64_t timestamp = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());

        std::vector<uint8_t> envelope;
        envelope.reserve(64 + raw_payload.size() + 32);

        // Header: [4B] magic, [4B] version, [8B] timestamp, [4B] key_id_len, [key_id], [8B] payload_len
        append_u32(envelope, MAGIC_ENCRYPTED);
        append_u32(envelope, VERSION);
        append_u64(envelope, timestamp);

        uint32_t k_len = static_cast<uint32_t>(key_id.size());
        append_u32(envelope, k_len);
        envelope.insert(envelope.end(), key_id.begin(), key_id.end());

        uint64_t p_len = static_cast<uint64_t>(raw_payload.size());
        append_u64(envelope, p_len);

        // Payload
        envelope.insert(envelope.end(), raw_payload.begin(), raw_payload.end());

        // Compute HMAC over entire envelope constructed so far
        auto mac = crypto::hmac_sha256(
            reinterpret_cast<const uint8_t*>(secret_key.data()), secret_key.size(),
            envelope.data(), envelope.size());

        // Append 32B HMAC
        envelope.insert(envelope.end(), mac.begin(), mac.end());
        return envelope;
    }

    static std::vector<uint8_t> verify_and_unpack(const std::vector<uint8_t>& envelope,
                                                  const std::string& secret_key) {
        if (envelope.size() < 4 + 4 + 8 + 4 + 8 + 32) {
            throw std::runtime_error("verify_and_unpack: envelope truncated");
        }

        // Verify MAC
        size_t mac_offset = envelope.size() - 32;
        auto expected_mac = crypto::hmac_sha256(
            reinterpret_cast<const uint8_t*>(secret_key.data()), secret_key.size(),
            envelope.data(), mac_offset);

        if (!crypto::constant_time_compare(envelope.data() + mac_offset, expected_mac.data(), 32)) {
            throw std::runtime_error("verify_and_unpack: HMAC verification failed (corrupted or forged snapshot)");
        }

        // Parse header
        size_t offset = 0;
        uint32_t magic = read_u32(envelope, offset);
        if (magic != MAGIC_ENCRYPTED) {
            throw std::runtime_error("verify_and_unpack: invalid encrypted magic header");
        }

        uint32_t version = read_u32(envelope, offset);
        if (version != VERSION) {
            throw std::runtime_error("verify_and_unpack: unsupported envelope version");
        }

        offset += 8; // skip timestamp
        uint32_t k_len = read_u32(envelope, offset);
        offset += k_len; // skip key_id

        uint64_t p_len = read_u64(envelope, offset);
        if (offset + p_len != mac_offset) {
            throw std::runtime_error("verify_and_unpack: payload length mismatch");
        }

        std::vector<uint8_t> payload(envelope.begin() + offset, envelope.begin() + offset + p_len);
        return payload;
    }

private:
    static void append_u32(std::vector<uint8_t>& buf, uint32_t val) {
        buf.push_back(static_cast<uint8_t>(val & 0xFF));
        buf.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
        buf.push_back(static_cast<uint8_t>((val >> 16) & 0xFF));
        buf.push_back(static_cast<uint8_t>((val >> 24) & 0xFF));
    }

    static void append_u64(std::vector<uint8_t>& buf, uint64_t val) {
        for (int i = 0; i < 8; ++i) buf.push_back(static_cast<uint8_t>((val >> (i * 8)) & 0xFF));
    }

    static uint32_t read_u32(const std::vector<uint8_t>& buf, size_t& offset) {
        uint32_t val = static_cast<uint32_t>(buf[offset]) |
                      (static_cast<uint32_t>(buf[offset + 1]) << 8) |
                      (static_cast<uint32_t>(buf[offset + 2]) << 16) |
                      (static_cast<uint32_t>(buf[offset + 3]) << 24);
        offset += 4;
        return val;
    }

    static uint64_t read_u64(const std::vector<uint8_t>& buf, size_t& offset) {
        uint64_t val = 0;
        for (int i = 0; i < 8; ++i) {
            val |= (static_cast<uint64_t>(buf[offset + i]) << (i * 8));
        }
        offset += 8;
        return val;
    }
};

} // namespace adaptq

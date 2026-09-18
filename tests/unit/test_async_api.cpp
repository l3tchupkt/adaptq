#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "../../include/adaptq_async.h"

struct TestUserData {
    std::atomic<int> calls{0};
    std::atomic<adaptq_error_t> last_status{ADAPTQ_ERR_ALLOC};
};

static void test_callback(adaptq_error_t status, void *user_data) {
    auto *ud = static_cast<TestUserData *>(user_data);
    if (ud) {
        ud->last_status = status;
        ud->calls++;
    }
}

TEST_CASE("AsyncQueue: lifecycle and validation", "[c-api][async]") {
    // Invalid worker count
    adaptq_async_queue_t bad_q = adaptq_async_queue_create(0, 10);
    REQUIRE(bad_q == nullptr);

    adaptq_async_queue_t bad_q2 = adaptq_async_queue_create(-2, 10);
    REQUIRE(bad_q2 == nullptr);

    // Valid queue
    adaptq_async_queue_t q = adaptq_async_queue_create(2, 64);
    REQUIRE(q != nullptr);
    REQUIRE(adaptq_async_queue_pending_count(q) == 0);

    adaptq_async_queue_wait_all(q);
    adaptq_async_queue_destroy(q);
}

TEST_CASE("AsyncQueue: argument boundary validation", "[c-api][async]") {
    adaptq_async_queue_t q = adaptq_async_queue_create(1, 10);
    REQUIRE(q != nullptr);

    float vec[16] = {0.0f};
    TestUserData ud;

    // NULL queue
    REQUIRE(adaptq_async_append_submit(nullptr, (adaptq_ctx_t)0x1, vec, vec, 0, 16, test_callback, &ud) == ADAPTQ_ERR_INVALID_ARG);
    // NULL ctx
    REQUIRE(adaptq_async_append_submit(q, nullptr, vec, vec, 0, 16, test_callback, &ud) == ADAPTQ_ERR_INVALID_ARG);
    // NULL key/val
    REQUIRE(adaptq_async_append_submit(q, (adaptq_ctx_t)0x1, nullptr, vec, 0, 16, test_callback, &ud) == ADAPTQ_ERR_INVALID_ARG);
    REQUIRE(adaptq_async_append_submit(q, (adaptq_ctx_t)0x1, vec, nullptr, 0, 16, test_callback, &ud) == ADAPTQ_ERR_INVALID_ARG);
    // Invalid dim
    REQUIRE(adaptq_async_append_submit(q, (adaptq_ctx_t)0x1, vec, vec, 0, 0, test_callback, &ud) == ADAPTQ_ERR_INVALID_ARG);

    // Compute NULL checks
    REQUIRE(adaptq_async_compute_submit(nullptr, (adaptq_ctx_t)0x1, vec, vec, 16, test_callback, &ud) == ADAPTQ_ERR_INVALID_ARG);
    REQUIRE(adaptq_async_compute_submit(q, nullptr, vec, vec, 16, test_callback, &ud) == ADAPTQ_ERR_INVALID_ARG);
    REQUIRE(adaptq_async_compute_submit(q, (adaptq_ctx_t)0x1, nullptr, vec, 16, test_callback, &ud) == ADAPTQ_ERR_INVALID_ARG);
    REQUIRE(adaptq_async_compute_submit(q, (adaptq_ctx_t)0x1, vec, nullptr, 16, test_callback, &ud) == ADAPTQ_ERR_INVALID_ARG);

    adaptq_async_queue_destroy(q);
}

TEST_CASE("AsyncQueue: end-to-end task execution with mock context", "[c-api][async]") {
    adaptq_async_queue_t q = adaptq_async_queue_create(4, 128);
    REQUIRE(q != nullptr);

    // Create real context
    adaptq_ctx_t ctx = adaptq_create(64, 4, 128, 42, 0.0f, 0);
    REQUIRE(ctx != nullptr);

    const int N_ITEMS = 20;
    std::vector<TestUserData> user_data_items(N_ITEMS);
    std::vector<float> sample(64, 0.1f);

    for (int i = 0; i < N_ITEMS; ++i) {
        adaptq_error_t err = adaptq_async_append_submit(
            q, ctx, sample.data(), sample.data(), i, 64,
            test_callback, &user_data_items[i]
        );
        REQUIRE(err == ADAPTQ_OK);
    }

    // Wait for all appends to complete
    adaptq_async_queue_wait_all(q);

    for (int i = 0; i < N_ITEMS; ++i) {
        REQUIRE(user_data_items[i].calls == 1);
        REQUIRE(user_data_items[i].last_status == ADAPTQ_OK);
    }

    // Submit async compute
    TestUserData compute_ud;
    std::vector<float> out(64, 0.0f);
    adaptq_error_t err = adaptq_async_compute_submit(
        q, ctx, sample.data(), out.data(), 64,
        test_callback, &compute_ud
    );
    REQUIRE(err == ADAPTQ_OK);

    adaptq_async_queue_wait_all(q);
    REQUIRE(compute_ud.calls == 1);
    REQUIRE(compute_ud.last_status == ADAPTQ_OK);

    adaptq_destroy(ctx);
    adaptq_async_queue_destroy(q);
}

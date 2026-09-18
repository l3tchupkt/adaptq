#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include "adaptq.h"

/* -----------------------------------------------------------------------
 * AdapTQ Async C API — Asynchronous Multi-Threaded Batch Inference Queue
 * ----------------------------------------------------------------------- */

typedef void *adaptq_async_queue_t;

typedef void (*adaptq_async_callback_t)(adaptq_error_t status, void *user_data);

/**
 * Create an asynchronous worker queue with a dedicated worker thread pool.
 * @param num_workers Number of worker threads (must be >= 1)
 * @param queue_capacity Maximum backlog before enqueue blocks (0 for unbounded)
 * @return Handle to the async queue, or NULL on error.
 */
adaptq_async_queue_t adaptq_async_queue_create(int num_workers, int queue_capacity);

/**
 * Destroy the async queue, draining pending tasks and terminating worker threads.
 */
void adaptq_async_queue_destroy(adaptq_async_queue_t queue);

/**
 * Submit an asynchronous append task.
 */
adaptq_error_t adaptq_async_append_submit(
    adaptq_async_queue_t queue,
    adaptq_ctx_t ctx,
    const float *key,
    const float *val,
    int token_pos,
    int dim,
    adaptq_async_callback_t cb,
    void *user_data
);

/**
 * Submit an asynchronous compute task.
 */
adaptq_error_t adaptq_async_compute_submit(
    adaptq_async_queue_t queue,
    adaptq_ctx_t ctx,
    const float *query,
    float *out,
    int dim,
    adaptq_async_callback_t cb,
    void *user_data
);

/**
 * Synchronously wait for all pending and in-flight tasks in the queue to complete.
 */
void adaptq_async_queue_wait_all(adaptq_async_queue_t queue);

/**
 * Return the number of pending tasks currently queued.
 */
size_t adaptq_async_queue_pending_count(adaptq_async_queue_t queue);

#ifdef __cplusplus
}
#endif

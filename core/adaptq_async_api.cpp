#include "../include/adaptq_async.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace {

struct AsyncTask {
    std::function<void()> work;
};

class AsyncWorkerQueue {
public:
    AsyncWorkerQueue(int num_workers, int max_capacity)
        : max_capacity_(max_capacity), stop_(false), active_tasks_(0) {
        if (num_workers < 1) num_workers = 1;
        workers_.reserve(num_workers);
        for (int i = 0; i < num_workers; ++i) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    ~AsyncWorkerQueue() {
        shutdown();
    }

    bool enqueue(std::function<void()> task_fn) {
        std::unique_lock<std::mutex> lock(mu_);
        if (stop_) return false;

        if (max_capacity_ > 0) {
            cv_capacity_.wait(lock, [this]() {
                return stop_ || tasks_.size() < static_cast<size_t>(max_capacity_);
            });
            if (stop_) return false;
        }

        tasks_.push(AsyncTask{std::move(task_fn)});
        active_tasks_++;
        cv_task_.notify_one();
        return true;
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_done_.wait(lock, [this]() {
            return tasks_.empty() && active_tasks_ == 0;
        });
    }

    size_t pending_count() {
        std::lock_guard<std::mutex> lock(mu_);
        return tasks_.size();
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(mu_);
            if (stop_) return;
            stop_ = true;
            cv_task_.notify_all();
            cv_capacity_.notify_all();
        }
        for (auto &w : workers_) {
            if (w.joinable()) {
                w.join();
            }
        }
    }

private:
    void worker_loop() {
        while (true) {
            AsyncTask task;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_task_.wait(lock, [this]() {
                    return stop_ || !tasks_.empty();
                });

                if (stop_ && tasks_.empty()) {
                    return;
                }

                task = std::move(tasks_.front());
                tasks_.pop();
                cv_capacity_.notify_one();
            }

            // Execute work
            if (task.work) {
                task.work();
            }

            {
                std::lock_guard<std::mutex> lock(mu_);
                active_tasks_--;
                if (tasks_.empty() && active_tasks_ == 0) {
                    cv_done_.notify_all();
                }
            }
        }
    }

    int max_capacity_;
    std::vector<std::thread> workers_;
    std::queue<AsyncTask> tasks_;
    std::mutex mu_;
    std::condition_variable cv_task_;
    std::condition_variable cv_capacity_;
    std::condition_variable cv_done_;
    bool stop_;
    std::atomic<size_t> active_tasks_;
};

} // anonymous namespace

extern "C" {

adaptq_async_queue_t adaptq_async_queue_create(int num_workers, int queue_capacity) {
    if (num_workers <= 0) return nullptr;
    try {
        auto *q = new AsyncWorkerQueue(num_workers, queue_capacity);
        return static_cast<adaptq_async_queue_t>(q);
    } catch (...) {
        return nullptr;
    }
}

void adaptq_async_queue_destroy(adaptq_async_queue_t queue) {
    if (!queue) return;
    auto *q = static_cast<AsyncWorkerQueue *>(queue);
    delete q;
}

adaptq_error_t adaptq_async_append_submit(
    adaptq_async_queue_t queue,
    adaptq_ctx_t ctx,
    const float *key,
    const float *val,
    int token_pos,
    int dim,
    adaptq_async_callback_t cb,
    void *user_data
) {
    if (!queue || !ctx || !key || !val || dim <= 0) {
        return ADAPTQ_ERR_INVALID_ARG;
    }

    auto *q = static_cast<AsyncWorkerQueue *>(queue);
    std::vector<float> k_copy(key, key + dim);
    std::vector<float> v_copy(val, val + dim);

    bool ok = q->enqueue([ctx, k = std::move(k_copy), v = std::move(v_copy), token_pos, cb, user_data]() {
        adaptq_error_t err = ADAPTQ_OK;
        try {
            adaptq_append(ctx, k.data(), v.data(), token_pos);
        } catch (...) {
            err = ADAPTQ_ERR_ALLOC;
        }
        if (cb) {
            cb(err, user_data);
        }
    });

    return ok ? ADAPTQ_OK : ADAPTQ_ERR_ALLOC;
}

adaptq_error_t adaptq_async_compute_submit(
    adaptq_async_queue_t queue,
    adaptq_ctx_t ctx,
    const float *query,
    float *out,
    int dim,
    adaptq_async_callback_t cb,
    void *user_data
) {
    if (!queue || !ctx || !query || !out || dim <= 0) {
        return ADAPTQ_ERR_INVALID_ARG;
    }

    auto *q = static_cast<AsyncWorkerQueue *>(queue);
    std::vector<float> q_copy(query, query + dim);

    bool ok = q->enqueue([ctx, q_in = std::move(q_copy), out, cb, user_data]() {
        adaptq_error_t err = ADAPTQ_OK;
        try {
            adaptq_compute(ctx, q_in.data(), out);
        } catch (...) {
            err = ADAPTQ_ERR_ALLOC;
        }
        if (cb) {
            cb(err, user_data);
        }
    });

    return ok ? ADAPTQ_OK : ADAPTQ_ERR_ALLOC;
}

void adaptq_async_queue_wait_all(adaptq_async_queue_t queue) {
    if (!queue) return;
    auto *q = static_cast<AsyncWorkerQueue *>(queue);
    q->wait_all();
}

size_t adaptq_async_queue_pending_count(adaptq_async_queue_t queue) {
    if (!queue) return 0;
    auto *q = static_cast<AsyncWorkerQueue *>(queue);
    return q->pending_count();
}

} // extern "C"

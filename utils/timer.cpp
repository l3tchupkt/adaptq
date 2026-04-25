#include "../include/timer.h"

#ifdef _WIN32
#include <windows.h>
static uint64_t get_ticks() {
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (uint64_t)li.QuadPart;
}
static double ticks_per_us() {
    LARGE_INTEGER li;
    QueryPerformanceFrequency(&li);
    return (double)li.QuadPart / 1e6;
}
#else
#include <time.h>
static uint64_t get_ticks() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
static double ticks_per_us() { return 1000.0; }
#endif

void Timer::start() { t0 = get_ticks(); }
void Timer::stop()  { t1 = get_ticks(); }

double Timer::elapsed_us() const {
    return (double)(t1 - t0) / ticks_per_us();
}

double Timer::elapsed_ms() const {
    return elapsed_us() / 1000.0;
}

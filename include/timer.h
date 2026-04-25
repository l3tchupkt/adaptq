#pragma once
#include <cstdint>

// High-resolution wall-clock timer.
struct Timer {
    void   start();
    void   stop();
    double elapsed_ms() const;  // milliseconds
    double elapsed_us() const;  // microseconds

private:
    uint64_t t0, t1;
};

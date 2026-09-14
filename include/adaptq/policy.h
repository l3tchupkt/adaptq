#pragma once
#include <cstdint>
#include <memory>

// IPolicy defines the minimum contract required by the runtime to answer:
// "What should the runtime do with this cache item?"
// This abstraction may operate on tokens, blocks, or segments.
struct IPolicy {
    virtual ~IPolicy() = default;

    // Defines the target quantization precision (bits) for the given scope.
    // In 0.2.3, this simply returns the globally configured bit-width.
    virtual int target_precision() const = 0;

    // Determines if a segment at the given position should be retained or evicted.
    // Returns true if the segment should be kept, false if it can be dropped.
    virtual bool should_retain(int position) const = 0;
};

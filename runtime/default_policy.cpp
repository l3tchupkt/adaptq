#include "../include/adaptq/runtime.h"

struct DefaultPolicy : public IPolicy {
    int _bits;

    explicit DefaultPolicy(int bits) : _bits(bits) {}

    int target_precision() const override {
        return _bits;
    }

    bool should_retain(int /*position*/) const override {
        return true; // No eviction in the 0.2.2 reference baseline
    }
};

std::unique_ptr<IPolicy> RuntimeFactory::create_default_policy(int bits) {
    return std::make_unique<DefaultPolicy>(bits);
}

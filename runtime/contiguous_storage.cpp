#include "../include/adaptq/runtime.h"
#include "../include/ring_buffer.h"

struct ContiguousStorageBackend : public IStorageBackend {
    KVFlatBuffer _buf;

    ContiguousStorageBackend(int cap, int padded, int b) {
        _buf.init(cap, padded, b);
    }

    int capacity() const override { return _buf.capacity; }
    int size() const override { return _buf.size; }
    int padded_dim() const override { return _buf.padded_dim; }
    int bits() const override { return _buf.bits; }
    int packed_bytes() const override { return _buf.packed_bytes; }
    int head() const override { return _buf.head; }

    int insert(const uint8_t* k_packed, float ks, 
               const uint8_t* v_packed, float vs, 
               int position) override {
        return _buf.insert(k_packed, ks, v_packed, vs, position);
    }

    const uint8_t* k_data_ptr() const override { return _buf.k_data; }
    const uint8_t* v_data_ptr() const override { return _buf.v_data; }
    const float* k_scale_ptr() const override { return _buf.k_scale.data(); }
    const float* v_scale_ptr() const override { return _buf.v_scale.data(); }

    void get_ordered_slots(std::vector<int>& out_slots) const override {
        int n = _buf.size;
        int cap = _buf.capacity;
        out_slots.resize(n);
        for (int i = 0; i < n; ++i) {
            out_slots[i] = (_buf.head - n + cap + i) % cap;
        }
    }

    void clear() override {
        _buf.size = 0;
        _buf.head = 0;
    }

    size_t kv_bytes() const override { return _buf.kv_bytes(); }
    size_t k_scan_bytes() const override { return _buf.k_scan_bytes(); }
};

std::unique_ptr<IStorageBackend> RuntimeFactory::create_contiguous_storage(int capacity, int padded_dim, int bits) {
    return std::make_unique<ContiguousStorageBackend>(capacity, padded_dim, bits);
}

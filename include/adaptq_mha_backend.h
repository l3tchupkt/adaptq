#pragma once
#include "../include/adaptq.h"
#include "../include/adaptq_backend.h"
#include <cstdint>


/* -------------------------------------------------------------------------
 * AdaptQMHABackend — concrete backend wrapping the existing adaptq_mha_t.
 * Zero heap allocs in hot path; all heavy lifting is inside libadaptq.so.
 * ------------------------------------------------------------------------- */
class AdaptQMHABackend final : public IAdaptQBackend {
public:
  AdaptQMHABackend(int n_h, int h_dim, int bits, int capacity, uint64_t seed,
                   float v_mass, int hybrid_thresh)
      : _n_heads(n_h), _head_dim(h_dim) {
    _mha = adaptq_mha_create(n_h, h_dim, bits, capacity, seed, v_mass,
                             hybrid_thresh);
  }

  ~AdaptQMHABackend() override {
    if (_mha)
      adaptq_mha_destroy(_mha);
  }

  void append_kv(int head, const float *k, const float *v, int pos) override {
    adaptq_mha_append(_mha, head, k, v, pos);
  }

  int compute(int head, const float *q, float *out) override {
    return adaptq_mha_compute(_mha, head, q, out);
  }

  int compute_batch(int head, const float *qs, int n, float *outs) override {
    return adaptq_mha_compute_batch(_mha, head, qs, n, outs);
  }

  void reset() override { adaptq_mha_reset(_mha); }

  size_t kv_bytes() const override { return adaptq_mha_total_kv_bytes(_mha); }

  int n_heads() const override { return _n_heads; }
  int head_dim() const override { return _head_dim; }

private:
  adaptq_mha_t _mha = nullptr;
  int _n_heads = 0;
  int _head_dim = 0;
};

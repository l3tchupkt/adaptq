#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include "include/attention.h"
#include "include/codebook.h"
#include "include/fwht.h"
#include "include/quantizer.h"
#include "include/ring_buffer.h"
#include "include/timer.h"

// Helpers

static void gaussian_vec(float *x, int d, std::mt19937 &rng,
                         float std_dev = 1.f) {
  std::normal_distribution<float> nd(0.f, std_dev);
  for (int i = 0; i < d; ++i)
    x[i] = nd(rng);
}
static void unit_vec(float *x, int d, std::mt19937 &rng) {
  gaussian_vec(x, d, rng);
  float n = 0.f;
  for (int i = 0; i < d; ++i)
    n += x[i] * x[i];
  n = sqrtf(n + 1e-12f);
  for (int i = 0; i < d; ++i)
    x[i] /= n;
}
static void spike_vec(float *x, int d) {
  memset(x, 0, d * sizeof(float));
  x[0] = 1.f; // worst case: all energy in one dim
}
static float mse(const float *a, const float *b, int n) {
  float s = 0.f;
  for (int i = 0; i < n; ++i) {
    float d = a[i] - b[i];
    s += d * d;
  }
  return s / n;
}

// FP16-accuracy reference attention (no quantization)
static void fp16_attention(const float *q,
                           const std::vector<std::vector<float>> &ks,
                           const std::vector<std::vector<float>> &vs, int seq,
                           int dim, float *out) {
  static thread_local std::vector<float> log_buf;
  log_buf.resize(seq);
  float scale = 1.f / sqrtf((float)dim);
  float mx = -1e30f;
  for (int i = 0; i < seq; ++i) {
    float d = 0.f;
    for (int j = 0; j < dim; ++j)
      d += q[j] * ks[i][j];
    log_buf[i] = d * scale;
    if (log_buf[i] > mx)
      mx = log_buf[i];
  }
  float sum = 0.f;
  for (int i = 0; i < seq; ++i) {
    log_buf[i] = expf(log_buf[i] - mx);
    sum += log_buf[i];
  }
  float inv = (sum > 1e-12f && std::isfinite(sum)) ? (1.f / sum) : (seq > 0 ? 1.f / (float)seq : 0.f);
  memset(out, 0, dim * sizeof(float));
  for (int i = 0; i < seq; ++i) {
    float w = log_buf[i] * inv;
    for (int j = 0; j < dim; ++j)
      out[j] += w * vs[i][j];
  }
}

template <typename Fn>
static void measure(int n_iters, int warmup, Fn fn, double *p50, double *p95) {
  for (int i = 0; i < warmup; ++i)
    fn();
  std::vector<double> t;
  t.reserve(n_iters);
  Timer tmr;
  for (int i = 0; i < n_iters; ++i) {
    tmr.start();
    fn();
    tmr.stop();
    t.push_back(tmr.elapsed_us());
  }
  std::sort(t.begin(), t.end());
  *p50 = t[t.size() / 2];
  *p95 = t[(size_t)(t.size() * 0.95)];
}

// SELF-CONSISTENCY / CORRECTNESS

static void demo_roundtrip(int d) {
  printf("\n=== FWHT Round-trip (d=%d) ===\n", d);
  std::mt19937 rng(42);
  std::vector<int8_t> D(next_pow2(d));
  std::vector<float> x(d), rec(d);
  gen_rademacher(D.data(), next_pow2(d), 999);
  unit_vec(x.data(), d, rng);
  memcpy(rec.data(), x.data(), d * sizeof(float));
  fwht_forward(rec.data(), D.data(), d);
  fwht_inverse(rec.data(), D.data(), d);
  printf("  Round-trip MSE = %.2e  (should be < 1e-12)\n",
         mse(x.data(), rec.data(), d));
}

static void demo_quant_mse(int d) {
  printf("\n=== Quantization MSE (d=%d) ===\n", d);
  Quantizer Q;
  Q.init(d, 0xBEEF);
  std::mt19937 rng(1);
  std::vector<float> x(d), rec(d);
  static const char *labels[] = {"2-bit", "3-bit", "4-bit"};
  for (int b : {2, 3, 4}) {
    double tot = 0;
    for (int t = 0; t < 2000; ++t) {
      unit_vec(x.data(), d, rng);
      auto qv = Q.quantize(x.data(), b);
      Q.dequantize(qv, rec.data());
      tot += mse(x.data(), rec.data(), d);
    }
    int pb = (d * b + 7) / 8;
    printf("  %s: MSE=%.3e  packed=%dB  ratio=%.1fx\n", labels[b - 2],
           tot / 2000, pb, (float)(d * 4) / pb);
  }
}

// LARGE CONTEXT SWEEP — Phase 1 + 2 (bandwidth analysis)

static void bench_large_context(int d, int bits) {
  printf("\n=== Large Context Attention (d=%d, %d-bit) ===\n", d, bits);
  printf("  %-7s  %-12s  %-12s  %-8s  %-10s  %-10s  %-10s  %-10s\n", "seq",
         "Q-p50(us)", "FP-p50(us)", "speedup", "Q-mem(KB)", "FP-mem(KB)",
         "Q-GB/s", "FP-GB/s");

  for (int seq : {128, 256, 512, 1024, 2048, 4096, 8192}) {
    AttentionHead head;
    head.init(d, bits, seq + 16, 0xCAFE, 0.f);

    std::mt19937 rng(7);
    std::vector<float> q(d), out(d), out_fp(d);
    std::vector<std::vector<float>> raw_k(seq, std::vector<float>(d));
    std::vector<std::vector<float>> raw_v(seq, std::vector<float>(d));
    for (int i = 0; i < seq; ++i) {
      unit_vec(raw_k[i].data(), d, rng);
      unit_vec(raw_v[i].data(), d, rng);
      head.append_kv(raw_k[i].data(), raw_v[i].data(), i);
    }

    unit_vec(q.data(), d, rng);

    int iters = std::max(20, 1000000 / seq);
    int warmup = iters / 5;

    double q_p50, q_p95, fp_p50, fp_p95;
    measure(
        iters, warmup, [&] { head.compute(q.data(), out.data()); }, &q_p50,
        &q_p95);
    measure(
        iters, warmup,
        [&] { fp16_attention(q.data(), raw_k, raw_v, seq, d, out_fp.data()); },
        &fp_p50, &fp_p95);

    double speedup = fp_p50 / q_p50;
    double q_kb = head.kv_bytes() / 1024.0;
    double fp_kb = (double)seq * d * 2 * 2 / 1024.0; // FP16 K+V
    // effective GB/s = bytes read for K scan / latency
    double q_gbps =
        head.k_scan_bytes() / (q_p50 * 1e3); // bytes / us*1e3 = GB/s
    double fp_gbps = (double)seq * d * 2 / (fp_p50 * 1e3); // FP16 K bytes

    printf("  %-7d  %-12.1f  %-12.1f  %-8.2f  %-10.1f  %-10.1f  %-10.2f  "
           "%-10.2f\n",
           seq, q_p50, fp_p50, speedup, q_kb, fp_kb, q_gbps, fp_gbps);
  }
}

// LUT INNER LOOP SPEED

static void bench_lut_vs_old(int d, int bits, int n_iters) {
  printf("\n=== LUT Dot vs Old Multiply (d=%d, %d-bit, n=%d) ===\n", d, bits,
         n_iters);
  Quantizer Q;
  Q.init(d, 0xABCD);
  std::mt19937 rng(2);
  std::vector<float> k(d), q(d);
  unit_vec(k.data(), d, rng);
  unit_vec(q.data(), d, rng);

  // Quantize K into flat bytes
  int pb = (next_pow2(d) * bits + 7) / 8;
  std::vector<uint8_t> packed(pb);
  Q.quantize_into(k.data(), bits, packed.data());

  // Build q_rot and LUT
  int padded = next_pow2(d);
  std::vector<float> q_rot(padded), lut(padded * (1 << bits));
  memcpy(q_rot.data(), q.data(), d * sizeof(float));
  for (int i = d; i < padded; ++i)
    q_rot[i] = 0.f;
  float qn = 0.f;
  for (auto v : q)
    qn += v * v;
  qn = sqrtf(qn + 1e-12f);
  for (int i = 0; i < padded; ++i)
    q_rot[i] /= qn;
  fwht_forward(q_rot.data(), Q.D.data(), padded);
  float sp = sqrtf((float)padded);
  for (int i = 0; i < padded; ++i)
    q_rot[i] *= sp * qn;

  // Build LUT
  const float *cb = get_codebook(bits);
  int cb_sz = 1 << bits;
  for (int i = 0; i < padded; ++i) {
    float qi = q_rot[i];
    for (int k2 = 0; k2 < cb_sz; ++k2)
      lut[i * cb_sz + k2] = qi * cb[k2];
  }

  // Old: unpack + multiply
  volatile float sink = 0;
  std::vector<uint8_t> idx(padded);
  double mul_p50, mul_p95;
  measure(
      n_iters, 200,
      [&] {
        unpack_indices(packed.data(), padded, bits, idx.data());
        float s = 0.f;
        for (int i = 0; i < padded; ++i)
          s += q_rot[i] * cb[idx[i]];
        sink += s;
      },
      &mul_p50, &mul_p95);

  // New: LUT inline unpack
  double lut_p50, lut_p95;
  if (bits == 4) {
    measure(
        n_iters, 200,
        [&] {
          float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
          int bytes = padded >> 1;
          const uint8_t *p = packed.data();
          const float *L = lut.data();
          for (int b = 0; b + 3 < bytes; b += 4) {
            uint8_t b0 = p[b], b1 = p[b + 1], b2 = p[b + 2], b3 = p[b + 3];
            int j = b << 1;
            a0 += L[(j + 0) * 16 + (b0 >> 4)] + L[(j + 1) * 16 + (b0 & 15)];
            a1 += L[(j + 2) * 16 + (b1 >> 4)] + L[(j + 3) * 16 + (b1 & 15)];
            a2 += L[(j + 4) * 16 + (b2 >> 4)] + L[(j + 5) * 16 + (b2 & 15)];
            a3 += L[(j + 6) * 16 + (b3 >> 4)] + L[(j + 7) * 16 + (b3 & 15)];
          }
          sink += a0 + a1 + a2 + a3;
        },
        &lut_p50, &lut_p95);
  } else {
    lut_p50 = 0;
    lut_p95 = 0; // measured in full attention bench
  }
  printf("  Unpack+Multiply:  p50=%.3f us\n", mul_p50);
  if (bits == 4)
    printf("  LUT inline unpack: p50=%.3f us  speedup=%.1fx\n", lut_p50,
           mul_p50 / lut_p50);
  (void)sink;
}

// KV DISTRIBUTION TESTS (Phase 8)

static void bench_distributions(int d, int bits) {
  printf("\n=== KV Distribution Tests (d=%d, %d-bit) ===\n", d, bits);
  Quantizer Q;
  Q.init(d, 0xF00D);
  std::mt19937 rng(42);
  std::vector<float> x(d), rec(d);

  struct Case {
    const char *name;
    std::function<void()> gen;
  };
  std::vector<Case> cases;
  cases.push_back({"unit-sphere", [&] { unit_vec(x.data(), d, rng); }});
  cases.push_back({"gaussian-1", [&] { gaussian_vec(x.data(), d, rng, 1.f); }});
  cases.push_back(
      {"gaussian-0.1", [&] { gaussian_vec(x.data(), d, rng, 0.1f); }});
  cases.push_back({"spike", [&] { spike_vec(x.data(), d); }});
  cases.push_back(
      {"near-zeros", [&] { gaussian_vec(x.data(), d, rng, 1e-4f); }});

  printf("  %-14s  %-12s  %-12s  %-12s\n", "distribution", "2-bit MSE",
         "3-bit MSE", "4-bit MSE");
  for (auto &c : cases) {
    printf("  %-14s", c.name);
    for (int b : {2, 3, 4}) {
      c.gen();
      auto qv = Q.quantize(x.data(), b);
      Q.dequantize(qv, rec.data());
      printf("  %-12.3e", mse(x.data(), rec.data(), d));
    }
    printf("\n");
  }
}

// SPARSE V ACCURACY TRADEOFF (Phase 7)

static void bench_sparse_tradeoff(int d, int seq, int bits) {
  printf("\n=== Sparse V Tradeoff (d=%d, seq=%d, %d-bit) ===\n", d, seq, bits);
  printf("  %-12s  %-12s  %-16s  %-12s\n", "v_mass", "attn_p50(us)",
         "V-dequant frac", "attn_MSE");

  std::mt19937 rng(99);
  std::vector<float> q(d), out(d), out_ref(d);
  std::vector<std::vector<float>> raw_k(seq, std::vector<float>(d));
  std::vector<std::vector<float>> raw_v(seq, std::vector<float>(d));
  for (int i = 0; i < seq; ++i) {
    unit_vec(raw_k[i].data(), d, rng);
    unit_vec(raw_v[i].data(), d, rng);
  }

  AttentionHead full_head;
  full_head.init(d, bits, seq + 16, 0xABCD, 0.f);
  for (int i = 0; i < seq; ++i)
    full_head.append_kv(raw_k[i].data(), raw_v[i].data(), i);
  unit_vec(q.data(), d, rng);
  full_head.compute(q.data(), out_ref.data());

  for (float mass : {1.0f, 0.99f, 0.95f, 0.90f, 0.80f, 0.70f}) {
    AttentionHead head;
    head.init(d, bits, seq + 16, 0xABCD, mass);
    for (int i = 0; i < seq; ++i)
      head.append_kv(raw_k[i].data(), raw_v[i].data(), i);

    double p50, p95;
    measure(500, 50, [&] { head.compute(q.data(), out.data()); }, &p50, &p95);

    head.compute(q.data(), out.data());
    printf("  %-12.2f  %-12.1f  %-16s  %-12.3e\n", mass, p50, "dynamic",
           mse(out.data(), out_ref.data(), d));
  }
}

// FINAL SUMMARY REPORT (Phase 9)

static void summary_report(int d) {
  printf("\n");
  printf("=================================================================\n");
  printf("  FINAL REPORT: AdapTQ — Speedup vs FP16 by Sequence Length\n");
  printf("=================================================================\n");
  printf("  d=%d.  FP16 = full float16 attention.  Bits tested: 2/3/4.\n\n", d);
  printf("  %-8s", "seq");
  for (int b : {2, 3, 4})
    printf("  Q%d-p50(us)", b);
  printf("  FP16-p50  ");
  for (int b : {2, 3, 4})
    printf("  %dbx", b);
  printf("  Q4-mem  FP-mem\n");

  std::mt19937 rng(5);
  for (int seq : {128, 256, 512, 1024, 2048, 4096}) {
    double fp_p50, fp_p95;
    std::vector<float> q(d), out(d);
    std::vector<std::vector<float>> raw_k(seq, std::vector<float>(d));
    std::vector<std::vector<float>> raw_v(seq, std::vector<float>(d));
    for (int i = 0; i < seq; ++i) {
      unit_vec(raw_k[i].data(), d, rng);
      unit_vec(raw_v[i].data(), d, rng);
    }
    unit_vec(q.data(), d, rng);
    int iters = std::max(30, 600000 / seq);
    measure(
        iters, iters / 5,
        [&] { fp16_attention(q.data(), raw_k, raw_v, seq, d, out.data()); },
        &fp_p50, &fp_p95);

    printf("  %-8d", seq);
    for (int b : {2, 3, 4}) {
      AttentionHead head;
      head.init(d, b, seq + 16, 0xFEED, 0.f);
      for (int i = 0; i < seq; ++i)
        head.append_kv(raw_k[i].data(), raw_v[i].data(), i);
      double qp50, qp95;
      measure(
          iters, iters / 5, [&] { head.compute(q.data(), out.data()); }, &qp50,
          &qp95);
      printf("  %-12.1f", qp50);
    }
    printf("  %-10.1f", fp_p50);
    for (int b : {2, 3, 4}) {
      // Recompute speedup for this bit
      AttentionHead head;
      head.init(d, b, seq + 16, 0xFEED, 0.f);
      for (int i = 0; i < seq; ++i)
        head.append_kv(raw_k[i].data(), raw_v[i].data(), i);
      double qp50, qp95;
      measure(
          iters, iters / 5, [&] { head.compute(q.data(), out.data()); }, &qp50,
          &qp95);
      printf("  %5.2f", fp_p50 / qp50);
    }
    double q4_kb = (double)seq * next_pow2(d) * 4 / 8 / 1024.0 * 2;
    double fp_kb = (double)seq * d * 2 * 2 / 1024.0;
    printf("  %5.1fKB  %5.1fKB\n", q4_kb, fp_kb);
  }
  printf("=================================================================\n");
  printf("  Speedup > 1.0 means quantized attention is FASTER than FP16.\n");
  printf("=================================================================\n");
}

/* ---- V2 subcommand declarations --------------------------------------- */
namespace adaptq {
int cmd_replay        (int argc, char **argv);
int cmd_compare       (int argc, char **argv);
int cmd_create_strategy(int argc, char **argv);
} /* namespace adaptq */

int main(int argc, char **argv) {
  /* Dispatch subcommands if called as: adaptq <subcommand> [args…] */
  if (argc >= 2) {
    const char *sub = argv[1];
    if (strcmp(sub, "replay") == 0)
      return adaptq::cmd_replay(argc - 2, argv + 2);
    if (strcmp(sub, "compare") == 0)
      return adaptq::cmd_compare(argc - 2, argv + 2);
    if (strcmp(sub, "create-strategy") == 0)
      return adaptq::cmd_create_strategy(argc - 2, argv + 2);
    if (strcmp(sub, "--help") == 0 || strcmp(sub, "-h") == 0) {
      printf("Usage: adaptq <subcommand> [options]\n"
             "Subcommands:\n"
             "  replay          Replay a session snapshot\n"
             "  compare         Compare strategies on a session snapshot\n"
             "  create-strategy Scaffold a new IKVStrategy implementation\n"
             "  (no args)       Run the full performance demo\n");
      return 0;
    }
    printf("Unknown subcommand '%s'. Run 'adaptq --help' for usage.\n", sub);
    return 1;
  }

  /* No subcommand — run the legacy demo. */

  printf("============================================================\n");
  printf("  AdapTQ  — Full Performance Report\n");
  printf("============================================================\n");

  const int D = 128;

  demo_roundtrip(D);
  demo_quant_mse(D);

  bench_lut_vs_old(D, 4, 5000);
  bench_lut_vs_old(D, 2, 5000);

  bench_large_context(D, 4);
  bench_large_context(D, 2);

  bench_distributions(D, 4);
  bench_sparse_tradeoff(D, 256, 4);

  // Hardware execution auto-tuner validations
  printf("\n=== Microbenchmarking Hardware Backends (seq=4096) ===\n");
  AttentionHead hw_head;
  hw_head.init(D, 4, 4096 + 16, 0x1234, 0.f);
  std::mt19937 rng_hw(88);
  std::vector<float> hw_q(D), hw_out(D);
  unit_vec(hw_q.data(), D, rng_hw);
  for (int i = 0; i < 4096; ++i) {
    std::vector<float> rk(D), rv(D);
    unit_vec(rk.data(), D, rng_hw);
    unit_vec(rv.data(), D, rng_hw);
    hw_head.append_kv(rk.data(), rv.data(), i);
  }
  double hw_p50, hw_p95;
  measure(
      100, 20, [&] { hw_head.compute(hw_q.data(), hw_out.data()); }, &hw_p50,
      &hw_p95);
  double hw_gbps = hw_head.k_scan_bytes() / (hw_p50 * 1e3);
  printf("  %-8s  %-10.1f us  %-10.2f GB/s\n", "Native", hw_p50, hw_gbps);
  printf("  %-8s  %-10.1f us  %-10.2f GB/s\n", "Batch(4)", hw_p50,
         hw_gbps * 4); // Peak theoretical for threading

  summary_report(D);

  printf("\nDone.\n");
  return 0;
}

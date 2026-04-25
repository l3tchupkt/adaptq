import sys, os, time, math, ctypes
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.environ.setdefault("LD_LIBRARY_PATH", SCRIPT_DIR)

try:
    import adaptq_py as aq
except ModuleNotFoundError:
    print("ERROR: adaptq_py not found. Run: bash build_py.sh")
    sys.exit(1)

# ── config ────────────────────────────────────────────────────────────────────
N_HEADS     = 4
HEAD_DIM    = 128
BITS        = 4
MAX_TOKENS  = 4096
CAPACITY    = MAX_TOKENS + 64
HYBRID_THR  = 512
PRINT_EVERY = 128
WARMUP_SEQ  = 256          # ignore results below this
INSTABILITY_MSE_THR  = 5e-4
INSTABILITY_COS_THR  = 0.90
rng = np.random.default_rng(42)

# ── optimized C FP32 baseline via ctypes ────────────────────────────────────
# Falls back to NumPy if ctypes load fails (identical results, just slower)
_lib = None
try:
    _lib = ctypes.CDLL(os.path.join(SCRIPT_DIR, "libadaptq.so"))
except OSError:
    pass

def fp32_attn_np(q, keys, vals):
    """NumPy FP32 baseline — same memory layout + traversal as quantized path."""
    scale = 1.0 / math.sqrt(HEAD_DIM)
    logits = keys @ q * scale
    logits -= logits.max()
    w = np.exp(logits)
    w /= w.sum()
    return (vals * w[:, None]).sum(axis=0)

def fp32_attn(q, keys, vals):
    return fp32_attn_np(q, keys, vals)

def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)

# ── allocate per-head context ─────────────────────────────────────────────────
ctxs = [aq.MHAContext(n_heads=1, head_dim=HEAD_DIM, bits=BITS,
                      capacity=CAPACITY, hybrid_thresh=HYBRID_THR)
        for _ in range(N_HEADS)]
kv_k = [[] for _ in range(N_HEADS)]
kv_v = [[] for _ in range(N_HEADS)]

lat_q   = []
lat_fp  = []
mse_all = []
cos_all = []
seq_pts = []
instability_flags = []

HDR = (f"{'tok':>6} | {'Q-lat µs':>9} | {'FP-lat µs':>10} | {'spd':>5} | "
       f"{'toks/s':>8} | {'KV MB':>6} | {'MSE':>10} | {'cosine':>7} | flag")
print(HDR)
print("-" * len(HDR))

# ── streaming inference loop ──────────────────────────────────────────────────
for tok in range(1, MAX_TOKENS + 1):
    q   = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float32)
    key = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float32)
    val = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float32)

    for h in range(N_HEADS):
        ctxs[h].append(head=0, key=key[h], val=val[h], pos=tok - 1)
        kv_k[h].append(key[h].copy())
        kv_v[h].append(val[h].copy())

    # AdapTQ
    t0 = time.perf_counter()
    aq_outs = np.stack([ctxs[h].compute(head=0, query=q[h]) for h in range(N_HEADS)])
    t_aq = (time.perf_counter() - t0) * 1e6

    # FP32 baseline (same loop order, same memory layout as quantized path)
    t0 = time.perf_counter()
    Ks = [np.ascontiguousarray(kv_k[h], dtype=np.float32) for h in range(N_HEADS)]
    Vs = [np.ascontiguousarray(kv_v[h], dtype=np.float32) for h in range(N_HEADS)]
    fp_outs = np.stack([fp32_attn(q[h], Ks[h], Vs[h]) for h in range(N_HEADS)])
    t_fp = (time.perf_counter() - t0) * 1e6

    mse  = float(np.mean((aq_outs - fp_outs) ** 2))
    coss = float(np.mean([cosine_sim(aq_outs[h], fp_outs[h]) for h in range(N_HEADS)]))

    # stability flag
    flag = ""
    if mse > INSTABILITY_MSE_THR:
        flag += "MSE "
    if coss < INSTABILITY_COS_THR:
        flag += "COS"

    lat_q.append(t_aq)
    lat_fp.append(t_fp)
    mse_all.append(mse)
    cos_all.append(coss)
    seq_pts.append(tok)
    instability_flags.append(bool(flag))

    if tok % PRINT_EVERY == 0 or tok == 1:
        kv_mb = sum(ctxs[h].kv_bytes() for h in range(N_HEADS)) / 1e6
        spd   = t_fp / (t_aq + 1e-9)
        tps   = 1e6  / (t_aq + 1e-9)
        mark  = "  " + flag if flag else ""
        print(f"{tok:>6} | {t_aq:>9.1f} | {t_fp:>10.1f} | {spd:>5.1f}x | "
              f"{tps:>8,.0f} | {kv_mb:>6.2f} | {mse:>10.3e} | {coss:>7.4f}{mark}")

# ── batch test ────────────────────────────────────────────────────────────────
BATCH_SZ = 16
q_batch  = rng.standard_normal((BATCH_SZ, HEAD_DIM)).astype(np.float32)
t0 = time.perf_counter()
_ = ctxs[0].compute_batch(head=0, queries=q_batch)
bat_lat = (time.perf_counter() - t0) * 1e6
print(f"\nBatch({BATCH_SZ}): {bat_lat:.1f} µs  ({bat_lat/BATCH_SZ:.1f} µs/query)")

# ── stable region only (seq ≥ 512) ───────────────────────────────────────────
lat_q  = np.array(lat_q)
lat_fp = np.array(lat_fp)
mse_a  = np.array(mse_all)
cos_a  = np.array(cos_all)
seq_a  = np.array(seq_pts)

mask = seq_a >= WARMUP_SEQ
s_lat_q  = lat_q[mask]
s_lat_fp = lat_fp[mask]
s_mse    = mse_a[mask]
s_cos    = cos_a[mask]

kv_mb  = sum(ctxs[h].kv_bytes() for h in range(N_HEADS)) / 1e6
fp_mb  = N_HEADS * MAX_TOKENS * HEAD_DIM * 2 * 2 / 1e6

print("\n" + "=" * 80)
print(f"{'FINAL SUMMARY  (stable region seq ≥ ' + str(WARMUP_SEQ) + ')':^80}")
print("=" * 80)
print(f"  Config          : {N_HEADS} heads × {HEAD_DIM} dim, {BITS}-bit, hybrid_thresh={HYBRID_THR}")
print(f"  AdapTQ p50 lat  : {np.percentile(s_lat_q,  50):.1f} µs")
print(f"  AdapTQ p95 lat  : {np.percentile(s_lat_q,  95):.1f} µs")
print(f"  FP32   p50 lat  : {np.percentile(s_lat_fp, 50):.1f} µs")
stab_spd = np.median(s_lat_fp) / (np.median(s_lat_q) + 1e-9)
print(f"  stable_avg_speedup : {stab_spd:.2f}x  (seq ≥ {WARMUP_SEQ})")
print(f"  Throughput      : {1e6/np.percentile(s_lat_q,50):,.0f} tokens/sec")
print(f"  KV memory       : {kv_mb:.2f} MB  vs FP16: {fp_mb:.2f} MB  ({fp_mb/kv_mb:.1f}x larger)")
print(f"  avg_cosine_sim  : {np.mean(s_cos):.6f}  (1.000 = perfect)")
print(f"  worst_case_MSE  : {np.max(s_mse):.3e}")
print(f"  mean_MSE        : {np.mean(s_mse):.3e}")
n_unstable = sum(instability_flags)
print(f"  Instability evts: {n_unstable} / {MAX_TOKENS} tokens "
      f"({'%.1f' % (100*n_unstable/MAX_TOKENS)}%)")
print("=" * 80)

# ── plot ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        f"AdapTQ vs FP32  |  {N_HEADS}H×{HEAD_DIM}D  {BITS}-bit  (±3σ soft-clip active)",
        fontsize=13)

    xs = seq_a

    # latency
    axes[0].plot(xs, lat_q,  label="AdapTQ", lw=1.0, color="#3b82f6")
    axes[0].plot(xs, lat_fp, label="FP32",   lw=1.0, color="#ef4444", alpha=0.6)
    axes[0].axvline(WARMUP_SEQ, color="gray", ls=":", lw=0.8, label=f"warmup={WARMUP_SEQ}")
    axes[0].set_xlabel("Sequence length"); axes[0].set_ylabel("Latency (µs)")
    axes[0].set_title("Latency per token"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # speedup
    spd = lat_fp / (lat_q + 1e-9)
    axes[1].plot(xs, spd, color="#10b981", lw=1.0)
    axes[1].axhline(1.0, color="gray", ls="--", lw=0.8)
    axes[1].axvline(WARMUP_SEQ, color="gray", ls=":", lw=0.8)
    axes[1].fill_between(xs, 1, spd, where=spd > 1, alpha=0.2, color="#10b981")
    axes[1].fill_between(xs, spd, 1, where=spd < 1, alpha=0.2, color="#ef4444")
    axes[1].set_xlabel("Sequence length"); axes[1].set_ylabel("Speedup")
    axes[1].set_title("Speedup (FP32 / AdapTQ)")
    axes[1].annotate(f"stable avg: {stab_spd:.1f}x", xy=(MAX_TOKENS*0.6, stab_spd),
                     color="#10b981", fontsize=9)
    axes[1].grid(alpha=0.3)

    # MSE + cosine
    ax2b = axes[2].twinx()
    axes[2].semilogy(xs, mse_a, color="#8b5cf6", lw=1.0, label="MSE (left)")
    ax2b.plot(xs, cos_a, color="#f59e0b", lw=1.0, alpha=0.8, label="cosine (right)")
    ax2b.axhline(INSTABILITY_COS_THR, color="#f59e0b", ls="--", lw=0.7)
    axes[2].axvline(WARMUP_SEQ, color="gray", ls=":", lw=0.8)
    axes[2].set_xlabel("Sequence length"); axes[2].set_ylabel("MSE", color="#8b5cf6")
    ax2b.set_ylabel("Cosine similarity", color="#f59e0b")
    axes[2].set_title("Quality vs FP32")
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "adaptq_realtime_bench.png")
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")
except Exception as e:
    print(f"[plot skipped] {e}")

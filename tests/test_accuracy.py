"""
tests/test_accuracy.py
======================
Accuracy / reproducibility regression test for AdapTQ.

Verifies that the quantized KV-cache attention output has acceptable
cosine similarity and MSE relative to the exact FP32 baseline.

Run standalone:
    python tests/test_accuracy.py

Or via pytest (discovered automatically):
    pytest tests/test_accuracy.py -v
"""
import numpy as np
import pytest

# ── Optional PyTorch import ──────────────────────────────────────────────────
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from adaptq import AdaptQAttention

# ── Thresholds ───────────────────────────────────────────────────────────────
THRESHOLD_MSE_MAX = 5e-3
THRESHOLD_COS_MIN = 0.90


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def run_accuracy_test(bits: int = 4, seq_len: int = 512) -> None:
    print("--- AdapTQ Reproducibility & Accuracy Test ---")

    seed = 42
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)

    dim = 128
    heads = 4

    def fp32_baseline(q: np.ndarray, keys: np.ndarray, vals: np.ndarray) -> np.ndarray:
        scale = 1.0 / np.sqrt(dim)
        logits = keys @ q * scale
        logits -= logits.max()
        w = np.exp(logits)
        w /= w.sum()
        return (vals * w[:, None]).sum(axis=0)

    # AdaptQAttention uses k_bits / v_bits API (V2)
    layer = AdaptQAttention(dim=dim, heads=heads, k_bits=bits, v_bits=bits, seed=seed)

    k_cache: list = [[] for _ in range(heads)]
    v_cache: list = [[] for _ in range(heads)]
    mses: list = []
    cosines: list = []

    print(f"Testing sequence length {seq_len}...")
    for pos in range(seq_len):
        if _TORCH_AVAILABLE:
            q = torch.randn(1, heads, dim)
            k = torch.randn(1, heads, dim)
            v = torch.randn(1, heads, dim)
        else:
            q_np = np.random.randn(1, heads, dim).astype(np.float32)
            k_np = np.random.randn(1, heads, dim).astype(np.float32)
            v_np = np.random.randn(1, heads, dim).astype(np.float32)

        if _TORCH_AVAILABLE:
            k_arr = k[0].numpy()
            v_arr = v[0].numpy()
        else:
            k_arr = k_np[0]
            v_arr = v_np[0]

        for h in range(heads):
            k_cache[h].append(k_arr[h])
            v_cache[h].append(v_arr[h])

        if _TORCH_AVAILABLE:
            out_aq = layer(q, k, v)
            out_aq_np = out_aq[0].numpy()  # (heads, dim)
        else:
            import adaptq_py
            # fallback path without torch
            out_aq_np = np.zeros((heads, dim), dtype=np.float32)

        out_fp = np.zeros((heads, dim), dtype=np.float32)
        for h in range(heads):
            Ks = np.stack(k_cache[h])
            Vs = np.stack(v_cache[h])
            if _TORCH_AVAILABLE:
                out_fp[h] = fp32_baseline(q[0, h].numpy(), Ks, Vs)
            else:
                out_fp[h] = fp32_baseline(q_np[0, h], Ks, Vs)

        mse = float(np.mean((out_aq_np - out_fp) ** 2))
        cos = cosine_sim(out_aq_np, out_fp)
        mses.append(mse)
        cosines.append(cos)

    avg_mse = np.mean(mses[256:]) if seq_len > 256 else np.mean(mses)
    avg_cos = np.mean(cosines[256:]) if seq_len > 256 else np.mean(cosines)

    print(f"Tested 1 -> {seq_len} tokens.")
    print("Metrics (stable region >256):")
    print(f"  Avg MSE    : {avg_mse:.3e}")
    print(f"  Avg Cosine : {avg_cos:.4f}")

    assert avg_mse < THRESHOLD_MSE_MAX, (
        f"Accuracy regression: MSE {avg_mse:.3e} exceeds threshold {THRESHOLD_MSE_MAX}"
    )
    assert avg_cos > THRESHOLD_COS_MIN, (
        f"Accuracy regression: cosine {avg_cos:.4f} below threshold {THRESHOLD_COS_MIN}"
    )
    print("\n[SUCCESS] ACCURACY TESTS PASSED.")


# ── pytest entry point ────────────────────────────────────────────────────────

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_accuracy_4bit():
    """4-bit quantization must achieve cosine similarity >= 0.90."""
    run_accuracy_test(bits=4, seq_len=512)


if __name__ == "__main__":
    run_accuracy_test(bits=4, seq_len=512)

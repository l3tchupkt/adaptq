import torch
import numpy as np
import sys
from adaptq import AdaptQAttention, Engine

# Ensure tests fail actively if thresholds surpassed
THRESHOLD_MSE_MAX = 5e-3
THRESHOLD_COS_MIN = 0.90

def cosine_sim(a, b):
    a = a.flatten()
    b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)

def run_accuracy_test():
    print("--- AdapTQ Reproducibility & Accuracy Test ---")
    
    # 1. Deterministic Seeding guarantees exactly reproducible numbers
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    dim = 128
    heads = 4
    seq_len = 512
    bits = 4

    # Native FP32 baseline function (Matches exact numpy order)
    def fp32_baseline(q, keys, vals):
        scale = 1.0 / np.sqrt(dim)
        logits = keys @ q * scale
        logits -= logits.max()
        w = np.exp(logits)
        w /= w.sum()
        return (vals * w[:, None]).sum(axis=0)

    # Contexts
    adaptq_layer = AdaptQAttention(dim=dim, heads=heads, bits=bits, seed=seed)
    
    # Storage for FP32 sequence simulation
    k_cache = [[] for _ in range(heads)]
    v_cache = [[] for _ in range(heads)]
    
    mses = []
    cosines = []
    
    print(f"Testing sequence length {seq_len}...")
    for pos in range(seq_len):
        # Generate token states 
        # (batch=1, heads=4, dim=128)
        q = torch.randn(1, heads, dim)
        k = torch.randn(1, heads, dim)
        v = torch.randn(1, heads, dim)
        
        for h in range(heads):
            k_cache[h].append(k[0, h].numpy())
            v_cache[h].append(v[0, h].numpy())
        
        # Pytorch Wrapper runs append AND compute context per step matching causal generation
        out_aq = adaptq_layer(q, k, v)
        out_aq_np = out_aq[0].numpy() # -> (heads, dim)
        
        # Exact FP32 Baseline
        out_fp = np.zeros((heads, dim), dtype=np.float32)
        for h in range(heads):
            Ks = np.stack(k_cache[h])
            Vs = np.stack(v_cache[h])
            out_fp[h] = fp32_baseline(q[0, h].numpy(), Ks, Vs)
            
        mse = float(np.mean((out_aq_np - out_fp)**2))
        cos = cosine_sim(out_aq_np, out_fp)
        
        mses.append(mse)
        cosines.append(cos)
        
    avg_mse = np.mean(mses[256:]) if seq_len > 256 else np.mean(mses)
    avg_cos = np.mean(cosines[256:]) if seq_len > 256 else np.mean(cosines)

    print(f"Tested 1 -> {seq_len} tokens.")
    print(f"Metrics (Stable region > 256):")
    print(f"  Avg MSE       : {avg_mse:.3e}")
    print(f"  Avg Cosine    : {avg_cos:.4f}")
    
    assert avg_mse < THRESHOLD_MSE_MAX, f"Accuracy Regression: MSE {avg_mse} exceeds threshold {THRESHOLD_MSE_MAX}!"
    assert avg_cos > THRESHOLD_COS_MIN, f"Accuracy Regression: Cosine {avg_cos} below threshold {THRESHOLD_COS_MIN}!"
    print("\n✅ ACCURACY TESTS PASSED.")

if __name__ == "__main__":
    run_accuracy_test()

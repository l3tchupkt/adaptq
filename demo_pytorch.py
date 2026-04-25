import torch
import torch.nn.functional as F
import time
from adaptq import AdaptQAttention, Engine

def generate_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).bool()

def main():
    print("=== Pipeline Demo: PyTorch vs AdapTQ Attention ===")
    
    batch = 1
    heads = 8
    dim = 64 # Head dimension
    seq_len = 1000 # Total sequence length to simulate

    print(f"Configuration: Heads={heads}, Dim={dim}, Seq={seq_len}, Bits=4\n")

    # 1. Initialize AdapTQ via Torch Wrapper
    # Hybrid threshold set to 256 to force quantization path for validation
    engine = AdaptQAttention(dim=dim, heads=heads, bits=4)
    # Underlying engine modification for hybrid threshold
    engine.engine._ctx = __import__('adaptq_py').MHAContext(n_heads=heads, head_dim=dim, bits=4, capacity=2048, seed=42, v_mass=0.95, hybrid_thresh=256)

    # Pre-generate exact deterministic dummy inputs representing model activations
    torch.manual_seed(100)
    # Shape: (batch, heads, seq_len, dim)
    queries = torch.randn(batch, heads, seq_len, dim)
    keys = torch.randn(batch, heads, seq_len, dim)
    values = torch.randn(batch, heads, seq_len, dim)

    # 2. Torch Baseline Execution (Forced Iterative to match Generation)
    print("Running PyTorch Iterative Generation...")
    torch_latencies = []
    torch_outputs = []
    
    # Simulating standard iterative token generation
    for pos in range(seq_len):
        # Current token Q: (B, H, 1, D)
        q_pos = queries[:, :, pos:pos+1, :]
        # KV cache up to current pos: (B, H, seq, D)
        k_cache = keys[:, :, :pos+1, :]
        v_cache = values[:, :, :pos+1, :]
        
        t0 = time.perf_counter()
        # Exact scaled dot product
        out_fp = F.scaled_dot_product_attention(q_pos, k_cache, v_cache)
        torch_latencies.append((time.perf_counter() - t0) * 1e6)
        torch_outputs.append(out_fp.squeeze(2)) # -> (B, H, D)
        
    torch_out = torch.stack(torch_outputs, dim=2) # -> (B, H, seq_len, D)

    # 3. AdapTQ Execution
    print("Running AdapTQ Generation (Zero-copy Append)...")
    aq_latencies = []
    aq_outputs = []
    
    for pos in range(seq_len):
        # Extract slices representing current KV projections
        k_pos = keys[:, :, pos, :] # -> (B, H, D)
        v_pos = values[:, :, pos, :]
        q_pos = queries[:, :, pos, :]
        
        t0 = time.perf_counter()
        
        # Forward pass: internal mechanism appends KV automatically to flat-buffers 
        out_aq = engine(q_pos, k_pos, v_pos)
        
        aq_latencies.append((time.perf_counter() - t0) * 1e6)
        aq_outputs.append(out_aq) # -> (B, H, D)
        
    aq_out = torch.stack(aq_outputs, dim=2)

    # 4. Benchmarks & Comparisons
    print("\n--- Benchmark Overview ---")
    
    # Ignoring warmup behavior or hybrid threshold bounds
    stable_torch_p50 = torch.tensor(torch_latencies[256:]).median().item()
    stable_aq_p50 = torch.tensor(aq_latencies[256:]).median().item()
    
    print(f"PyTorch p50 Latency : {stable_torch_p50:.2f} µs")
    print(f"AdapTQ p50 Latency  : {stable_aq_p50:.2f} µs")
    print(f"Stable Speedup      : {stable_torch_p50 / stable_aq_p50:.2f}x\n")

    print(f"Memory (FP32 Cache) : {batch * heads * seq_len * dim * 2 * 4 / 1024**2:.2f} MB")
    print(f"Memory (AdapTQ)     : {engine.engine.kv_bytes / 1024**2:.2f} MB\n")

    # Calculate MSE and Cosine
    aq_flat = aq_out[:, :, 256:, :].flatten()
    fp_flat = torch_out[:, :, 256:, :].flatten()
    
    mse = F.mse_loss(aq_flat, fp_flat).item()
    cosine = F.cosine_similarity(aq_flat.unsqueeze(0), fp_flat.unsqueeze(0)).item()
    
    print(f"Quality Metrics (seq > 256, strictly Quantized layer):")
    print(f"  Mean Squared Error : {mse:.3e}")
    print(f"  Cosine Similarity  : {cosine:.4f}")

if __name__ == "__main__":
    main()

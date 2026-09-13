import numpy as np

from adaptq import AdapTQ


def main():
    print("--- AdapTQ Example ---")
    
    # 1. Initialize AdapTQ context
    n_heads = 2
    head_dim = 128
    engine = AdapTQ.create(n_heads=n_heads, head_dim=head_dim, bits=4, capacity=2048)
    
    print(f"Initialized AdapTQ engine with {engine.kv_bytes} bytes starting memory usage.\n")

    # 2. Append mock Keys / Values
    print("Streaming 100 tokens of K/V context...")
    for seq_pos in range(100):
        # Fake model outputs for a token
        k = np.random.randn(head_dim).astype(np.float32)
        v = np.random.randn(head_dim).astype(np.float32)
        
        # Append to head 0
        engine.append_kv(head=0, key=k, val=v, pos=seq_pos)

    print(f"Cache memory usage after 100 tokens: {engine.kv_bytes / 1024:.2f} KB\n")

    # 3. Compute attention
    print("Computing attention for new query...")
    query = np.random.randn(head_dim).astype(np.float32)
    output = engine.compute(head=0, query=query)

    print("\nAttention output (first 5 elements):")
    print(output[:5])
    print("\nDone!")

if __name__ == "__main__":
    main()

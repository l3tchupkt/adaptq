import numpy as np
from adaptq.core import Engine

def test_engine_asymmetric():
    dim = 64
    heads = 2
    
    # K=4, V=2
    engine = Engine(dim=dim, heads=heads, k_bits=4, v_bits=2, capacity=128, seed=42)
    
    k = np.random.randn(heads, dim).astype(np.float32)
    v = np.random.randn(heads, dim).astype(np.float32)
    q = np.random.randn(heads, dim).astype(np.float32)
    
    engine.append(k, v)
    out = engine.compute(q)
    print("K=4, V=2 OK:", out.shape)

    # K=16, V=2 (Layer 0 simulation)
    engine_l0 = Engine(dim=dim, heads=heads, k_bits=16, v_bits=2, capacity=128, seed=42)
    engine_l0.append(k, v)
    out_l0 = engine_l0.compute(q)
    print("K=16, V=2 OK:", out_l0.shape)
    
    # K=16, V=16
    engine_dense = Engine(dim=dim, heads=heads, k_bits=16, v_bits=16, capacity=128, seed=42)
    engine_dense.append(k, v)
    out_dense = engine_dense.compute(q)
    print("K=16, V=16 OK:", out_dense.shape)
    
if __name__ == "__main__":
    test_engine_asymmetric()

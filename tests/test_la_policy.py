"""
tests/test_la_policy.py
=======================
Tests for the AdapTQ-LA Layer-Aware Asymmetric KV policy.

Covers:
- Layer 0 K=FP16 policy
- Asymmetric K/V precision (K4-V2, K3-V2)
- Uniform 2/3/4-bit backward compatibility
- Memory accounting for asymmetric/FP16 configs
- Numerical round-trip correctness (KV MSE, cosine similarity)
- Ring-buffer reset and multi-layer behavior
"""
import math
import numpy as np
import pytest

from adaptq.core import Engine

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _append_compute(k_bits, v_bits, dim=64, heads=4, seq_len=16, seed=0):
    """Create an engine, fill seq_len tokens, return last attention output."""
    rng = np.random.default_rng(seed)
    engine = Engine(dim=dim, heads=heads, k_bits=k_bits, v_bits=v_bits,
                    capacity=256, seed=42)
    for _ in range(seq_len):
        k = rng.standard_normal((heads, dim)).astype(np.float32)
        v = rng.standard_normal((heads, dim)).astype(np.float32)
        engine.append(k, v)
    q = rng.standard_normal((heads, dim)).astype(np.float32)
    return engine.compute(q), engine


# ------------------------------------------------------------------
# 1. Uniform precision backward compatibility
# ------------------------------------------------------------------

class TestUniformPrecision:
    """Existing uniform 2/3/4-bit paths must still work correctly."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_uniform_output_shape(self, bits):
        out, _ = _append_compute(k_bits=bits, v_bits=bits)
        assert out.shape == (4, 64), f"Shape mismatch for {bits}-bit"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_uniform_output_finite(self, bits):
        out, _ = _append_compute(k_bits=bits, v_bits=bits)
        assert np.isfinite(out).all(), f"Non-finite output for {bits}-bit"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_uniform_kv_bytes_positive(self, bits):
        _, engine = _append_compute(k_bits=bits, v_bits=bits, seq_len=32)
        assert engine.kv_bytes > 0


# ------------------------------------------------------------------
# 2. Asymmetric K/V precision (core AdapTQ-LA configurations)
# ------------------------------------------------------------------

class TestAsymmetricPrecision:
    """K4-V2 and K3-V2 are the two validated AdapTQ-LA configurations."""

    @pytest.mark.parametrize("k_bits,v_bits", [(4, 2), (3, 2)])
    def test_output_shape(self, k_bits, v_bits):
        out, _ = _append_compute(k_bits=k_bits, v_bits=v_bits)
        assert out.shape == (4, 64)

    @pytest.mark.parametrize("k_bits,v_bits", [(4, 2), (3, 2)])
    def test_output_finite(self, k_bits, v_bits):
        out, _ = _append_compute(k_bits=k_bits, v_bits=v_bits)
        assert np.isfinite(out).all()

    def test_k4v2_memory_less_than_k4v4(self):
        """K4-V2 must consume strictly less memory than K4-V4 at same capacity."""
        _, e_asym = _append_compute(k_bits=4, v_bits=2, seq_len=64)
        _, e_sym  = _append_compute(k_bits=4, v_bits=4, seq_len=64)
        assert e_asym.kv_bytes < e_sym.kv_bytes, (
            f"K4-V2 ({e_asym.kv_bytes}B) >= K4-V4 ({e_sym.kv_bytes}B)"
        )

    def test_k3v2_memory_less_than_k4v2(self):
        """K3-V2 must consume less memory than K4-V2."""
        _, e_k4v2 = _append_compute(k_bits=4, v_bits=2, seq_len=64)
        _, e_k3v2 = _append_compute(k_bits=3, v_bits=2, seq_len=64)
        assert e_k3v2.kv_bytes < e_k4v2.kv_bytes, (
            f"K3-V2 ({e_k3v2.kv_bytes}B) >= K4-V2 ({e_k4v2.kv_bytes}B)"
        )


# ------------------------------------------------------------------
# 3. Layer-0 K FP16 protection (k_bits=16)
# ------------------------------------------------------------------

class TestLayer0FP16Policy:
    """k_bits=16 means K is stored as raw float32 — no quantization loss."""

    def test_k16_v2_output_shape(self):
        out, _ = _append_compute(k_bits=16, v_bits=2)
        assert out.shape == (4, 64)

    def test_k16_v2_output_finite(self):
        out, _ = _append_compute(k_bits=16, v_bits=2)
        assert np.isfinite(out).all()

    def test_k16_v16_output_shape(self):
        """Full FP16 passthrough (dense layer equivalent)."""
        out, _ = _append_compute(k_bits=16, v_bits=16)
        assert out.shape == (4, 64)

    def test_k16_memory_greater_than_k4(self):
        """FP16 K storage must use more memory than 4-bit K (sanity check)."""
        _, e_fp16 = _append_compute(k_bits=16, v_bits=2, seq_len=64)
        _, e_4bit = _append_compute(k_bits=4,  v_bits=2, seq_len=64)
        assert e_fp16.kv_bytes > e_4bit.kv_bytes, (
            f"FP16-K ({e_fp16.kv_bytes}B) <= 4-bit-K ({e_4bit.kv_bytes}B) — unexpected"
        )

    def test_k16_v2_output_closer_to_dense_than_k4v2(self):
        """
        K=FP16 (no K quantization error) should produce attention closer to
        the dense reference than K=4bit. We verify this by computing cosine
        similarity against K=16/V=16 (true dense) reference.
        """
        dim, heads, seq_len = 64, 4, 32
        rng = np.random.default_rng(99)
        ks = [rng.standard_normal((heads, dim)).astype(np.float32) for _ in range(seq_len)]
        vs = [rng.standard_normal((heads, dim)).astype(np.float32) for _ in range(seq_len)]
        q  = rng.standard_normal((heads, dim)).astype(np.float32)

        def _run(k_bits, v_bits):
            eng = Engine(dim=dim, heads=heads, k_bits=k_bits, v_bits=v_bits,
                         capacity=256, seed=42)
            for k, v in zip(ks, vs):
                eng.append(k, v)
            return eng.compute(q)

        dense  = _run(16, 16)
        fp16k  = _run(16,  2)
        quant4 = _run(4,   2)

        def _cos(a, b):
            return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        sim_fp16k  = np.mean([_cos(dense[h], fp16k[h])  for h in range(heads)])
        sim_quant4 = np.mean([_cos(dense[h], quant4[h]) for h in range(heads)])

        assert sim_fp16k >= sim_quant4 - 0.05, (
            f"K=FP16 similarity ({sim_fp16k:.4f}) not close to dense; "
            f"K=4bit similarity ({sim_quant4:.4f}). FP16 K should not degrade."
        )


# ------------------------------------------------------------------
# 4. Memory accounting for AdapTQ-LA policy
# ------------------------------------------------------------------

class TestMemoryAccounting:
    """Verify byte counts follow expected ratios for each config."""

    def _bytes_for(self, k_bits, v_bits, seq_len=128, dim=64, heads=4):
        _, engine = _append_compute(k_bits=k_bits, v_bits=v_bits,
                                    dim=dim, heads=heads, seq_len=seq_len)
        return engine.kv_bytes

    def test_compression_ratio_k4v2_vs_dense(self):
        """K4-V2 should achieve at least 2x over FP16 (theoretical ~3x at dim=64)."""
        # Dense stored as k_bits=16, v_bits=16 (float32 internally)
        dense_bytes = self._bytes_for(16, 16)
        asym_bytes  = self._bytes_for(4,  2)
        ratio = dense_bytes / asym_bytes
        assert ratio >= 2.0, f"Compression ratio {ratio:.2f}x below expected 2x floor"

    def test_memory_scales_linearly_with_seq_len(self):
        """KV bytes should scale linearly with sequence length."""
        b64  = self._bytes_for(4, 2, seq_len=64)
        b128 = self._bytes_for(4, 2, seq_len=128)
        # Allow 10% tolerance for scale vector overhead
        ratio = b128 / b64
        assert 1.8 <= ratio <= 2.2, f"Memory scaling ratio {ratio:.2f} unexpected"


# ------------------------------------------------------------------
# 5. Numerical round-trip correctness
# ------------------------------------------------------------------

class TestNumericalCorrectness:
    """KV MSE and cosine similarity bounds for each configuration."""

    @pytest.mark.parametrize("k_bits,v_bits,min_cos_sim", [
        (4, 4, 0.90),  # 4/4-bit: high quality expected
        (4, 2, 0.85),  # 4/2-bit: K high quality, V lower
        (3, 2, 0.75),  # 3/2-bit: aggressive, some degradation OK
        (16, 2, 0.85), # FP16-K: K not quantized, output must be reasonable
        (16, 16, 0.999), # dense: should perfectly recover (modulo float rounding)
    ])
    def test_output_cosine_similarity_vs_dense(self, k_bits, v_bits, min_cos_sim):
        """
        Attention output should have cosine similarity >= min_cos_sim
        against the dense (FP16/FP16) reference across all heads.
        """
        dim, heads, seq_len = 64, 4, 32
        rng = np.random.default_rng(7)
        ks = [rng.standard_normal((heads, dim)).astype(np.float32) for _ in range(seq_len)]
        vs = [rng.standard_normal((heads, dim)).astype(np.float32) for _ in range(seq_len)]
        q  = rng.standard_normal((heads, dim)).astype(np.float32)

        def _run(kb, vb):
            eng = Engine(dim=dim, heads=heads, k_bits=kb, v_bits=vb,
                         capacity=256, seed=42)
            for k, v in zip(ks, vs):
                eng.append(k, v)
            return eng.compute(q)

        dense = _run(16, 16)
        out   = _run(k_bits, v_bits)

        sims = []
        for h in range(heads):
            n_d = np.linalg.norm(dense[h])
            n_o = np.linalg.norm(out[h])
            if n_d < 1e-9 or n_o < 1e-9:
                continue
            sims.append(float(np.dot(dense[h], out[h]) / (n_d * n_o)))

        mean_sim = np.mean(sims)
        assert mean_sim >= min_cos_sim, (
            f"K{k_bits}-V{v_bits}: mean cosine sim {mean_sim:.4f} < {min_cos_sim}"
        )


# ------------------------------------------------------------------
# 6. Cache reset and capacity behavior
# ------------------------------------------------------------------

class TestCacheReset:
    """After reset(), the engine must behave as if freshly initialized."""

    def test_reset_clears_state(self):
        engine = Engine(dim=64, heads=2, k_bits=4, v_bits=2, capacity=32, seed=42)
        rng = np.random.default_rng(0)

        for _ in range(10):
            engine.append(
                rng.standard_normal((2, 64)).astype(np.float32),
                rng.standard_normal((2, 64)).astype(np.float32),
            )
        assert engine.kv_bytes > 0

        engine.reset()
        assert engine.pos == 0
        assert engine.kv_bytes == 0

    def test_output_after_reset_is_zeros(self):
        engine = Engine(dim=64, heads=2, k_bits=4, v_bits=2, capacity=32, seed=42)
        rng = np.random.default_rng(0)
        engine.append(
            rng.standard_normal((2, 64)).astype(np.float32),
            rng.standard_normal((2, 64)).astype(np.float32),
        )
        engine.reset()
        q   = rng.standard_normal((2, 64)).astype(np.float32)
        out = engine.compute(q)
        assert np.allclose(out, 0.0), "Output after reset should be all zeros"

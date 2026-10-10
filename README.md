<div align="center">
  <h1>AdapTQ</h1>
  <p><b>KV-Cache Quantization for CPU-Based LLM Inference</b></p>

  <a href="https://pypi.org/project/adaptq/">
    <img src="https://img.shields.io/pypi/v/adaptq?color=blue&label=PyPI" alt="PyPI version">
  </a>
  <a href="https://github.com/l3tchupkt/adaptq/actions/workflows/integration.yml">
    <img src="https://github.com/l3tchupkt/adaptq/actions/workflows/integration.yml/badge.svg" alt="Build Status">
  </a>
  <a href="https://pypi.org/project/adaptq/">
    <img src="https://img.shields.io/pypi/pyversions/adaptq" alt="Python Versions">
  </a>
  <a href="https://github.com/l3tchupkt/adaptq/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
</div>

<br>

**AdapTQ** is a CPU-oriented KV-cache quantization engine for Large Language Model inference. It implements Rademacher sign randomization combined with the Fast Walsh-Hadamard Transform (FWHT) to precondition KV vectors before scalar quantization, substantially reducing quantization distortion compared to direct scalar quantization.

**Current version: 0.2.3**

> **Research note:** AdapTQ-LA (Layer-Aware Asymmetric KV-Cache Quantization) is an active research configuration. See [Research Configuration](#research-configuration-adaptq-la) and the [companion research repository](https://github.com/l3tchupkt/adaptq-research) for validated experimental results.

---

## What AdapTQ Does

AdapTQ stores KV cache vectors in a quantized form using this pipeline per token:

1. **Rademacher sign randomization** — multiply each dimension by a fixed ±1 sign vector (sampled once per layer, stored as bytes)
2. **FWHT** — apply Fast Walsh-Hadamard Transform in O(d log d) to homogenize outliers
3. **Scalar quantization** — apply Max-Lloyd codebook quantization at 2, 3, or 4 bits
4. **Packed storage** — store bit-packed indices and per-vector scale factors in a flat ring buffer

During attention, queries undergo the same rotation; dot products are computed directly from packed indices via a centroid lookup table without full dequantization.

---

## Current Capabilities (v0.2.3)

- **Supported bit widths:** 2-bit, 3-bit, 4-bit quantization; `bits=16` for FP32 passthrough (unquantized)
- **Asymmetric K/V precision:** K and V can be quantized at independent bit widths
- **C++17 core** with AVX2 SIMD path (x86 only); scalar fallback for non-AVX2
- **Python bindings** via pybind11
- **Hybrid execution:** Short sequences below a configurable threshold use raw FP32 cache (no decode overhead)
- **Ring buffer:** Fixed capacity, O(1) append and eviction
- **Multi-backend Python runtime** for HuggingFace Transformers and llama-cpp-python

---

## Quick Start

### Installation

```bash
pip install adaptq
```

With backend dependencies:

```bash
pip install adaptq[transformers]     # HuggingFace support
pip install adaptq[llama]            # llama-cpp-python support
```

### Basic API

```python
import numpy as np
from adaptq import Engine

# Create a single-layer KV cache engine
# K at 4-bit, V at 2-bit — the AdapTQ-LA research configuration
engine = Engine(dim=64, heads=4, k_bits=4, v_bits=2, capacity=2048, seed=42)

# Append K and V for each generated token
k = np.random.randn(4, 64).astype(np.float32)  # shape: (heads, dim)
v = np.random.randn(4, 64).astype(np.float32)
engine.append(k, v)

# Compute attention output for the current query
q = np.random.randn(4, 64).astype(np.float32)
out = engine.compute(q)  # shape: (heads, dim)

# Check KV cache memory usage
print(f"KV bytes: {engine.kv_bytes}")

# Reset between sequences
engine.reset()
```

### PyTorch Integration

```python
import torch
from adaptq.torch_adapter import AdaptQAttention

# Wraps the C++ engine in a PyTorch nn.Module
layer = AdaptQAttention(dim=64, heads=4, k_bits=4, v_bits=2, capacity=2048)

q = torch.randn(1, 4, 64)  # (batch=1, heads, dim)
k = torch.randn(1, 4, 64)
v = torch.randn(1, 4, 64)
out = layer(q, k, v)        # (1, 4, 64)
```

**Constraints:** CPU only; batch size must be 1.

---

## Research Configuration: AdapTQ-LA

**AdapTQ-LA** (Layer-Aware Asymmetric KV-Cache Quantization) is a research configuration evaluated in the companion repository. It applies the following layer-aware policy to a multi-layer LLM:

| Layer | K precision | V precision |
| :--- | :--- | :--- |
| Layer 0 | FP16 (unquantized) | Configured precision |
| Layers 1+ | k_bits | v_bits |

**Validated configurations:** K4-V2 and K3-V2 with Layer-0 K FP16 protection.
**Evaluated models:** Qwen2-0.5B, TinyLlama-1.1B.
**Evaluated context lengths:** 512 and 1024 tokens.

This policy is implementable with the current API by creating one `Engine` per layer and using `k_bits=16` for Layer 0:

```python
# AdapTQ-LA: one Engine per layer
engines = []
for layer_idx in range(num_layers):
    k_bits = 16 if layer_idx == 0 else 4  # Layer 0: FP16 K passthrough
    v_bits = 2
    engines.append(Engine(dim=head_dim, heads=n_kv_heads,
                           k_bits=k_bits, v_bits=v_bits,
                           capacity=max_ctx, seed=42 ^ layer_idx))
```

### Evaluated Perplexity (Qwen2-0.5B, seed 2026, context 1024, next-token PPL)

| Configuration | PPL | KV Memory |
| :--- | :--- | :--- |
| Dense FP16 | 1.006 | 4.21 MB |
| Uniform 4-bit | 16.65 | 3.81 MB |
| Uniform 3-bit | 264.3 | 2.96 MB |
| **AdapTQ-LA K4-V2** | **1.026** | **3.40 MB** |
| **AdapTQ-LA K3-V2** | **1.241** | **3.00 MB** |

Source: `data/results/v2_evaluation_Qwen_Qwen2-0.5B.json` in the research repository.

### Evaluated Memory Compression (Production C++ core, dim=64, heads=14, context=512)

| Configuration | KV Bytes | vs Dense |
| :--- | :--- | :--- |
| Dense (FP16) | 3,670,016 | 1.0× |
| AdapTQ-LA K4-V2 | 344,064 | **~10.7×** |
| AdapTQ-LA K3-V2 | 286,720 | **~12.8×** |

> **Important:** Compression ratios depend on head dimension, context length, and the FP16 Layer-0 overhead. These numbers reflect the specific benchmark configuration above. Do not generalize without measuring your target configuration.

### Benchmark Limitations

- All kernel benchmarks are **isolated attention-step measurements** on CPU. No end-to-end generation speedup has been demonstrated.
- The Python HF integration path introduces framework overhead that prevents reliable E2E throughput measurement through the research simulator.
- Only two small models (0.5B, 1.1B) have been evaluated. Generalization to larger models is unknown.
- Only next-token perplexity has been measured. Downstream task accuracy (MMLU, GSM8K) has not been evaluated.

---

## Building from Source

```bash
git clone https://github.com/l3tchupkt/adaptq
cd adaptq

# Build C++ extension
pip install -e ".[dev]"

# Or build manually
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Run tests:

```bash
pytest tests/ -v
```

Run isolated benchmarks:

```bash
python -m benchmarks.benchmark_adaptq_la --dim 64 --heads 14 --seq 512
```

---

## Project Structure

```
core/          # FWHT, codebooks, quantizer
attention/     # Attention computation (AVX2 + scalar)
adaptq/        # Python package
  core.py      # Engine wrapper
  torch_adapter.py  # PyTorch AdaptQAttention
  runtime_py/  # Multi-backend Python runtimes
tests/         # Unit and integration tests
benchmarks/    # Isolated kernel benchmarks
adapters/      # pybind11 C++/Python bridge
```

---

## Research Repository

All experiments, raw JSON results, methodology documentation, and the scientific manuscript are in the companion repository:

[https://github.com/l3tchupkt/adaptq-research](https://github.com/l3tchupkt/adaptq-research)

---

## Citation

```bibtex
@article{adaptq_la_2026,
  author  = {Lakshmikanthan K.},
  title   = {{AdapTQ-LA}: Static Layer-Aware Asymmetric {KV}-Cache Quantization},
  year    = {2026},
  url     = {https://github.com/l3tchupkt/adaptq}
}
```

## License

MIT License. See [LICENSE](LICENSE).

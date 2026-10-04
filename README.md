<div align="center">
  <h1>AdapTQ</h1>
  <p><b>Adaptive Streaming Vector Quantization KV Cache for LLMs</b></p>

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
  <a href="https://github.com/l3tchupkt/adaptq">
    <img src="https://img.shields.io/badge/C++-17-blue.svg" alt="C++17">
  </a>
</div>

<br>

**AdapTQ** is an open-source C++17 KV cache quantization library for Large Language Model inference.

It runs entirely on the CPU and integrates with PyTorch and Hugging Face `transformers`. By leveraging Fast Walsh-Hadamard Transforms (FWHT) and AVX2 SIMD optimizations, AdapTQ dynamically quantizes and dequantizes Key-Value states into 2-bit, 3-bit, and 4-bit representations.

> **Research Note:**
> This repository contains the production library. For the reproducible empirical validation, raw benchmark datasets, generated figures, and the LaTeX research paper evaluating this library, please see our dedicated [AdapTQ-Research repository](https://github.com/l3tchupkt/adaptq-research).

## ✨ Key Features

- **Extreme Memory Compression**: Up to ~6.4× smaller KV cache footprints via 2-bit, 3-bit, and 4-bit Max-Lloyd quantization.
- **Unified SIMD Pipeline**: 2/3/4-bit decoding shares a single, quad-unrolled branchless loop using AVX2 intrinsics. No scalar fallbacks.
- **Hybrid Execution**: Automatically routes short sequences (≤ 256 tokens) to FP32 and long sequences to quantized SIMD, maximizing speed without data copying.
- **Deterministic Replay**: Save `.aqss` session snapshots to disk, branch conversations at any token, and replay states.

---

## 🚀 Quick Start

### 1. Installation

Install directly from PyPI (includes pre-built C++ extensions for Linux/Windows/macOS):

```bash
pip install adaptq
```

*To install with specific backend dependencies:*
```bash
pip install adaptq[transformers]   # For Hugging Face support
```

### 2. Hugging Face Transformers Integration

AdapTQ seamlessly injects itself into any standard `transformers` generation pipeline:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adaptq import create_adapter

model_id = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Wrap the model with AdapTQ (4-bit quantization, 2048 capacity)
adapter = create_adapter("transformers", model=model, bits=4, capacity=2048)

# Generate normally! The KV cache is now fully compressed and managed in C++.
inputs = tokenizer("The future of AI on edge devices is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## 📜 License

MIT License. See `LICENSE` for details.

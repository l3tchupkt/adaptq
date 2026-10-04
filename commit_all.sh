#!/bin/bash
git checkout -b feature/research-paper-evaluation

git add core/ include/ tests/unit/
git commit -m "Fix #229: Correct codebook symmetries and Rademacher seeds"

git add adaptq/research/packed_kv.py tests/test_packed_kv.py adaptq/research/packed_kv_pt.py scratch_fwht.py
git commit -m "Feature #230: Implement Python packed KV cache and PyTorch vectorized references"

git add adaptq/runtime_py/backends/transformers_hf.py tests/test_hf_integration_path.py test_generation.py test_logits.py
git commit -m "Feature #231: Implement Hugging Face exact KV interception"

git add adaptq/research/validation.py run_val.sh run_gpu_val.sh run_cpu_val.sh *.log *.json
git commit -m "Feature #232: Implement GPU-accelerated end-to-end statistical quality validation"

git add CMakeLists.txt benchmarks/ README.md research/README.md research/paper/
git commit -m "Feature #233: Overhaul benchmarking and author LaTeX paper"

echo "Branch feature/research-paper-evaluation created and commits staged."

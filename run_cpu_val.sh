#!/bin/bash
set -e

echo "=== RUNNING CPU MATHEMATICAL ORACLE VALIDATION ==="
export CUDA_VISIBLE_DEVICES=""

# Qwen2-0.5B
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m adaptq.research.validation --model Qwen/Qwen2-0.5B --lengths 128 1024 4096 --bits 2 3 4 --seeds 42 > qwen_cpu_oracle.log 2>&1

# TinyLlama-1.1B
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m adaptq.research.validation --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --lengths 128 1024 4096 --bits 2 3 4 --seeds 42 > tinyllama_cpu_oracle.log 2>&1

echo "CPU mathematical oracle complete"
